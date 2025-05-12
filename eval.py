from lib import *
from math_verify import parse, verify
import json
import datetime
import subprocess
import threading
import time
from system_prompt import SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import psutil
import argparse


# =================== argparse 解析命令行参数 ===================
def parse_args():
    parser = argparse.ArgumentParser(
        description="模型评测脚本 (Model evaluation script)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/llama-3.2-3b-instruct-bnb-4bit",
        help="模型名称 (model name)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="实验名称 (experiment name)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=300,
        help="评测步数 (evaluation step)",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="不使用 chat_template.json (Do not use chat_template.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式，输出详细日志 (Debug mode, print detailed logs)",
    )
    return parser.parse_args()


model_ckpt_dir = lambda model_name: Path(
    f"/root/lanyun-tmp/r1/exp/{model_name.replace('/','_')}/ckpt"
)

model_exp_ckpt_dir = lambda model_name, exp_name, step: Path(
    f"/root/lanyun-tmp/r1/exp/{model_name.replace('/','_')}/{exp_name}/eval"
)

model_exp_step_ckpt_dir = lambda model_name, exp_name, step: Path(
    f"/root/lanyun-tmp/r1/exp/{model_name.replace('/','_')}/{exp_name}/eval/checkpoint-{step}"
)

model_exp_step_ckpt_baidu_dir = lambda model_name, exp_name, step: Path(
    f"/share/proj/r1/exp/{model_name.replace('/','_')}/{exp_name}/ckpt/checkpoint-{step}"
)

model_exp_step_ckpt_merged_dir = lambda model_name, exp_name, step: Path(
    f"/root/lanyun-tmp/r1/exp/{model_name.replace('/','_')}/{exp_name}/eval/checkpoint-{step}_merged"
    if exp_name
    else f"/root/lanyun-tmp/r1/exp/{model_name.replace('/','_')}/ckpt"
)


gsm8k_test_path = Path("data/raw/gsm8k_test.jsonl")

llm_api_url = "http://127.0.0.1:23333/v1"
llm_api_key = "sk-jiangyj"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = Path("eval") / f"{timestamp}.jsonl"
out_file.parent.mkdir(exist_ok=True)


def download_ckpt_and_merge(model_name: str, exp_name: str, step: int):
    print(f"准备下载模型 ...")
    subprocess.run(
        f"curl sh.jyj.cx/baidu | bash -s - d {str(model_exp_step_ckpt_baidu_dir(model_name, exp_name, step))} {str(model_exp_ckpt_dir(model_name, exp_name, step))}",
        shell=True,
    )
    print(f"模型下载完毕，准备加载模型并合并 LoRA ...")

    from unsloth import FastLanguageModel

    print(f"Unsloth 加载完毕，开始加载模型 ...")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_exp_step_ckpt_dir(model_name, exp_name, step))
        )
        print(f"模型加载完毕，准备合并（使用 merged_16bit 格式） ...")

        model.save_pretrained_merged(
            str(model_exp_step_ckpt_merged_dir(model_name, exp_name, step)),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"模型合并完毕")

    except Exception as e:
        print(f"模型合并失败：{e}")

        # 如果是 TensorFlow 权重问题，尝试使用 from_tf=True 参数
        if "Use `from_tf=True`" in str(e):
            print(f"检测到 TensorFlow 权重问题，尝试使用 from_tf=True 参数重新加载...")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    str(model_exp_step_ckpt_dir(model_name, exp_name, step)),
                    from_tf=True,
                )
                print(f"使用 from_tf=True 参数加载成功，准备合并 ...")

                model.save_pretrained_merged(
                    str(model_exp_step_ckpt_merged_dir(model_name, exp_name, step)),
                    tokenizer,
                    save_method=(
                        "merged_16bit"
                        # "merged_4bit" if "phi" in model_name.lower() else "merged_16bit"
                    ),
                )
                print(f"模型合并完毕")
            except Exception as e2:
                print(f"使用 from_tf=True 参数后仍然合并失败：{e2}")


def get_chat_template(model_name: str):
    if "llama" in model_name.lower():
        return "chat_templates/llama.jinja"
    elif "phi" in model_name.lower():
        return "chat_templates/phi.jinja"
    else:
        return "chat_templates/qwen.jinja"


def deploy_model_lmdeploy(
    model_name: str,
    exp_name: str = None,
    step: int = None,
    no_chat_template: bool = False,
    debug: bool = False,
):
    """在后台运行 lmdeploy 服务器"""
    import os

    lmdeploy_path1 = "/root/miniconda/envs/lmdeploy/bin/lmdeploy"
    lmdeploy_path2 = "/usr/local/miniforge3/envs/lmdeploy/bin/lmdeploy"

    if os.path.exists(lmdeploy_path1):
        print(f"找到 lmdeploy 路径: {lmdeploy_path1}")
        lmdeploy_cmd = lmdeploy_path1
    elif os.path.exists(lmdeploy_path2):
        print(f"找到 lmdeploy 路径: {lmdeploy_path2}")
        lmdeploy_cmd = lmdeploy_path2
    else:
        print("未找到 lmdeploy，尝试创建环境并安装...")
        try:
            print(
                "执行: conda create -n lmdeploy python=3.10 && conda activate lmdeploy && pip install uv && uv pip install lmdeploy"
            )
            subprocess.run(
                "source ~/.zshrc && conda create -n lmdeploy python=3.10 -y && conda activate lmdeploy && pip install uv && uv pip install lmdeploy",
                shell=True,
                check=True,
                executable="/bin/zsh",
            )

            # 安装后再次检查路径
            if os.path.exists(lmdeploy_path1):
                print(f"安装成功，找到 lmdeploy 路径: {lmdeploy_path1}")
                lmdeploy_cmd = lmdeploy_path1
            elif os.path.exists(lmdeploy_path2):
                print(f"安装成功，找到 lmdeploy 路径: {lmdeploy_path2}")
                lmdeploy_cmd = lmdeploy_path2
            else:
                raise FileNotFoundError(
                    "安装后仍无法找到 lmdeploy 可执行文件，请手动检查安装情况"
                )
        except Exception as e:
            print(f"安装 lmdeploy 失败: {str(e)}")
            raise FileNotFoundError(f"无法找到或安装 lmdeploy: {str(e)}")

    print(f"启动 lmdeploy 服务器，使用模型: {model_name}")
    cmd = [
        lmdeploy_cmd,
        "serve",
        "api_server",
        (
            model_exp_step_ckpt_merged_dir(model_name, exp_name, step)
            if exp_name
            else model_ckpt_dir(model_name)
        ),
    ]
    if not no_chat_template:
        cmd += ["--chat-template", get_chat_template(model_name)]
    cmd += [
        "--model-name",
        model_name,
        "--api-keys",
        "sk-jiangyj",
        "--tp",
        "1",
    ]
    if debug:
        subprocess.run(cmd)
    else:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def deploy_model_sglang(
    model_name: str,
    exp_name: str = None,
    step: int = None,
    no_chat_template: bool = False,
    debug: bool = False,
):
    """在后台运行 sglang 服务器"""
    import os

    # 假设 sglang 安装在 conda 环境或全局
    sglang_path1 = "/root/miniconda/envs/sglang/bin/python"
    sglang_path2 = "/usr/local/miniforge3/envs/sglang/bin/python"

    if os.path.exists(sglang_path1):
        print(f"找到 sglang 路径: {sglang_path1}")
        sglang_cmd = sglang_path1
    elif os.path.exists(sglang_path2):
        print(f"找到 sglang 路径: {sglang_path2}")
        sglang_cmd = sglang_path2
    else:
        print("未找到 sglang，尝试创建环境并安装...")
        try:
            print(
                "执行: conda create -n sglang python=3.12 && conda activate sglang && pip install uv && uv pip install 'sglang[all]>=0.4.6.post3'"
            )
            subprocess.run(
                'source ~/.zshrc && conda create -n sglang python=3.12 -y && conda activate sglang && pip install uv && uv pip install "sglang[all]>=0.4.6.post3"',
                shell=True,
                check=True,
                executable="/bin/zsh",
            )
            if os.path.exists(sglang_path1):
                print(f"安装成功，找到 sglang 路径: {sglang_path1}")
                sglang_cmd = sglang_path1
            elif os.path.exists(sglang_path2):
                print(f"安装成功，找到 sglang 路径: {sglang_path2}")
                sglang_cmd = sglang_path2
            else:
                raise FileNotFoundError(
                    "安装后仍无法找到 sglang 可执行文件，请手动检查安装情况"
                )
        except Exception as e:
            print(f"安装 sglang 失败: {str(e)}")
            raise FileNotFoundError(f"无法找到或安装 sglang: {str(e)}")

    print(f"启动 sglang 服务器，使用模型: {model_name}")
    # sglang 启动命令
    # python -m sglang.launch_server --model-path <模型路径> [--tp 1] [--其他参数]
    model_path = (
        str(model_exp_step_ckpt_merged_dir(model_name, exp_name, step))
        if exp_name
        else str(model_ckpt_dir(model_name))
    )
    cmd = [
        sglang_cmd,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        "1",
        "--port",
        "23333",
        "--api-key",
        "sk-jiangyj",
    ]
    if not no_chat_template:
        cmd += ["--chat-template", get_chat_template(model_name)]
    if debug:
        subprocess.run(cmd)
    else:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def deploy_model_vllm(
    model_name: str,
    exp_name: str = None,
    step: int = None,
    no_chat_template: bool = False,
    debug: bool = False,
):
    """在后台运行 vllm 服务器"""
    import os

    vllm_path1 = "/usr/local/miniforge3/envs/vllm/bin/vllm"
    vllm_path2 = "/root/miniconda/envs/vllm/bin/vllm"

    if os.path.exists(vllm_path1):
        print(f"找到 vllm 路径: {vllm_path1}")
        vllm_cmd = vllm_path1
    elif os.path.exists(vllm_path2):
        print(f"找到 vllm 路径: {vllm_path2}")
        vllm_cmd = vllm_path2
    else:
        print("未找到 vllm，尝试创建环境并安装...")
        try:
            print(
                "执行: conda create -n vllm python=3.12 && conda activate vllm && pip install uv && uv pip install vllm"
            )
            subprocess.run(
                "source ~/.zshrc && conda create -n vllm python=3.12 -y && conda activate vllm && pip install uv && uv pip install vllm",
                shell=True,
                check=True,
                executable="/bin/zsh",
            )
            if os.path.exists(vllm_path1):
                print(f"安装成功，找到 vllm 路径: {vllm_path1}")
                vllm_cmd = vllm_path1
            elif os.path.exists(vllm_path2):
                print(f"安装成功，找到 vllm 路径: {vllm_path2}")
                vllm_cmd = vllm_path2
            else:
                raise FileNotFoundError(
                    "安装后仍无法找到 vllm 可执行文件，请手动检查安装情况"
                )
        except Exception as e:
            print(f"安装 vllm 失败: {str(e)}")
            raise FileNotFoundError(f"无法找到或安装 vllm: {str(e)}")

    print(f"启动 vllm 服务器，使用模型: {model_name}")
    model_path = (
        str(model_exp_step_ckpt_merged_dir(model_name, exp_name, step))
        if exp_name
        else str(model_ckpt_dir(model_name))
    )
    cmd = [
        vllm_cmd,
        "serve",
        model_path,
        "--served-model-name",
        model_name,
        "--port",
        "23333",
        "--api-key",
        "sk-jiangyj",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "4096",
    ]
    if not no_chat_template:
        cmd += ["--chat-template", get_chat_template(model_name)]
    if debug:
        subprocess.run(cmd)
    else:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def load_gsm8k_test_data():
    """加载并处理GSM8K测试数据集"""

    if not gsm8k_test_path.exists():
        print(f"数据文件不存在，下载中：{gsm8k_test_path}")
        import requests

        test_data_url = "https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/test.jsonl"
        resp = requests.get(
            test_data_url,
            proxies={
                "http": "http://127.0.0.1:30001",
                "https": "http://127.0.0.1:30001",
            },
        )
        # 确保响应成功
        if resp.status_code != 200:
            print(f"下载失败，状态码：{resp.status_code}")
            return

        # 确保目标目录存在
        gsm8k_test_path.parent.mkdir(parents=True, exist_ok=True)
        gsm8k_test_path.write_text(resp.text)
        print(f"数据文件下载完毕：{gsm8k_test_path}")

    with gsm8k_test_path.open("r") as f:
        gsm8k_test_data = [json.loads(line) for line in f]
    return [
        {"question": d["question"], "answer": d["answer"].split("####")[-1].strip()}
        for d in gsm8k_test_data
    ]


def get_llm_response(llm: LLM, question: str, answer: str):
    resp = llm.chat(
        question,
        # context=SYSTEM_PROMPT,
        silent=True,
    )
    result = {
        "response": resp,
        "question": question,
        "answer": answer,
    }
    with out_file.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def verify_and_calculate_accuracy(result_file: Path):
    # 统计正确答案的数量
    correct_count = 0
    total_count = 0

    # 读取输出文件并进行验证和统计
    with result_file.open("r") as f:
        for line in tqdm(f, desc="Verifying and calculating accuracy"):
            try:
                result = json.loads(line)
                total_count += 1

                parsed_resp = parse(result["response"])
                parsed_answer = parse(result["answer"])
                parsed_answer_2 = parse(result["answer"] + "%")

                is_correct = verify(parsed_answer, parsed_resp) or verify(
                    parsed_answer_2, parsed_resp
                )

                # 添加验证结果到字典
                result["parsed_response"] = str(parsed_resp)
                result["parsed_answer"] = str(parsed_answer)
                result["correct"] = is_correct

                if is_correct:
                    correct_count += 1

            except json.JSONDecodeError:
                print(f"无法解析JSON行: {line}")
            except Exception as e:
                print(f"验证或统计出错: {e}")

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"总题目数: {total_count}")
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    exp_name = args.exp_name
    step = args.step
    no_chat_template = args.no_chat_template
    debug = args.debug

    print("=" * 100)
    print(f"模型名称: {model_name}")
    print(f"实验名称: {exp_name}")
    print(f"评测步数: {step}")
    print(f"使用 Chat Template: {not no_chat_template}")
    print("=" * 100)

    # 检查合并后的ckpt目录是否存在，不存在则先下载并合并
    if not model_exp_step_ckpt_merged_dir(model_name, exp_name, step).exists():
        download_ckpt_and_merge(model_name, exp_name, step)

    # 判断是否用 vllm
    if "phi" in model_name.lower():
        threading.Thread(
            target=deploy_model_vllm,
            args=(model_name, exp_name, step, no_chat_template, debug),
            daemon=True,
        ).start()
        print("正在启动 vllm 服务器 ...")
    else:
        threading.Thread(
            target=deploy_model_lmdeploy,
            args=(model_name, exp_name, step, no_chat_template, debug),
            daemon=True,
        ).start()
        print("正在启动 lmdeploy 服务器 ...")
    time.sleep(5)

    gsm8k_test_data = load_gsm8k_test_data()
    print(f"从 GSM8K 测试集加载了 {len(gsm8k_test_data)} 个示例")
    print(f"预测结果将保存到： {out_file}")

    llm = LLM(model_name, base_url=llm_api_url, key=llm_api_key, timeout=25)

    # 循环检测 llm_api_url 是否可用，直到获得非 connection error 的响应
    print("正在检测 API 服务是否就绪...")
    max_retries = 100
    retry_interval = 2  # 秒

    for attempt in range(max_retries):
        try:
            # 尝试发送一个简单的请求来检测服务是否可用
            response = llm.chat("你好", silent=True)
            print(f"API 服务已就绪，收到响应: {response[:20]}...")
            break
        except Exception as e:
            print(f"API 服务未就绪，错误: {e}")
            time.sleep(retry_interval)

    else:
        print("警告: 达到最大重试次数，API 服务可能仍未就绪，但将继续执行后续操作")

    # 批量获取 LLM 响应并写入文件
    try:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(get_llm_response, llm, d["question"], d["answer"])
                for d in gsm8k_test_data
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Getting LLM responses"
            ):
                try:
                    # 设置超时时间为60秒
                    result = future.result()
                except TimeoutError:
                    print("任务超时，已跳过")
                except Exception as e:
                    print(f"任务执行出错: {e}")

    except KeyboardInterrupt:
        print("用户手动停止")

    # 验证和统计
    accuracy = verify_and_calculate_accuracy(out_file)
    print("=" * 100)
    print(f"模型名称: {model_name}")
    print(f"实验名称: {exp_name}")
    print(f"评测步数: {step}")
    print(f"使用 Chat Template: {not no_chat_template}")
    print(f"准确率: {accuracy:.2%}")
    print("=" * 100)

    # 获取所有包含 'lmdeploy' 或 'vllm' 的进程 pid，并用 kill -9 杀掉
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
            if "lmdeploy" in cmdline or "vllm" in cmdline:
                pid = proc.info["pid"]
                try:
                    subprocess.run(["kill", "-9", str(pid)])
                    print(
                        f"已杀死进程 pid: {pid}，命令行包含: {'lmdeploy' if 'lmdeploy' in cmdline else 'vllm'}"
                    )
                except Exception as e:
                    print(f"杀死进程 {pid} 失败: {e}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
