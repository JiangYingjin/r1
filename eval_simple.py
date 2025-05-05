from lib import *
from math_verify import parse, verify
import json
import datetime
import subprocess
import threading
import time
from system_prompt import SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

model_name = "Qwen/Qwen2.5-3B-Instruct"

# exp_name = "gsmplus600_course_1"
exp_name = "course_2"
step = 180

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
)


gsm8k_test_path = Path("data/raw/gsm8k_test.jsonl")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = Path("eval") / f"{timestamp}.jsonl"

llm = LLM(model_name, base_url="http://127.0.0.1:23333/v1", key="sk-jiangyj")


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
            str(model_exp_step_ckpt_dir(model_name, exp_name, step)),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"模型合并完毕")

    except Exception as e:
        print(f"模型合并失败：{e}")


def deploy_model_lmdeploy(model_name: str, exp_name: str = None, step: int = None):
    """在后台运行 lmdeploy 服务器"""
    subprocess.run(
        [
            "/root/miniconda/envs/lmdeploy/bin/lmdeploy",
            "serve",
            "api_server",
            (
                model_exp_step_ckpt_merged_dir(model_name, exp_name, step)
                if exp_name
                else model_ckpt_dir(model_name)
            ),
            "--chat-template",
            "chat_template.json",
            "--model-name",
            model_name,
            "--api-keys",
            "sk-jiangyj",
            "--tp",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_gsm8k_test_data():
    """加载并处理GSM8K测试数据集"""
    with gsm8k_test_path.open("r") as f:
        gsm8k_test_data = [json.loads(line) for line in f]
    return [
        {"question": d["question"], "answer": d["answer"].split("####")[-1].strip()}
        for d in gsm8k_test_data
    ]


def get_llm_response(question: str, answer: str):
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
    # 检查合并后的ckpt目录是否存在，不存在则先下载并合并
    if not model_exp_step_ckpt_merged_dir(model_name, exp_name, step).exists():
        download_ckpt_and_merge(model_name, exp_name, step)

    # 启动lmdeploy服务器，传递必要参数
    threading.Thread(
        target=deploy_model_lmdeploy, args=(model_name, exp_name, step), daemon=True
    ).start()
    print("正在启动 lmdeploy 服务器 ...")

    gsm8k_test_data = load_gsm8k_test_data()
    print(f"Loaded {len(gsm8k_test_data)} examples from GSM8K test set")
    print(f"Predict results will be saved to: {out_file}")

    # 批量获取 LLM 响应并写入文件
    try:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(get_llm_response, d["question"], d["answer"])
                for d in gsm8k_test_data
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Getting LLM responses"
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"任务执行出错: {e}")

    except KeyboardInterrupt:
        print("用户手动停止")

    # 验证和统计
    accuracy = verify_and_calculate_accuracy(out_file)
