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

model = "Qwen2.5-3B-Instruct"
gsm8k_test_path = Path("data/raw/gsm8k_test.jsonl")
out_dir = Path("eval")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = out_dir / f"{timestamp}.jsonl"


llm = LLM(model, base_url="http://127.0.0.1:23333/v1", key="sk-jiangyj")


def load_gsm8k_test_data():
    """加载并处理GSM8K测试数据集"""
    with gsm8k_test_path.open("r") as f:
        gsm8k_test_data = [json.loads(line) for line in f]
    return [
        {"question": d["question"], "answer": d["answer"].split("####")[-1].strip()}
        for d in gsm8k_test_data
    ]


def run_lmdeploy_server():
    """在后台运行 lmdeploy 服务器"""
    subprocess.run(
        [
            "/root/miniconda/envs/lmdeploy/bin/lmdeploy",
            "serve",
            "api_server",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/ckpt",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/better_reward_3/ckpt/checkpoint-100_merged",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/better_reward_3/ckpt/checkpoint-200_merged",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/better_reward_3/ckpt/checkpoint-300_merged",
            "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/gsmplus600_course_1/ckpt/checkpoint-200_merged",
            "--chat-template",
            "chat_template.json",
            "--model-name",
            "Qwen2.5-3B-Instruct",
            "--api-keys",
            "sk-jiangyj",
            "--tp",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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

    # # 打开错误结果文件
    # wrong_file_path = result_file.with_name(f"{result_file.stem}_wrong.jsonl")
    # wrong_file = wrong_file_path.open("w", encoding="utf-8")

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

                # 重新写入文件（可选，如果需要更新文件内容）
                # with open(result_file, "r+") as f_write:
                #     lines = f_write.readlines()
                #     f_write.seek(0)
                #     lines[total_count - 1] = json.dumps(result, ensure_ascii=False) + "\n"
                #     f_write.writelines(lines)

                if is_correct:
                    correct_count += 1
                else:
                    # 将错误结果写入文件
                    # wrong_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    pass

            except json.JSONDecodeError:
                print(f"无法解析JSON行: {line}")
            except Exception as e:
                print(f"验证或统计出错: {e}")

    # # 关闭错误结果文件
    # wrong_file.close()

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0

    # 打印统计结果
    print(f"总题目数: {total_count}")
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.2%}")

    return accuracy


if __name__ == "__main__":
    # 在后台线程中启动 lmdeploy 服务器
    server_thread = threading.Thread(target=run_lmdeploy_server, daemon=True)
    server_thread.start()
    # 等待一段时间，确保服务器有足够时间启动
    print("正在启动 lmdeploy 服务器，请稍候...")

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

    # accuracy = verify_and_calculate_accuracy(Path("eval/20250503_232504.jsonl"))
