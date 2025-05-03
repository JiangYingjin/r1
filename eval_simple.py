from lib import *
from math_verify import parse, verify
import json
import datetime
from system_prompt import SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

model = "Qwen2.5-3B-Instruct"
gsm8k_test_path = Path("eval/datasets/gsm8k_test.jsonl")
out_dir = Path("eval")

with gsm8k_test_path.open("r") as f:
    # gsm8k_test_data = [json.loads(line) for line in f][:222]
    gsm8k_test_data = [json.loads(line) for line in f]


def get_gsm8k_final_answer(answer: str):
    return answer.split("####")[-1]


gsm8k_test_data = [
    {"question": d["question"], "answer": get_gsm8k_final_answer(d["answer"]).strip()}
    for d in gsm8k_test_data
]


# 打印读取的数据量
print(f"Loaded {len(gsm8k_test_data)} examples from GSM8K test set")
# print(gsm8k_test_data[:5])

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = out_dir / f"{timestamp}.jsonl"
print(f"Predict results will be saved to: {out_file}")

llm = LLM(model, base_url="http://127.0.0.1:23333/v1", key="sk-jiangyj")


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
