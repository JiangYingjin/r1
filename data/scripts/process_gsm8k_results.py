import json
from pathlib import Path
from math_verify import parse, verify

from tqdm import tqdm

# 输入文件路径
input_file = Path("eval/datasets/gsm8k_math_resp.jsonl")
# 确保输入文件存在
if not input_file.exists():
    raise FileNotFoundError(f"输入文件不存在: {input_file}")

# 初始化结果字典
results = {}

# 从JSONL文件中读取数据
with input_file.open("r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing lines"):
        try:
            # 解析JSON行
            data = json.loads(line)

            # 提取字段
            question_id = data.get("id")
            question = data.get("question")
            response = data.get("response")
            answer = data.get("answer")

            # 如果id不存在于结果字典中，则创建新条目
            if question_id not in results:
                # 解析answer并存储
                parsed_answer = parse(answer)

                if not parsed_answer:
                    print(f"解析answer失败 (ID: {question_id}): {answer}")
                    continue

                # 创建新的字典条目
                results[question_id] = {
                    "id": question_id,
                    "question": question,
                    "answer": answer,
                    "response": [],
                    "parsed_response": [],
                    "parsed_answer": str(parsed_answer),
                    "parsed_answer_math_verify": parsed_answer,  # 存储解析后的对象，而不是字符串
                    "pass": 0,
                    "attempts": 0,
                }

            parsed_response = parse(response)

            # 将response添加到列表
            results[question_id]["response"].append(response)
            results[question_id]["parsed_response"].append(str(parsed_response))

            # 验证response是否与answer一致
            if verify(
                results[question_id]["parsed_answer_math_verify"],
                parsed_response,
            ):
                results[question_id]["pass"] += 1

            # 增加attempts计数
            results[question_id]["attempts"] += 1

        except json.JSONDecodeError:
            print(f"无法解析JSON行: {line[:50]}...")
        except Exception as e:
            print(f"处理数据时出错: {e}")

# 输出统计信息
total_ids = len(results)
total_attempts = sum(item["attempts"] for item in results.values())
total_passes = sum(item["pass"] for item in results.values())

print(f"总共处理了 {total_ids} 个不同的ID")
print(f"总尝试次数: {total_attempts}")
print(f"总通过次数: {total_passes}")
print(f"通过率: {total_passes / total_attempts:.2%}")

# 输出每个ID的统计信息
print("\n每个ID的统计信息:")
for question_id, data in sorted(results.items()):
    pass_rate = data["pass"] / data["attempts"] if data["attempts"] > 0 else 0
    print(
        f"ID: {question_id}, 通过: {data['pass']}/{data['attempts']} ({pass_rate:.2%})"
    )

# 准备用于保存的结果（删除不可序列化的对象）
serializable_results = {}
for question_id, data in results.items():
    serializable_data = data.copy()
    # 删除不可序列化的对象
    if "parsed_answer_math_verify" in serializable_data:
        del serializable_data["parsed_answer_math_verify"]
    serializable_results[question_id] = serializable_data

# 将结果保存到文件 (JSONL格式，按ID升序)
output_file = Path("eval/datasets/gsm8k_math_resp_analysis.jsonl")
# 获取结果列表并按ID排序
sorted_results = sorted(serializable_results.values(), key=lambda x: x["id"])
with output_file.open("w", encoding="utf-8") as f:
    for result in sorted_results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\n结果已保存到: {output_file}")
