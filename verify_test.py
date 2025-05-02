import jsonlines
from pathlib import Path
from pprint import pp


evalscope_math_pred_path = Path(
    "eval/qwen2.5_3b_instruct/20250502_092243/predictions/Qwen2.5-3B-Instruct/math.jsonl"
)

evalscope_math_pred = []

with jsonlines.open(evalscope_math_pred_path, "r") as f:
    for data in f:
        evalscope_math_pred.append(data)

math_pred_processed = []
for item in evalscope_math_pred:
    math_pred_processed.append(
        {
            "pred": item["choices"][0]["message"]["content"],
            "solution": item["raw_input"]["solution"],
            "gold": item["raw_input"]["answer"],
        }
    )

from math_verify import parse, verify

# 统计通过率
passed_count = 0
total_count = len(math_pred_processed)

for item in math_pred_processed:
    parsed_pred = parse(item["pred"])
    parsed_solution = parse(item["solution"])
    parsed_gold = parse(item["gold"])
    passed = verify(parsed_pred, parsed_solution) or verify(parsed_pred, parsed_gold)
    if passed:
        passed_count += 1

print(f"\n通过率: {passed_count}/{total_count} = {passed_count/total_count:.2%}\n")

# 随机采样10条打印
import random

samples = random.sample(math_pred_processed, min(10, total_count))

print("随机采样10条示例:")
print("-" * 80)
for item in samples:
    parsed_pred = parse(item["pred"])
    parsed_solution = parse(item["solution"])
    passed = verify(parsed_pred, parsed_solution, strict=False) or verify(
        parsed_pred, parse(item["gold"]), strict=False
    )
    print(f"原始答案: {item['pred']}")
    print(f"标准答案: {item['solution']}")
    print(f"解析后的预测: {parsed_pred}")
    print(f"解析后的答案: {parsed_solution}")
    print(f"是否通过: {passed}")
    print("-" * 80)
