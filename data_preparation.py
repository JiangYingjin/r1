from datasets import load_dataset, Dataset
from system_prompt import SYSTEM_PROMPT
from math_verify import parse, verify


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore

    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [  # 必须包含列 prompt！
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],  # 用于在奖励函数中获取奖励
            # "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


def get_math_questions() -> Dataset:
    data = load_dataset("json", data_files="eval/datasets/math_500.jsonl")["train"]  # type: ignore

    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [  # 必须包含列 prompt！
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],  # 用于在奖励函数中获取奖励
        }
    )  # type: ignore
    return data


if __name__ == "__main__":
    math_dataset = get_math_questions()
    print("加载并转换后的数学数据集样本:")
    for i in range(min(3, len(math_dataset))):
        print(f"样本 {i+1}:")
        print(f"  Prompt: {math_dataset[i]['prompt']}")
        print(f"  Answer: {math_dataset[i]['answer']}")
        print("-" * 20)
