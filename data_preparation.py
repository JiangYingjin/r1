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


def get_gsmplus600_questions(split="train") -> Dataset:
    """加载并组织 gsmplus_600 数据集"""
    # 注意：gsmplus_600.jsonl 是一个本地文件，使用 json 格式加载
    data = load_dataset("json", data_files="data/processed/gsmplus_600.jsonl")[split]  # type: ignore

    data = data.map(
        lambda x: {  # type: ignore
            "id": x["id"],  # 保留原始ID
            "question": x["question"],  # 保留原始问题
            "answer": x["answer"],  # 用于在奖励函数中获取奖励
            "difficulty": x["difficulty"],  # 保留难度信息
            "prompt": [  # 必须包含列 prompt！
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
        }
    )  # type: ignore
    return data  # type: ignore


if __name__ == "__main__":
    # math_dataset = get_math_questions()
    # print("加载并转换后的数学数据集样本:")
    # for i in range(min(3, len(math_dataset))):
    #     print(f"样本 {i+1}:")
    #     print(f"  Prompt: {math_dataset[i]['prompt']}")
    #     print(f"  Answer: {math_dataset[i]['answer']}")
    #     print("-" * 20)

    # print("\n" + "=" * 40 + "\n")

    gsmplus600_dataset = get_gsmplus600_questions()
    print("加载并转换后的gsmplus_600数据集样本:")
    for i in range(min(3, len(gsmplus600_dataset))):
        print(f"样本 {i+1}:")
        print(f"  ID: {gsmplus600_dataset[i]['id']}")
        print(f"  Question: {gsmplus600_dataset[i]['question']}")
        print(f"  Answer: {gsmplus600_dataset[i]['answer']}")
        print(f"  Difficulty: {gsmplus600_dataset[i]['difficulty']}")
        print(f"  Prompt: {gsmplus600_dataset[i]['prompt']}")
        print("-" * 20)
