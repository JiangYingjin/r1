from datasets import load_dataset, Dataset, IterableDataset
from system_prompt import SYSTEM_PROMPT
from math_verify import parse, verify


# class MapToIterableDataset(IterableDataset):
#     def __init__(self, dataset: Dataset):
#         self.dataset = dataset
#         self._epoch = 0

#     def __iter__(self):
#         for i in range(len(self.dataset)):
#             yield self.dataset[i]

#     # def __len__(self):
#     #     return len(self.dataset)


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


def get_gsmplus600_questions(split="train") -> IterableDataset:
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
    )

    # TRL==0.17.0 之前没有控制数据集 shuffle 的入参，推测可以通过 IterableDataset 来避免数据集被 shuffle
    # return MapToIterableDataset(data)
    return data


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
    count = 0
    for sample in gsmplus600_dataset:
        if count >= 3:
            break
        print(f"样本 {count+1}:")
        print(f"  ID: {sample['id']}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Difficulty: {sample['difficulty']}")
        print(f"  Prompt: {sample['prompt']}")
        print("-" * 20)
        count += 1
