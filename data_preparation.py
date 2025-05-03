from datasets import load_dataset, Dataset
from system_prompt import SYSTEM_PROMPT


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
            "answer": x["answer"], # 用于在奖励函数中获取奖励
            # "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore
