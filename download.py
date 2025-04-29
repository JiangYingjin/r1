import torch
from unsloth import FastLanguageModel
from data_preparation import get_gsm8k_questions


def download_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    return model, tokenizer


if __name__ == "__main__":
    models_to_download = [
        "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
    ]
    for model_name in models_to_download:
        model, tokenizer = download_model(model_name)

    questions = get_gsm8k_questions()
