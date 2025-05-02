import torch
import yaml
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from rewards import correctness_reward, format_reward, length_reward
from data_preparation import get_gsm8k_questions
from system_prompt import SYSTEM_PROMPT


def load_config(config_path="configs/train_config.yml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config()

    exp_name = config["experiment"]["name"]
    model_name = config["experiment"]["model"]
    output_root = config["paths"]["output_root"]
    ckpt_out_dir = f"{output_root}/exp/{model_name.replace('/','_')}/{exp_name}/ckpt"

    # 初始化 wandb
    wandb.init(
        project="r1",
        name=exp_name,
        notes=config["experiment"]["description"],
        config=config,
    )

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["sequence"]["max_prompt_length"]
        + config["sequence"]["max_completion_length"],
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=config["lora"]["rank"],
        gpu_memory_utilization=config["gpu"]["memory_utilization"],
        dtype=torch.bfloat16,
    )

    # peft model
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["rank"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=config["lora"]["rank"],
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=config["random_state"],
    )

    dataset = get_gsm8k_questions()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=GRPOConfig(
            use_vllm=True,  # use vLLM for fast inference!
            learning_rate=5e-6,  # 通常建议尝试使用 2e-4、1e-4、5e-5、2e-5 等数值
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,  # Increase to 4 for smoother training；相当于将批次大小增加到自身大小以上，但不会影响内存消耗！如果您想要更平滑的训练损失曲线，我们通常建议增加此值。
            num_generations=config["training"]["grpo_num_generations"],
            max_prompt_length=config["sequence"]["max_prompt_length"],
            max_completion_length=config["sequence"]["max_completion_length"],
            max_steps=config["training"]["total_steps"],
            save_steps=config["training"]["save_steps"],
            max_grad_norm=0.1,
            output_dir=ckpt_out_dir,
            report_to="wandb",
        ),
        reward_funcs=[
            format_reward.format_reward_func,
            correctness_reward.calculate_math_correctness_reward,
            length_reward.calculate_length_reward,
        ],
    )
    trainer.train()


if __name__ == "__main__":
    main()
