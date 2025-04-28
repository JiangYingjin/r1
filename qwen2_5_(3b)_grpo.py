import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from reward_fn import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)
from data_preparation import get_gsm8k_questions
import datetime


model_name = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"

total_steps = 4000
save_steps = 200

grpo_num_generations = 8

lora_rank = 64  # Larger rank = smarter, but slower, Choose any number > 0 ! Suggested 8, 16, 32, 64, 128

max_seq_length = 3072  # Can increase for longer reasoning traces
max_completion_length = 512
max_prompt_length = max_seq_length - max_completion_length

gpu_memory_utilization = 0.9  # Reduce if out of memory

random_state = 3407  # seed

exp_run_timestamp = datetime.datetime.now().strftime("%d_%H%M%S")
ckpt_out_dir = (
    f"/root/lanyun-tmp/{model_name.replace('/','_')}/exp/{exp_run_timestamp}/ckpt"
)


# model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=gpu_memory_utilization,  # Reduce if out of memory
    dtype=torch.bfloat16,
)

# peft model
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=random_state,
)


dataset = get_gsm8k_questions()


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=5e-6,
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
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=grpo_num_generations,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=total_steps,
        save_steps=save_steps,
        max_grad_norm=0.1,
        output_dir=ckpt_out_dir,
        report_to="wandb",  # Can use Weights & Biases
    ),
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
)
trainer.train()
