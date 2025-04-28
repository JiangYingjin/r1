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
from args import get_args


def main():
    args = get_args()

    exp_name = args.exp_name
    model_name = args.model_name
    ckpt_out_dir = (
        f"/root/lanyun-tmp/r1/exp/{model_name.replace('/','_')}/{exp_name}/ckpt"
    )

    max_seq_length = args.max_prompt_length + args.max_completion_length

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,  # Reduce if out of memory
        dtype=torch.bfloat16,
    )

    # peft model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=args.random_state,
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
            num_generations=args.grpo_num_generations,  # Decrease if out of memory
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            # num_train_epochs=1,  # Set to 1 for a full training run
            max_steps=args.total_steps,
            save_steps=args.save_steps,
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


if __name__ == "__main__":
    main()
