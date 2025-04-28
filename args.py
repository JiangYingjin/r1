import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, required=True, help="实验名称（一般为实验运行时间）"
    )
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--total_steps", type=int, default=4000, help="总训练步数")
    parser.add_argument(
        "--save_steps", type=int, default=200, help="保存检查点的步数间隔"
    )
    parser.add_argument(
        "--grpo_num_generations", type=int, default=8, help="GRPO组内生成数量"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA秩，更大的秩 = 更智能，但更慢，建议选择 8, 16, 32, 64, 128",
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=512, help="最大提示长度"
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=2560, help="最大完成长度"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU内存利用率，如果内存不足可以降低此值",
    )
    parser.add_argument("--random_state", type=int, default=3407, help="随机种子")
    return parser.parse_args()
