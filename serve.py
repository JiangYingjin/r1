import subprocess

if __name__ == "__main__":
    subprocess.run(
        [
            "/root/miniconda/envs/lmdeploy/bin/lmdeploy",
            "serve",
            "api_server",
            "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/ckpt",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/better_reward_3/ckpt/checkpoint-100_merged",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/better_reward_3/ckpt/checkpoint-200_merged",
            # "/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/better_reward_3/ckpt/checkpoint-300_merged",
            "--chat-template",
            "chat_template.json",
            "--model-name",
            "Qwen2.5-3B-Instruct",
            "--api-keys",
            "sk-jiangyj",
            "--tp",
            "1",
        ]
    )
