import wandb
from pathlib import Path
import json

api = wandb.Api()

# Run 的信息
entity_name = "J.Ac"  #  W&B 用户名或团队名
project_name = "r1"  # 项目名称

# Run 的唯一 ID
run_ids = {
    "better_reward_3": "mrg9gnub",
    "better_reward_qwen2.5_1.5b": "crkuw5bh",
    "better_reward_llama": "ha2nhs99",
    "gsmplus600_course_1": "jaevmcxh",
    "course_2": "rxq1u1xy",
    "20250427_181546": "zr5cm18h",
}

# 确保保存目录存在
save_dir = Path("plots/data/wandb")
save_dir.mkdir(parents=True, exist_ok=True)

# 获取指定的 Run 对象并下载所有指标
for run_key, run_id in run_ids.items():
    try:
        run = api.run(f"{entity_name}/{project_name}/{run_id}")
        print(f"成功获取 Run: {run.name}")
    except Exception as e:
        print(f"获取 Run 失败: {e}")
        continue

    # 获取所有历史数据
    try:
        history = run.scan_history()
        metrics = [dict(row) for row in history]  # 保留所有字段

        # 保存为 JSON 格式
        save_path = save_dir / f"{run_key}.json"
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"数据已保存到 {save_path}")
    except Exception as e:
        print(f"获取历史数据失败: {e}")
