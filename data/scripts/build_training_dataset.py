#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
构建训练数据集脚本

该脚本根据采样数据构建训练数据集，将样本分成三个训练阶段：
- 阶段1（前100个样本）：75简单、25中等
- 阶段2（第101~400个样本）：45简单、195中等、60困难
- 阶段3（第401~600个样本）：20简单、80中等、100困难

每个阶段内的样本会随机打乱，最终生成一个jsonl格式的数据集文件。
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any


def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data, file_path):
    """保存JSON文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl_file(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl_file(data, file_path):
    """保存JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    # 定义文件路径
    base_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    simple_file = base_dir / "gsm8k_sampled_简单.json"
    medium_file = base_dir / "gsm8k_sampled_中等.json"
    hard_file = base_dir / "gsm8k_sampled_困难.json"
    raw_data_file = raw_dir / "gsm8k_math_id.jsonl"
    output_json_file = base_dir / "gsm8k_training_dataset.json"
    output_jsonl_file = base_dir / "gsmplus_600.jsonl"

    # 加载采样数据
    simple_data = load_json_file(simple_file)
    medium_data = load_json_file(medium_file)
    hard_data = load_json_file(hard_file)

    # 获取样本ID列表
    simple_ids = simple_data["ids"]
    medium_ids = medium_data["ids"]
    hard_ids = hard_data["ids"]

    # 验证样本数量
    print(f"简单样本数量: {len(simple_ids)}")
    print(f"中等样本数量: {len(medium_ids)}")
    print(f"困难样本数量: {len(hard_ids)}")

    # 定义各阶段所需的样本数量
    stage_requirements = {
        "stage1": {"simple": 75, "medium": 25, "hard": 0},
        "stage2": {"simple": 45, "medium": 195, "hard": 60},
        "stage3": {"simple": 20, "medium": 80, "hard": 100},
    }

    # 初始化各阶段的样本ID列表
    stage1_ids = []
    stage2_ids = []
    stage3_ids = []

    # 分配简单样本
    simple_start_idx = 0

    # 阶段1的简单样本
    stage1_simple_count = stage_requirements["stage1"]["simple"]
    stage1_simple_ids = simple_ids[simple_start_idx : simple_start_idx + stage1_simple_count]
    stage1_ids.extend([{"id": id, "difficulty": "简单"} for id in stage1_simple_ids])
    simple_start_idx += stage1_simple_count

    # 阶段2的简单样本
    stage2_simple_count = stage_requirements["stage2"]["simple"]
    stage2_simple_ids = simple_ids[simple_start_idx : simple_start_idx + stage2_simple_count]
    stage2_ids.extend([{"id": id, "difficulty": "简单"} for id in stage2_simple_ids])
    simple_start_idx += stage2_simple_count

    # 阶段3的简单样本
    stage3_simple_count = stage_requirements["stage3"]["simple"]
    stage3_simple_ids = simple_ids[simple_start_idx : simple_start_idx + stage3_simple_count]
    stage3_ids.extend([{"id": id, "difficulty": "简单"} for id in stage3_simple_ids])

    # 分配中等样本
    medium_start_idx = 0

    # 阶段1的中等样本
    stage1_medium_count = stage_requirements["stage1"]["medium"]
    stage1_medium_ids = medium_ids[medium_start_idx : medium_start_idx + stage1_medium_count]
    stage1_ids.extend([{"id": id, "difficulty": "中等"} for id in stage1_medium_ids])
    medium_start_idx += stage1_medium_count

    # 阶段2的中等样本
    stage2_medium_count = stage_requirements["stage2"]["medium"]
    stage2_medium_ids = medium_ids[medium_start_idx : medium_start_idx + stage2_medium_count]
    stage2_ids.extend([{"id": id, "difficulty": "中等"} for id in stage2_medium_ids])
    medium_start_idx += stage2_medium_count

    # 阶段3的中等样本
    stage3_medium_count = stage_requirements["stage3"]["medium"]
    stage3_medium_ids = medium_ids[medium_start_idx : medium_start_idx + stage3_medium_count]
    stage3_ids.extend([{"id": id, "difficulty": "中等"} for id in stage3_medium_ids])

    # 分配困难样本
    hard_start_idx = 0

    # 阶段2的困难样本
    stage2_hard_count = stage_requirements["stage2"]["hard"]
    stage2_hard_ids = hard_ids[hard_start_idx : hard_start_idx + stage2_hard_count]
    stage2_ids.extend([{"id": id, "difficulty": "困难"} for id in stage2_hard_ids])
    hard_start_idx += stage2_hard_count

    # 阶段3的困难样本
    stage3_hard_count = stage_requirements["stage3"]["hard"]
    stage3_hard_ids = hard_ids[hard_start_idx : hard_start_idx + stage3_hard_count]
    stage3_ids.extend([{"id": id, "difficulty": "困难"} for id in stage3_hard_ids])

    # 随机打乱各阶段内的样本顺序
    random.seed(42)  # 设置随机种子，确保结果可复现
    random.shuffle(stage1_ids)
    random.shuffle(stage2_ids)
    random.shuffle(stage3_ids)

    # 验证各阶段样本数量
    print(f"阶段1样本数量: {len(stage1_ids)}")
    print(f"阶段2样本数量: {len(stage2_ids)}")
    print(f"阶段3样本数量: {len(stage3_ids)}")

    # 构建训练数据集
    training_dataset = {
        "total_samples": len(stage1_ids) + len(stage2_ids) + len(stage3_ids),
        "stages": {
            "stage1": {"sample_count": len(stage1_ids), "samples": stage1_ids},
            "stage2": {"sample_count": len(stage2_ids), "samples": stage2_ids},
            "stage3": {"sample_count": len(stage3_ids), "samples": stage3_ids},
        },
    }

    # 保存训练数据集
    save_json_file(training_dataset, output_json_file)
    print(f"训练数据集已保存至: {output_json_file}")

    # 输出各阶段的难度分布统计
    for stage_name, samples in [
        ("阶段1", stage1_ids),
        ("阶段2", stage2_ids),
        ("阶段3", stage3_ids),
    ]:
        difficulty_counts = {"简单": 0, "中等": 0, "困难": 0}
        for sample in samples:
            difficulty_counts[sample["difficulty"]] += 1

        print(f"\n{stage_name}难度分布:")
        print(f"  简单: {difficulty_counts['简单']}")
        print(f"  中等: {difficulty_counts['中等']}")
        print(f"  困难: {difficulty_counts['困难']}")

    # 加载原始数据
    print("\n加载原始数据文件...")
    raw_data = load_jsonl_file(raw_data_file)

    # 创建ID到原始数据的映射
    id_to_raw_data = {item["id"]: item for item in raw_data}

    # 构建最终的数据集
    final_dataset = []

    # 添加阶段1的样本（前100个）
    for sample in stage1_ids:
        sample_id = sample["id"]
        if sample_id in id_to_raw_data:
            raw_item = id_to_raw_data[sample_id]
            final_dataset.append({
                "id": sample_id,
                "question": raw_item["question"],
                "answer": raw_item["answer"],
                "difficulty": sample["difficulty"]
            })

    # 添加阶段2的样本（第101~400个）
    for sample in stage2_ids:
        sample_id = sample["id"]
        if sample_id in id_to_raw_data:
            raw_item = id_to_raw_data[sample_id]
            final_dataset.append({
                "id": sample_id,
                "question": raw_item["question"],
                "answer": raw_item["answer"],
                "difficulty": sample["difficulty"]
            })

    # 添加阶段3的样本（第401~600个）
    for sample in stage3_ids:
        sample_id = sample["id"]
        if sample_id in id_to_raw_data:
            raw_item = id_to_raw_data[sample_id]
            final_dataset.append({
                "id": sample_id,
                "question": raw_item["question"],
                "answer": raw_item["answer"],
                "difficulty": sample["difficulty"]
            })

    # 验证最终数据集的样本数量
    print(f"最终数据集样本数量: {len(final_dataset)}")

    # 保存最终数据集
    save_jsonl_file(final_dataset, output_jsonl_file)
    print(f"最终数据集已保存至: {output_jsonl_file}")


if __name__ == "__main__":
    main()
