#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
from collections import defaultdict
from pathlib import Path

def main():
    # 输入文件路径
    input_file = Path("data/processed/gsm8k_attempts_pass_stats.json")
    # 输出目录
    output_dir = Path("data/processed")

    # 确保输入文件存在
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取统计数据
    with input_file.open("r", encoding="utf-8") as f:
        stats_data = json.load(f)

    # 提取ID列表
    id_lists = stats_data["id_lists"]

    # 按准确率分组的ID列表
    acc_groups = defaultdict(list)

    # 遍历所有attempts和pass组合，计算准确率并分组
    for attempts_str, passes_dict in id_lists.items():
        attempts = int(attempts_str)
        if attempts == 0:  # 避免除以零
            continue
            
        for passes_str, ids in passes_dict.items():
            passes = int(passes_str)
            
            # 计算准确率
            accuracy = passes / attempts
            
            # 将ID添加到对应准确率组
            for id_val in ids:
                acc_groups[accuracy].append(id_val)

    # 采样配置：准确率 -> (难度类别, 样本数量)
    sampling_config = {
        0.8: ("简单", 80),
        0.7: ("简单", 60),
        0.6: ("中等", 150),
        0.5: ("中等", 150),
        0.4: ("困难", 90),
        0.3: ("困难", 70)
    }

    # 采样结果
    sampled_results = {}
    total_sampled = 0

    # 打印可用的准确率组及其大小
    print("可用的准确率组:")
    for acc, ids in sorted(acc_groups.items()):
        print(f"准确率 {acc:.1f}: {len(ids)} 个样本")

    # 对每个目标准确率进行采样
    for target_acc, (difficulty, sample_size) in sampling_config.items():
        # 查找最接近的准确率组
        closest_acc = min(acc_groups.keys(), key=lambda x: abs(x - target_acc))
        
        # 如果没有完全匹配的准确率组，打印警告
        if closest_acc != target_acc:
            print(f"警告: 未找到准确率为 {target_acc} 的组，使用最接近的 {closest_acc}")
        
        # 获取该准确率组的ID列表
        available_ids = acc_groups[closest_acc]
        
        # 确定实际可采样的数量
        actual_sample_size = min(sample_size, len(available_ids))
        
        if actual_sample_size < sample_size:
            print(f"警告: 准确率 {closest_acc} 的组中只有 {len(available_ids)} 个样本，少于请求的 {sample_size} 个")
        
        # 随机采样
        sampled_ids = random.sample(available_ids, actual_sample_size)
        
        # 记录采样结果
        sampled_results[closest_acc] = {
            "difficulty": difficulty,
            "requested_size": sample_size,
            "actual_size": actual_sample_size,
            "ids": sampled_ids
        }
        
        total_sampled += actual_sample_size

    # 打印采样结果摘要
    print("\n采样结果摘要:")
    print(f"{'准确率':<10}{'难度':<10}{'请求数量':<10}{'实际数量':<10}")
    print("-" * 40)
    
    for acc, result in sorted(sampled_results.items()):
        print(f"{acc:<10.1f}{result['difficulty']:<10}{result['requested_size']:<10}{result['actual_size']:<10}")
    
    print("-" * 40)
    print(f"总计: {total_sampled} 个样本")

    # 将采样结果保存到文件
    output_file = output_dir / "gsm8k_sampled_by_accuracy.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "sampling_config": {str(k): v for k, v in sampling_config.items()},
            "sampled_results": {str(k): v for k, v in sampled_results.items()},
            "total_sampled": total_sampled
        }, f, ensure_ascii=False, indent=2)

    print(f"\n采样结果已保存到: {output_file}")

    # # 为每个难度类别创建单独的ID列表文件
    # difficulty_groups = defaultdict(list)
    
    # for acc, result in sampled_results.items():
    #     difficulty = result["difficulty"]
    #     ids = result["ids"]
    #     difficulty_groups[difficulty].extend(ids)
    
    # for difficulty, ids in difficulty_groups.items():
    #     output_file = output_dir / f"gsm8k_sampled_{difficulty}.json"
    #     with output_file.open("w", encoding="utf-8") as f:
    #         json.dump({
    #             "difficulty": difficulty,
    #             "count": len(ids),
    #             "ids": ids
    #         }, f, ensure_ascii=False, indent=2)
        
    #     print(f"{difficulty}难度的 {len(ids)} 个样本ID已保存到: {output_file}")

if __name__ == "__main__":
    main()