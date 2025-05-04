#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
from pathlib import Path

# 输入文件路径
input_file = Path("data/processed/gsm8k_math_resp_analysis.jsonl")
# 输出目录
output_dir = Path("data/processed")

# 确保输入文件存在
if not input_file.exists():
    raise FileNotFoundError(f"输入文件不存在: {input_file}")

# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

# 初始化统计字典
# 格式: stats[attempts][pass] = count
stats = defaultdict(lambda: defaultdict(int))
# 初始化id列表字典
# 格式: id_lists[attempts][pass] = [id1, id2, ...]
id_lists = defaultdict(lambda: defaultdict(list))

# 从JSONL文件中读取数据并统计
total_lines = 0
with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1
        try:
            # 解析JSON行
            data = json.loads(line)
            
            # 提取attempts、pass和id字段
            attempts = data.get("attempts", 0)
            passes = data.get("pass", 0)
            question_id = data.get("id", "unknown")
            
            # 更新统计信息
            stats[attempts][passes] += 1
            # 记录id
            id_lists[attempts][passes].append(question_id)
            
        except json.JSONDecodeError:
            print(f"无法解析JSON行: {line[:50]}...")
        except Exception as e:
            print(f"处理数据时出错: {e}")

# 输出统计信息
print(f"总共处理了 {total_lines} 行数据")
print("\n每种attempts对应的不同pass统计:")

# 按attempts排序
for attempts in sorted(stats.keys()):
    print(f"\nattempts = {attempts}:")
    total_for_attempts = sum(stats[attempts].values())
    
    # 按pass排序
    for passes in sorted(stats[attempts].keys()):
        count = stats[attempts][passes]
        percentage = count / total_for_attempts * 100 if total_for_attempts > 0 else 0
        print(f"  pass = {passes}: {count} 行 ({percentage:.2f}%)")
    
    print(f"  总计: {total_for_attempts} 行")

# 创建一个表格形式的统计结果
print("\n统计表格 (行: attempts, 列: pass):")
all_attempts = sorted(stats.keys())
all_passes = sorted(set(passes for attempt_stats in stats.values() for passes in attempt_stats.keys()))

# 打印表头
header = "attempts\\pass"
for passes in all_passes:
    header += f"\t{passes}"
header += "\t总计"
print(header)

# 打印表格内容
for attempts in all_attempts:
    row = f"{attempts}"
    row_total = 0
    for passes in all_passes:
        count = stats[attempts][passes]
        row_total += count
        row += f"\t{count}"
    row += f"\t{row_total}"
    print(row)

# 打印列总计
row = "总计"
col_totals = defaultdict(int)
for attempts in all_attempts:
    for passes in all_passes:
        col_totals[passes] += stats[attempts][passes]

for passes in all_passes:
    row += f"\t{col_totals[passes]}"

# 总行数
row += f"\t{total_lines}"
print(row)

# 将统计结果保存到JSON文件
output_stats = {
    "stats": {str(a): {str(p): stats[a][p] for p in stats[a]} for a in stats},
    "id_lists": {str(a): {str(p): id_lists[a][p] for p in id_lists[a]} for a in id_lists},
    "summary": {
        "total_lines": total_lines,
        "all_attempts": [a for a in all_attempts],
        "all_passes": [p for p in all_passes],
        "col_totals": {str(p): col_totals[p] for p in all_passes}
    }
}

# 保存统计结果
output_stats_file = output_dir / "gsm8k_attempts_pass_stats.json"
with output_stats_file.open("w", encoding="utf-8") as f:
    json.dump(output_stats, f, ensure_ascii=False)

print(f"\n统计结果已保存到: {output_stats_file}")

# 创建一个更易读的文本报告
report_file = output_dir / "gsm8k_attempts_pass_report.txt"
with report_file.open("w", encoding="utf-8") as f:
    f.write(f"总共处理了 {total_lines} 行数据\n")
    f.write("\n每种attempts对应的不同pass统计:\n")
    
    # 按attempts排序
    for attempts in sorted(stats.keys()):
        f.write(f"\nattempts = {attempts}:\n")
        total_for_attempts = sum(stats[attempts].values())
        
        # 按pass排序
        for passes in sorted(stats[attempts].keys()):
            count = stats[attempts][passes]
            percentage = count / total_for_attempts * 100 if total_for_attempts > 0 else 0
            f.write(f"  pass = {passes}: {count} 行 ({percentage:.2f}%)\n")
            f.write(f"    ID列表: {', '.join(map(str, id_lists[attempts][passes]))}\n")
        
        f.write(f"  总计: {total_for_attempts} 行\n")
    
    # 创建一个表格形式的统计结果
    f.write("\n统计表格 (行: attempts, 列: pass):\n")
    
    # 写入表头
    header = "attempts\\pass"
    for passes in all_passes:
        header += f"\t{passes}"
    header += "\t总计"
    f.write(header + "\n")
    
    # 写入表格内容
    for attempts in all_attempts:
        row = f"{attempts}"
        row_total = 0
        for passes in all_passes:
            count = stats[attempts][passes]
            row_total += count
            row += f"\t{count}"
        row += f"\t{row_total}"
        f.write(row + "\n")
    
    # 写入列总计
    row = "总计"
    for passes in all_passes:
        row += f"\t{col_totals[passes]}"
    
    # 总行数
    row += f"\t{total_lines}"
    f.write(row + "\n")

print(f"详细报告已保存到: {report_file}")