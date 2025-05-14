import sys
import yaml  # 新增：用于读取 YAML 文件
import os  # 新增：用于创建输出目录

sys.path.append("/root/proj/r1")

import matplotlib.pyplot as plt
import numpy as np
import math  # 用于向上向下取整

# --- 图表样式与字体设置 ---
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams["font.serif"] = ["Noto Serif CJK JP"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

base_fontsize = 13
title_fontsize = base_fontsize + 7  # 例如 20
axes_label_fontsize = base_fontsize + 3  # 例如 16
tick_label_fontsize = base_fontsize + 1  # 例如 14
legend_fontsize = base_fontsize  # 例如 12
value_on_point_fontsize = base_fontsize  # 例如 12 (点上数值增大)

plt.rcParams["axes.labelsize"] = axes_label_fontsize
plt.rcParams["xtick.labelsize"] = tick_label_fontsize
plt.rcParams["ytick.labelsize"] = tick_label_fontsize
plt.rcParams["legend.fontsize"] = legend_fontsize
plt.rcParams["lines.markersize"] = 9  # 稍微再增大标记
plt.rcParams["lines.linewidth"] = 2.4

# 颜色方案
color_baseline_orig = "#000000"  # 纯黑色 (基线)
# color_baseline_orig = "#5D4037"       # 深棕色 (Brown Darken-2 - Material Design)
color_baseline_orig_fig6 = "#A1887F"  # 灰褐色 (Brown Lighten-1 - Material Design)
linestyle_baseline_orig_fig6 = ":"  # 基线线条样式 (点线)
linewidth_baseline_orig_fig6 = 2.4  # 基线线条宽度
color_baseline_prompt = "#BF360C"  # 深橙红色 (Deep Orange Darken-4 - Material Design)
color_rl_tuned = "#0D47A1"  # 非常深的蓝色 (Blue Darken-4 - Material Design)

grid_main_color = "#999999"  # 稍微浅一点的灰色
grid_minor_color = "#cccccc"

# 标记样式
marker_baseline_orig = "o"
marker_rl_tuned = "o"

# 线条样式
linestyle_baseline_prompt = "--"
linestyle_rl_tuned = "-"

# --- 数据定义 ---
# 读取 YAML 数据
with open("plots/data/acc.yml", "r", encoding="utf-8") as f:
    acc_data = yaml.safe_load(f)

# 输出目录
output_dir = "plots/output"
os.makedirs(output_dir, exist_ok=True)


# 定义每个 figure 的 RL 步数顺序（自动推断）
def extract_steps_and_acc(fig_dict):
    steps = []
    accs = []
    for k, v in fig_dict.items():
        if k.startswith("基线"):
            continue
        # 提取步数（如 "50步" -> 50）
        step = int(k.replace("步", ""))
        steps.append(step)
        accs.append(v)
    # 步数排序
    steps_acc = sorted(zip(steps, accs), key=lambda x: x[0])
    steps_sorted = [x[0] for x in steps_acc]
    accs_sorted = [x[1] for x in steps_acc]
    return steps_sorted, accs_sorted


# --- 循环绘制 fig1-fig6 ---
for fig_idx in range(1, 7):
    log_x = fig_idx == 6
    fig_key = f"fig{fig_idx}"
    fig_dict = acc_data[fig_key]
    baseline_no_template_accuracy = fig_dict.get("基线")
    baseline_with_template_accuracy = fig_dict.get("基线+对话模版")
    rl_steps, rl_accuracies = extract_steps_and_acc(fig_dict)
    if log_x:
        # 对数x轴不允许0
        rl_steps_plot = [step for step in rl_steps if step > 0]
        rl_accuracies_plot = [
            rl_accuracies[i] for i, step in enumerate(rl_steps) if step > 0
        ]
    else:
        rl_steps_plot = [0] + rl_steps
        rl_accuracies_plot = [baseline_with_template_accuracy] + rl_accuracies

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 基线 ---
    if log_x:
        ax.axhline(
            y=baseline_no_template_accuracy,
            color=color_baseline_orig_fig6,
            linestyle=linestyle_baseline_orig_fig6,
            linewidth=linewidth_baseline_orig_fig6,
            label="基线",
        )
    else:
        ax.plot(
            0,
            baseline_no_template_accuracy,
            marker=marker_baseline_orig,
            markeredgecolor=color_baseline_orig,
            markerfacecolor="none",
            label="基线",
            linestyle="None",
            markersize=plt.rcParams["lines.markersize"],
        )

    # --- 基线+模板 ---
    ax.axhline(
        y=baseline_with_template_accuracy,
        color=color_baseline_prompt,
        linestyle=linestyle_baseline_prompt,
        linewidth=plt.rcParams["lines.linewidth"],
        label="基线 + 对话模版",
    )

    # --- RL微调曲线 ---
    ax.plot(
        rl_steps_plot,
        rl_accuracies_plot,
        marker=marker_rl_tuned,
        markerfacecolor=color_rl_tuned,
        markeredgecolor=color_rl_tuned,
        color=color_rl_tuned,
        linestyle=linestyle_rl_tuned,
        linewidth=plt.rcParams["lines.linewidth"] + 0.4,
        label=(
            "RL 微调"
            if log_x
            else ("RL 微调（本文方法）" if fig_idx <= 3 else "RL 微调")
        ),
    )

    # --- RL点数值标注 ---
    for i, step in enumerate(rl_steps_plot):
        if log_x:
            # fig6特殊偏移
            if step in [100, 300]:
                y_offset = 0.2
                x_offset = 0
            elif step == 1000:
                y_offset = 0.4
                x_offset = 100

            elif step == 7000:
                y_offset = 0
                x_offset = -1600
            else:
                y_offset = 0.4
                x_offset = 0
            ax.text(
                step + x_offset,
                rl_accuracies_plot[i] + y_offset,
                f"{rl_accuracies_plot[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_on_point_fontsize,
                color=color_rl_tuned,
                fontweight="semibold",
            )
        else:
            # 其它fig统一偏移，fig2特殊处理
            vertical_offset = 0.4
            horizontal_align = "center"
            x_offset = 0
            y_offset = vertical_offset
            if fig_idx == 2 and (step == 0 or step == 50):
                x_offset = -8
                y_offset = vertical_offset
            ax.text(
                step + x_offset,
                rl_accuracies_plot[i] + y_offset,
                f"{rl_accuracies_plot[i]:.2f}",
                ha=horizontal_align,
                va="bottom",
                fontsize=value_on_point_fontsize,
                color=color_rl_tuned,
                fontweight="semibold",
            )
    # 单独标注原始基线的值（仅线性x轴）
    if not log_x:
        ax.text(
            0,
            baseline_no_template_accuracy + 0.4,
            f"{baseline_no_template_accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=value_on_point_fontsize,
            color=color_baseline_orig,
            fontweight="semibold",
        )

    # --- 坐标轴设置 ---
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(rl_steps_plot)
        ax.set_xticklabels([str(int(x)) for x in rl_steps_plot])
        x_max = max(rl_steps_plot)
        ax.set_xlim(min(rl_steps_plot) * 0.8, x_max * 1.2)
        all_y_values = [baseline_no_template_accuracy] + rl_accuracies_plot
        y_min_data = min(all_y_values)
        y_max_data = max(all_y_values)
        y_min_plot = math.floor(y_min_data) - 1
        y_max_plot = math.ceil(y_max_data) + 1.5
        ax.set_ylim(y_min_plot, y_max_plot)
        major_y_ticks = np.arange(y_min_plot, y_max_plot + 1, 2)
        ax.set_yticks(major_y_ticks)
        minor_y_ticks = np.arange(y_min_plot, y_max_plot + 0.2, 0.2)
        ax.set_yticks(minor_y_ticks, minor=True)
    else:
        x_max = max([0] + rl_steps)
        ax.set_xlim(-40, x_max + 40)
        all_y_values = [baseline_no_template_accuracy] + rl_accuracies_plot
        y_min_data = min(all_y_values)
        y_max_data = max(all_y_values)
        y_min_plot = math.floor(y_min_data)
        y_max_plot = (
            math.ceil(y_max_data) + 0.5
            if y_max_data % 1 < 0.5
            else math.ceil(y_max_data) + 1
        )
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.set_xticks(rl_steps_plot)
        major_y_ticks = np.arange(y_min_plot, y_max_plot + 1, 1)
        ax.set_yticks(major_y_ticks)
        minor_y_ticks = np.arange(y_min_plot, y_max_plot + 0.2, 0.2)
        ax.set_yticks(minor_y_ticks, minor=True)

    # --- 标签 ---
    ax.set_xlabel(
        "RL 训练步数（对数尺度）" if log_x else "RL 训练步数（Steps）",
        fontweight="bold",
        labelpad=20,
    )
    ax.set_ylabel("评估准确率（%）", fontweight="bold", labelpad=20)

    # --- 添加网格线 ---
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="-",
        alpha=0.7,
        linewidth=1.2,
        color=grid_main_color,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        alpha=0.25,
        linewidth=0.5,
        color=grid_minor_color,
    )
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.4, linewidth=0.6)

    # --- 添加图例 ---
    legend = ax.legend(
        loc="lower left" if log_x else "lower right",
        frameon=True,
        shadow=False,
        borderpad=0.8,
        labelspacing=0.7,
        handletextpad=0.8,
        markerscale=0.9,
        fontsize=legend_fontsize,
    )
    legend.get_frame().set_edgecolor("darkgray")
    legend.get_frame().set_linewidth(0.7)

    # --- 优化外观 ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(
        axis="both", which="major", direction="out", length=6, width=1, pad=10
    )
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("black")

    plt.tight_layout(pad=2.2)
    file_name = os.path.join(output_dir, f"fig{fig_idx}.png")
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    print(f"图表已保存为: {file_name}")
    plt.close(fig)
