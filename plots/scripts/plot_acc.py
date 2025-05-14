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
    fig_key = f"fig{fig_idx}"
    fig_dict = acc_data[fig_key]
    baseline_no_template_accuracy = fig_dict.get("基线")
    baseline_with_template_accuracy = fig_dict.get("基线+对话模版")
    rl_steps, rl_accuracies = extract_steps_and_acc(fig_dict)
    rl_steps = [0] + rl_steps
    rl_accuracies = [baseline_with_template_accuracy] + rl_accuracies

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 对fig6应用对数x轴 ---
    if fig_idx == 6:
        # 1. 设置X轴为对数尺度
        # 中文：设置X轴为对数尺度，适合步数跨度较大时展示
        # English: Set x-axis to log scale for better visualization when step range is large
        # 注意：0不能用于对数轴，需去除0点
        rl_steps_log = [step for step in rl_steps if step > 0]
        rl_accuracies_log = [
            rl_accuracies[i] for i, step in enumerate(rl_steps) if step > 0
        ]
        # 6. 绘制基线点和基线+模板线（x=最小步数）
        ax.axhline(
            y=baseline_no_template_accuracy,
            color=color_baseline_orig_fig6,
            linestyle=linestyle_baseline_orig_fig6,
            linewidth=linewidth_baseline_orig_fig6,
            label=f"基线",
        )
        ax.axhline(
            y=baseline_with_template_accuracy,
            color=color_baseline_prompt,
            linestyle=linestyle_baseline_prompt,
            linewidth=plt.rcParams["lines.linewidth"],
            label="基线 + 对话模版",
        )
        # 2. 绘制RL微调曲线（对数轴）
        ax.plot(
            rl_steps_log,
            rl_accuracies_log,
            marker=marker_rl_tuned,
            markerfacecolor=color_rl_tuned,
            markeredgecolor=color_rl_tuned,
            color=color_rl_tuned,
            linestyle=linestyle_rl_tuned,
            linewidth=plt.rcParams["lines.linewidth"] + 0.4,
            label="RL 微调",
        )
        # 3. 设置对数尺度
        ax.set_xscale("log")
        # 4. 设置主刻度点为实际步数
        ax.set_xticks(rl_steps_log)
        # 5. 设置刻度标签为整数
        ax.set_xticklabels([str(int(x)) for x in rl_steps_log])
        # 7. 标注RL点数值
        for i, step in enumerate(rl_steps_log):
            if step in [100, 300]:
                ax.text(
                    step,
                    rl_accuracies_log[i] + 0.2,
                    f"{rl_accuracies_log[i]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_on_point_fontsize,
                    color=color_rl_tuned,
                    fontweight="semibold",
                )
            elif step == 1000:
                ax.text(
                    step + 100,
                    rl_accuracies_log[i] + 0.4,
                    f"{rl_accuracies_log[i]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_on_point_fontsize,
                    color=color_rl_tuned,
                    fontweight="semibold",
                )
            elif step == 7000:
                ax.text(
                    step - 1600,
                    rl_accuracies_log[i],
                    f"{rl_accuracies_log[i]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_on_point_fontsize,
                    color=color_rl_tuned,
                    fontweight="semibold",
                )
            else:
                ax.text(
                    step,
                    rl_accuracies_log[i] + 0.4,
                    f"{rl_accuracies_log[i]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_on_point_fontsize,
                    color=color_rl_tuned,
                    fontweight="semibold",
                )
        # 9. 其余设置同原代码
        ax.set_xlabel("RL 训练步数（对数尺度）", fontweight="bold", labelpad=20)
        ax.set_ylabel("评估准确率（%）", fontweight="bold", labelpad=20)
        x_max = max(rl_steps_log)
        ax.set_xlim(min(rl_steps_log) * 0.8, x_max * 1.2)
        all_y_values = [baseline_no_template_accuracy] + rl_accuracies_log
        y_min_data = min(all_y_values)
        y_max_data = max(all_y_values)
        y_min_plot = math.floor(y_min_data) - 1
        y_max_plot = math.ceil(y_max_data) + 1.5
        ax.set_ylim(y_min_plot, y_max_plot)
        # 主刻度每隔 1
        major_y_ticks = np.arange(y_min_plot, y_max_plot + 1, 2)
        ax.set_yticks(major_y_ticks)
        # 次刻度每隔 0.2
        minor_y_ticks = np.arange(y_min_plot, y_max_plot + 0.2, 0.2)
        ax.set_yticks(minor_y_ticks, minor=True)
        # 主网格线（每隔 1，较粗且深色）
        ax.grid(
            True,
            which="major",
            axis="y",
            linestyle="-",
            alpha=0.7,
            linewidth=1.2,
            color=grid_main_color,
        )
        # 次网格线（每隔 0.2，较细且淡色）
        ax.grid(
            True,
            which="minor",
            axis="y",
            linestyle=":",
            alpha=0.25,
            linewidth=0.5,
            color=grid_minor_color,
        )
        # x 轴主网格线保持原样
        ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.4, linewidth=0.6)
        legend = ax.legend(
            loc="lower left",
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

        # --- 增强边框感（Enhance border/spine for academic style）---
        for spine in ["left", "bottom", "top", "right"]:
            ax.spines[spine].set_visible(True)  # 确保四周边框都显示
            ax.spines[spine].set_linewidth(1.2)  # 增加边框粗细
            ax.spines[spine].set_color("black")  # 统一为黑色
        ax.tick_params(
            axis="both", which="major", direction="out", length=6, width=1, pad=10
        )
        plt.tight_layout(pad=2.2)
        file_name = os.path.join(output_dir, f"fig{fig_idx}.png")
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"图表已保存为: {file_name}")
        plt.close(fig)
        continue

    # --- 开始绘图 ---
    # 1. 绘制 基线 - 一个点 (空心)
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

    # 2. 绘制 基线 + 对话模版 - 水平虚线
    ax.axhline(
        y=baseline_with_template_accuracy,
        color=color_baseline_prompt,
        linestyle=linestyle_baseline_prompt,
        linewidth=plt.rcParams["lines.linewidth"],
        label="基线 + 对话模版",
    )

    # 3. 绘制 RL 微调曲线 (实心圆点)
    ax.plot(
        rl_steps,
        rl_accuracies,
        marker=marker_rl_tuned,
        markerfacecolor=color_rl_tuned,
        markeredgecolor=color_rl_tuned,
        color=color_rl_tuned,
        linestyle=linestyle_rl_tuned,
        linewidth=plt.rcParams["lines.linewidth"] + 0.4,
        label="RL 微调（本文方法）" if fig_idx <= 3 else "RL 微调",
    )

    # 4. 在每个点上显示数值
    # 标注 RL 曲线上的点
    for i, step in enumerate(rl_steps):
        # 默认参数
        vertical_offset = 0.4  # 增大垂直偏移，给更大的字留空间
        horizontal_align = "center"
        text_color = color_rl_tuned
        fontweight_value = "semibold"  # 加粗数值
        current_step_display = step
        x_offset = 0
        y_offset = vertical_offset

        # 仅在fig2时，第0步和第50步的RL点数值标注在点的左上方
        # For fig2, annotate the value of RL points at step 0 and 50 at the upper left of the point
        if fig_idx == 2 and (step == 0 or step == 50):
            horizontal_align = "center"  # 左对齐
            x_offset = -8  # 向左偏移一点（单位为数据坐标，适当调整）
            y_offset = vertical_offset  # 向上偏移更多
            # 中文注释：左上方显示
            # English: show at upper left

        ax.text(
            current_step_display + x_offset,
            rl_accuracies[i] + y_offset,
            f"{rl_accuracies[i]:.2f}",
            ha=horizontal_align,
            va="bottom",
            fontsize=value_on_point_fontsize,
            color=text_color,
            fontweight=fontweight_value,
        )

    # 单独标注原始基线的值
    ax.text(
        0,
        baseline_no_template_accuracy + vertical_offset,
        f"{baseline_no_template_accuracy:.2f}",  # 基线数值也放点上方
        ha="center",
        va="bottom",
        fontsize=value_on_point_fontsize,
        color=color_baseline_orig,
        fontweight=fontweight_value,
    )

    # --- 设置图表标题和标签 ---
    # ax.set_title(f"{model_name_display} 在 GSM8K 上的评估准确率", fontweight="bold", pad=30, fontsize=title_fontsize)
    ax.set_xlabel("RL 训练步数（Steps）", fontweight="bold", labelpad=20)
    ax.set_ylabel("评估准确率（%）", fontweight="bold", labelpad=20)

    # --- 设置坐标轴范围和刻度 ---
    x_max = max([0] + rl_steps)
    ax.set_xlim(-40, x_max + 40)
    all_y_values = [baseline_no_template_accuracy] + rl_accuracies
    y_min_data = min(all_y_values)
    y_max_data = max(all_y_values)
    y_min_plot = math.floor(y_min_data)
    y_max_plot = (
        math.ceil(y_max_data) + 0.5
        if y_max_data % 1 < 0.5
        else math.ceil(y_max_data) + 1
    )
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_xticks([0] + rl_steps)
    # 主刻度每隔 1
    major_y_ticks = np.arange(y_min_plot, y_max_plot + 1, 1)
    ax.set_yticks(major_y_ticks)
    # 次刻度每隔 0.2
    minor_y_ticks = np.arange(y_min_plot, y_max_plot + 0.2, 0.2)
    ax.set_yticks(minor_y_ticks, minor=True)

    # --- 添加网格线 ---
    # 主网格线（每隔 1，较粗且深色）
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="-",
        alpha=0.7,
        linewidth=1.2,
        color=grid_main_color,
    )
    # 次网格线（每隔 0.2，较细且淡色）
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        alpha=0.25,
        linewidth=0.5,
        color=grid_minor_color,
    )
    # x 轴主网格线保持原样
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.4, linewidth=0.6)

    # --- 添加图例 ---
    legend = ax.legend(
        loc="lower right",  # 图例的位置
        frameon=True,  # 是否显示图例边框
        shadow=False,  # 是否给图例添加阴影
        borderpad=0.8,  # 图例边框和内容之间的内边距（以字体大小为单位）
        labelspacing=0.7,  # 图例条目之间的垂直间距（以字体大小为单位）
        handletextpad=0.8,  # 图例标记（如线、点）和文字之间的水平间距（以字体大小为单位）
        markerscale=0.9,  # 图例中标记（marker）的缩放比例
        fontsize=legend_fontsize,  # 图例文字的字体大小
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
    )  # 增大pad

    # --- 增强边框感（Enhance border/spine for academic style）---
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)  # 确保四周边框都显示
        ax.spines[spine].set_linewidth(1.2)  # 增加边框粗细
        ax.spines[spine].set_color("black")  # 统一为黑色

    # --- 显示并保存图表 ---
    plt.tight_layout(pad=2.2)
    file_name = os.path.join(output_dir, f"fig{fig_idx}.png")
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    print(f"图表已保存为: {file_name}")
    plt.close(fig)

try:
    plt.show()
except KeyboardInterrupt:
    print("\n用户中断显示图表，程序退出。")
except Exception as e:
    print(f"显示图表时发生错误: {e}")
