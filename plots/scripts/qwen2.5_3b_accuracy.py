import sys

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

# --- 数据定义 ---
model_name_display = "Qwen2.5-3B-Instruct (4-bit)"
model_name_file = (
    model_name_display.replace(" ", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("/", "_")
)

rl_steps = np.array([0, 50, 100, 150, 200, 250, 300])

baseline_no_template_accuracy = 73.82
baseline_with_template_accuracy = 78.22
rl_accuracies = np.array(
    [baseline_with_template_accuracy, 79.08, 79.45, 80.29, 81.05, 81.50, 82.25]
)

# 颜色方案
color_baseline_orig = "#000000"  # 纯黑色 (基线)
color_baseline_prompt = "#FF8C00"  # 深橙色
color_rl_tuned = "#1f77b4"  # Matplotlib 默认蓝色

# 标记样式
marker_baseline_orig = "o"
marker_rl_tuned = "o"

# 线条样式
linestyle_baseline_prompt = "--"
linestyle_rl_tuned = "-"


# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(10, 8))  # 调整图表尺寸以适应更大的字体和更舒展的布局

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
    label="RL 微调（本文方法）",
)

# 4. 在每个点上显示数值
show_values_on_points = True
if show_values_on_points:
    # 标注 RL 曲线上的点
    for i, step in enumerate(rl_steps):
        vertical_offset = 0.4  # 增大垂直偏移，给更大的字留空间
        horizontal_align = "center"
        text_color = color_rl_tuned
        fontweight_value = "semibold"  # 加粗数值

        current_step_display = step

        ax.text(
            current_step_display,
            rl_accuracies[i] + vertical_offset,
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
# ax.set_title(
#     f"{model_name_display} 在 GSM8K 上的评估准确率",
#     fontweight="bold",
#     pad=30,
#     fontsize=title_fontsize,
# )
ax.set_xlabel("RL 训练步数（Steps）", fontweight="bold", labelpad=20)
ax.set_ylabel("评估准确率（%）", fontweight="bold", labelpad=20)

# --- 设置坐标轴范围和刻度 ---
ax.set_xlim(-40, rl_steps[-1] + 40)

# 计算Y轴范围
all_y_values = np.concatenate(([baseline_no_template_accuracy], rl_accuracies))
y_min_data = np.min(all_y_values)
y_max_data = np.max(all_y_values)

y_min_plot = math.floor(y_min_data)
y_max_plot = math.ceil(y_max_data) + 0.5
ax.set_ylim(y_min_plot, y_max_plot)

ax.set_xticks(rl_steps)

major_y_ticks = np.arange(y_min_plot, y_max_plot + 1, 2)  # 每2%一个主刻度，确保覆盖范围
ax.set_yticks(major_y_ticks)

# --- 添加图例 ---
legend = ax.legend(
    loc="lower right",  # 图例的位置。Location of the legend, e.g. "lower right" means bottom right corner.
    frameon=True,  # 是否显示图例边框。Whether to draw a frame (box) around the legend.
    shadow=False,  # 是否给图例添加阴影。Whether to draw a shadow behind the legend.
    borderpad=0.8,  # 图例边框和内容之间的内边距（以字体大小为单位）。Padding between the legend border and the content (in font-size units).
    labelspacing=0.7,  # 图例条目之间的垂直间距（以字体大小为单位）。Vertical space between the legend entries (in font-size units).
    handletextpad=0.8,  # 图例标记（如线、点）和文字之间的水平间距（以字体大小为单位）。Padding between the legend handle and the text (in font-size units).
    markerscale=0.9,  # 图例中标记（marker）的缩放比例。The relative size of legend markers compared with the original plot.
    fontsize=legend_fontsize,  # 图例文字的字体大小。Font size of the legend text.
)
legend.get_frame().set_edgecolor("darkgray")
legend.get_frame().set_linewidth(0.7)

# --- 添加网格线 ---
ax.grid(True, which="major", linestyle=":", alpha=0.4, linewidth=0.6)  # 网格线更淡一些

# --- 优化外观 ---
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1)
ax.spines["bottom"].set_linewidth(1)
ax.tick_params(
    axis="both", which="major", direction="out", length=6, width=1, pad=10
)  # 增大pad

# --- 显示并保存图表 ---
plt.tight_layout(pad=2.2)
file_name = f"{model_name_file}_accuracy_plot_v6.png"
plt.savefig(file_name, dpi=300, bbox_inches="tight")
print(f"图表已保存为: {file_name}")

try:
    plt.show()
except KeyboardInterrupt:
    print("\n用户中断显示图表，程序退出。")
except Exception as e:
    print(f"显示图表时发生错误: {e}")
