import sys
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append("/root/proj/r1")

# ================== 样式与参数 ==================
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FONT_SERIF = ["Noto Serif CJK JP"]
FONT_SANS = ["Noto Sans CJK JP"]
FONT_FAMILY = "serif"

BASE_FONTSIZE = 13
PARAMS = {
    "font.serif": FONT_SERIF,
    "font.sans-serif": FONT_SANS,
    "font.family": FONT_FAMILY,
    "axes.unicode_minus": False,
    "axes.labelsize": BASE_FONTSIZE + 3,
    "xtick.labelsize": BASE_FONTSIZE + 1,
    "ytick.labelsize": BASE_FONTSIZE + 1,
    "legend.fontsize": BASE_FONTSIZE,
    "lines.markersize": 9,
    "lines.linewidth": 2.4,
}

COLORS = {
    "baseline_orig": "#000000",
    "baseline_orig_fig6": "#A1887F",
    "baseline_prompt": "#BF360C",
    "rl_tuned": "#014f86",
    # "rl_tuned": "#0D47A1",
    "grid_main": "#999999",
    "grid_minor": "#cccccc",
}

LINESTYLES = {
    "baseline_orig_fig6": ":",
    "baseline_prompt": "--",
    "rl_tuned": "-",
}

MARKERS = {
    "baseline_orig": "o",
    "rl_tuned": "o",
}


# ================== 样式设置函数 ==================
def set_plot_style():
    plt.style.use(PLOT_STYLE)
    for k, v in PARAMS.items():
        plt.rcParams[k] = v


# ================== 数据处理 ==================
def extract_steps_and_acc(fig_dict):
    steps, accs = [], []
    for k, v in fig_dict.items():
        if k.startswith("基线"):
            continue
        step = int(k.replace("步", ""))
        steps.append(step)
        accs.append(v)
    steps_acc = sorted(zip(steps, accs), key=lambda x: x[0])
    steps_sorted = [x[0] for x in steps_acc]
    accs_sorted = [x[1] for x in steps_acc]
    return steps_sorted, accs_sorted


# ================== 偏移量查表 ==================
def get_point_offset(fig_idx, step, log_x):
    if log_x:
        if step in [100, 300]:
            return 0, 0.2
        elif step == 1000:
            return 100, 0.4
        elif step == 7000:
            return -1600, 0
        else:
            return 0, 0.4
    else:
        if fig_idx == 2 and (step == 0 or step == 50):
            return -8, 0.4
        return 0, 0.4


# ================== 坐标轴设置 ==================
def set_axis(ax, rl_steps_plot, rl_accuracies_plot, baseline, log_x):
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(rl_steps_plot)
        ax.set_xticklabels([str(int(x)) for x in rl_steps_plot])
        x_max = max(rl_steps_plot)
        ax.set_xlim(min(rl_steps_plot) * 0.8, x_max * 1.2)
        all_y_values = [baseline] + list(rl_accuracies_plot)
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
        x_max = max([0] + list(rl_steps_plot))
        ax.set_xlim(-40, x_max + 40)
        all_y_values = [baseline] + list(rl_accuracies_plot)
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


# ================== 主绘图函数 ==================
def plot_figure(fig_idx, fig_dict, output_dir):
    log_x = fig_idx == 6
    baseline = fig_dict.get("基线")
    baseline_tmpl = fig_dict.get("基线+对话模版")
    rl_steps, rl_accs = extract_steps_and_acc(fig_dict)
    if log_x:
        rl_steps_plot, rl_accs_plot = zip(
            *[(s, a) for s, a in zip(rl_steps, rl_accs) if s > 0]
        )
    else:
        rl_steps_plot = [0] + rl_steps
        rl_accs_plot = [baseline_tmpl] + rl_accs

    fig, ax = plt.subplots(figsize=(10, 8))
    # --- 基线 ---
    if log_x:
        ax.axhline(
            y=baseline,
            color=COLORS["baseline_orig_fig6"],
            linestyle=LINESTYLES["baseline_orig_fig6"],
            linewidth=2.4,
            label="基线",
        )
    else:
        ax.plot(
            0,
            baseline,
            marker=MARKERS["baseline_orig"],
            markeredgecolor=COLORS["baseline_orig"],
            markerfacecolor="none",
            label="基线",
            linestyle="None",
            markersize=plt.rcParams["lines.markersize"],
        )
    # --- 基线+模板 ---
    ax.axhline(
        y=baseline_tmpl,
        color=COLORS["baseline_prompt"],
        linestyle=LINESTYLES["baseline_prompt"],
        linewidth=plt.rcParams["lines.linewidth"],
        label="基线 + 对话模版",
    )
    # --- RL微调曲线 ---
    ax.plot(
        rl_steps_plot,
        rl_accs_plot,
        marker=MARKERS["rl_tuned"],
        markerfacecolor=COLORS["rl_tuned"],
        markeredgecolor=COLORS["rl_tuned"],
        color=COLORS["rl_tuned"],
        linestyle=LINESTYLES["rl_tuned"],
        linewidth=plt.rcParams["lines.linewidth"] + 0.4,
        label=(
            "RL 微调"
            if log_x
            else ("RL 微调（本文方法）" if fig_idx <= 3 else "RL 微调")
        ),
    )
    # --- RL点数值标注 ---
    for i, step in enumerate(rl_steps_plot):
        x_offset, y_offset = get_point_offset(fig_idx, step, log_x)
        ax.text(
            step + x_offset,
            rl_accs_plot[i] + y_offset,
            f"{rl_accs_plot[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=BASE_FONTSIZE,
            color=COLORS["rl_tuned"],
            fontweight="semibold",
        )
    # 单独标注原始基线的值（仅线性x轴）
    if not log_x:
        ax.text(
            0,
            baseline + 0.4,
            f"{baseline:.2f}",
            ha="center",
            va="bottom",
            fontsize=BASE_FONTSIZE,
            color=COLORS["baseline_orig"],
            fontweight="semibold",
        )
    # --- 坐标轴设置 ---
    set_axis(ax, rl_steps_plot, rl_accs_plot, baseline, log_x)
    # --- 标签 ---
    ax.set_xlabel(
        "RL 训练步数（对数尺度）" if log_x else "RL 训练步数（Steps）",
        fontweight="bold",
        labelpad=20,
    )
    ax.set_ylabel("评估准确率（%）", fontweight="bold", labelpad=20)
    # --- 网格线 ---
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="-",
        alpha=0.7,
        linewidth=1.2,
        color=COLORS["grid_main"],
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        alpha=0.25,
        linewidth=0.5,
        color=COLORS["grid_minor"],
    )
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.4, linewidth=0.6)
    # --- 图例 ---
    legend = ax.legend(
        loc="lower left" if log_x else "lower right",
        frameon=True,
        shadow=False,
        borderpad=0.8,
        labelspacing=0.7,
        handletextpad=0.8,
        markerscale=0.9,
        fontsize=BASE_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("darkgray")
    legend.get_frame().set_linewidth(0.7)
    # --- 外观优化 ---
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("black")
    plt.tight_layout(pad=2.2)
    file_name = os.path.join(output_dir, f"{fig_idx}_acc.png")
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    print(f"图表已保存为: {file_name}")
    plt.close(fig)


# ================== 主流程 ==================
def main():
    set_plot_style()
    with open("plots/data/acc.yml", "r", encoding="utf-8") as f:
        acc_data = yaml.safe_load(f)
    output_dir = "plots/output"
    os.makedirs(output_dir, exist_ok=True)
    for fig_idx in range(1, 7):
        plot_figure(fig_idx, acc_data[fig_idx], output_dir)


if __name__ == "__main__":
    main()
