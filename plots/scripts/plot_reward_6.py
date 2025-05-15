import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# --- 图表样式与字体设置 ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.serif"] = ["Noto Serif CJK JP"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

# 单组数据配置
run_name = "20250427_181546"
color = "#D55E00"  # 主色
label = "Qwen2.5-3B-Instruct-QLoRA (4-bit) 简单奖励函数 + 长训练"


def plot_smoothed_timeseries_full_range(
    x_values,
    y_values,
    smoothing_window_size=50,
    outlier_detection_window_size=20,
    outlier_std_factor=3,
    original_color_hex="#a9d6e5",
    smoothed_color_hex="#014f86",
    boundary_mode="reflect",
    plot_raw=True,
    **plot_kwargs,
):
    if len(x_values) != len(y_values):
        raise ValueError("x_values 和 y_values 的长度必须相同。")
    if len(y_values) == 0:
        print("没有数据可供绘制。")
        return None, None
    # --- 1. 异常值处理 (裁剪) ---
    if len(y_values) > 0:
        sorted_y = np.sort(y_values)
        idx_85 = int(0.85 * len(sorted_y))
        val_85 = sorted_y[idx_85]
        extreme_threshold = 3 * val_85
        mask = y_values <= extreme_threshold
        x_values = x_values[mask]
        y_values = y_values[mask]
    y_series = pd.Series(y_values)
    rolling_mean = y_series.rolling(
        window=outlier_detection_window_size, center=True, min_periods=1
    ).mean()
    rolling_std = y_series.rolling(
        window=outlier_detection_window_size, center=True, min_periods=1
    ).std()
    global_std_fallback = y_series.std() if not y_series.std() == 0 else 1.0
    rolling_std_filled = rolling_std.fillna(global_std_fallback)
    lower_bound = rolling_mean - outlier_std_factor * rolling_std_filled
    upper_bound = rolling_mean + outlier_std_factor * rolling_std_filled
    y_clipped = np.clip(y_values, lower_bound.to_numpy(), upper_bound.to_numpy())
    # --- 2. 计算移动平均 ---
    if len(y_clipped) >= smoothing_window_size:
        y_smoothed = uniform_filter1d(
            y_clipped, size=smoothing_window_size, mode=boundary_mode, origin=0
        )
        x_smoothed = x_values
    elif len(y_clipped) > 0:
        effective_window = min(len(y_clipped), smoothing_window_size)
        y_smoothed = uniform_filter1d(
            y_clipped, size=effective_window, mode=boundary_mode, origin=0
        )
        x_smoothed = x_values
    else:
        y_smoothed = np.array([])
        x_smoothed = np.array([])
    # --- 3. 绘图 ---
    if plot_raw:
        plt.plot(
            x_values,
            y_clipped,
            color=original_color_hex,
            alpha=0.25,
            linewidth=1.2,
            label=None,
            zorder=1,
        )
    (line,) = plt.plot(
        x_smoothed,
        y_smoothed,
        color=smoothed_color_hex,
        linewidth=2.2,
        zorder=2,
        **plot_kwargs,
    )
    return line, y_smoothed


if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    json_path = f"plots/data/wandb/{run_name}.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"未找到文件: {json_path}")
        exit(1)
    if len(data) == 0:
        print(f"文件为空: {json_path}")
        exit(1)
    data = data[:-1]
    x = np.array(
        [
            item["train/global_step"]
            for item in data
            if "train/global_step" in item and "train/reward" in item
        ]
    )
    y = np.array(
        [
            item["train/reward"]
            for item in data
            if "train/global_step" in item and "train/reward" in item
        ]
    )
    # 只保留从100步开始的数据
    mask = x >= 100
    x = x[mask]
    y = y[mask]
    # 绘图
    plot_smoothed_timeseries_full_range(
        x_values=x,
        y_values=y,
        smoothing_window_size=200,
        outlier_detection_window_size=30,
        outlier_std_factor=2.5,
        original_color_hex=color + "55",
        smoothed_color_hex=color,
        boundary_mode="reflect",
        plot_raw=True,
        label=label,
    )
    # 统计分析
    mean_y = np.mean(y) if len(y) > 0 else float("nan")
    std_y = np.std(y) if len(y) > 0 else float("nan")
    print("\n================= 总奖励统计/Total Reward Statistics =================")
    print(
        f"全步骤 {run_name} (Qwen2.5-3B-Instruct-QLoRA (4-bit) 长训练) 总奖励均值/Mean: {mean_y:.4f}, 标准差/Std: {std_y:.4f}, 样本数/N: {len(y)}"
    )
    print(
        "==========================================================================\n"
    )
    plt.xlabel("RL 训练步数（对数尺度）", fontsize=13, fontweight="bold", labelpad=15)
    plt.ylabel("总奖励值", fontsize=13, fontweight="bold", labelpad=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    ax = plt.gca()
    # x轴对数尺度，刻度与 plot_kl.py 6号图一致
    ax.set_xscale("log")
    ax.set_ylim(top=3.9)
    xticks_major = [100, 200, 300, 500, 1000, 2000, 4000, 7000]
    ax.set_xticks(xticks_major)
    ax.set_xticklabels([str(xx) for xx in xticks_major])
    ax.set_xlim(min(x) * 0.8, max(x) * 1.2)
    ax.xaxis.grid(True, which="major")
    ax.xaxis.grid(False, which="minor")
    # 设置所有边框为黑色且加粗
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")
    plt.tight_layout()
    save_path = "plots/output/6_reward.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已保存: {save_path}")
