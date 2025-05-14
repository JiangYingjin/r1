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

fig_run_name = {
    5: "course_2",
    1: "better_reward_3",
    2: "better_reward_qwen2.5_1.5b",
    3: "better_reward_llama",
}

color_map = {
    5: "#D55E00",  # Deep Orange (Protagonist)
    1: "#0072B2",  # Darker Blue (Control 1)
    2: "#56B4E9",  # Lighter/Sky Blue (Control 3)
    3: "#009E73",  # Teal/Green (Control 2)
}
label_map = {
    5: "Qwen2.5-3B-Instruct-QLoRA (4-bit) + 长响应奖励函数",
    1: "Qwen2.5-3B-Instruct-QLoRA (4-bit)",
    2: "Qwen2.5-1.5B-Instruct-QLoRA (4-bit)",
    3: "Llama-3.2-3B-Instruct-QLoRA (4-bit)",
}


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
    group_y_after_100 = {}
    for fig_key, run_name in fig_run_name.items():
        json_path = f"plots/data/wandb/{run_name}.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"未找到文件: {json_path}")
            continue
        if len(data) == 0:
            print(f"文件为空: {json_path}")
            continue
        data = data[:-1]
        x = np.array(
            [
                item["train/global_step"]
                for item in data
                if "train/global_step" in item and "train/completion_length" in item
            ]
        )
        y = np.array(
            [
                item["train/completion_length"]
                for item in data
                if "train/global_step" in item and "train/completion_length" in item
            ]
        )
        # 只保留1~5在1~300步的数据
        mask = (x >= 1) & (x <= 300)
        x = x[mask]
        y = y[mask]
        mask_100 = x > 100
        group_y_after_100[fig_key] = y[mask_100]
        # # 调试信息
        # print(f"[DEBUG] fig_key={fig_key}, run_name={run_name}")
        # print(f"  x.shape={x.shape}, y.shape={y.shape}")
        # print(
        #     f"  x范围: min={x.min() if len(x)>0 else 'NA'}, max={x.max() if len(x)>0 else 'NA'}"
        # )
        # print(f"  mask_100.sum={mask_100.sum()}, 100步后y长度={len(y[mask_100])}")
        # if len(y[mask_100]) > 0:
        #     print(f"  100步后y前5: {y[mask_100][:5]}")
        # else:
        #     print(f"  100步后y为空")
        color = color_map[fig_key]
        label = label_map[fig_key]
        plot_smoothed_timeseries_full_range(
            x_values=x,
            y_values=y,
            smoothing_window_size=10,
            outlier_detection_window_size=30,
            outlier_std_factor=2.5,
            original_color_hex=color + "55",
            smoothed_color_hex=color,
            boundary_mode="reflect",
            plot_raw=True,
            label=label,
        )
    # # --- 统计分析 ---
    # print("\n[DEBUG] group_y_after_100 keys:", list(group_y_after_100.keys()))
    # for k, v in group_y_after_100.items():
    #     print(
    #         f"[DEBUG] fig_key={k}, 100步后y长度={len(v)}, 前5: {v[:5] if len(v)>0 else '[]'}"
    #     )
    # 5组
    y_5 = group_y_after_100.get(5, np.array([]))
    # 1,2,3组
    y_1 = group_y_after_100.get(1, np.array([]))
    y_2 = group_y_after_100.get(2, np.array([]))
    y_3 = group_y_after_100.get(3, np.array([]))
    y_123 = np.concatenate([y_1, y_2, y_3])
    # 计算均值和标准差
    mean_5 = np.mean(y_5) if len(y_5) > 0 else float("nan")
    std_5 = np.std(y_5) if len(y_5) > 0 else float("nan")
    mean_1 = np.mean(y_1) if len(y_1) > 0 else float("nan")
    std_1 = np.std(y_1) if len(y_1) > 0 else float("nan")
    mean_2 = np.mean(y_2) if len(y_2) > 0 else float("nan")
    std_2 = np.std(y_2) if len(y_2) > 0 else float("nan")
    mean_3 = np.mean(y_3) if len(y_3) > 0 else float("nan")
    std_3 = np.std(y_3) if len(y_3) > 0 else float("nan")
    mean_123 = np.mean(y_123) if len(y_123) > 0 else float("nan")
    std_123 = np.std(y_123) if len(y_123) > 0 else float("nan")
    ratio = mean_5 / mean_123 if mean_123 != 0 else float("nan")
    # 打印结果
    print(
        "\n================= 响应长度统计/Response Length Statistics ================="
    )
    print(
        f"100步后 5 (Qwen2.5-3B-Instruct-QLoRA (4-bit) + 长响应奖励函数) 响应长度均值/Mean: {mean_5:.2f}, 标准差/Std: {std_5:.2f}, 样本数/N: {len(y_5)}"
    )
    print(
        f"100步后 1 (Qwen2.5-3B-Instruct-QLoRA (4-bit)) 响应长度均值/Mean: {mean_1:.2f}, 标准差/Std: {std_1:.2f}, 样本数/N: {len(y_1)}"
    )
    print(
        f"100步后 2 (Qwen2.5-1.5B-Instruct-QLoRA (4-bit)) 响应长度均值/Mean: {mean_2:.2f}, 标准差/Std: {std_2:.2f}, 样本数/N: {len(y_2)}"
    )
    print(
        f"100步后 3 (Llama-3.2-3B-Instruct-QLoRA (4-bit)) 响应长度均值/Mean: {mean_3:.2f}, 标准差/Std: {std_3:.2f}, 样本数/N: {len(y_3)}"
    )
    print(
        f"100步后 1,2,3 总体响应长度均值/Mean (all 1,2,3): {mean_123:.2f}, 标准差/Std: {std_123:.2f}, 样本数/N: {len(y_123)}"
    )
    print(
        f"5 (Qwen2.5-3B-Instruct-QLoRA (4-bit) + 长响应奖励函数) 是 1,2,3 总均值的/Ratio: {ratio:.3f} 倍"
    )
    print(
        "==========================================================================\n"
    )
    plt.xlabel("RL 训练步数", fontsize=13, fontweight="bold", labelpad=15)
    plt.ylabel("模型响应长度", fontsize=13, fontweight="bold", labelpad=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    ax = plt.gca()
    ax.set_ylim(top=1500)  # 设置y轴最大值为1600
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")
    plt.tight_layout()
    save_path = "plots/output/response_length.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已保存: {save_path}")
