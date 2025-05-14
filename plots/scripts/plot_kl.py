import json
import numpy as np
import pandas as pd  # 导入 pandas 用于辅助处理（如异常值检测的滚动统计）
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # 导入 scipy 的一维均匀滤波器

# --- 图表样式与字体设置 ---
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams["font.serif"] = ["Noto Serif CJK JP"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

fig_run_name = {
    1: "better_reward_3",
    2: "better_reward_qwen2.5_1.5b",
    3: "better_reward_llama",
    4: "gsmplus600_course_1",
    5: "course_2",
    6: "20250427_181546",
}


def plot_smoothed_timeseries_full_range(
    x_values,  # X轴的数据点
    y_values,  # Y轴的数据点
    smoothing_window_size=50,  # 滑动平均的窗口大小
    outlier_detection_window_size=20,  # 用于异常值检测的滚动统计窗口大小
    outlier_std_factor=3,  # 定义异常值的标准差倍数（超过此倍数则视为异常）
    original_color_hex="#a9d6e5",  # 原始数据线条的十六进制颜色代码（浅蓝色）
    smoothed_color_hex="#014f86",  # 平滑后数据线条的十六进制颜色代码（深蓝色）
    figsize=(8, 6),  # 图形的尺寸，格式为 (宽度, 高度)
    # figsize=(12, 6),  # 图形的尺寸，格式为 (宽度, 高度)
    title="指标随训练步数的变化",  # 图表的标题
    xlabel="train/global_step",  # X轴的标签
    ylabel="数值",  # Y轴的标签
    boundary_mode="reflect",  # 边界处理模式，例如 'reflect', 'nearest', 'mirror', 'constant', 'wrap'
):
    """
    绘制原始时间序列数据及其滑动平均值，覆盖完整数据范围，
    应用异常值裁剪，并采用 WandB 风格的美化。

    参数:
        x_values (np.ndarray): X坐标数据。
        y_values (np.ndarray): Y坐标数据。
        smoothing_window_size (int): 移动平均的窗口大小。
        outlier_detection_window_size (int): 用于计算滚动均值/标准差以检测异常值的窗口大小。
        outlier_std_factor (float): 数据点与滚动均值的差超过此标准差倍数时，被视为异常值并进行裁剪。
        original_color_hex (str): 原始数据线的十六进制颜色。
        smoothed_color_hex (str): 平滑数据线的十六进制颜色。
        figsize (tuple): 图形尺寸。
        title (str): 图表标题。
        xlabel (str): X轴标签。
        ylabel (str): Y轴标签。
        boundary_mode (str): 传递给 uniform_filter1d 的边界处理模式。
                             'reflect' (反射) 模式通常能在边界处提供较好的平滑效果。
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values 和 y_values 的长度必须相同。")
    if len(y_values) == 0:
        print("没有数据可供绘制。")
        return

    # --- 1. 异常值处理 (裁剪) ---
    # 先做极端异常值舍弃
    if len(y_values) > 0:
        sorted_y = np.sort(y_values)
        idx_85 = int(0.85 * len(sorted_y))
        val_85 = sorted_y[idx_85]
        extreme_threshold = 3 * val_85
        mask = y_values <= extreme_threshold
        if not np.all(mask):
            print(f"有 {np.sum(~mask)} 个点被极端值舍弃 (>{extreme_threshold:.4f})")
        x_values = x_values[mask]
        y_values = y_values[mask]
    # 使用 pandas Series 以方便进行滚动计算
    y_series = pd.Series(y_values)

    # 计算用于异常值检测的滚动均值和标准差
    # center=True 表示窗口围绕当前点居中
    # min_periods=1 确保即使在序列的开头/结尾，当窗口数据不足时也能计算出值
    rolling_mean = y_series.rolling(
        window=outlier_detection_window_size, center=True, min_periods=1
    ).mean()
    rolling_std = y_series.rolling(
        window=outlier_detection_window_size, center=True, min_periods=1
    ).std()

    # 如果滚动标准差为0或NaN（例如在平坦区域），则使用全局标准差作为备用值
    # 或者一个小的正数，以避免除以零或过度敏感的裁剪
    global_std_fallback = y_series.std() if not y_series.std() == 0 else 1.0
    rolling_std_filled = rolling_std.fillna(global_std_fallback)

    # 定义异常值的上下界
    lower_bound = rolling_mean - outlier_std_factor * rolling_std_filled
    upper_bound = rolling_mean + outlier_std_factor * rolling_std_filled

    # 将原始 y_values 裁剪到计算出的上下界之间
    # 注意：rolling_mean 和 rolling_std 是 Pandas Series，需要转换为 NumPy 数组与 y_values 进行运算
    y_clipped = np.clip(y_values, lower_bound.to_numpy(), upper_bound.to_numpy())

    # --- 2. 计算移动平均 ---
    # 使用 scipy.ndimage.uniform_filter1d 进行一维均匀滤波（即移动平均）
    # 这个函数能很好地处理边界，并返回与输入等长的数组。
    # 'size' 参数即为窗口大小。
    # 'mode' 参数控制边界如何处理。'reflect' 模式通过反射序列两端的数据来填充边界，
    # 使得滤波器在边界处也能获得足够的虚拟数据点进行计算，通常能产生自然的平滑效果。
    # 'origin=0' 参数控制滤波器的原点。对于奇数窗口大小，这相当于标准的中心化移动平均。
    # 对于偶数窗口大小，结果会相对于中心有半个像素的偏移，但对于可视化而言通常可以接受。

    if len(y_clipped) >= smoothing_window_size:  # 确保有足够数据进行有意义的平滑
        y_smoothed = uniform_filter1d(
            y_clipped, size=smoothing_window_size, mode=boundary_mode, origin=0
        )
        x_smoothed = x_values  # 输出与输入等长，所以x轴坐标不变
    elif len(y_clipped) > 0:  # 如果数据点少于平滑窗口，但仍有数据
        print(
            f"警告: 数据长度 ({len(y_clipped)}) 小于平滑窗口大小 ({smoothing_window_size})。"
            "平滑效果在端点处可能不佳或使用较小的有效窗口。"
        )
        # 回退方案：使用数据长度和窗口大小中较小的一个作为实际滤波窗口
        effective_window = min(len(y_clipped), smoothing_window_size)
        y_smoothed = uniform_filter1d(
            y_clipped, size=effective_window, mode=boundary_mode, origin=0
        )
        x_smoothed = x_values
    else:  # 没有数据
        y_smoothed = np.array([])
        x_smoothed = np.array([])

    # --- 3. 绘图 ---
    plt.figure(figsize=figsize)  # 创建新的图形，并设置大小

    # 绘制原始 (裁剪后) 数据
    plt.plot(
        x_values,
        y_clipped,
        color=original_color_hex,
        alpha=0.4,
        linewidth=1.5,
        label="原始值",
    )

    # 绘制平滑后的数据
    if len(y_smoothed) > 0:
        plt.plot(
            x_smoothed,
            y_smoothed,
            color=smoothed_color_hex,
            linewidth=2.5,
            label=f"平滑值",
            # label=f"滑动平均值",
            # label=f"滑动平均 (窗口: {smoothing_window_size}, 边界: {boundary_mode})",
        )

    # plt.title(
    #     title, fontsize=16, fontweight="bold"
    # )  # 设置图表标题，并指定字体大小和粗细
    plt.xlabel(
        xlabel, fontsize=13, fontweight="bold", labelpad=15
    )  # 设置X轴标签，并指定字体大小
    plt.ylabel(
        ylabel, fontsize=13, fontweight="bold", labelpad=15
    )  # 设置Y轴标签，并指定字体大小

    plt.xticks(fontsize=13)  # 设置X轴刻度标签的字体大小
    plt.yticks(fontsize=13)  # 设置Y轴刻度标签的字体大小

    # plt.legend(
    #     loc="upper left",
    #     frameon=True,
    #     shadow=False,
    #     borderpad=0.8,
    #     labelspacing=0.7,
    #     handletextpad=0.8,
    #     markerscale=0.9,
    #     fontsize=12,
    # )  # 显示图例，并指定图例字体大小
    plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域，防止标签重叠
    # plt.show()  # 显示图形


# 也可以尝试其他的 boundary_mode，例如：
# boundary_mode='nearest' # 使用最近的边界值填充
# boundary_mode='mirror'  # 类似反射，但边界点只使用一次
# boundary_mode='constant', cval=0.0 # 使用常数填充，cval指定常数值

if __name__ == "__main__":
    # 批量处理 fig_run_name 中的每个 run_name
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
        # 去掉最后一个数据点
        data = data[:-1]
        # 提取 x 和 y
        x = np.array(
            [
                item["train/global_step"]
                for item in data
                if "train/global_step" in item and "train/kl" in item
            ]
        )
        y = np.array(
            [
                item["train/kl"]
                for item in data
                if "train/global_step" in item and "train/kl" in item
            ]
        )
        # 只保留6_kl从100步开始的数据
        if fig_key == 6:
            mask = x >= 100
            x = x[mask]
            y = y[mask]
        # 只保留1~5_kl在1~300步的数据
        if fig_key in [1, 2, 3, 4, 5]:
            mask = (x >= 1) & (x <= 300)
            x = x[mask]
            y = y[mask]
        # 绘图并保存
        plt.figure()
        # 针对第6个图，调整滑动窗口为500，其余为50
        smoothing_window = 300 if fig_key == 6 else 50
        plot_smoothed_timeseries_full_range(
            x_values=x,
            y_values=y,
            smoothing_window_size=smoothing_window,
            outlier_detection_window_size=30,
            outlier_std_factor=2.5,
            xlabel="RL 训练步数",
            ylabel="KL 散度",
            boundary_mode="reflect",
        )
        # 完全对齐 plot_acc.py 的 6_acc x轴设置
        if fig_key == 6:
            ax = plt.gca()
            ax.set_xscale("log")
            # 只保留指定的x轴grid和刻度
            xticks_major = [100, 200, 300, 500, 1000, 2000, 4000, 7000]
            ax.set_xticks(xticks_major)
            ax.set_xticklabels([str(xx) for xx in xticks_major])
            ax.set_xlim(min(x) * 0.8, max(x) * 1.2)
            # 只显示指定的纵向grid（major grid）
            ax.xaxis.grid(True, which="major")  # 只画主刻度grid
            ax.xaxis.grid(False, which="minor")  # 不画次刻度grid
        # 设置所有边框为黑色且加粗
        ax = plt.gca()
        for spine in ["left", "bottom", "top", "right"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color("black")
        save_path = f"plots/output/{fig_key}_kl.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"已保存: {save_path}")
