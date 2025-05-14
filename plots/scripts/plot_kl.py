import json  # 新增导入

import numpy as np
import pandas as pd  # 导入 pandas 用于辅助处理（如异常值检测的滚动统计）
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d  # 导入 scipy 的一维均匀滤波器


def plot_smoothed_timeseries_full_range(
    x_values,  # X轴的数据点
    y_values,  # Y轴的数据点
    smoothing_window_size=50,  # 滑动平均的窗口大小
    outlier_detection_window_size=20,  # 用于异常值检测的滚动统计窗口大小
    outlier_std_factor=3,  # 定义异常值的标准差倍数（超过此倍数则视为异常）
    original_color_hex="#a9d6e5",  # 原始数据线条的十六进制颜色代码（浅蓝色）
    smoothed_color_hex="#014f86",  # 平滑后数据线条的十六进制颜色代码（深蓝色）
    figsize=(12, 6),  # 图形的尺寸，格式为 (宽度, 高度)
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
    sns.set_style("whitegrid")  # 设置 Seaborn 绘图风格为 "whitegrid"（白色网格背景）

    # 绘制原始 (裁剪后) 数据
    plt.plot(
        x_values,
        y_clipped,
        color=original_color_hex,
        alpha=0.4,
        linewidth=1.5,
        label="原始数据 (异常值已裁剪)",
    )

    # 绘制平滑后的数据
    if len(y_smoothed) > 0:
        plt.plot(
            x_smoothed,
            y_smoothed,
            color=smoothed_color_hex,
            linewidth=2.5,
            label=f"滑动平均 (窗口: {smoothing_window_size}, 边界: {boundary_mode})",
        )

    plt.title(
        title, fontsize=16, fontweight="bold"
    )  # 设置图表标题，并指定字体大小和粗细
    plt.xlabel(xlabel, fontsize=12)  # 设置X轴标签，并指定字体大小
    plt.ylabel(ylabel, fontsize=12)  # 设置Y轴标签，并指定字体大小

    plt.xticks(fontsize=10)  # 设置X轴刻度标签的字体大小
    plt.yticks(fontsize=10)  # 设置Y轴刻度标签的字体大小

    sns.despine()  # 移除顶部和右侧的坐标轴边框，使图形更简洁（类似 WandB 风格）
    plt.legend(fontsize=10)  # 显示图例，并指定图例字体大小
    plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域，防止标签重叠
    plt.show()  # 显示图形


# # --- 生成示例数据 (模仿原始图像的特征) ---
# np.random.seed(42)  # 设置随机种子以保证结果可复现
# num_points = 280  # 数据点数量
# x = np.arange(num_points)  # 生成X轴数据 (0, 1, ..., num_points-1)

# # 基础趋势 (二次函数 + 正弦波以增加一些变化)
# y_trend = 0.0001 * (x - 50) ** 2 - 0.5 + 0.3 * np.sin(x / 30)
# # 噪声
# y_noise = np.random.normal(0, 0.4, num_points)  # 生成高斯噪声
# # 添加一些稀疏的、较大的 "异常值"
# num_outliers = 15  # 异常值数量
# outlier_indices = np.random.choice(
#     num_points, num_outliers, replace=False
# )  # 随机选择异常值的位置
# outlier_magnitudes = np.random.uniform(-2, 2, num_outliers)  # 为异常值生成随机的幅度

# # 原始信号（趋势+噪声）
# y_signal_base = y_trend + y_noise
# # 使趋势在末端更强
# growth_factor = np.linspace(1, 2.5, num_points)  # 生成一个线性增长因子
# y_signal = y_signal_base * growth_factor  # 将信号与增长因子相乘
# y_signal[outlier_indices] += (
#     outlier_magnitudes * 1.5
# )  # 在指定位置添加异常值（异常值也受增长影响）


# # --- 调用绘图函数 ---
# plot_smoothed_timeseries_full_range(
#     x_values=x,
#     y_values=y_signal,
#     smoothing_window_size=50,
#     outlier_detection_window_size=30,  # 用于异常值检测的窗口
#     outlier_std_factor=2.5,  # 定义异常值的标准差阈值
#     title="训练奖励随训练步数的变化 (Reflect边界)",
#     xlabel="全局步数",
#     ylabel="累积奖励",
#     boundary_mode="reflect",  # 指定边界处理模式为 'reflect'
# )


# 你也可以尝试其他的 boundary_mode，例如：
# boundary_mode='nearest' # 使用最近的边界值填充
# boundary_mode='mirror'  # 类似反射，但边界点只使用一次
# boundary_mode='constant', cval=0.0 # 使用常数填充，cval指定常数值

if __name__ == "__main__":
    # 1. 读取 JSON 文件
    json_path = "plots/data/wandb/better_reward_3.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 2. 去掉最后一个数据点
    data = data[:-1]
    # 3. 提取 x 和 y
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
    # 4. 调用绘图函数
    plot_smoothed_timeseries_full_range(
        x_values=x,
        y_values=y,
        smoothing_window_size=50,
        outlier_detection_window_size=30,
        outlier_std_factor=2.5,
        title="KL散度随训练步数的变化 (Reflect边界)\nKL Divergence vs. Training Steps (Reflect)",
        xlabel="全局步数 Global Step",
        ylabel="KL散度 KL Divergence",
        boundary_mode="reflect",
    )
