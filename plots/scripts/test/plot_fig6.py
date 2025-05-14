import matplotlib.pyplot as plt
import numpy as np
import math

# --- 图表样式与字体设置 ---
plt.style.use('seaborn-v0_8-whitegrid')

preferred_serif_fonts = [
    "Noto Serif CJK JP", "Songti SC", "SimSun",
    "WenQuanYi Zen Hei", "serif",
]
preferred_sans_serif_fonts = [
    "Noto Sans CJK JP", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "Heiti SC", "Microsoft YaHei", "sans-serif",
]
plt.rcParams["font.serif"] = preferred_serif_fonts
plt.rcParams["font.sans-serif"] = preferred_sans_serif_fonts
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

base_fontsize = 12
title_fontsize = base_fontsize + 5
axes_label_fontsize = base_fontsize + 2
tick_label_fontsize = base_fontsize
legend_fontsize = base_fontsize
value_on_point_fontsize = base_fontsize - 2 # 点上数值的字号

plt.rcParams['axes.labelsize'] = axes_label_fontsize
plt.rcParams['xtick.labelsize'] = tick_label_fontsize
plt.rcParams['ytick.labelsize'] = tick_label_fontsize
plt.rcParams['legend.fontsize'] = legend_fontsize
plt.rcParams['lines.markersize'] = 7 # RL曲线的标记大小
plt.rcParams['lines.linewidth'] = 2.0 # 基础线条宽度

# --- 数据定义 ---
model_name_display = "Qwen2.5-3B (4-bit) 长训练（简单奖励）"
model_name_file = model_name_display.replace(' ', '_').replace('(', '').replace(')', '').replace('/','_')

# RL训练数据 (从100步开始)
long_train_steps_rl = np.array([100, 200, 300, 1000, 2000, 4000, 7000])
long_train_accuracies_rl = np.array([77.33, 76.50, 77.26, 70.43, 68.92, 70.51, 63.05])

# X=0 时的基准值
baseline_no_template_accuracy = 73.82
baseline_with_template_accuracy = 78.22

# 颜色方案
color_baseline_orig = '#778899'             # 浅石板灰 (LightSlateGray) - 作为最底层的基线
color_baseline_prompt = '#FF8C00'           # 深橙色 (DarkOrange)
color_rl_long_train = 'forestgreen'          # 森绿色

# 标记样式 (RL曲线使用)
marker_rl_long_train = 's' # 方形标记

# 线条样式
linestyle_baseline_orig = ':'               # 点线 (更细微)
linestyle_baseline_prompt = '--'            # 虚线
linestyle_rl_tuned = '-'                    # 实线


fig, ax = plt.subplots(figsize=(12, 7))

# 1. 设置X轴为对数尺度
ax.set_xscale('log')

# 2. 绘制 "基线" 的水平参考线
ax.axhline(y=baseline_no_template_accuracy,
           color=color_baseline_orig,
           linestyle=linestyle_baseline_orig,
           linewidth=plt.rcParams['lines.linewidth'] - 0.2, # 比对话模版线略细
           label=f'基线 ({baseline_no_template_accuracy:.2f}%)')

# 3. 绘制 "基线 + 对话模版" 的水平参考线
ax.axhline(y=baseline_with_template_accuracy,
           color=color_baseline_prompt,
           linestyle=linestyle_baseline_prompt,
           linewidth=plt.rcParams['lines.linewidth'],
           label=f'基线 + 对话模版 ({baseline_with_template_accuracy:.2f}%)')

# 4. 绘制RL训练曲线 (从100步开始)
ax.plot(long_train_steps_rl, long_train_accuracies_rl,
        marker=marker_rl_long_train,
        color=color_rl_long_train,
        linestyle=linestyle_rl_tuned,
        linewidth=plt.rcParams['lines.linewidth'] + 0.3, # RL曲线略粗
        label='RL (简单奖励+长训练)')

# 5. 在RL曲线的点上显示数值
show_values_on_points = True
if show_values_on_points:
    for i, step in enumerate(long_train_steps_rl):
        ax.text(step, long_train_accuracies_rl[i] + 0.5, f'{long_train_accuracies_rl[i]:.2f}',
                ha='center', va='bottom', fontsize=value_on_point_fontsize, color='black') # 数值颜色统一为黑色以保证清晰

# --- 设置图表标题和标签 ---
ax.set_title(f'{model_name_display} 性能变化', fontweight='bold', fontsize=title_fontsize, pad=20)
ax.set_xlabel('RL 训练步数', fontweight='bold', labelpad=15)
ax.set_ylabel('GSM8K 准确率 (%)', fontweight='bold', labelpad=15)

# --- 设置坐标轴刻度和范围 ---
x_ticks_to_show = np.unique(np.sort(long_train_steps_rl))
ax.set_xticks(x_ticks_to_show)
ax.set_xticklabels([str(int(s)) for s in x_ticks_to_show])
ax.set_xlim(long_train_steps_rl[0] * 0.8, long_train_steps_rl[-1] * 1.2) # X轴范围基于RL数据点

all_y_values_plot = np.concatenate(([baseline_no_template_accuracy, baseline_with_template_accuracy], long_train_accuracies_rl))
y_min_plot = math.floor(np.min(all_y_values_plot)) - 1
y_max_plot = math.ceil(np.max(all_y_values_plot)) + 1
ax.set_ylim(y_min_plot, y_max_plot)
ax.set_yticks(np.arange(y_min_plot, y_max_plot + 1, 2))

# --- 图例 ---
legend = ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=True, borderpad=0.8, labelspacing=0.7)
legend.get_frame().set_edgecolor('darkgray')
legend.get_frame().set_linewidth(0.6)


# --- 网格线 ---
ax.grid(True, which="major", ls=":", alpha=0.6, linewidth=0.7)

# ... (其他美化和保存代码) ...
plt.tight_layout(pad=1.5)
file_name = f"{model_name_file}_log_plot_two_lines.png"
plt.savefig(file_name, dpi=300, bbox_inches='tight')
print(f"图表已保存为: {file_name}")
plt.show()