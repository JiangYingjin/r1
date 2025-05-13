import matplotlib.pyplot as plt

# --- Matplotlib Configuration for Dissertation Figures (Chinese Support) ---

# 1. Define preferred font lists based on installed fonts and academic recommendations.
#    Matplotlib will try fonts in the order they appear in these lists.

# For Serif fonts (ideal for axis labels, figure text resembling body text):
# We prioritize Noto Serif CJK JP (Simplified Chinese variant).
# If your matplotlib setup resolves "Noto Serif CJK JP" using the "Noto Serif CJK JP"
# files you have, that's ideal. Otherwise, explicitly listing "Noto Serif CJK JP"
# (as found in your font list) is a direct way.
preferred_serif_fonts = [
    "Noto Serif CJK JP",  # Preferred: Specific Simplified Chinese Serif (ideal if resolved)
    "Noto Serif CJK JP",  # From your installed list (contains SC glyphs)
    "Songti SC",  # Common system fallback for Simplified Chinese Serif
    "SimSun",  # Another common Windows Simplified Chinese Serif
    "WenQuanYi Zen Hei",  # Fallback to a Sans-Serif CJK if no Serif CJK is found
    "serif",  # Generic system serif fallback
]

# For Sans-serif fonts (ideal for titles, annotations, modern-looking text):
# Similar logic for Noto Sans CJK JP.
preferred_sans_serif_fonts = [
    "Noto Sans CJK JP",  # Preferred: Specific Simplified Chinese Sans-serif
    "Noto Sans CJK JP",  # From your installed list (contains SC glyphs)
    "WenQuanYi Micro Hei",  # From your installed list - good CJK Sans-serif
    "WenQuanYi Zen Hei",  # From your installed list - good CJK Sans-serif
    "Heiti SC",  # Common system fallback for Simplified Chinese Sans-serif
    "Microsoft YaHei",  # Common Windows Simplified Chinese Sans-serif
    "sans-serif",  # Generic system sans-serif fallback
]

# Apply the font lists to matplotlib's rcParams
plt.rcParams["font.serif"] = preferred_serif_fonts
plt.rcParams["font.sans-serif"] = preferred_sans_serif_fonts

# 2. Set the default font family.
#    For academic papers, using a serif font for axis labels (like body text)
#    and a sans-serif font for titles is a common and good practice.
#    Let's default to 'serif' for general elements like axis labels.
plt.rcParams["font.family"] = "serif"

# 3. Ensure correct display of minus sign with Unicode fonts
plt.rcParams["axes.unicode_minus"] = False

# 4. Optional: Adjust default font sizes for consistency in your dissertation.
#    Uncomment and modify these as needed.
# plt.rcParams['font.size'] = 10  # Base font size for text elements
# plt.rcParams['axes.titlesize'] = 14 # Font size for axes titles (e.g., plot title)
# plt.rcParams['axes.labelsize'] = 12 # Font size for x and y labels
# plt.rcParams['xtick.labelsize'] = 10 # Font size for x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 10 # Font size for y-axis tick labels
# plt.rcParams['legend.fontsize'] = 10 # Font size for legend
# plt.rcParams['figure.titlesize'] = 16 # Font size for suptitle (figure-level title)

# --- End of Configuration ---

# --- Verification and Important Notes ---
print("Matplotlib configuration for Chinese dissertation figures applied.")
print(f"Default font family set to: {plt.rcParams['font.family']}")
print(f"Serif font list: {plt.rcParams['font.serif']}")
print(f"Sans-serif font list: {plt.rcParams['font.sans-serif']}")

# IMPORTANT:
# 1. Font Cache: If you've recently installed these fonts or changed configurations,
#    matplotlib might not pick them up immediately. You may need to clear
#    matplotlib's font cache. The cache file is typically located in a directory
#    like `~/.cache/matplotlib/` (Linux/macOS) or `C:\\Users\\<username>\\.matplotlib\\` (Windows).
#    Delete any `fontlist-vXXX.json` or similar `.cache` files in that directory,
#    and then restart your Python kernel/environment.
#
# 2. Specificity: The names 'Noto Serif CJK JP' and 'Noto Sans CJK JP' are standard
#    names for the Simplified Chinese variants. Matplotlib's font manager will attempt
#    to find them. If it can't, it will proceed down the list. Since your installed
#    list explicitly showed 'Noto Serif CJK JP' and 'Noto Sans CJK JP', these are
#    included as they are known to be available and typically cover SC glyphs.
#
# 3. University Guidelines: Always prioritize any specific typographic guidelines
#    provided by your university or department for dissertations.
#
# 4. Font Manager Check (Optional): You can list all fonts matplotlib recognizes:
#    fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
#    print("\\nAvailable font names to Matplotlib:")
#    for font_name in sorted(list(fonts)):
#        if "CJK" in font_name or "WenQuanYi" in font_name or "Song" in font_name or "Hei" in font_name:
#             print(font_name)


# --- Example Usage ---
# This demonstrates how to use the configured defaults and override if needed.

# plt.figure(figsize=(7, 5)) # Adjust figure size as needed for your dissertation

# Title: Typically use a Sans-serif font for clarity and modern look.
# Since default is serif, we specify fontname here.
# plt.title("图表示例：不同处理方法的比较", fontname='Noto Sans CJK JP', fontsize=14, fontweight='bold')
# If Noto Sans CJK JP isn't found, it will try Noto Sans CJK JP, then WenQuanYi Micro Hei etc.

# Axis labels will use the default serif font (e.g., Noto Serif CJK JP)
# plt.xlabel("实验条件 (单位)", fontsize=12)
# plt.ylabel("观测指标 (单位)", fontsize=12)

# Example data
# x_data = ['处理A', '处理B', '处理C', '对照组']
# y_data = [25.3, 28.1, 22.5, 15.0]
# plt.bar(x_data, y_data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Adding text annotation - can also specify font properties
# plt.text(x_data[1], y_data[1] + 0.5, f'{y_data[1]} units',
#          ha='center', fontname='Noto Sans CJK JP', fontsize=10)

# Legend - will use default font settings unless overridden
# (create some dummy lines for legend example)
# plt.plot([], [], color='#1f77b4', label='方案一')
# plt.plot([], [], color='#ff7f0e', label='方案二')
# plt.legend(title="图例说明")


# plt.xticks(fontname='Noto Serif CJK JP') # Ensure ticks also use desired font if needed
# plt.yticks(fontname='Noto Serif CJK JP')

# plt.tight_layout() # Adjust layout to prevent labels from overlapping
# plt.savefig("dissertation_figure_example_chinese.png", dpi=300) # Save with high DPI for print
# plt.show()

# 配置支持中英文的字体
preferred_serif_fonts = [
    "Noto Serif CJK JP",
    "Noto Serif CJK JP",
    "Songti SC",
    "SimSun",
    "WenQuanYi Zen Hei",
    "serif",
]
preferred_sans_serif_fonts = [
    "Noto Sans CJK JP",
    "Noto Sans CJK JP",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Heiti SC",
    "Microsoft YaHei",
    "sans-serif",
]
plt.rcParams["font.serif"] = preferred_serif_fonts
plt.rcParams["font.sans-serif"] = preferred_sans_serif_fonts
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

print("Matplotlib 中文/English 字体配置已应用。")
print(f"默认字体: {plt.rcParams['font.family']}")
print(f"Serif 字体列表: {plt.rcParams['font.serif']}")
print(f"Sans-serif 字体列表: {plt.rcParams['font.sans-serif']}")

# 示例用法 Example usage
plt.figure(figsize=(7, 5))
plt.title(
    "图表示例/Example: 不同处理方法的比较",
    fontname="Noto Sans CJK JP",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("实验条件 (单位)/Condition", fontsize=12)
plt.ylabel("观测指标 (单位)/Metric", fontsize=12)
x_data = ["处理A", "处理B", "处理C", "对照组"]
y_data = [25.3, 28.1, 22.5, 15.0]
plt.bar(x_data, y_data, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
plt.text(
    x_data[1],
    y_data[1] + 0.5,
    f"{y_data[1]} units",
    ha="center",
    fontname="Noto Sans CJK JP",
    fontsize=10,
)
plt.legend(["方案一/Plan A", "方案二/Plan B"], title="图例说明/Legend")
plt.xticks(fontname="Noto Serif CJK JP")
plt.yticks(fontname="Noto Serif CJK JP")
plt.tight_layout()
plt.savefig("dissertation_figure_example_chinese.png", dpi=300)
plt.show()
