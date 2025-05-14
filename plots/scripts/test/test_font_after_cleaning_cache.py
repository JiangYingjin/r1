import matplotlib.font_manager

print("--- Matplotlib 当前识别的相关字体 ---")
# 获取所有字体
font_manager = matplotlib.font_manager.fontManager
# 筛选并打印可能相关的字体名称
relevant_fonts_found = False
for font in sorted(font_manager.ttflist, key=lambda x: x.name):
    name_lower = font.name.lower()
    # 扩展关键词以匹配您安装的字体
    keywords = ["noto", "cjk", "wenquan", "wqy", "song", "hei", "simsun", "yahei"]
    if any(keyword in name_lower for keyword in keywords):
        print(f"  名称: '{font.name}', 文件路径: '{font.fname}'")
        relevant_fonts_found = True

if not relevant_fonts_found:
    print("  未能从 Matplotlib 的字体列表中找到明确的 CJK 或常用中文字体。")
    print("  这可能意味着字体安装未被 Matplotlib 正确检测到，或者字体缓存未正确重建。")
print("------------------------------------")
