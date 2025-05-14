import matplotlib
import os
import shutil  # 导入 shutil 模块

# 获取 Matplotlib 缓存目录
cache_dir = matplotlib.get_cachedir()
print(f"Matplotlib 缓存目录: {cache_dir}")

if os.path.exists(cache_dir):
    try:
        # 更彻底地删除整个缓存目录
        shutil.rmtree(cache_dir)
        print(f"已成功删除整个缓存目录: {cache_dir}")
        print("请务必【重启】您的 Python 内核/环境，以便 Matplotlib 重建缓存。")
    except OSError as e:
        print(f"删除缓存目录 {cache_dir} 时出错: {e}")
        print("请尝试手动删除此目录，然后重启 Python 内核/环境。")
else:
    print(f"缓存目录 {cache_dir} 不存在，无需删除。")

# 提示用户重启
print("\n*** 重要提示 ***")
print(
    "字体缓存已尝试清除。您现在【必须】重启您的 Python 内核（例如，如果您在 Jupyter Notebook 中，请选择 Kernel -> Restart Kernel）。"
)
print("如果不重启，Matplotlib 将不会重新扫描字体，问题可能依旧存在。")
