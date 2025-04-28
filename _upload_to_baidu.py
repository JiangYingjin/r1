import os
import subprocess
import glob
import shutil


def upload_to_baidu(local_path, remote_path):
    """上传文件夹到百度网盘"""
    cmd = f"baidu u {local_path} {remote_path}"
    print(f"执行命令: {cmd}")
    with open("_upload.log", "a") as log_file:
        print(f"执行命令: {cmd}\n")
        result = subprocess.run(
            cmd, shell=True, stdout=log_file, stderr=log_file, text=True
        )
    if result.returncode == 0:
        print(f"上传成功: {local_path} -> {remote_path}")
        # 上传成功后移动文件夹
        uploaded_dir = os.path.expanduser("~/lanyun-fs/outputs/uploaded")
        os.makedirs(uploaded_dir, exist_ok=True)
        dest_path = os.path.join(uploaded_dir, os.path.basename(local_path))
        shutil.move(local_path, dest_path)
        print(f"已移动到: {dest_path}")
    else:
        print(f"上传失败: {local_path}")
        print(f"错误信息: {result.stderr}")


def main():
    """主函数，列出所有文件夹并上传"""
    base_dir = os.path.expanduser("~/lanyun-fs/outputs")
    remote_dir = "/share/proj/r1/ckpt"

    # 获取所有文件夹
    folders = [f for f in glob.glob(f"{base_dir}/*") if os.path.isdir(f)]

    if not folders:
        print(f"在 {base_dir} 中没有找到文件夹")
        return

    print(f"找到 {len(folders)} 个文件夹:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {os.path.basename(folder)}")

    # 逐个上传文件夹
    for folder in folders:
        upload_to_baidu(folder, remote_dir)


if __name__ == "__main__":
    main()
