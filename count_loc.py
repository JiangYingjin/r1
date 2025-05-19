import os
import fnmatch

# 要统计的文件类型
EXTS = [".py", ".sh", ".yml", ".jinja"]
# 要跳过的目录前缀
SKIP_DIRS = [
    ".git",
    "__pycache__",
    "data/raw",
    "eval",
    "examples",
    "grpo",
    "unsloth",
    "wandb",
    "rewards/better_reward_3",
]


def should_skip_dir(dirname):
    for skip in SKIP_DIRS:
        if (
            dirname == skip
            or dirname.startswith(skip)
            or dirname.startswith(f"{skip}/")
        ):
            return True
    return False


def count_file_lines(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"无法读取文件 {filepath}: {e}")
        return 0


def count_dir_lines(target_dir):
    file_counts = []
    total = 0
    for root, dirs, files in os.walk(target_dir, topdown=True):
        rel_root = os.path.relpath(root, target_dir)
        if rel_root == ".":
            rel_root = ""
        skip = False
        for d in SKIP_DIRS:
            if rel_root == d or rel_root.startswith(d) or rel_root.startswith(f"{d}/"):
                skip = True
                break
        if skip:
            dirs[:] = []
            continue
        for file in files:
            for ext in EXTS:
                if file.endswith(ext):
                    filepath = os.path.join(root, file)
                    count = count_file_lines(filepath)
                    file_counts.append((filepath, count))
                    total += count
                    break
    return file_counts, total


def main():
    # 统计全局
    all_file_counts, all_total = count_dir_lines(".")
    dir_counts = {}
    print("每个文件的代码行数：")
    for filepath, count in all_file_counts:
        print(f"{filepath}: {count}")
        filepath_split = filepath.split("/")
        if len(filepath_split) == 2:
            dir_name = filepath_split[0]
            if dir_name not in dir_counts:
                dir_counts[dir_name] = 0
            dir_counts[dir_name] += count
        elif len(filepath_split) > 2:
            dir_name = filepath_split[1]
            if dir_name not in dir_counts:
                dir_counts[dir_name] = 0
            dir_counts[dir_name] += count
    print(f"总代码行数: {all_total}")
    print("\n---\n")
    print("每个一级文件夹的总代码行数：")
    for dir_name, count in sorted(dir_counts.items()):
        print(f"{dir_name}: {count}")


if __name__ == "__main__":
    main()
