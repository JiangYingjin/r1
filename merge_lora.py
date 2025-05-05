from unsloth import FastLanguageModel
from pathlib import Path
from tqdm import tqdm

CKPT_DIR = Path("/root/lanyun-tmp/r1/exp/Qwen_Qwen2.5-3B-Instruct/gsmplus600_course_1/ckpt")


def find_checkpoints(ckpt_dir: Path):
    """查找所有 checkpoint-xxxx 目录，返回 Path对象列表"""
    checkpoints = []
    for item in ckpt_dir.iterdir():
        if (
            item.is_dir()
            and item.name.startswith("checkpoint-")
            and not item.name.endswith("_merged")
        ):
            checkpoints.append(item)

    print(f"找到 {len(checkpoints)} 个 checkpoint：")
    for checkpoint_path in checkpoints:
        print(f"  {checkpoint_path.name}")

    return checkpoints


def merge_and_save(ckpt_dir: Path):

    merged_dir = Path(ckpt_dir).parent / (ckpt_dir.name + "_merged")

    if merged_dir.exists():
        print(f"{merged_dir.name} 已存在，跳过处理")
        return

    try:
        print(f"正在处理 {ckpt_dir.name}...")
        model, tokenizer = FastLanguageModel.from_pretrained(str(ckpt_dir))
        print(f"成功加载 {ckpt_dir.name} 模型")

        print(f"正在将模型保存到 {merged_dir}，使用 merged_16bit 格式...")
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"{ckpt_dir.name} 已成功保存为 merged_16bit 格式")
    except Exception as e:
        print(f"合并失败: {e}")


def main():
    checkpoints = find_checkpoints(CKPT_DIR)
    for ckpt_dir in tqdm(checkpoints):
        merge_and_save(ckpt_dir)


if __name__ == "__main__":
    main()

"""
save_method=
    merged_16bit (你使用的): 合并后保存为 16-bit (通常是 float16 或 bfloat16)。这是一个很好的平衡点，模型大小适中，精度损失小，加载后可以直接使用标准 Transformers 库进行推理。
    merged_4bit: 合并后保存为 4-bit 量化模型。最节省空间，推理速度最快（在支持的环境下）。适用于部署资源受限（显存、磁盘）且对轻微精度下降不敏感的场景。这是 Unsloth 推荐的部署方式之一。
    merged_8bit: 合并后保存为 8-bit 量化模型。介于 16-bit 和 4-bit 之间，是空间、速度和精度的一个折中选项。
    lora: 只保存适配器。适用于 分享微调结果（只需分享小小的适配器文件），或者希望保留基础模型不变，在不同任务间切换适配器的场景。加载推理时需要额外步骤来合并。
"""
