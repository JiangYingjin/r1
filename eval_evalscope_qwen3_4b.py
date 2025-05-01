from evalscope import TaskConfig, run_task
from evalscope.collections import CollectionSchema, DatasetInfo
from evalscope.collections import (
    WeightedSampler,  # 数据集加权采样
    StratifiedSampler,  # 数据集分层采样
    UniformSampler,  # 数据集均匀采样
)
from evalscope.utils.io_utils import dump_jsonl_data
from pathlib import Path

DATASET_TOTAL_SIZE = 333
DATASET_PATH = Path("eval/datasets/math.jsonl")

EVAL_BATCH_SIZE = 8

dataset_schema = CollectionSchema(
    name="math",
    datasets=[
        DatasetInfo(name="gsm8k", task_type="math", tags=["math"]),
        DatasetInfo(name="math_500", task_type="math", tags=["math"]),
        DatasetInfo(name="aime24", task_type="math", tags=["math"]),
    ],
)

task_cfg = TaskConfig(
    model="Qwen/Qwen3-4B",
    eval_type="service",
    api_url="http://127.0.0.1:23333/v1/chat/completions",
    api_key="sk-jiangyj",
    # 数据集名称固定为 data_collection，表示评测混合数据集
    datasets=["data_collection"],
    dataset_args={
        "data_collection": {
            "dataset_id": DATASET_PATH,  # 评测数据集的路径，可以是本地路径，也可以是modelscope上的数据集id
            "filters": {"remove_until": "</think>"},  # 过滤掉思考的内容
        }
    },
    eval_batch_size=EVAL_BATCH_SIZE,
    generation_config={
        "max_tokens": 32768,  # 最大生成token数，建议设置为较大值避免输出截断
        "temperature": 0.6,  # 采样温度 (qwen 报告推荐值)
        "top_p": 0.95,  # top-p采样 (qwen 报告推荐值)
        "top_k": 20,  # top-k采样 (qwen 报告推荐值)
        "n": 1,  # 每个请求产生的回复数量
    },
    timeout=10 * 60 * 000,  # 超时时间
    stream=True,  # 是否使用流式输出
    work_dir="eval/qwen3_4b",  # 评估过程保存路径
    outputs="eval/qwen3_4b",  # 评估结果保存路径
    # limit=100,  # 设置为100条数据进行测试
)

if __name__ == "__main__":
    if not DATASET_PATH.exists():
        sampler = UniformSampler(dataset_schema)
        mixed_data = sampler.sample(DATASET_TOTAL_SIZE)
        dump_jsonl_data(mixed_data, DATASET_PATH)
    run_task(task_cfg=task_cfg)
