from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="Qwen/Qwen3-4B",
    api_url="http://127.0.0.1:23333/v1/chat/completions",
    api_key="sk-jiangyj",
    eval_type="service",
    datasets=[
        "data_collection",
    ],
    dataset_args={
        "data_collection": {
            "dataset_id": "modelscope/EvalScope-Qwen3-Test",
            "filters": {"remove_until": "</think>"},  # 过滤掉思考的内容
        }
    },
    eval_batch_size=128,
    generation_config={
        "max_tokens": 30000,  # 最大生成token数，建议设置为较大值避免输出截断
        "temperature": 0.6,  # 采样温度 (qwen 报告推荐值)
        "top_p": 0.95,  # top-p采样 (qwen 报告推荐值)
        "top_k": 20,  # top-k采样 (qwen 报告推荐值)
        "n": 1,  # 每个请求产生的回复数量
    },
    timeout=60000,  # 超时时间
    stream=True,  # 是否使用流式输出
    limit=100,  # 设置为100条数据进行测试
    outputs="eval/qwen3_4b",  # 评估结果保存路径
)

run_task(task_cfg=task_cfg)
