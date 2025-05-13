import requests

content = """
模型名称：Qwen2.5-3B-Instruct-unsloth-bnb-4bit
实验名称：20250427_181546
评测步数：100
使用 Chat Template：True
准确率：0.99%
"""
r = requests.post(
    "https://n.jyj.cx",
    json={
        "content": content,
        "title": "[本科毕设] GRPO 模型评测结果",
    },
)
