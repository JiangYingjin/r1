import json
import os
from lib import LLM  # 从lib.py导入LLM类
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 输入文件路径
stats_file = Path("data/processed/gsm8k_attempts_pass_stats.json")
resp_file = Path("data/processed/gsm8k_math_resp.jsonl")
raw_file = Path("data/raw/gsm8k_math_id.jsonl")

# 每个ID请求的次数
num_requests_per_id = 5
max_workers = 32

# 实例化LLM
llm = LLM("Qwen2.5-3B-Instruct", base_url="http://127.0.0.1:23333/v1", key="sk-jiangyj")

def process_single_request(data, request_index):
    """处理单个LLM请求"""
    question_id = data["id"]
    question = data["question"]
    answer = data["answer"]

    try:
        # 使用实例化的llm进行聊天
        response = llm.chat(question, silent=True)
        output_data = {
            "id": question_id,
            "response": response,
            "question": question,
            "answer": answer,
        }
        return json.dumps(output_data, ensure_ascii=False)
    except Exception as e:
        print(f"处理请求时出错: {e}")
        return None

def main():
    # 确保输入文件存在
    if not stats_file.exists():
        raise FileNotFoundError(f"统计文件不存在: {stats_file}")
    if not resp_file.exists():
        raise FileNotFoundError(f"响应文件不存在: {resp_file}")
    if not raw_file.exists():
        raise FileNotFoundError(f"原始文件不存在: {raw_file}")

    # 读取attempts=5 && pass=1的ID列表
    with stats_file.open("r", encoding="utf-8") as f:
        stats_data = json.load(f)
        target_ids = stats_data["id_lists"]["5"]["3"]
    
    print(f"找到 {len(target_ids)} 个符合条件的ID")

    # 读取原始问题数据
    raw_data = {}
    with raw_file.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            raw_data[item["id"]] = item
    
    # 准备要处理的项目
    items_to_process = []
    for id in target_ids:
        if id in raw_data:
            items_to_process.append(raw_data[id])
        else:
            print(f"警告: ID {id} 在原始数据中未找到")
    
    print(f"准备处理 {len(items_to_process)} 个项目")

    # 使用ThreadPoolExecutor并行处理请求
    all_futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in items_to_process:
            for i in range(num_requests_per_id):
                future = executor.submit(process_single_request, item, i)
                all_futures.append(future)
        
        # 将结果追加到响应文件
        with resp_file.open("a", encoding="utf-8") as outfile:
            for future in tqdm(
                as_completed(all_futures),
                total=len(all_futures),
                desc="处理请求",
            ):
                try:
                    result_line = future.result()
                    if result_line:
                        outfile.write(result_line + "\n")
                except Exception as e:
                    print(f"处理结果时出错: {e}")
    
    print("处理完成。")

if __name__ == "__main__":
    main()