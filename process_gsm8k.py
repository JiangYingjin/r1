import json
import os
from lib import LLM  # Assuming lib.py contains the LLM class
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

input_file = Path("eval/datasets/gsm8k_math_id.jsonl")
output_file = Path("eval/datasets/gsm8k_math_resp.jsonl")
num_requests_per_id = 5
max_workers = 32

# Instantiate the LLM
llm = LLM("Qwen2.5-3B-Instruct", base_url="http://127.0.0.1:23333/v1", key="sk-jiangyj")

# Ensure the output directory exists
output_file.parent.mkdir(parents=True, exist_ok=True)


def process_single_request(data, request_index):
    """Processes a single LLM request for an item."""
    question_id = data["id"]
    question = data["question"]
    answer = data["answer"]

    try:
        # Use the instantiated llm_instance to chat
        response = llm.chat(question, silent=True)
        output_data = {
            "id": question_id,
            "response": response,
            "question": question,
            "answer": answer,
        }
        return json.dumps(output_data, ensure_ascii=False)
    except Exception as e:
        return None


if __name__ == "__main__":
    items_to_process = []
    with input_file.open("r", encoding="utf-8") as infile:
        for line in infile:
            items_to_process.append(json.loads(line))

    all_futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in items_to_process:
            for i in range(num_requests_per_id):
                future = executor.submit(process_single_request, item, i)
                all_futures.append(future)

        with output_file.open("w", encoding="utf-8") as outfile:
            for future in tqdm(
                as_completed(all_futures),
                total=len(all_futures),
                desc="Processing requests",
            ):
                try:
                    result_line = future.result()
                    if result_line:
                        outfile.write(result_line + "\n")
                except Exception as e:
                    # This exception handling might be redundant if caught in process_single_request,
                    # but kept for safety.
                    print(f"A future generated an exception: {e}")

    print("Processing complete.")
