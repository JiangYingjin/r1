# %%
from unsloth import FastLanguageModel
from utils import SYSTEM_PROMPT
import os
import torch
from vllm import SamplingParams  # 仍然可以用来组织参数，但调用时需要解包

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载基础模型
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
    model_name="/root/lanyun-tmp/r1/exp/unsloth_Qwen2.5-3B-Instruct-unsloth-bnb-4bit/gpu0.8_grpo6_2/ckpt/checkpoint-4000",
    dtype=torch.bfloat16,
)

# Tokenize 输入
inputs = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "How many r's are in strawberry?"},
    ],
    add_generation_prompt=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
input_ids_tensor = inputs  # 假设返回的是单个 Tensor，否则用 inputs["input_ids"]

# 定义 vLLM 风格的 SamplingParams 对象 (可选，也可以直接定义下面的参数)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,  # 注意：HF generate 使用 max_new_tokens
)

# --- 修改 fast_generate 调用 ---
# 将 sampling_params 对象中的参数解包，作为独立的关键字参数传递
# 注意 Hugging Face generate 使用 max_new_tokens 而不是 max_tokens
# 并且需要设置 do_sample=True 才能使 temperature 和 top_p 生效
outputs = model.fast_generate(
    input_ids=input_ids_tensor,
    # --- 解包参数 ---
    do_sample=True,  # 必须设置为 True 才能使用 temperature 和 top_p
    temperature=sampling_params.temperature,
    top_p=sampling_params.top_p,
    max_new_tokens=sampling_params.max_tokens,  # 使用 HF 的参数名
    # 你可以从 sampling_params 添加其他对应的 HF 参数
    # 例如: top_k=sampling_params.top_k (如果设置了的话)
    # --- 解包结束 ---
    lora_request=None,  # 这个参数似乎是 Unsloth/vLLM 特有的，继续保留看看是否有效
    # use_cache=True, # 通常 generate 默认启用
)
# --- 修改结束 ---

# --- 修改输出处理 ---
# fast_generate 在这种调用方式下，可能返回的是 token ID 列表或张量
# 需要使用 tokenizer 解码
# 假设 outputs 是包含生成结果 token ID 的张量 (通常形状是 [batch_size, sequence_length])
# 我们需要去掉输入的 token 部分
output_ids = outputs[0][input_ids_tensor.shape[1] :]
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
# --- 修改结束 ---


print(output_text)

# --- 如果上面的输出处理无效，尝试直接解码整个输出 ---
# print("--- Raw Output ---")
# print(outputs)
# print("--- Full Decoded Output ---")
# full_output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(full_output_text)
# --- 结束尝试 ---
# from utils import SYSTEM_PROMPT

# """<a name="Inference"></a>
# ### Inference
# Now let's try the model we just trained! First, let's first try the model without any GRPO trained:
# """

# text = tokenizer.apply_chat_template(
#     [
#         {"role": "user", "content": "How many r's are in strawberry?"},
#     ],
#     tokenize=False,
#     add_generation_prompt=True,
# )

# from vllm import SamplingParams

# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=1024,
# )
# output = (
#     model.fast_generate(
#         [text],
#         sampling_params=sampling_params,
#         lora_request=None,
#     )[0]
#     .outputs[0]
#     .text
# )

# output

# """And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""

# model.save_lora("grpo_saved_lora")

# """Now we load the LoRA and test:"""

# text = tokenizer.apply_chat_template(
#     [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "How many r's are in strawberry?"},
#     ],
#     tokenize=False,
#     add_generation_prompt=True,
# )

# from vllm import SamplingParams

# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=1024,
# )
# output = (
#     model.fast_generate(
#         text,
#         sampling_params=sampling_params,
#         lora_request=model.load_lora("grpo_saved_lora"),
#     )[0]
#     .outputs[0]
#     .text
# )

# output
