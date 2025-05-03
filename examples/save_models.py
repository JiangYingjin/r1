from unsloth import FastLanguageModel

for step in range(2000, 4001, 200):
    print(f"正在处理 checkpoint-{step}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"/root/lanyun-tmp/r1/exp/unsloth_Qwen2.5-3B-Instruct-unsloth-bnb-4bit/gpu0.8_grpo6_2/ckpt/checkpoint-{step}",
    )
    print(f"成功加载 checkpoint-{step} 模型")

    save_path = f"ckpt_merged/{step}"
    print(f"正在将模型保存到 {save_path}，使用 merged_16bit 格式...")
    model.save_pretrained_merged(
        save_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"checkpoint-{step} 已成功保存为 merged_16bit 格式")

"""
merged_16bit (你使用的): 合并后保存为 16-bit (通常是 float16 或 bfloat16)。这是一个很好的平衡点，模型大小适中，精度损失小，加载后可以直接使用标准 Transformers 库进行推理。
merged_4bit: 合并后保存为 4-bit 量化模型。最节省空间，推理速度最快（在支持的环境下）。适用于部署资源受限（显存、磁盘）且对轻微精度下降不敏感的场景。这是 Unsloth 推荐的部署方式之一。
merged_8bit: 合并后保存为 8-bit 量化模型。介于 16-bit 和 4-bit 之间，是空间、速度和精度的一个折中选项。
lora: 只保存适配器。适用于 分享微调结果（只需分享小小的适配器文件），或者希望保留基础模型不变，在不同任务间切换适配器的场景。加载推理时需要额外步骤来合并。
"""

exit()


# 保存为 GGUF 格式 并进行 q4_k_m 量化（经测试可用）
model.save_pretrained_gguf(
    "gguf_model",
    tokenizer,
    quantization_method="q4_k_m",
)


# Merge to 16bit
if False:
    model.save_pretrained_merged(
        "model",
        tokenizer,
        save_method="merged_16bit",
    )
if False:
    model.push_to_hub_merged(
        "hf/model", tokenizer, save_method="merged_16bit", token=""
    )

# Merge to 4bit
if False:
    model.save_pretrained_merged(
        "model",
        tokenizer,
        save_method="merged_4bit",
    )
if False:
    model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")

# Just LoRA adapters
if False:
    model.save_pretrained_merged(
        "model",
        tokenizer,
        save_method="lora",
    )
if False:
    model.push_to_hub_merged("hf/model", tokenizer, save_method="lora", token="")

"""### GGUF / llama.cpp Conversion
To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
* `q8_0` - Fast conversion. High resource use, but generally acceptable.
* `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
* `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

[**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
"""

# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf(
        "model",
        tokenizer,
    )
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if 1:
    model.push_to_hub_gguf(
        "hf/model", tokenizer, quantization_method="q4_k_m", token=""
    )

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model",  # Change hf to your username!
        tokenizer,
        quantization_method=[
            "q4_k_m",
            "q8_0",
            "q5_k_m",
        ],
        token="",
    )

"""Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>

"""
