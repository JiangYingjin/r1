export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_USE_V1=0

python qwen2_5_\(3b\)_grpo.py | tee logs/$(date +"%d_%H%M").log