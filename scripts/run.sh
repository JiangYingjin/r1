export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_USE_V1=0

export CUDA_VISIBLE_DEVICES=1

script_name=qwen2_5_\(3b\)_grpo.py
# script_name=t1.py

python $script_name | tee logs/$(date +"%d_%H%M").log