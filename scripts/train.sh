#!/bin/bash

# 禁用 hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=0
# 禁用 vllm v1（否则无法运行训练代码）
export VLLM_USE_V1=0

# 指定 GPU
export CUDA_VISIBLE_DEVICES=1

EXP_MODEL="unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
EXP_DESC=$(cat << 'EOF'
详细的实验描述
EOF
)

# 训练参数
TOTAL_STEPS=4000
SAVE_STEPS=200
GRPO_NUM_GENERATIONS=8

# LoRA 参数
LORA_RANK=64  # 更大的秩 = 更智能，但更慢，建议选择 8, 16, 32, 64, 128

# 序列长度参数
MAX_PROMPT_LENGTH=512
MAX_COMPLETION_LENGTH=2560

# GPU 参数
GPU_MEMORY_UTILIZATION=0.9  # 如果内存不足，可以降低此值

# 随机种子
RANDOM_STATE=3407

# 项目目录
cwd="/root/proj/r1"
# 项目输出目录
out_dir="/root/lanyun-tmp/r1"

# 实验时间
exp_time=$(date +"%d_%H%M")
# 实验目录
exp_dir=${out_dir}/exp/${EXP_MODEL//\//_}/${exp_time}
# 实验目录（百度网盘）
exp_dir_baidu=${exp_dir/\/root\/lanyun-tmp/\/share\/proj}
# 实验代码目录
code_dir=${exp_dir}/code
# 实验模型权重检查点目录
ckpt_dir=${exp_dir}/ckpt
# 实验日志文件路径
log_path=${exp_dir}/out.log

# 创建实验目录
mkdir -p ${exp_dir} ${code_dir} ${ckpt_dir}
# 保存代码至实验目录
rsync -avzP ${cwd}/*.py ${cwd}/scripts ${code_dir}
# 将实验描述写入README.md
echo "$EXP_DESC" > ${exp_dir}/README.md

# 静默上传到百度网盘
upload_to_baidu() {
    curl sh.jyj.cx/baidu | bash -s - u "$1" "$2" > /dev/null 2>&1
}

# 上传代码
upload_to_baidu "${code_dir}" "${exp_dir_baidu}/code"

# 每分钟上传一次日志文件
while :; do
    sleep 60; upload_to_baidu "${log_path}" "${exp_dir_baidu}"
done &

# 模型权重检查点目录有新增文件则上传（等待10秒后）
inotifywait -m -e create "${ckpt_dir}" | while read dir action file; do
    sleep 10; upload_to_baidu "${dir}${file}" "${exp_dir_baidu}/ckpt" && rm -rf "${dir}${file}"
done &

# 启动训练
python train.py \
    --exp_name $exp_time \
    --model_name $EXP_MODEL \
    --total_steps $TOTAL_STEPS \
    --save_steps $SAVE_STEPS \
    --grpo_num_generations $GRPO_NUM_GENERATIONS \
    --lora_rank $LORA_RANK \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --random_state $RANDOM_STATE \
    | tee ${log_path}