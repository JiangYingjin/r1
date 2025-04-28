#!/bin/bash

# 加载配置文件
CONFIG_FILE="configs/train_config.yml"

# 使用 yq 工具读取 YAML 配置
export HF_HUB_ENABLE_HF_TRANSFER=$(yq '.environment.hf_hub_enable_hf_transfer' $CONFIG_FILE)
export VLLM_USE_V1=$(yq '.environment.vllm_use_v1' $CONFIG_FILE)
export CUDA_VISIBLE_DEVICES=$(yq '.gpu.visible_devices' $CONFIG_FILE)

# 从配置文件读取实验描述
EXP_MODEL=$(yq '.experiment.model' $CONFIG_FILE)
EXP_DESC=$(yq '.experiment.description' $CONFIG_FILE)

# 从配置文件读取训练参数
TOTAL_STEPS=$(yq '.training.total_steps' $CONFIG_FILE)
SAVE_STEPS=$(yq '.training.save_steps' $CONFIG_FILE)
GRPO_NUM_GENERATIONS=$(yq '.training.grpo_num_generations' $CONFIG_FILE)

# 从配置文件读取 LoRA 参数
LORA_RANK=$(yq '.lora.rank' $CONFIG_FILE)

# 从配置文件读取序列长度参数
MAX_PROMPT_LENGTH=$(yq '.sequence.max_prompt_length' $CONFIG_FILE)
MAX_COMPLETION_LENGTH=$(yq '.sequence.max_completion_length' $CONFIG_FILE)

# 从配置文件读取 GPU 参数
GPU_MEMORY_UTILIZATION=$(yq '.gpu.memory_utilization' $CONFIG_FILE)

# 从配置文件读取随机种子
RANDOM_STATE=$(yq '.random_state' $CONFIG_FILE)

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
rsync -avzP ${cwd}/*.py ${cwd}/scripts ${cwd}/configs ${code_dir}
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