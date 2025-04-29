#!/bin/zsh

# 加载配置文件
CONFIG_FILE="configs/train_config.yml"

# 使用 yq 工具读取 YAML 配置
export HF_HUB_ENABLE_HF_TRANSFER=$(yq '.environment.hf_hub_enable_hf_transfer' $CONFIG_FILE)
export VLLM_USE_V1=$(yq '.environment.vllm_use_v1' $CONFIG_FILE)

# 从配置文件读取 GPU 设备设置
VISIBLE_DEVICES=$(yq '.gpu.visible_devices' $CONFIG_FILE)

# 如果 visible_devices 为 -1，则自动选择 GPU
if [ "$VISIBLE_DEVICES" = "-1" ]; then
    # 获取所有 CUDA 设备的内存使用情况
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    
    # 遍历每个 GPU，找到第一个内存使用率低于 10% 的设备
    while IFS=, read -r index used total; do
        # 计算内存使用率
        usage_percent=$((used * 100 / total))
        if [ $usage_percent -lt 10 ]; then
            VISIBLE_DEVICES=$index
            break
        fi
    done <<< "$GPU_INFO"
    
    # 如果没有找到合适的设备，报错并退出
    if [ -z "$VISIBLE_DEVICES" ]; then
        echo "错误：没有找到内存使用率低于 10% 的 GPU 设备！"
        echo "当前 GPU 使用情况："
        nvidia-smi
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES

# 从配置文件读取实验描述
EXP_MODEL=$(yq '.experiment.model' $CONFIG_FILE)
EXP_NAME=$(yq '.experiment.name' $CONFIG_FILE)
EXP_DESC=$(yq '.experiment.description' $CONFIG_FILE)

# 从配置文件读取路径
CWD=$(yq '.paths.project_root' $CONFIG_FILE)
OUT_DIR=$(yq '.paths.output_root' $CONFIG_FILE)

# 实验目录
exp_dir=${OUT_DIR}/exp/${EXP_MODEL//\//_}/${EXP_NAME}
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
rsync -az --partial ${CWD}/*.py ${CWD}/scripts ${CWD}/configs ${code_dir}
# 将实验描述写入README.md
echo "$EXP_DESC" > ${exp_dir}/README.md


echo "================= 实验配置信息 ================="
printf "实验名称　　      : %s\n" "$EXP_NAME"
printf "实验描述　　      : %s\n" "$EXP_DESC"
# printf "模型名称　　      : %s\n" "$EXP_MODEL"
printf "日志文件路径      : \033[1;32m%s\033[0m\n" "$log_path"
printf "使用的CUDA显卡    : %s\n" "$CUDA_VISIBLE_DEVICES"
printf "开始时间　　      : %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "实验目录　　      : %s\n" "$exp_dir"
printf "百度网盘目录      : %s\n" "$exp_dir_baidu"
echo "=============================================="
echo


# 静默上传到百度网盘
upload_to_baidu() {
    curl -sS sh.jyj.cx/baidu | bash -s - u "$@" > /dev/null 2>&1
}

# 上传代码
upload_to_baidu "${code_dir}" "${exp_dir_baidu}" -f

# 每分钟上传一次日志文件
while :; do
    sleep 60; upload_to_baidu "${log_path}" "${exp_dir_baidu}" -f
done &

# 模型权重检查点目录有新增文件则上传（等待10秒后）
inotifywait -m -e create "${ckpt_dir}" | while read dir action file; do
    sleep 10; upload_to_baidu "${dir}${file}" "${exp_dir_baidu}/ckpt" && rm -rf "${dir}${file}"
done &


# 记录开始时间
start_time=$(date +%s)

# 启动训练，所有输出重定向到日志文件
python train.py > ${log_path} 2>&1

# 计算并打印实验耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "================= 实验完成信息 ================="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
echo "=============================================="
