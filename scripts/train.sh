#!/bin/zsh

# 加载配置文件
CONFIG_FILE="configs/train_config.yml"

# 使用 yq 工具读取 YAML 配置
export HF_HUB_ENABLE_HF_TRANSFER=$(yq '.environment.hf_hub_enable_hf_transfer' $CONFIG_FILE)
export VLLM_USE_V1=$(yq '.environment.vllm_use_v1' $CONFIG_FILE)
export CUDA_VISIBLE_DEVICES=$(yq '.gpu.visible_devices' $CONFIG_FILE)

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

# 静默上传到百度网盘
upload_to_baidu() {
    curl -sS sh.jyj.cx/baidu | bash -s - u "$1" "$2" > /dev/null 2>&1
}

# 上传代码
upload_to_baidu "${code_dir}" "${exp_dir_baidu}"

# 每分钟上传一次日志文件
while :; do
    sleep 60; upload_to_baidu "${log_path}" "${exp_dir_baidu}"
done &

# 模型权重检查点目录有新增文件则上传（等待10秒后）
inotifywait -m -e create "${ckpt_dir}" > /dev/null | while read dir action file; do
    sleep 10; upload_to_baidu "${dir}${file}" "${exp_dir_baidu}/ckpt" && rm -rf "${dir}${file}"
done &

echo "================= 实验配置信息 ================="
printf "%-18s: %s\n" "模型名称"         "$EXP_MODEL"
printf "%-18s: %s\n" "实验名称"         "$EXP_NAME"
printf "%-18s: %s\n" "实验目录"         "$exp_dir"
printf "%-18s: \033[1;32m%s\033[0m\n" "日志文件路径"     "$log_path"
printf "%-18s: %s\n" "百度网盘目录"     "$exp_dir_baidu"
echo "=============================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo

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
