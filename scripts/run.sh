export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_USE_V1=0

export CUDA_VISIBLE_DEVICES=1

script_name="qwen2_5_(3b)_grpo.py"
# script_name=t1.py

cwd="/root/proj/r1"
out_dir="/root/lanyun-tmp/r1"

exp_model="unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
exp_desc=$(cat << 'EOF'
详细的实验描述
EOF
)

exp_time=$(date +"%d_%H%M")
exp_dir=${out_dir}/exp/${exp_model//\//_}/${exp_time}
exp_dir_baidu=${exp_dir/\/root\/lanyun-tmp/\/share\/proj}
code_dir=${exp_dir}/code
ckpt_dir=${exp_dir}/ckpt
log_path=${exp_dir}/out.log

# 创建实验目录
mkdir -p ${exp_dir} ${code_dir} ${ckpt_dir}
# 保存代码至实验目录
rsync -avzP ${cwd}/*.py ${cwd}/scripts ${code_dir}
# 将实验描述写入README.md
echo "$exp_desc" > ${exp_dir}/README.md

# 静默上传到百度网盘
upload_to_baidu() {
    curl sh.jyj.cx/baidu | bash -s - u "$1" "$2" > /dev/null 2>&1
}

# 每分钟上传一次日志文件
while :; do
    sleep 60; upload_to_baidu "${log_path}" "${exp_dir_baidu}"
done &

# ckpt 目录有新增文件则上传（等待10秒后）
inotifywait -m -e create "${ckpt_dir}" | while read dir action file; do
    sleep 10; upload_to_baidu "${dir}${file}" "${exp_dir_baidu}/ckpt" && rm -rf "${dir}${file}"
done &

python $script_name | tee ${log_path}