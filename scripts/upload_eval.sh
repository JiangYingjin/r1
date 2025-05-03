#!/bin/zsh

# 静默上传到百度网盘
upload_to_baidu() {
    curl -sS sh.jyj.cx/baidu | bash -s - u "$@" > /dev/null 2>&1
}

# 源目录和目标目录
SOURCE_DIR="/root/proj/r1/eval"
DEST_DIR="/share/proj/r1"

echo "开始每隔5分钟上传 ${SOURCE_DIR} 到百度网盘 ${DEST_DIR}"

# 每隔5分钟上传一次
while :; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 正在上传 ${SOURCE_DIR}..."
    upload_to_baidu "${SOURCE_DIR}" "${DEST_DIR}" -f
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 上传完成，等待5分钟..."
    sleep 300
done
