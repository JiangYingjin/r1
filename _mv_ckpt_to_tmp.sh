OUTPUT_DIR="/root/proj/r1/outputs"
TARGET_DIR="$HOME/lanyun-tmp/outputs"

mkdir -p "$TARGET_DIR"

echo "开始监控 $OUTPUT_DIR 目录..."

inotifywait -m -e create "$OUTPUT_DIR" |
while read file_path action file; do
    echo "DEBUG: Detected event - path: '$file_path', action: '$action', file: '$file'" # 添加这行
    echo "DEBUG: Checking if '$file_path$file' is a directory..." # 添加这行

    # 检查是否是目录创建事件
    if [[ "$action" == *"CREATE"* ]] && [ -d "$file_path$file" ]; then
        echo "DEBUG: Condition met - processing folder '$file'" # 添加这行

        folder_name="$file"
        time1=$(date +"%d_%H%M%S")
        echo "发现新文件夹: $folder_name 在时间为 $time1 时"

        echo "等待10秒后移动文件夹..."
        sleep 10

        target="${TARGET_DIR}/${folder_name}_${time1}"
        echo "移动 $file_path$file 到 $target"
        mv "$file_path$file" "$target"

        echo "移动完成!"
    else
        echo "DEBUG: Condition NOT met. Action: '$action', Is directory: $([ -d "$file_path$file" ] && echo "true" || echo "false")" # 添加这行
    fi
done