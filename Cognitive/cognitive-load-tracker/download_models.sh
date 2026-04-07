#!/bin/bash
# 下载 L2CS-Net 模型权重
# 使用方法: ./download_models.sh

MODEL_URL="https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"
MODEL_FILE="L2CSNet_gaze360.pkl"
MODEL_DIR="models"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "模型文件已存在: $MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "请手动下载模型文件:"
echo "1. 访问: $MODEL_URL"
echo "2. 进入 Gaze360 文件夹"
echo "3. 下载 L2CSNet_gaze360.pkl"
echo "4. 将文件放到 models/ 目录"
echo ""
echo "或者使用 gdown (如果已安装):"
echo "  pip install gdown"
echo "  gdown --folder $MODEL_URL"
