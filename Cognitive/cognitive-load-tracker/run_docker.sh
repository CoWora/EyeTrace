#!/bin/bash
# ===========================================
# 认知负荷数据采集系统 - Docker 运行脚本 (Linux/macOS)
# ===========================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==========================================="
echo "  认知负荷数据采集系统 Docker 启动器"
echo "==========================================="
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[错误] Docker 未安装${NC}"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo -e "${RED}[错误] Docker 未运行${NC}"
    echo "请启动 Docker"
    exit 1
fi

echo -e "${GREEN}[1/5]${NC} 检查 GPU 支持..."
GPU_MODE=1
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "      ${YELLOW}GPU 模式不可用，将使用 CPU 版本${NC}"
    GPU_MODE=0
else
    echo -e "      ${GREEN}GPU 检测成功${NC}"
fi

echo -e "${GREEN}[2/5]${NC} 检查 NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null && [ $GPU_MODE -eq 1 ]; then
    echo -e "${YELLOW}      警告: NVIDIA Container Toolkit 未安装${NC}"
    echo "      请参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

echo -e "${GREEN}[3/5]${NC} 检查模型文件..."
if [ ! -f "models/L2CSNet_gaze360.pkl" ]; then
    echo -e "${YELLOW}      警告: 模型文件不存在${NC}"
    echo "      请下载 L2CSNet_gaze360.pkl 并放到 models/ 目录"
    echo "      下载地址: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"
fi

echo -e "${GREEN}[4/5]${NC} 构建 Docker 镜像..."

# 创建必要目录
mkdir -p data/cognitive_study

if [ $GPU_MODE -eq 1 ]; then
    echo "      正在构建 GPU 版本镜像..."
    docker build -t cognitive-load-tracker:latest .
else
    echo "      正在构建 CPU 版本镜像..."
    docker build -t cognitive-load-tracker:cpu -f Dockerfile.cpu .
fi

echo -e "${GREEN}[5/5]${NC} 启动容器..."
echo ""

# 设置 X11 显示
export DISPLAY=${DISPLAY:-:0}

# 停止并删除已存在的容器
docker stop cognitive-tracker cognitive-tracker-cpu 2>/dev/null || true
docker rm cognitive-tracker cognitive-tracker-cpu 2>/dev/null || true

if [ $GPU_MODE -eq 1 ]; then
    docker run --gpus all \
        -e DISPLAY=$DISPLAY \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --device /dev/video0:/dev/video0 \
        --network=host \
        --name cognitive-tracker \
        -it cognitive-load-tracker:latest
else
    docker run \
        -e DISPLAY=$DISPLAY \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --device /dev/video0:/dev/video0 \
        --network=host \
        --name cognitive-tracker-cpu \
        -it cognitive-load-tracker:cpu
fi

echo ""
echo "容器已停止"
