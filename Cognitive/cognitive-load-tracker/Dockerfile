# 认知负荷数据采集系统 Docker 配置
# 基础镜像: Ubuntu 22.04 + CUDA 11.8 + Python 3.10

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# 安装系统依赖和 Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    wget \
    unzip \
    libv4l-0 \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接使 python3 可用
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖 (先安装 PyTorch)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    opencv-python>=4.5.0 \
    pandas>=1.3.0 \
    pyautogui>=0.9.50 \
    pygetwindow>=0.0.9

# 安装 L2CS-Net
RUN pip install --no-cache-dir git+https://github.com/Ahmednull/L2CS-Net.git

# 复制项目文件
COPY cognitive_study/ ./cognitive_study/
COPY models/ ./models/

# 创建数据输出目录
RUN mkdir -p data/cognitive_study

# 安装模型权重下载脚本
RUN chmod +x /app/download_models.sh

# 暴露端口 (用于浏览器插件通信)
EXPOSE 8765

# 默认启动命令
CMD ["python3", "cognitive_study/aoi_collector_v3_2.py"]
