# Docker 部署指南

本目录包含使用 Docker 运行认知负荷数据采集系统的配置文件。

## 前提条件

### Windows
- Docker Desktop for Windows
- WSL 2 (Windows Subsystem for Linux)
- NVIDIA GPU + WSL 2 支持 (可选，用于 GPU 加速)

### Linux
- Docker
- NVIDIA Container Toolkit (GPU 支持)
- NVIDIA GPU + CUDA 11.8

### macOS
- Docker Desktop
- 支持 Xavier/NVIDIA GPU (需 eGPU)

## 文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile` | GPU 版本 - 需要 NVIDIA GPU |
| `Dockerfile.cpu` | CPU 版本 - 无 GPU 时使用 |
| `docker-compose.yml` | Docker Compose 配置 |
| `run_docker.bat` | Windows 启动脚本 |
| `run_docker.sh` | Linux/macOS 启动脚本 |
| `download_models.sh` | 模型下载脚本 |

## 快速开始

### 1. 下载模型文件

```bash
# 创建 models 目录
mkdir -p models

# 下载 L2CSNet_gaze360.pkl
# 从 Google Drive 下载: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd
# 放入 models/ 目录
```

### 2. Windows 用户

双击运行 `run_docker.bat` 或在命令行执行:

```cmd
run_docker.bat
```

### 3. Linux/macOS 用户

```bash
chmod +x run_docker.sh
./run_docker.sh
```

## Docker Compose 使用

### 构建并启动

```bash
# GPU 版本
docker-compose up --build

# CPU 版本
docker-compose up --build cognitive-tracker-cpu
```

### 后台运行

```bash
docker-compose up -d
docker-compose logs -f
```

### 停止

```bash
docker-compose down
```

## 手动构建

### GPU 版本

```bash
docker build -t cognitive-load-tracker:latest .
```

### CPU 版本

```bash
docker build -t cognitive-load-tracker:cpu -f Dockerfile.cpu .
```

## 手动运行

### GPU 版本

```bash
docker run --gpus all \
    -e DISPLAY=$DISPLAY \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --device /dev/video0:/dev/video0 \
    --network=host \
    -it cognitive-load-tracker:latest
```

### CPU 版本

```bash
docker run \
    -e DISPLAY=$DISPLAY \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --device /dev/video0:/dev/video0 \
    --network=host \
    -it cognitive-load-tracker:cpu
```

## 故障排除

### Docker 权限问题 (Linux)

```bash
sudo usermod -aG docker $USER
# 重新登录使更改生效
```

### X11 显示问题

确保 X server 允许 Docker 容器连接:

```bash
# Linux
xhost +local:docker
```

### NVIDIA GPU 问题

检查 NVIDIA Container Toolkit 是否正确安装:

```bash
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 摄像头设备问题

确保宿主机摄像头可用:

```bash
# Linux - 列出可用摄像头
ls -la /dev/video*
```

## 性能提示

- **GPU 版本**: 帧率 10-15 fps
- **CPU 版本**: 帧率 2-5 fps
- 推荐使用 NVIDIA GPU 以获得最佳性能

## 数据输出

容器运行后，数据保存在主机的 `data/cognitive_study/` 目录下。
