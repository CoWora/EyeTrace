@echo off
REM ===========================================
REM 认知负荷数据采集系统 - Docker 运行脚本 (Windows)
REM ===========================================

echo ===========================================
echo   认知负荷数据采集系统 Docker 启动器
echo ===========================================
echo.

REM 检查 Docker 是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker 未安装或未启动
    echo 请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM 检查 Docker 是否运行
docker info >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker 未运行
    echo 请启动 Docker Desktop
    pause
    exit /b 1
)

echo [1/4] 检查 GPU 支持...
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       GPU 模式不可用，将使用 CPU 版本
    set GPU_MODE=0
) else (
    echo       GPU 检测成功
    set GPU_MODE=1
)

REM 设置项目目录
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo.
echo [2/4] 检查模型文件...
if not exist "models\L2CSNet_gaze360.pkl" (
    echo       警告: 模型文件不存在
    echo       请下载 L2CSNet_gaze360.pkl 并放到 models/ 目录
    echo       下载地址: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd
)

echo.
echo [3/4] 构建 Docker 镜像...
if %GPU_MODE%==1 (
    echo       正在构建 GPU 版本镜像...
    docker build -t cognitive-load-tracker:latest .
) else (
    echo       正在构建 CPU 版本镜像...
    docker build -t cognitive-load-tracker:cpu -f Dockerfile.cpu .
)

if errorlevel 1 (
    echo [错误] Docker 镜像构建失败
    pause
    exit /b 1
)

echo.
echo [4/4] 启动容器...
echo.

REM 设置 X11 显示
set DISPLAY=host.docker.internal:0

if %GPU_MODE%==1 (
    docker run --gpus all ^
        -e DISPLAY=%DISPLAY% ^
        -v "%PROJECT_DIR%data:/app/data" ^
        -v "%PROJECT_DIR%models:/app/models" ^
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw ^
        --network=host ^
        --name cognitive-tracker ^
        -it cognitive-load-tracker:latest
) else (
    docker run ^
        -e DISPLAY=%DISPLAY% ^
        -v "%PROJECT_DIR%data:/app/data" ^
        -v "%PROJECT_DIR%models:/app/models" ^
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw ^
        --network=host ^
        --name cognitive-tracker-cpu ^
        -it cognitive-load-tracker:cpu
)

echo.
echo 容器已停止
pause
