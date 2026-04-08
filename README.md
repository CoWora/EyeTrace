# EyeTrace

基于视线追踪的认知负荷采集与建模项目（含：采集端、聚类/分类/实时预测）。

## 目录结构

- `Cognitive/cognitive-load-tracker/`: 采集端（摄像头 + L2CS-Net 视线估计 + AOI/任务/眨眼等记录）
- `Model/ET_model/`: 特征提取、聚类、监督分类、实时预测输出（含 `realtime_session_monitor.py`）

更详细的使用说明请查看各子目录 README：

- `Cognitive/cognitive-load-tracker/README.md`
- `Model/ET_model/README.md`
- `WORKLOG.md` — 全项目工作日志（含立项、技术选型、各阶段关键决策）
- `Model/ET_model/WORKLOG.md` — 模型模块详细日志

## 重要：隐私与大文件

本仓库默认 **不提交** 以下内容（已在根 `.gitignore` 中屏蔽）：
- 采集数据：`data/`、`Cognitive/cognitive-load-tracker/data/` 等
- 预训练大权重文件：`*.pkl`（如 `L2CSNet_gaze360.pkl`）、`*.pth`、`*.pt`、`*.onnx`、`*.tflite`
- 实时预测输出：`Model/ET_model/realtime_predictions*.jsonl`
- 可能含个人信息的材料：`*.docx/*.pdf` 等
- 浏览器扩展目录：`luogu-gaze-extension/`
- 采集端隐私数据目录：`Cognitive/cognitive-load-tracker/cognitive_data/` 等

如果需要分享演示数据，建议放到 `examples/`（脱敏、小体积），或用网盘/Release 单独提供下载链接。

## 最小运行提示（面向开发者）

### 1) 采集端（Python）

进入 `Cognitive/cognitive-load-tracker/`，按该目录 `requirements.txt` 说明安装。

> 说明：L2CS-Net 权重文件 `L2CSNet_gaze360.pkl` **不要提交到 Git**；请本地放到采集脚本能找到的位置（通常是 `Cognitive/cognitive-load-tracker/models/`）。

### 2) 实时预测（Python）

进入 `Model/ET_model/`，按 `Model/ET_model/README.md` 的流程：
- 聚类生成 `outputs/`
- 训练分类器生成 `outputs_supervised/`
- 运行 `realtime_session_monitor.py` 生成实时预测 jsonl（这些 jsonl 默认不提交）

### 3) 浏览器扩展

本项目当前不包含/不需要浏览器扩展；相关目录已在根 `.gitignore` 中默认忽略。

### 4) 一键总控脚本

从项目根目录使用 `EyeTrace_controller.py` 可一键启动采集与预测系统：

```bash
cd C:\Users\YNS\Desktop\EyeTrace

# 只做数据采集
py -3.10 EyeTrace_controller.py

# 采集 + 准实时预测 + 可视化面板
py -3.10 EyeTrace_controller.py --with-monitor
```

| 命令 | 采集 | 监控 | 面板 |
|------|:----:|:----:|:----:|
| `EyeTrace_controller.py` | ✅ 前台运行 | ❌ | ❌ |
| `EyeTrace_controller.py --with-monitor` | ✅ 前台运行 | ✅ 后台运行 | ✅ 后台运行 |

> 使用 `--with-monitor` 时，监控和面板在后台运行，不影响采集窗口显示。采集结束后会自动终止后台进程。