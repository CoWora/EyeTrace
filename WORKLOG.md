# EyeTrace 项目工作日志

> 本日志面向"可复现、可追溯"，记录项目从立项到实现的全流程关键节点。
> 运行环境：Windows + `py -3.10`（示例命令以项目根目录 `C:\Users\YNS\Desktop\EyeTrace` 为工作目录）。

---

## 总体目标

打造面向算法学习与智能教育的"认知负荷实时量化与调优"系统，融合计算机视觉、认知科学与机器学习，构建可深度解析学习者注意力与认知状态的智能监测与干预平台，为精准教学与个性化学习提供数据中枢与决策引擎。

---

## 阶段一：立项与方案设计（2025 年底）

### 1.1 需求梳理

基于洛谷算法题解题场景，提出以下核心需求：

| 方向 | 内容 |
|------|------|
| 多模态认知感知 | 高精度人脸与眼动视觉管线，提取注视分布、注视时长、扫视模式、眨眼等特征 |
| 特征工程与模型 | 设计时间/空间/视觉搜索/生理 proxy 等特征，训练认知负荷分类与回归模型 |
| 实时监测与可视化 | 实现认知热力图、注视轨迹、负荷曲线等可视化 |
| 智能教学决策 | 基于负荷水平自动生成教学与学习建议 |

### 1.2 技术选型

| 模块 | 方案 | 说明 |
|------|------|------|
| 视线追踪 | L2CS-Net + L2CSNet_gaze360.pkl | 基于 CNN 的回归型视线估计，在 GazeCapture 数据集预训练 |
| 眨眼检测 | EAR（Eye Aspect Ratio）指标 | 基于 6 个关键点坐标比值，阈值 < 0.2 判定眨眼 |
| AOI 区域识别 | 浏览器插件 + DOM 解析 | 插件负责区域划分，数据送回 Python 端处理 |
| 注视检测 | I-VT（Velocity-based Interest Threshold） | 基于速度阈值的眨眼/注视分类 |
| 聚类分析 | KMeans / 层次聚类 / DBSCAN | 对 session 或 task 级特征向量进行无监督聚类 |
| 分类器 | SVM / XGBoost | 将聚类标签固化为可对新数据预测的监督模型 |

### 1.3 设计文档

| 文档 | 说明 |
|------|------|
| `项目概述.md` | 项目目的、意义、内容、预期效果 |
| `系统架构设计.md` | 整体架构、模块划分、数据流 |
| `算法设计说明.md` | 视线追踪、眨眼检测、AOI 转移等核心算法 |
| `实时监测vs离线分析方案对比.md` | 两种分析方案的权衡 |
| `眼动数据到认知负荷的量化方法.md` | 特征提取与量化方法 |
| `认知负荷指标说明.md` | 负荷等级定义与划分依据 |

---

## 阶段二：数据采集系统（2025 年底 - 2026 年初）

### 2.1 采集脚本迭代

| 版本 | 文件 | 说明 |
|------|------|------|
| v3 | `aoi_collector_v3.py` | 早期版本，基础功能实现 |
| v3.2（当前） | `aoi_collector_v3_2.py` | 稳定版本，含 L2CS-Net 视线追踪 + AOI 检测 + 眨眼检测 + 注视检测 + 窗口检测 |
| v3.2 备份 | `aoi_collector_v3_2_backup_20260227.py` | v3.2 的备份存档 |

### 2.2 辅助脚本

| 脚本 | 说明 |
|------|------|
| `aoi_config_tool.py` | AOI 区域配置文件管理工具 |
| `aoi_analyzer.py` | AOI 数据离线分析脚本 |

### 2.3 数据输出格式

每个 session 目录（形如 `data/20260124_140152/`）包含：

| 文件 | 内容 |
|------|------|
| `gaze_data.csv` | 视线坐标时间序列 |
| `fixations.csv` | 注视事件列表 |
| `blinks.csv` | 眨眼事件列表 |
| `events.csv` | 通用事件（含任务切换） |
| `aoi_transitions.csv` | AOI 区域转移记录 |
| `tasks.csv` | 任务信息（含难度/主观评分） |
| `session_meta.json` | session 元信息 |

### 2.4 硬件与依赖

- GPU：NVIDIA（支持 CUDA，帧率 10-15fps；无 GPU 时 CPU 帧率 2-5fps）
- 视线模型：`Cognitive/models/L2CSNet_gaze360.pkl`（不纳入 Git，需单独放置）

---

## 阶段三：聚类与特征工程（2026 年 1 月 - 2 月）

### 3.1 特征提取

脚本：`Model/ET_model/cluster_cognitive_data.py`

特征前缀：
- `fix__`：注视相关（时长、次数、AOI 区域熵等）
- `blink__`：眨眼相关（次数）
- `trans__`：AOI 转移相关（次数、同区域比例、AOI 熵等）
- `task__`：任务相关（时长、难度、主观评分等）

特征数量：187 维（task 级）

### 3.2 聚类算法

支持三种：无监督聚类（KMeans / 层次聚类 / DBSCAN）

关键参数：
- `--unit session|task`：聚类单位
- `--feature_weights_json`：可选的特征加权配置（作用于 z-score 之后）

### 3.3 聚类产物

| 目录 | 说明 |
|------|------|
| `Model/ET_model/outputs/` | session 级聚类输出（早期） |
| `Model/ET_model/outputs_task_cluster/` | **task 级聚类输出（当前主用）** |

task 级聚类（k=6）的 cluster → 负荷映射：

| cluster | 相对负荷等级 | 负荷标签 |
|---------|------------|---------|
| 0 | Level 2 | 低负荷 / 轻量任务型 |
| 1 | Level 2 | 低负荷 / 轻量任务型 |
| 2 | Level 3 | 中高负荷 / 信息整合型 |
| 3 | Level 4 | 高负荷 / 持续专注解题型 |
| 4 | Level 3 | 中高负荷 / 信息整合型 |
| 5 | Level 1 | 极低负荷 / 轻松浏览型 |

### 3.4 特征加权

配置文件：`Model/ET_model/feature_weights_task.json`

关键加权维度：
- `task__duration__mean` ×3.0（任务持续时间最能反映认知负荷）
- `trans__n` ×2.5（AOI 转移次数）
- `task__subjective_effort__mean` ×1.8（主观 Effort 评分）

---

## 阶段四：监督模型训练（2026 年 2 月）

### 4.1 训练脚本

脚本：`Model/ET_model/train_classifier.py`

训练数据：
- 特征：`Model/ET_model/outputs_task_cluster/features.csv`（50 个 task 级样本）
- 标签：`Model/ET_model/outputs_task_cluster/clusters.csv`

> 注：训练时过滤了 `::task=none` 等无效任务，实际参与样本 50 个，覆盖 cluster 0、1、2、3。

### 4.2 模型指标（SVM）

| 指标 | 值 |
|------|------|
| 交叉验证准确率 | 74% ± 8% |
| 测试集准确率 | 100%（10 个留出样本） |
| 样本数 | 50 |
| 特征数 | 187 |
| 算法 | SVM |

> 由于样本量有限，指标仅供参考。后续在数据量增加后建议重新评估。

### 4.3 模型文件（已纳入 Git）

- `Model/ET_model/outputs_supervised_task/model_svm.joblib`
- `Model/ET_model/outputs_task_cluster/pca_model.joblib`

---

## 阶段五：实时预测系统（2026 年 2 月 - 3 月）

### 5.1 核心脚本

| 脚本 | 说明 |
|------|------|
| `Model/ET_model/realtime_session_monitor.py` | 监控采集目录，调用模型预测 |
| `Model/ET_model/realtime_dashboard.py` | Tkinter 可视化面板，实时展示预测结果 |
| `EyeTrace_controller.py` | 总控脚本，一键启动采集+监控+面板 |

### 5.2 实时预测输出

文件：`Model/ET_model/realtime_predictions_task_supervised.jsonl`

每行一个预测样本，字段包括：
- `session_dir`：采集目录路径
- `sample_key`：样本标识（`session_id::task=task_xxx`）
- `predicted_cluster`：预测的 cluster ID
- `coordinates_2d`：PCA 2D 坐标
- `relative_load_level`：相对认知负荷等级（1-4）
- `relative_load_label`：相对认知负荷标签
- `probabilities`：各类别预测概率

### 5.3 试跑记录

| 日期 | session | 结论 |
|------|---------|------|
| 2026-02-27 | `20260227_233556_realtime` | 实时链路闭环验证成功；后续需在更长 session 上验证稳定性 |

### 5.4 一键启动方式

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 EyeTrace_controller.py --with-monitor
```

启动顺序：监控脚本（后台）→ 预测面板（后台）→ 采集窗口（前台）
采集结束后自动终止后台进程。

---

## 阶段六：浏览器扩展（早期探索）

### 6.1 洛谷视线交互脚本

文档：`luogu_gaze_interaction_README.md`

功能：
- 基于 MediaPipe Face Detection 的人脸检测与跟踪
- 准心跟随人脸中心
- DOM 元素高亮
- 数据导出（JSON 格式）

当前状态：**已归档**，不作为 EyeTrace 主采集方案（改用 Python 端 L2CS-Net）。

---

## 目录结构一览

```
EyeTrace/
├── EyeTrace_controller.py          # 总控脚本
├── README.md                      # 项目入口说明
├── WORKLOG.md                     # 本文件：全项目工作日志
├── .gitignore
│
├── Cognitive/
│   └── cognitive-load-tracker/
│       ├── README.md              # 采集端说明
│       ├── requirements.txt
│       ├── cognitive_study/
│       │   ├── aoi_collector_v3_2.py      # 当前采集脚本
│       │   ├── aoi_config_tool.py         # AOI 配置工具
│       │   └── aoi_analyzer.py            # 离线分析
│       └── models/
│           └── L2CSNet_gaze360.pkl        # 视线模型（不纳入 Git）
│
├── Model/
│   └── ET_model/
│       ├── README.md               # 模型模块说明
│       ├── WORKLOG.md              # 模型模块详细日志
│       ├── requirements.txt
│       ├── predict_utils.py                # 预测核心工具
│       ├── cluster_cognitive_data.py       # 聚类脚本
│       ├── summarize_cluster_load.py      # 负荷映射脚本
│       ├── train_classifier.py            # 监督训练脚本
│       ├── realtime_session_monitor.py    # 实时监控脚本
│       ├── realtime_dashboard.py          # 实时面板
│       ├── offline_task_dashboard.py      # 离线任务浏览面板
│       ├── feature_weights_task.json       # 特征加权配置
│       ├── outputs_task_cluster/          # task 级聚类输出（纳入 Git）
│       │   ├── features.csv
│       │   ├── clusters.csv
│       │   ├── pca_model.joblib
│       │   ├── cluster_load_mapping.csv
│       │   └── cluster_load_summary.csv
│       └── outputs_supervised_task/       # 任务级监督模型（纳入 Git）
│           ├── model_svm.joblib
│           ├── metrics.json
│           └── cv_predictions_last_fold.csv
│
├── luogu_gaze_interaction_README.md  # 浏览器扩展文档（已归档）
└── data/                              # 采集数据（不纳入 Git）
```

---

## 当前推荐工作流

### 从零开始（新数据）

```bash
cd C:\Users\YNS\Desktop\EyeTrace

# 1. 采集数据
py -3.10 EyeTrace_controller.py

# 2. 任务级聚类（带加权）
py -3.10 Model\ET_model\cluster_cognitive_data.py ^
  --data_root data --unit task --k 6 ^
  --out_dir Model\ET_model\outputs_task_cluster ^
  --feature_weights_json Model\ET_model\feature_weights_task.json

# 3. 生成相对负荷映射
py -3.10 Model\ET_model\summarize_cluster_load.py ^
  --features Model\ET_model\outputs_task_cluster\features.csv ^
  --clusters Model\ET_model\outputs_task_cluster\clusters.csv ^
  --out_dir Model\ET_model\outputs_task_cluster

# 4. 重新训练监督模型（当样本量显著增加时）
py -3.10 Model\ET_model\train_classifier.py ^
  --features Model\ET_model\outputs_task_cluster\features.csv ^
  --labels Model\ET_model\outputs_task_cluster\clusters.csv ^
  --algo svm --out_dir Model\ET_model\outputs_supervised_task

# 5. 实时监控（采集+监控+面板一键启动）
py -3.10 EyeTrace_controller.py --with-monitor
```

### 直接使用已有模型（无需重新训练）

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 EyeTrace_controller.py --with-monitor
```

克隆仓库后，`outputs_task_cluster/` 和 `outputs_supervised_task/` 中的文件已纳入 Git，模型可直接使用，无需重新训练。

---

## 后续可选项

| 方向 | 说明 |
|------|------|
| 模型重训练 | 随着数据量增加，建议重新训练监督模型并重新评估指标 |
| 覆盖全部 cluster | 当前监督模型覆盖 cluster 0-3，cluster 4-5 样本不足；后续可补充数据后重新训练 |
| 实时验证 | 在更丰富的 session 上验证实时预测与主观体验的一致性 |
| 教学建议生成 | 基于负荷等级自动生成教学建议（需进一步开发） |
| 浏览器扩展 | 未来可考虑将实时监控能力集成到浏览器插件中 |
