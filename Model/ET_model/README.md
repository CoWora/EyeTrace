# 眼动时序数据聚类与实时认知负荷预测

本模块负责对眼动时序数据进行聚类分析，基于聚类结果训练监督分类器，并支持实时任务级认知负荷预测。

## 功能概览

| 功能 | 脚本 | 说明 |
|------|------|------|
| 聚类 | `cluster_cognitive_data.py` | KMeans / 层次聚类 / DBSCAN，支持 session 级和 task 级 |
| 特征提取 | `eyerunn_cluster/` | 从多 CSV 时序数据中提取统计与时序特征 |
| 负荷映射 | `summarize_cluster_load.py` | 自动将 cluster 映射为相对认知负荷等级（1-4） |
| 监督训练 | `train_classifier.py` | SVM / XGBoost，将聚类结果固化为可预测模型 |
| 实时预测 | `realtime_session_monitor.py` | 监听采集目录，实时调用任务级模型预测认知负荷 |
| 预测可视化 | `realtime_dashboard.py` | Tkinter 面板，实时展示最新预测结果 |

## 1. 安装依赖

```bash
cd Model/ET_model
python -m pip install -r requirements.txt
```

## 聚类产物说明

聚类完成后在指定 `--out_dir` 下生成：

| 文件 | 说明 |
|------|------|
| `features.csv` | 每个样本的特征向量（索引为 `sample_key`） |
| `clusters.csv` | 聚类标签（`sample_key, cluster`） |
| `embedding_2d.csv` | PCA 2D 坐标（便于可视化） |
| `cluster_plot.png` | 聚类散点图 |
| `pca_model.joblib` | 包含 pipeline（imputer + scaler + weighter）与 PCA，用于后续对齐 transform |
| `cluster_load_summary.csv` | 每个 cluster 的关键特征均值 + 相对认知负荷等级 |
| `cluster_load_mapping.csv` | cluster → 相对认知负荷等级的精简映射表 |

## 当前已训练模型（已纳入 Git 跟踪）

### 任务级聚类 + 监督模型

训练基于 **task 级聚类**（粒度：每个 session 中的每个 task 一条样本），使用加权特征突出认知负荷相关维度。

**模型路径：**
- 聚类输出 & 模板：`Model/ET_model/outputs_task_cluster/`
- 监督分类器：`Model/ET_model/outputs_supervised_task/model_svm.joblib`

**聚类结果（6 个 cluster）：**

| cluster | 相对负荷等级 | 负荷标签 |
|---------|------------|---------|
| 0 | Level 2 | 低负荷 / 轻量任务型 |
| 1 | Level 2 | 低负荷 / 轻量任务型 |
| 2 | Level 3 | 中高负荷 / 信息整合型 |
| 3 | Level 4 | 高负荷 / 持续专注解题型 |
| 4 | Level 3 | 中高负荷 / 信息整合型 |
| 5 | Level 1 | 极低负荷 / 轻松浏览型 |

> 注意：监督分类器训练时过滤了 `::task=none` 等无效任务，实际参与训练的样本数为 50，对应 4 个 cluster（0、1、2、3）。后续新增数据后可重新训练以覆盖全部 cluster。

**监督模型指标（SVM，5 折交叉验证）：**
- 训练样本：50 个，187 特征
- 交叉验证准确率：74% ± 8%
- 测试集准确率：100%（10 个留出样本）

> 由于样本量有限，指标仅供参考。后续在数据量增加后建议重新评估。

**关键配置：**
- 特征加权配置：`Model/ET_model/feature_weights_task.json`
- 强调维度：`task__duration__mean`（×3.0）、`task__subjective_effort__mean`（×1.8）、`trans__n`（×2.5）等
- 特征前缀筛选：`fix__ / blink__ / trans__ / task__`

## 实时预测使用方法

### 通过总控脚本一键启动（推荐）

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 EyeTrace_controller.py --with-monitor
```

这会同时启动：采集窗口（前台）+ 准实时监控（后台）+ 预测面板（后台）。

### 直接调用监控脚本

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\realtime_session_monitor.py --watch_dirs data --interval 10
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--watch_dirs` | `Cognitive/data/cognitive_study data` | 要监控的数据根目录 |
| `--interval` | `10` | 轮询间隔（秒） |
| `--classifier_model` | `Model/ET_model/outputs_supervised_task/model_svm.joblib` | 任务级分类器 |
| `--pca_model` | `Model/ET_model/outputs_task_cluster/pca_model.joblib` | 任务级 PCA 模型 |
| `--features_template` | `Model/ET_model/outputs_task_cluster/features.csv` | 特征模板（对齐列名/顺序） |
| `--log_jsonl` | `Model/ET_model/realtime_predictions_task_supervised.jsonl` | 预测结果输出文件 |
| `--run_once` | - | 只扫描一次后退出（调试用） |

### 实时预测输出字段

每条预测记录包含：

| 字段 | 说明 |
|------|------|
| `session_dir` | 采集 session 目录路径 |
| `sample_key` | 样本标识，格式 `session_id::task=task_xxx` |
| `task_id` | 任务 ID（`task_001` 等） |
| `predicted_cluster` | 预测的 cluster ID |
| `coordinates_2d` | PCA 2D 坐标 `[x, y]` |
| `relative_load_level` | 相对认知负荷等级（1-4） |
| `relative_load_label` | 相对认知负荷标签（文本） |
| `probabilities` | 各 cluster 的预测概率分布 |

## 常见报错：特征维度不匹配

```
X has 191 features, but SimpleImputer is expecting 185 features as input.
```

含义：预测时的特征列数/顺序与模型训练时不一致。

解决方式：
1. **重新训练模型**：确保 `--features_template` 指向的 `features.csv` 与训练时使用的是同一份
2. **使用最新代码**：新版本在 `model_*.joblib` 中保存了 `feature_columns`，预测时会自动对齐并在不一致时打印警告

## 如果分组不满意怎么办

1) 打开 `outputs_task_cluster/clusters.csv`，手动改成你认为正确的类别
2) 重新运行 `train_classifier.py` 重新训练，模型会用你的修正结果再训练
3) 运行 `summarize_cluster_load.py --mapping_mode auto` 重新生成负荷映射
