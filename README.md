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

---

## Git 实践学习记录

### 一、学习资料来源

- **官方文档**: [Git 官方文档](https://git-scm.com/book/zh/v2)
- **GitHub Guides**: [GitHub Getting Started](https://docs.github.com/en/get-started)
- **廖雪峰 Git 教程**: [Git 教程](https://www.liaoxuefeng.com/wiki/896043488029600)
- **实践平台**: 本项目仓库 `C:\Users\YNS\Desktop\EyeTrace`

### 二、实践流程

#### 2.1 Git 安装与初始化

```powershell
# 检查 Git 是否已安装
git --version

# 全局配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 初始化本地仓库
cd "C:\Users\YNS\Desktop\EyeTrace"
git init
```

#### 2.2 基本工作流程

```powershell
# 查看当前状态
git status

# 添加文件到暂存区
git add <文件名>       # 添加单个文件
git add .             # 添加所有更改

# 提交到本地仓库
git commit -m "提交信息"

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v
```

#### 2.3 分支管理

```powershell
# 创建新分支
git branch <分支名>

# 切换分支
git checkout <分支名>
# 或
git switch <分支名>

# 创建并切换
git checkout -b <分支名>

# 合并分支
git merge <分支名>

# 删除分支
git branch -d <分支名>
```

#### 2.4 远程协作

```powershell
# 添加远程仓库
git remote add origin <仓库URL>

# 推送代码
git push -u origin master

# 拉取更新
git pull origin master
```

### 三、提交历史记录

| 提交哈希 | 日期 | 提交信息 | 主要内容 |
|---------|------|---------|---------|
| `30139c7` | 2026-04-08 | `docs: update README, WORKLOG and .gitignore for model upload` | 更新模型模块的README、WORKLOG，添加.gitignore配置，新增聚类输出文件(SVM模型、PCA模型、metrics.json、confusion_matrix.png等) |
| `325072a` | 2026-04-07 | `docs: update README, WORKLOG and .gitignore for model upload` | 优化.gitignore配置，扩展README.md内容，新增luogu_gaze_interaction_README.md，整理模型工作日志 |
| `b74640a` | 2026-04-07 | `Initial commit: EyeTrace cognitive load tracking system` | 项目初始化，提交EyeTrace认知负荷追踪系统的核心代码和文档 |

### 四、遇到的问题及解决方法

#### 问题1: 大文件误提交导致仓库膨胀

**问题描述**: 在初次提交时，误将模型权重文件(`.pkl`)、预训练模型文件(`.pth`)、大型输出文件等二进制文件提交到了Git仓库，导致仓库体积过大。

**解决方法**:
1. 在项目根目录和各子模块创建 `.gitignore` 文件
2. 配置忽略规则:

```gitignore
# 大模型文件
*.pkl
*.pth
*.pt
*.onnx
*.tflite

# 输出文件
outputs/
outputs_supervised/
outputs_task/

# 数据文件
data/
*.csv
```

3. 对于已提交的大文件，使用以下命令移除:

```powershell
# 从Git中移除（保留本地文件）
git rm --cached <大文件路径>

# 强制推送到远程（清除历史记录）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch <大文件路径>" \
  --prune-empty --tag-name-filter cat -- --all
```

**经验总结**: 养成"先忽略，后提交"的习惯。`.gitignore` 应在项目初始化时第一时间创建。

#### 问题2: Windows PowerShell 中 Git 命令执行报错

**问题描述**: 在使用 PowerShell 执行 `git log --oneline --all` 时，出现如下错误:

```
The token '&&' is not a valid statement separator in this version
```

**问题原因**: PowerShell 的命令分隔符与 Bash 不同，不支持 `&&` 运算符。

**解决方法**:

```powershell
# 方法1: 使用分号分隔命令
cd "C:\Users\YNS\Desktop\EyeTrace"; git log --oneline

# 方法2: 分两行执行
cd "C:\Users\YNS\Desktop\EyeTrace"
git log --oneline

# 方法3: 在Git Bash中执行
```

**经验总结**: 跨平台开发时需注意Shell语法的差异。Linux/Mac的Bash与Windows的PowerShell/CMD在管道符、命令分隔符等细节上存在区别。

#### 问题3: 提交信息不规范

**问题描述**: 最初提交时使用中文或简单的 "update" 作为提交信息，不便于后续追溯和维护。

**解决方法**: 采用 Conventional Commits 规范:

```bash
# 格式: <type>: <subject>
# type 类型:
# - feat: 新功能
# - fix: 修复bug
# - docs: 文档更新
# - style: 代码格式（不影响功能）
# - refactor: 重构
# - test: 测试相关
# - chore: 构建/工具相关

git commit -m "docs: update README for model upload"
git commit -m "fix: resolve merge conflict in config"
```

**经验总结**: 规范的提交信息是团队协作的基础，建议从一开始就养成好习惯。

### 五、Git 学习心得

#### 5.1 理解Git的核心概念

通过这次实践，我对Git的三大区域有了更清晰的认识:
- **工作区 (Working Directory)**: 实际编辑文件的目录
- **暂存区 (Staging Area)**: 临时存放即将提交的内容
- **版本库 (Repository)**: Git保存所有版本数据的地方

#### 5.2 分支管理的优势

分支机制让并行开发成为可能。在实际项目中，可以:
- `main/master`: 稳定版本，始终保持可发布状态
- `develop`: 开发分支，集成各功能模块
- `feature/*`: 功能分支，单独开发新功能
- `hotfix/*`: 紧急修复，快速修复生产环境问题

#### 5.3 版本控制不仅是备份

Git不仅是文件备份工具，更是一个项目管理工具:
- **追溯历史**: 可以查看每一行代码是谁在什么时候为什么而修改的
- **团队协作**: 多人可以同时工作在同一个项目上而互不干扰
- **实验性开发**: 可以在分支上大胆尝试，失败了就删除分支重来
- **代码审查**: 通过Pull Request可以让团队成员在合并前进行代码审查

#### 5.4 持续学习的方向

- 学习 `git rebase` 进行更优雅的分支整合
- 掌握 `git stash` 暂存未完成的修改
- 理解 `git cherry-pick` 选择性地合并提交
- 使用 `.gitignore` 规范管理敏感文件和构建产物
- 学习 Git Flow 或 Trunk-Based Development 等工作流

#### 5.5 最佳实践建议

1. **频繁提交，小步前进**: 每完成一个独立功能就提交
2. **提交前检查**: 使用 `git diff` 确认修改内容
3. **写好提交信息**: 清晰描述"做了什么"和"为什么这么做"
4. **善用 `.gitignore`**: 避免提交无关文件
5. **分支策略**: 根据项目规模选择合适的分支管理模型
6. **定期同步**: 经常 `pull` 远程更新，避免大量冲突

---

> 📝 本文档记录于 2026-04-08，作为 Git 版本控制学习的实践总结。