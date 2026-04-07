# 洛谷视线交互脚本 (Luogu Gaze Interaction)

## 📋 项目简介

这是一个基于 **Tampermonkey** 的用户脚本，专为洛谷网站（luogu.com.cn）设计。该脚本通过摄像头和 **MediaPipe** 人脸检测技术，实现准心跟随人脸中心的功能，并支持 DOM 元素高亮和数据采集导出。

## ✨ 核心功能

### 1. 人脸检测与跟踪
- 使用 **MediaPipe Face Detection** 模型进行实时人脸检测
- 准心自动跟随人脸中心位置
- 支持 `short` 和 `full` 两种检测模式（默认使用 `short` 模式以获得更快的响应速度）

### 2. 视觉反馈
- **准心标记**：屏幕中央显示黄色十字准心，实时跟随人脸位置
- **DOM 高亮**：自动高亮当前视线聚焦的页面元素（宽度 > 80px，高度 > 40px）
- **实时状态显示**：右下角控制面板显示当前运行状态

### 3. 数据采集与分析
- 自动记录视线数据（每 500ms 一次）
- 采集内容包括：
  - 时间戳
  - 视线坐标（X, Y）
  - 屏幕尺寸
  - 当前页面 URL
  - 聚焦元素的标签、类名、ID 和文本内容
- 支持一键导出为 JSON 格式文件

### 4. 用户界面
- 简洁的控制面板（右下角）
- 实时视频预览窗口
- 状态提示信息
- 一键关闭功能

## 🛠️ 技术栈

- **前端框架**：原生 JavaScript（ES6+）
- **人脸检测**：MediaPipe Face Detection v0.4.1635988162
- **摄像头 API**：MediaDevices.getUserMedia()
- **脚本管理器**：Tampermonkey / Greasemonkey

### 依赖的 MediaPipe 库
- `@mediapipe/camera_utils@0.3.1640029074`
- `@mediapipe/control_utils@0.6.1629159504`
- `@mediapipe/drawing_utils@0.3.1620248257`
- `@mediapipe/face_detection@0.4.1635988162`

## 📦 安装步骤

### 1. 安装脚本管理器
- **Chrome / Edge**：安装 [Tampermonkey](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo)
- **Firefox**：安装 [Tampermonkey](https://addons.mozilla.org/firefox/addon/tampermonkey/) 或 [Greasemonkey](https://addons.mozilla.org/firefox/addon/greasemonkey/)
- **Safari**：安装 [Tampermonkey](https://apps.apple.com/app/tampermonkey/id1482490089)

### 2. 安装脚本
1. 打开 Tampermonkey 管理面板
2. 点击"创建新脚本"
3. 将脚本代码粘贴到编辑器中
4. 保存（Ctrl+S 或 Cmd+S）

### 3. 首次使用
1. 访问 [洛谷网站](https://www.luogu.com.cn)
2. 脚本会自动运行并请求摄像头权限
3. 允许浏览器访问摄像头
4. 等待 MediaPipe 模型加载完成（首次加载可能需要几秒钟）

## 🎯 使用方法

### 基本操作
1. **启动**：访问洛谷网站后，脚本自动启动并请求摄像头权限
2. **授权**：在浏览器弹窗中允许访问摄像头
3. **等待加载**：首次使用时，MediaPipe 模型需要从 CDN 加载，请耐心等待
4. **开始使用**：模型加载完成后，准心会自动跟随您的脸部移动

### 数据导出
1. 点击右下角控制面板中的"导出数据"按钮
2. 系统会自动下载 JSON 格式的数据文件
3. 文件名格式：`luogu_gaze_data_[时间戳].json`

### 关闭脚本
- 点击控制面板右上角的 "✕" 按钮即可关闭脚本并释放摄像头资源

## ⚙️ 配置说明

### 检测模式
在脚本中修改以下参数可调整检测精度和速度：

```javascript
faceDetection.setOptions({
    model: 'short',  // 改为 'full' 可获得更高精度（但速度较慢）
    minDetectionConfidence: 0.5  // 检测置信度阈值（0-1）
});
```

### 数据采集频率
修改 `recordInterval` 变量可调整数据记录频率（单位：毫秒）：

```javascript
const recordInterval = 500;  // 默认 500ms，即每 0.5 秒记录一次
```

### 视频分辨率
在 `startCamera()` 函数中可调整摄像头分辨率：

```javascript
const constraints = {
    video: {
        width: { ideal: 640 },
        height: { ideal: 480 }
    }
};
```

## 🔧 故障排除

### 问题 1：摄像头无法启动
**可能原因**：
- 浏览器未授予摄像头权限
- Windows 隐私设置禁止浏览器访问摄像头
- 没有可用的摄像头设备

**解决方法**：
1. 检查浏览器权限设置（地址栏左侧的锁图标）
2. 检查 Windows 隐私设置 → 摄像头 → 允许应用访问摄像头
3. 确认摄像头设备正常工作

### 问题 2：MediaPipe 模型加载失败
**可能原因**：
- 网络连接问题，无法从 CDN 加载模型
- CDN 服务不可用

**解决方法**：
1. 检查网络连接
2. 脚本会自动降级为鼠标跟随模式
3. 刷新页面重试

### 问题 3：准心不跟随人脸
**可能原因**：
- 光线不足，人脸检测失败
- 人脸距离摄像头过远或过近
- 检测置信度阈值设置过高

**解决方法**：
1. 改善光照条件
2. 调整与摄像头的距离（建议 50-100cm）
3. 降低 `minDetectionConfidence` 值

### 问题 4：性能问题
**优化建议**：
- 使用 `short` 模式（默认）
- 降低视频分辨率
- 关闭其他占用资源的标签页

## 📊 数据格式说明

导出的 JSON 数据格式如下：

```json
[
  {
    "timestamp": 1234567890123,
    "gazeX": 640,
    "gazeY": 360,
    "screenWidth": 1920,
    "screenHeight": 1080,
    "url": "https://www.luogu.com.cn/problem/P1001",
    "hitElement": {
      "tagName": "DIV",
      "className": "problem-content",
      "id": "problem-content",
      "textContent": "题目描述..."
    }
  }
]
```

## 🔒 隐私与安全

- **本地处理**：所有人脸检测和数据采集均在本地浏览器完成
- **无服务器传输**：数据不会上传到任何服务器
- **用户控制**：可随时关闭脚本并删除采集的数据
- **权限最小化**：仅请求摄像头权限，不访问其他系统资源

## 📝 版本历史

### v3.0（当前版本）
- ✅ 集成 MediaPipe Face Detection
- ✅ 准心跟随人脸中心
- ✅ DOM 元素高亮功能
- ✅ 数据采集与导出
- ✅ 自动降级机制（模型加载失败时使用鼠标模式）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。

## ⚠️ 免责声明

本脚本仅供学习和研究使用。使用本脚本时，请确保：
- 遵守洛谷网站的使用条款
- 尊重他人隐私
- 不用于任何非法用途

---

**注意**：本脚本需要现代浏览器支持，建议使用 Chrome、Edge 或 Firefox 的最新版本。

