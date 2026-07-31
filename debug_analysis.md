# 采集数据为 0 的根本原因分析

## 问题 1：IndexedDB VersionError（已修复）

- 用户浏览器中已有 v4 数据库（之前骨架采集独立化版本创建的）
- 回滚后代码使用 DB_VERSION = 3
- `indexedDB.open(name, 3)` 打开已有的 v4 数据库会抛出 VersionError
- **修复**：将 `DB_VERSION` 改为 4，并添加 `skeleton_samples` 存储区的创建逻辑

## 问题 2：`handResults` 为 `null`

从截图分析：

- 左侧视频预览区域是**纯黑色**，没有摄像头画面
- 右侧骨架动画是 HandSkeletonAnim 的预设动画（"伸出食指"），不是实时检测
- 控制台日志：`[DataCollect] No hand detected, skipped N times. hands= null`

### 关键发现

从 `DataCollect.tsx` 第 100 行：`const bothReady = gloveConnected && cameraRunning;`

- 页面显示了采集界面（REC 0/20），说明 `bothReady = true`
- 但 `handResultsRef.current = null`

这说明：

1. `cameraRunning = true`（摄像头已启动）
2. `handResults = null`（MediaPipe 没有检测到手）

可能原因：

- 摄像头启动了但 MediaPipe 模型还在加载中
- 用户的手在摄像头视野中但 MediaPipe 检测失败
- **更可能**: 视频流正常但 HandCanvas 没有收到 videoRef（左侧全黑）

### HandCanvas 分析

HandCanvas 第 69 行: `if (showVideo && videoRef?.current && videoRef.current.readyState >= 2)`

- 如果 videoRef.current 为 null 或 readyState < 2，就显示黑色背景
- useHandTracking 中 videoRef 是通过 `videoRef.current = video` 赋值的
- 但 video 元素是动态创建的隐藏元素

### 结论

视频预览全黑 + handResults 为 null = **MediaPipe 检测循环可能没有正常运行**
但 cameraRunning = true 说明初始化流程走完了

最可能的原因：**MediaPipe CDN 加载问题**。用户在中国大陆时，`cdn.jsdelivr.net` 可能无法访问或加载很慢。
或者 MediaPipe 模型加载成功但推理失败（WebGL/WASM 问题）

### 但是...

用户说"摄像头检测到了手"，而且截图中确实有 REC 按钮显示。
这意味着 MediaPipe 在首页是工作的，但在 DataCollect 页面可能需要重新初始化。

**真正的问题**：`DataCollect` 页面有自己的 `useHandTracking` 实例，需要用户在这个页面重新启动摄像头。
用户可能是在首页启动了摄像头，然后导航到 DataCollect 页面，但首页的摄像头实例不会传递到 DataCollect。

不对，DataCollect 页面有自己的"启动摄像头"按钮，而且 bothReady = true 说明用户确实在这个页面启动了。

**最终结论**：问题出在 `setInterval` 的 100ms 间隔内，`handResults` 可能在某些帧之间被设为 `null`（MediaPipe 检测不到手的帧），而引用更新不及时。但日志显示是持续为 `null`，不是间歇性的。

需要在 DataCollect 中添加更多调试信息来确认 handResults 的状态变化。
