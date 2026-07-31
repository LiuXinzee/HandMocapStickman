# 手套 WebSocket 桥接服务

将柔性触觉手套的串口数据通过 WebSocket 实时转发给浏览器前端，实现手部动捕视频与触觉数据的同步采集。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动桥接服务

```bash
# 自动检测串口
python glove_ws_bridge.py

# 指定串口和参数
python glove_ws_bridge.py --port COM7 --baudrate 921600 --ws-port 8765

# 列出可用串口
python glove_ws_bridge.py --list-ports
```

### 3. 打开网页应用

启动桥接服务后，打开手部动捕网页，系统会自动连接 `ws://localhost:8765` 接收手套数据。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` / `-p` | 自动检测 | 串口端口号（如 COM7、/dev/ttyUSB0） |
| `--baudrate` / `-b` | 921600 | 波特率 |
| `--ws-port` / `-w` | 8765 | WebSocket 服务端口 |
| `--list-ports` / `-l` | - | 列出可用串口 |

## 数据协议

### 串口帧格式

- 帧头: `AA 55 03 99`
- 包1 (0x01): 128 字节传感器数据
- 包2 (0x02): 144 字节（128字节传感器 + 16字节IMU）
- 拼合: 256 字节传感器 + 16 字节四元数 = 272 字节

### WebSocket 推送格式

```json
{
    "type": "glove_frame",
    "timestamp": 1234567890.123456,
    "hand": 1,
    "sensor_data": [0, 12, 34, ...],
    "quaternion": [1.0, 0.0, 0.0, 0.0],
    "frame_id": 12345
}
```

## 同步机制

- 手套数据帧率: ~100Hz（取决于传感器硬件）
- 视频检测帧率: ~30fps
- 同步策略: 两路数据各自带高精度时间戳独立存储，导出时按时间戳最近邻匹配
