# 手套 WebSocket 桥接服务

该服务使用 Python 从串口读取柔性触觉手套数据，再通过 WebSocket 实时转发给浏览器或其他客户端。它适合无法直接使用 Web Serial API，或需要由独立进程统一读取串口的场景。

> 当前项目的主要页面默认通过 Web Serial API 直连手套，不依赖本服务。只有使用 `useGloveConnection` 或自行连接 WebSocket 的客户端才会接收本服务的数据。

## 安装依赖

在项目根目录执行：

```bash
python -m pip install -r glove_bridge/requirements.txt
```

## 启动服务

以下命令均在项目根目录执行。

```bash
# 自动选择检测到的第一个串口
python glove_bridge/glove_ws_bridge.py

# 指定串口、波特率和 WebSocket 端口
python glove_bridge/glove_ws_bridge.py --port COM7 --baudrate 921600 --ws-port 8765

# 列出可用串口
python glove_bridge/glove_ws_bridge.py --list-ports
```

服务启动后，客户端可连接：

```text
ws://localhost:8765
```

按 `Ctrl+C` 停止服务。

## 命令行参数

| 参数                  | 默认值   | 说明                                                            |
| --------------------- | -------- | --------------------------------------------------------------- |
| `--port` / `-p`       | 自动选择 | 串口名称，例如 Windows 下的 `COM7` 或 Linux 下的 `/dev/ttyUSB0` |
| `--baudrate` / `-b`   | `921600` | 串口波特率                                                      |
| `--ws-port` / `-w`    | `8765`   | WebSocket 服务端口                                              |
| `--list-ports` / `-l` | 无       | 列出可用串口后退出                                              |

串口参数固定为 8 个数据位、无校验位、1 个停止位（8N1）。WebSocket 服务监听 `0.0.0.0`，因此同一局域网内的其他设备也可使用运行服务的计算机 IP 地址连接。

## 串口协议

当前 Python 桥接脚本解析旧版 272 字节有效载荷格式：

- 固定帧头：`AA 55 03 99`
- 第一包：包序号 `0x01`，包含 128 字节传感器数据；整包共 134 字节
- 第二包：包序号 `0x02`，包含 128 字节传感器数据和 16 字节 IMU 四元数；整包共 150 字节
- 两包的数据段拼合后共 272 字节：前 256 字节为原始传感器值，最后 16 字节为四个小端 `float32` 四元数 `[w, x, y, z]`

浏览器直连解析器还支持第二包为 168 字节的扩展格式，其中额外包含加速度和姿态角；当前 `glove_ws_bridge.py` 尚不支持该扩展格式。完整协议见 [protocol_notes.md](./protocol_notes.md)。

## WebSocket 消息格式

每拼合出一帧，服务会发送一条 `glove_frame` JSON 消息：

```json
{
  "type": "glove_frame",
  "timestamp": 1234567890.123456,
  "hand": 1,
  "sensor_data": [0, 12, 34],
  "quaternion": [1.0, 0.0, 0.0, 0.0],
  "frame_id": 12345
}
```

示例中的 `sensor_data` 为便于展示而省略，实际包含 256 个取值范围为 `0` 至 `255` 的原始传感器值。各字段含义如下：

| 字段          | 含义                                                                |
| ------------- | ------------------------------------------------------------------- |
| `timestamp`   | 桥接服务收到第二包时通过 `time.time()` 记录的 Unix 时间戳，单位为秒 |
| `hand`        | 串口帧中的传感器类型；`1`（`0x01`）表示左手，`2`（`0x02`）表示右手  |
| `sensor_data` | 256 个原始传感器值，尚未按物理位置重排                              |
| `quaternion`  | IMU 四元数 `[w, x, y, z]`；数据无效时使用 `[1.0, 0.0, 0.0, 0.0]`    |
| `frame_id`    | 桥接服务生成的递增帧序号                                            |

客户端连接成功后还会收到一条 `connected` 消息。客户端可发送 `{"type":"ping"}` 检查连接，或发送 `{"type":"get_status"}` 查询串口状态、帧率、帧数和错误数。

## 数据同步说明

桥接服务只负责读取、加时间戳和转发手套数据，不负责录制视频或匹配视频帧。若通过 WebSocket 自行实现同步采集，应将手套帧与视频帧统一到同一时钟后，再按时间戳做最近邻匹配；不能直接将 Unix 秒时间戳与浏览器的 `performance.now()` 毫秒时间戳混用。
