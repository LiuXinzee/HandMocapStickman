"""
手套 WebSocket 桥接服务
========================
将串口手套数据通过 WebSocket 实时转发给浏览器前端。

使用方法:
    python glove_ws_bridge.py --port COM7 --baudrate 921600 --ws-port 8765

协议:
    手套帧格式: 帧头 AA 55 03 99 + 包序号(1B) + 传感器类型(1B) + 数据
    包1: 128字节数据
    包2: 自动识别 144/168 字节数据段
    拼合后: 256字节传感器 + 16字节IMU，可选加速度和姿态角

WebSocket 推送 JSON:
    {
        "type": "glove_frame",
        "timestamp": 1234567890.123456,  // 高精度时间戳(秒)
        "hand": 1,                        // 传感器类型标识
        "sensor_data": [0-255] x 256,     // 256个传感器值
        "quaternion": [w, x, y, z],       // IMU四元数
        "acceleration": [x, y, z] | null, // 扩展格式加速度
        "attitude": [yaw, roll, pitch] | null,
        "frame_id": 12345                 // 帧序号
    }
"""

import asyncio
import json
import math
import time
import struct
import sys
import argparse
import threading
import queue

try:
    import serial
except ImportError:
    print("请安装 pyserial: pip install pyserial")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("请安装 websockets: pip install websockets")
    sys.exit(1)

# ============================================================
# 帧头和包类型定义（与 serial_parser_two.py 保持一致）
# ============================================================
HEADER = bytes([0xAA, 0x55, 0x03, 0x99])
HEADER_LEN = len(HEADER)
PACKET_TYPE_1 = 0x01
PACKET_TYPE_2 = 0x02
PACKET_METADATA_LEN = 2
PACKET1_DATA_LEN = 128
PACKET2_DATA_LENS = (144, 168)


class GloveSerialReader:
    """串口数据读取和解析器（简化版，适配 WebSocket 桥接）"""

    def __init__(self, port, baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=500)
        self.frame_count = 0
        self.error_count = 0

        # 解析缓冲区
        self.buffer = bytearray()

        # 包1缓存（等待包2来拼合）
        self.packet1_cache = {}

        # 统计
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.current_fps = 0.0

    def connect(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0.01
            )
            print(f"[Bridge] 已连接串口 {self.port}, 波特率 {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"[Bridge] 连接串口失败: {e}")
            return False

    def disconnect(self):
        """断开串口"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[Bridge] 串口已关闭")

    def read_loop(self):
        """串口读取和解析循环（在独立线程中运行）"""
        self.running = True
        print("[Bridge] 串口读取线程启动")

        while self.running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue

            try:
                data = self.ser.read(256)
                if data:
                    read_timestamp = time.time()
                    self.buffer.extend(data)
                    self._parse_buffer(read_timestamp)
            except serial.SerialException as e:
                print(f"[Bridge] 串口读取错误: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"[Bridge] 解析错误: {e}")

        print("[Bridge] 串口读取线程退出")

    def _packet2_data_len(self, packet_start, buffer_len):
        """根据候选包体后的下一帧头判断包2长度；数据不足时返回 None。"""
        data_start = packet_start + HEADER_LEN + PACKET_METADATA_LEN
        for data_len in PACKET2_DATA_LENS:
            next_packet = data_start + data_len
            if buffer_len < next_packet + HEADER_LEN:
                return None
            if self.buffer[next_packet:next_packet + HEADER_LEN] == HEADER:
                return data_len
        return -1

    def _parse_buffer(self, read_timestamp=None):
        """从缓冲区中解析完整数据帧，并在一轮扫描后统一裁剪缓冲区。"""
        if read_timestamp is None:
            read_timestamp = time.time()

        cursor = 0
        buffer_len = len(self.buffer)
        minimum_packet_len = HEADER_LEN + PACKET_METADATA_LEN

        while buffer_len - cursor >= HEADER_LEN:
            header_pos = self.buffer.find(HEADER, cursor)
            if header_pos < 0:
                # 只保留可能是下一帧头前缀的末尾字节。
                cursor = max(cursor, buffer_len - HEADER_LEN + 1)
                break

            cursor = header_pos
            if buffer_len - cursor < minimum_packet_len:
                break

            packet_order = self.buffer[cursor + HEADER_LEN]
            sensor_type = self.buffer[cursor + HEADER_LEN + 1]

            if packet_order == PACKET_TYPE_1:
                data_len = PACKET1_DATA_LEN
            elif packet_order == PACKET_TYPE_2:
                data_len = self._packet2_data_len(cursor, buffer_len)
                if data_len is None:
                    break
                if data_len < 0:
                    # 两种候选长度都无法落到下一帧头，跳过当前假帧头重同步。
                    self.packet1_cache.pop(sensor_type, None)
                    self.error_count += 1
                    cursor += HEADER_LEN
                    continue
            else:
                # 无效包序号不应中断本轮扫描，继续寻找后续有效帧头。
                self.packet1_cache.pop(sensor_type, None)
                self.error_count += 1
                cursor += HEADER_LEN
                continue

            total_len = HEADER_LEN + PACKET_METADATA_LEN + data_len
            if buffer_len - cursor < total_len:
                break

            data_start = cursor + minimum_packet_len
            packet_end = cursor + total_len
            packet_data = bytes(self.buffer[data_start:packet_end])
            self._process_packet(
                packet_order,
                sensor_type,
                packet_data,
                read_timestamp,
            )
            cursor = packet_end

        if cursor:
            del self.buffer[:cursor]

    def _process_packet(self, packet_order, sensor_type, data, timestamp):
        """处理解析出的数据包"""
        if packet_order == PACKET_TYPE_1:
            # 缓存包1，等待包2
            self.packet1_cache[sensor_type] = data

        elif packet_order == PACKET_TYPE_2:
            # 检查是否有对应的包1
            if sensor_type in self.packet1_cache:
                packet1_data = self.packet1_cache.pop(sensor_type)

                # 拼合数据: 包1(128B) + 包2(144B/168B)
                combined_data = packet1_data + data

                # 解析传感器值（前256字节）
                sensor_values = list(combined_data[:256])

                # 四元数偏移固定；扩展字段只在 168 字节包2中存在。
                quaternion = None
                if len(combined_data) >= 272:
                    try:
                        imu_bytes = bytes(combined_data[256:272])
                        q = struct.unpack('<4f', imu_bytes)
                        # 检查有效性
                        magnitude = sum(v * v for v in q) ** 0.5
                        if 0.5 < magnitude < 2.0 and all(abs(v) < 10 for v in q):
                            quaternion = list(q)
                    except Exception:
                        pass

                acceleration = None
                attitude = None
                if len(combined_data) >= 296:
                    try:
                        acceleration_values = struct.unpack(
                            '<3f', combined_data[272:284]
                        )
                        attitude_values = struct.unpack(
                            '<3f', combined_data[284:296]
                        )
                        if all(math.isfinite(value) for value in acceleration_values):
                            acceleration = list(acceleration_values)
                        if all(math.isfinite(value) for value in attitude_values):
                            attitude = list(attitude_values)
                    except struct.error:
                        pass

                # 构建帧数据
                self.frame_count += 1
                frame = {
                    "type": "glove_frame",
                    "timestamp": timestamp,
                    "hand": sensor_type,
                    "sensor_data": sensor_values,
                    "quaternion": quaternion or [1.0, 0.0, 0.0, 0.0],
                    "acceleration": acceleration,
                    "attitude": attitude,
                    "frame_id": self.frame_count
                }

                # 放入队列
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # 队列满了，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                # FPS 统计
                self.fps_counter += 1
                now = time.time()
                if now - self.fps_last_time >= 1.0:
                    self.current_fps = self.fps_counter / (now - self.fps_last_time)
                    self.fps_counter = 0
                    self.fps_last_time = now

            else:
                self.error_count += 1


class GloveWebSocketBridge:
    """WebSocket 桥接服务器"""

    def __init__(self, serial_reader, ws_host="0.0.0.0", ws_port=8765):
        self.serial_reader = serial_reader
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.clients = set()
        self.running = False

    async def handler(self, websocket):
        """处理 WebSocket 连接"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"[Bridge] 客户端连接: {client_addr}")

        # 发送连接确认
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "Glove bridge connected",
            "port": self.serial_reader.port,
            "baudrate": self.serial_reader.baudrate
        }))

        try:
            async for message in websocket:
                # 处理来自前端的命令
                try:
                    cmd = json.loads(message)
                    await self._handle_command(websocket, cmd)
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[Bridge] 客户端断开: {client_addr}")

    async def _handle_command(self, websocket, cmd):
        """处理前端发来的命令"""
        cmd_type = cmd.get("type", "")

        if cmd_type == "ping":
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": time.time()
            }))

        elif cmd_type == "get_status":
            await websocket.send(json.dumps({
                "type": "status",
                "connected": self.serial_reader.ser is not None and self.serial_reader.ser.is_open,
                "fps": round(self.serial_reader.current_fps, 1),
                "frame_count": self.serial_reader.frame_count,
                "error_count": self.serial_reader.error_count,
                "port": self.serial_reader.port,
                "baudrate": self.serial_reader.baudrate
            }))

    async def broadcast_loop(self):
        """从串口队列取数据并广播给所有 WebSocket 客户端"""
        while self.running:
            frames_batch = []

            # 批量取出队列中的帧（减少 JSON 序列化次数）
            try:
                while len(frames_batch) < 10:
                    frame = self.serial_reader.frame_queue.get_nowait()
                    frames_batch.append(frame)
            except queue.Empty:
                pass

            if frames_batch and self.clients:
                # 逐帧发送（保持时序）
                for frame in frames_batch:
                    message = json.dumps(frame)
                    # 广播给所有连接的客户端
                    disconnected = set()
                    for client in self.clients.copy():
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                    self.clients -= disconnected

            # 短暂休眠，避免 CPU 空转
            await asyncio.sleep(0.005)  # 5ms，支持 ~200Hz 转发

    async def status_loop(self):
        """定期打印状态"""
        while self.running:
            await asyncio.sleep(5.0)
            print(
                f"[Bridge] 状态: FPS={self.serial_reader.current_fps:.1f}, "
                f"帧数={self.serial_reader.frame_count}, "
                f"错误={self.serial_reader.error_count}, "
                f"客户端={len(self.clients)}"
            )

    async def run(self):
        """启动 WebSocket 服务器"""
        self.running = True

        # 启动串口读取线程
        serial_thread = threading.Thread(
            target=self.serial_reader.read_loop, daemon=True
        )
        serial_thread.start()

        # 启动 WebSocket 服务器
        print(f"[Bridge] WebSocket 服务器启动: ws://{self.ws_host}:{self.ws_port}")
        print(f"[Bridge] 前端连接地址: ws://localhost:{self.ws_port}")
        print("[Bridge] 按 Ctrl+C 停止服务")

        async with websockets.serve(
            self.handler,
            self.ws_host,
            self.ws_port,
            ping_interval=20,
            ping_timeout=60
        ):
            await asyncio.gather(
                self.broadcast_loop(),
                self.status_loop()
            )


def list_serial_ports():
    """列出可用的串口"""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        if ports:
            print("\n可用串口:")
            for port in ports:
                print(f"  {port.device} - {port.description}")
        else:
            print("\n未检测到可用串口")
        return [p.device for p in ports]
    except Exception as e:
        print(f"列举串口失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="手套 WebSocket 桥接服务 - 将串口手套数据转发给浏览器"
    )
    parser.add_argument(
        "--port", "-p",
        type=str,
        default=None,
        help="串口端口号 (如 COM7 或 /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=921600,
        help="波特率 (默认: 921600)"
    )
    parser.add_argument(
        "--ws-port", "-w",
        type=int,
        default=8765,
        help="WebSocket 服务端口 (默认: 8765)"
    )
    parser.add_argument(
        "--list-ports", "-l",
        action="store_true",
        help="列出可用串口"
    )

    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        return

    if not args.port:
        ports = list_serial_ports()
        if ports:
            args.port = ports[0]
            print(f"\n自动选择串口: {args.port}")
        else:
            print("\n请使用 --port 参数指定串口")
            return

    # 创建串口读取器
    reader = GloveSerialReader(args.port, args.baudrate)
    if not reader.connect():
        return

    # 创建并启动 WebSocket 桥接
    bridge = GloveWebSocketBridge(reader, ws_port=args.ws_port)

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\n[Bridge] 正在停止...")
    finally:
        reader.disconnect()
        print("[Bridge] 服务已停止")


if __name__ == "__main__":
    main()
