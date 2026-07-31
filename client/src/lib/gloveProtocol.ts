/*
 * gloveProtocol.ts — 手套串口帧解析（协议核心，供单/双手 Hook 复用）
 *
 * 协议（矩侨精密 织物电子皮肤 高频版；V3 集成加速度）:
 *   帧头: 0xAA 0x55 0x03 0x99
 *   包1: [帧头4B][包序号0x01][传感器类型1B][数据128B]
 *   包2: [帧头4B][包序号0x02][传感器类型1B][数据 144B 或 168B]
 *        144B = 传感器后128B + IMU四元数16B                 → 拼合 272B（旧手套，无加速度）
 *        168B = 传感器后128B + 四元数16B + 加速度12B + 姿态角12B → 拼合 296B（带加速度手套）
 *   两包拼合:
 *     [0:256]   256B 传感器
 *     [256:272] 四元数 [w,x,y,z]（4×float32 LE）
 *     [272:284] 加速度 [x,y,z]（3×float32 LE）      —— 仅 296B 帧
 *     [284:296] 姿态角 [yaw,roll,pitch] 度（3×float32 LE）—— 仅 296B 帧
 *
 * 字节偏移与 glove_all_v3/serial_parser_two.py 的 SensorData 保持一致。
 *
 * 传感器类型:
 *   0x01 = LH (Left Hand)   0x02 = RH (Right Hand)
 *   0x03 = LF   0x04 = RF   0x05 = WB
 *
 * 波特率: 921600
 */
import { remapSensorData } from "@/lib/sensorMapping";

// 帧头
export const HEADER = new Uint8Array([0xaa, 0x55, 0x03, 0x99]);
export const HEADER_LEN = 4;
export const PACKET_TYPE_1 = 0x01;
export const PACKET_TYPE_2 = 0x02;

/** 带加速度手套的 type-2 数据段长度（默认）；旧手套为 144 */
export const TYPE2_LEN_WITH_ACC = 168;
export const TYPE2_LEN_LEGACY = 144;

// 传感器类型映射
export const SENSOR_TYPE_MAP: Record<number, string> = {
  0x01: "LH",
  0x02: "RH",
  0x03: "LF",
  0x04: "RF",
  0x05: "WB",
};

export interface GloveFrame {
  type: "glove_frame";
  /** performance.now() 高精度时间戳(ms) —— 与视觉帧同一时钟，利于同步 */
  timestamp: number;
  /** 传感器类型标识（若指定了 handType 则为指定值） */
  hand: number;
  /** "LH" | "RH" | ... */
  handLabel: string;
  /** 256个原始传感器值(0-255) */
  sensor_data: number[];
  /** 137个有效传感点(物理顺序) */
  mapped_data: number[];
  /** IMU 四元数 [w, x, y, z] */
  quaternion: [number, number, number, number];
  /** 加速度 [x, y, z]；旧手套(272B)为 null */
  acceleration: [number, number, number] | null;
  /** 姿态角 [yaw, roll, pitch]（度）；旧手套(272B)为 null */
  attitude: [number, number, number] | null;
  /** 帧序号 */
  frame_id: number;
}

export interface GloveParserOptions {
  /** 指定该数据流对应的手别（0x01=左/0x02=右）。设置后驱动 remap 与 handLabel，
   *  不再依赖固件的 sensor_type 字节（双口场景更稳健）。不设则用帧内 sensor_type。 */
  handType?: number;
  /** type-2 数据段长度：168=带加速度(默认)，144=旧手套 */
  type2Len?: number;
  /** 每解析出一帧回调 */
  onFrame: (frame: GloveFrame) => void;
}

/** 从小端字节读取 float32 数组 */
function readFloats(bytes: Uint8Array, offset: number, count: number): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset + offset, count * 4);
  const out: number[] = [];
  for (let i = 0; i < count; i++) out.push(view.getFloat32(i * 4, true));
  return out;
}

/**
 * 增量式手套帧解析器。喂入串口原始字节块，解析出完整帧后回调。
 * 逻辑与 serial_parser_two.py 的 find_packet/process_packet 一致（包1缓存等待包2拼合）。
 */
export class GloveParser {
  private buffer = new Uint8Array(0);
  private packet1Cache = new Map<number, Uint8Array>();
  private frameCount = 0;
  private readonly type2Len: number;

  constructor(private opts: GloveParserOptions) {
    this.type2Len = opts.type2Len ?? TYPE2_LEN_WITH_ACC;
  }

  /** 重置解析状态（重新连接时调用） */
  reset() {
    this.buffer = new Uint8Array(0);
    this.packet1Cache.clear();
    this.frameCount = 0;
  }

  get count() {
    return this.frameCount;
  }

  /** 喂入一块串口原始数据 */
  push(chunk: Uint8Array) {
    if (chunk.length === 0) return;
    const merged = new Uint8Array(this.buffer.length + chunk.length);
    merged.set(this.buffer, 0);
    merged.set(chunk, this.buffer.length);
    this.buffer = merged;
    this.parseBuffer();
  }

  private findHeader(buf: Uint8Array): number {
    for (let i = 0; i <= buf.length - HEADER_LEN; i++) {
      if (
        buf[i] === HEADER[0] &&
        buf[i + 1] === HEADER[1] &&
        buf[i + 2] === HEADER[2] &&
        buf[i + 3] === HEADER[3]
      ) {
        return i;
      }
    }
    return -1;
  }

  private parseBuffer() {
    let buf = this.buffer;

    while (buf.length >= HEADER_LEN) {
      const headerPos = this.findHeader(buf);
      if (headerPos === -1) {
        // 没找到帧头，保留末尾可能的部分帧头
        buf = buf.slice(Math.max(0, buf.length - HEADER_LEN + 1));
        break;
      }
      if (headerPos > 0) buf = buf.slice(headerPos);

      if (buf.length < HEADER_LEN + 2) break;

      const packetOrder = buf[HEADER_LEN];
      const sensorType = buf[HEADER_LEN + 1];

      let dataLen: number;
      if (packetOrder === PACKET_TYPE_1) {
        dataLen = 128;
      } else if (packetOrder === PACKET_TYPE_2) {
        dataLen = this.type2Len;
      } else {
        // 无效包序号，跳过该帧头继续找
        buf = buf.slice(HEADER_LEN);
        continue;
      }

      const totalLen = HEADER_LEN + 2 + dataLen;
      if (buf.length < totalLen) break; // 数据不完整，等待更多

      const packetData = buf.slice(HEADER_LEN + 2, totalLen);
      buf = buf.slice(totalLen);

      this.processPacket(packetOrder, sensorType, packetData);
    }

    this.buffer = buf;
  }

  private processPacket(
    packetOrder: number,
    sensorType: number,
    data: Uint8Array
  ) {
    if (packetOrder === PACKET_TYPE_1) {
      this.packet1Cache.set(sensorType, data);
      return;
    }
    // PACKET_TYPE_2
    const packet1Data = this.packet1Cache.get(sensorType);
    if (!packet1Data) return;
    this.packet1Cache.delete(sensorType);

    // 拼合: 包1(128B) + 包2(144B/168B)
    const combined = new Uint8Array(packet1Data.length + data.length);
    combined.set(packet1Data, 0);
    combined.set(data, packet1Data.length);

    if (combined.length < 272) return; // 传感器+四元数都不够，丢弃

    // 传感器值: 前256字节
    const sensorData: number[] = Array.from(combined.slice(0, 256));

    // 四元数: [256:272]
    let quaternion: [number, number, number, number] = [1, 0, 0, 0];
    const [qw, qx, qy, qz] = readFloats(combined, 256, 4);
    const qmag = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (
      qmag > 0.5 &&
      qmag < 2.0 &&
      isFinite(qw) &&
      isFinite(qx) &&
      isFinite(qy) &&
      isFinite(qz)
    ) {
      quaternion = [qw, qx, qy, qz];
    }

    // 加速度 [272:284] + 姿态角 [284:296]（仅 296B 帧）
    let acceleration: [number, number, number] | null = null;
    let attitude: [number, number, number] | null = null;
    if (combined.length >= 296) {
      const acc = readFloats(combined, 272, 3);
      if (acc.every((v) => isFinite(v))) {
        acceleration = [acc[0], acc[1], acc[2]];
      }
      const att = readFloats(combined, 284, 3);
      if (att.every((v) => isFinite(v))) {
        attitude = [att[0], att[1], att[2]];
      }
    }

    // 手别：优先用指定的 handType，否则用帧内 sensor_type
    const handType = this.opts.handType ?? sensorType;

    this.frameCount += 1;
    const frame: GloveFrame = {
      type: "glove_frame",
      timestamp: performance.now(),
      hand: handType,
      handLabel: SENSOR_TYPE_MAP[handType] || `0x${handType.toString(16)}`,
      sensor_data: sensorData,
      mapped_data: remapSensorData(sensorData, handType),
      quaternion,
      acceleration,
      attitude,
      frame_id: this.frameCount,
    };

    this.opts.onFrame(frame);
  }
}
