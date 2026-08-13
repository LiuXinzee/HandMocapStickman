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

/** 带加速度手套的 type-2 数据段长度；旧手套为 144 */
export const TYPE2_LEN_WITH_ACC = 168;
export const TYPE2_LEN_LEGACY = 144;
export type GloveType2Length =
  | typeof TYPE2_LEN_LEGACY
  | typeof TYPE2_LEN_WITH_ACC;

const PACKET_METADATA_LEN = 2;
const TYPE1_DATA_LEN = 128;
const WAIT_FOR_MORE_DATA = 0;
const INVALID_PACKET_LENGTH = -1;
const DEFAULT_FRAME_INTERVAL_MS = 1000 / 227;
const MIN_PLAUSIBLE_FRAME_INTERVAL_MS = 1;
const MAX_PLAUSIBLE_FRAME_INTERVAL_MS = 20;
const MIN_TIMESTAMP_STEP_MS = 0.001;

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
  /** 固定 type-2 数据段长度；不设置时根据下一帧头自动识别 144/168。 */
  type2Len?: GloveType2Length;
  /** Optional monotonic clock, primarily for deterministic stream tests. */
  now?: () => number;
  /** 每解析出一帧回调 */
  onFrame: (frame: GloveFrame) => void;
}

/** 从小端字节读取 float32 数组 */
function readFloats(
  bytes: Uint8Array,
  offset: number,
  count: number
): number[] {
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
  private lastFrameTimestamp: number | null = null;
  private estimatedFrameIntervalMs = DEFAULT_FRAME_INTERVAL_MS;
  private readonly type2Len: GloveType2Length | null;

  constructor(private opts: GloveParserOptions) {
    if (
      opts.type2Len !== undefined &&
      opts.type2Len !== TYPE2_LEN_LEGACY &&
      opts.type2Len !== TYPE2_LEN_WITH_ACC
    ) {
      throw new RangeError("type2Len must be 144 or 168");
    }
    this.type2Len = opts.type2Len ?? null;
  }

  /** 重置解析状态（重新连接时调用） */
  reset() {
    this.buffer = new Uint8Array(0);
    this.packet1Cache.clear();
    this.frameCount = 0;
    this.lastFrameTimestamp = null;
    this.estimatedFrameIntervalMs = DEFAULT_FRAME_INTERVAL_MS;
  }

  get count() {
    return this.frameCount;
  }

  get pendingByteCount() {
    return this.buffer.length;
  }

  /** 喂入一块串口原始数据 */
  push(chunk: Uint8Array) {
    if (chunk.length === 0) return;
    const receivedAt = this.opts.now?.() ?? performance.now();
    const merged = new Uint8Array(this.buffer.length + chunk.length);
    merged.set(this.buffer, 0);
    merged.set(chunk, this.buffer.length);
    this.buffer = merged;
    this.emitFrames(this.parseBuffer(), receivedAt);
  }

  private findHeader(buf: Uint8Array, from: number): number {
    let position = buf.indexOf(HEADER[0], from);
    while (position >= 0 && position <= buf.length - HEADER_LEN) {
      if (
        buf[position + 1] === HEADER[1] &&
        buf[position + 2] === HEADER[2] &&
        buf[position + 3] === HEADER[3]
      ) {
        return position;
      }
      position = buf.indexOf(HEADER[0], position + 1);
    }
    return -1;
  }

  private hasHeaderAt(buf: Uint8Array, position: number): boolean {
    return (
      position >= 0 &&
      position + HEADER_LEN <= buf.length &&
      buf[position] === HEADER[0] &&
      buf[position + 1] === HEADER[1] &&
      buf[position + 2] === HEADER[2] &&
      buf[position + 3] === HEADER[3]
    );
  }

  private resolveType2Length(buf: Uint8Array, packetStart: number): number {
    if (this.type2Len !== null) return this.type2Len;

    const dataStart = packetStart + HEADER_LEN + PACKET_METADATA_LEN;
    const legacyEnd = dataStart + TYPE2_LEN_LEGACY;
    if (buf.length < legacyEnd + HEADER_LEN) return WAIT_FOR_MORE_DATA;
    if (this.hasHeaderAt(buf, legacyEnd)) return TYPE2_LEN_LEGACY;

    const extendedEnd = dataStart + TYPE2_LEN_WITH_ACC;
    if (buf.length < extendedEnd + HEADER_LEN) return WAIT_FOR_MORE_DATA;
    if (this.hasHeaderAt(buf, extendedEnd)) return TYPE2_LEN_WITH_ACC;

    return INVALID_PACKET_LENGTH;
  }

  private parseBuffer(): GloveFrame[] {
    const buf = this.buffer;
    let cursor = 0;
    const frames: GloveFrame[] = [];

    while (buf.length - cursor >= HEADER_LEN) {
      const headerPos = this.findHeader(buf, cursor);
      if (headerPos === -1) {
        // 没找到完整帧头，只保留末尾可能的部分帧头。
        cursor = Math.max(cursor, buf.length - HEADER_LEN + 1);
        break;
      }

      const metadataEnd = headerPos + HEADER_LEN + PACKET_METADATA_LEN;
      if (buf.length < metadataEnd) {
        cursor = headerPos;
        break;
      }

      const packetOrder = buf[headerPos + HEADER_LEN];
      const sensorType = buf[headerPos + HEADER_LEN + 1];

      let dataLen: number;
      if (packetOrder === PACKET_TYPE_1) {
        dataLen = TYPE1_DATA_LEN;
      } else if (packetOrder === PACKET_TYPE_2) {
        dataLen = this.resolveType2Length(buf, headerPos);
        if (dataLen === WAIT_FOR_MORE_DATA) {
          cursor = headerPos;
          break;
        }
        if (dataLen === INVALID_PACKET_LENGTH) {
          this.packet1Cache.delete(sensorType);
          cursor = headerPos + 1;
          continue;
        }
      } else {
        // 无效包序号，跳过当前帧头并继续寻找下一个候选。
        this.packet1Cache.delete(sensorType);
        cursor = headerPos + HEADER_LEN;
        continue;
      }

      const packetEnd = metadataEnd + dataLen;
      if (buf.length < packetEnd) {
        cursor = headerPos;
        break;
      }

      const packetData = buf.subarray(metadataEnd, packetEnd);
      const frame = this.processPacket(packetOrder, sensorType, packetData);
      if (frame) frames.push(frame);
      cursor = packetEnd;
    }

    if (cursor > 0) {
      this.buffer = buf.slice(cursor);
    }
    return frames;
  }

  private emitFrames(frames: GloveFrame[], receivedAt: number) {
    if (frames.length === 0) return;

    let interval = this.estimatedFrameIntervalMs;
    let firstTimestamp = receivedAt - interval * (frames.length - 1);
    if (this.lastFrameTimestamp !== null) {
      const elapsed = receivedAt - this.lastFrameTimestamp;
      const observedInterval = elapsed / frames.length;
      if (
        observedInterval >= MIN_PLAUSIBLE_FRAME_INTERVAL_MS &&
        observedInterval <= MAX_PLAUSIBLE_FRAME_INTERVAL_MS
      ) {
        interval = observedInterval;
        this.estimatedFrameIntervalMs =
          this.estimatedFrameIntervalMs * 0.8 + observedInterval * 0.2;
        firstTimestamp = this.lastFrameTimestamp + interval;
      } else if (elapsed > 0 && elapsed < interval * frames.length) {
        // Fit queued frames into the measured arrival span without teaching
        // the long-term estimator an artificial USB read cadence.
        interval = elapsed / frames.length;
        firstTimestamp = this.lastFrameTimestamp + interval;
      } else if (elapsed >= interval * frames.length) {
        // A scheduling pause occurred; anchor the newest frame to arrival time.
        firstTimestamp = receivedAt - interval * (frames.length - 1);
      } else {
        // Equal clock ticks still need a strict order, but only advance by a
        // negligible amount rather than a full estimated sensor interval.
        interval = MIN_TIMESTAMP_STEP_MS;
        firstTimestamp = this.lastFrameTimestamp + interval;
      }
    }

    for (let index = 0; index < frames.length; index++) {
      let timestamp = firstTimestamp + interval * index;
      if (
        this.lastFrameTimestamp !== null &&
        timestamp <= this.lastFrameTimestamp
      ) {
        timestamp = this.lastFrameTimestamp + interval;
      }
      frames[index].timestamp = timestamp;
      this.lastFrameTimestamp = timestamp;
      this.opts.onFrame(frames[index]);
    }
  }

  private processPacket(
    packetOrder: number,
    sensorType: number,
    data: Uint8Array
  ): GloveFrame | null {
    if (packetOrder === PACKET_TYPE_1) {
      this.packet1Cache.set(sensorType, data.slice());
      return null;
    }
    // PACKET_TYPE_2
    const packet1Data = this.packet1Cache.get(sensorType);
    if (!packet1Data) return null;
    this.packet1Cache.delete(sensorType);

    // 拼合: 包1(128B) + 包2(144B/168B)
    const combined = new Uint8Array(packet1Data.length + data.length);
    combined.set(packet1Data, 0);
    combined.set(data, packet1Data.length);

    if (combined.length < 272) return null; // 传感器+四元数都不够，丢弃

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
      if (acc.every(v => isFinite(v))) {
        acceleration = [acc[0], acc[1], acc[2]];
      }
      const att = readFloats(combined, 284, 3);
      if (att.every(v => isFinite(v))) {
        attitude = [att[0], att[1], att[2]];
      }
    }

    // 手别：优先用指定的 handType，否则用帧内 sensor_type
    const handType = this.opts.handType ?? sensorType;

    this.frameCount += 1;
    const frame: GloveFrame = {
      type: "glove_frame",
      timestamp: 0,
      hand: handType,
      handLabel: SENSOR_TYPE_MAP[handType] || `0x${handType.toString(16)}`,
      sensor_data: sensorData,
      mapped_data: remapSensorData(sensorData, handType),
      quaternion,
      acceleration,
      attitude,
      frame_id: this.frameCount,
    };

    return frame;
  }
}
