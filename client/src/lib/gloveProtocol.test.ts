/**
 * GloveProtocol 帧解析回归测试。
 *
 * 背景：type2Len 曾经写死成 168，而现场手套是 144 固件。多读 24 字节会啃穿下一个
 * 包1的帧头，导致包1全被跳过、包2全配不上对，一帧都出不来——表现为"连上了但完全
 * 没数据"。这里同时构造两种固件的字节流，确保不指定长度时都能解析出来。
 */
import { describe, it, expect } from "vitest";
import {
  GloveParser,
  HEADER,
  HEADER_LEN,
  PACKET1_LEN,
  PACKET_TYPE_1,
  PACKET_TYPE_2,
  TYPE2_LEN_LEGACY,
  TYPE2_LEN_WITH_ACC,
  type GloveFrame,
} from "./gloveProtocol";

const SENSOR_TYPE_LH = 0x01;

function packet(order: number, body: Uint8Array): number[] {
  return [...HEADER, order, SENSOR_TYPE_LH, ...body];
}

function f32(values: number[]): Uint8Array {
  const buf = new Uint8Array(values.length * 4);
  const view = new DataView(buf.buffer);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

/** 造 n 组包1+包2。传感器值用 (i % 200) + 1 填充，保证非零、便于校验偏移 */
function makeStream(n: number, type2Len: number): Uint8Array {
  const quat = [0.5, 0.5, 0.5, 0.5];
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const sensorHalf = new Uint8Array(PACKET1_LEN).fill((i % 200) + 1);
    out.push(...packet(PACKET_TYPE_1, sensorHalf));

    const body = new Uint8Array(type2Len);
    body.set(sensorHalf, 0);
    body.set(f32(quat), PACKET1_LEN);
    if (type2Len === TYPE2_LEN_WITH_ACC) {
      body.set(f32([1, 2, 3]), PACKET1_LEN + 16); // 加速度
      body.set(f32([4, 5, 6]), PACKET1_LEN + 28); // 姿态角
    }
    out.push(...packet(PACKET_TYPE_2, body));
  }
  // 末尾补一个帧头，让最后一包也能完成长度探测
  out.push(...HEADER);
  return new Uint8Array(out);
}

function parse(stream: Uint8Array, chunkSize: number, type2Len?: number) {
  const frames: GloveFrame[] = [];
  const parser = new GloveParser({ type2Len, onFrame: f => frames.push(f) });
  for (let i = 0; i < stream.length; i += chunkSize) {
    parser.push(stream.slice(i, i + chunkSize));
  }
  return frames;
}

describe("GloveParser", () => {
  it("标记无效四元数，避免把占位姿态写入校准，同时保留传感器数据", () => {
    const stream = makeStream(2, TYPE2_LEN_LEGACY);
    stream.set(f32([0, 0, 0, 0]), 2 * (HEADER_LEN + 2) + 2 * PACKET1_LEN);
    const frames = parse(stream, 4096);
    expect(frames[0].quaternionValid).toBe(false);
    expect(frames[0].sensor_data.every(v => v === 1)).toBe(true);
    expect(frames[1].quaternionValid).toBe(true);
  });
  it("不指定 type2Len 时能解析 144 字节固件（272B 帧）", () => {
    const frames = parse(makeStream(20, TYPE2_LEN_LEGACY), 4096);
    expect(frames).toHaveLength(20);
    expect(frames[0].sensor_data).toHaveLength(256);
    expect(frames[0].sensor_data.every(v => v === 1)).toBe(true);
    expect(frames[0].quaternion).toEqual([0.5, 0.5, 0.5, 0.5]);
    expect(frames[0].acceleration).toBeNull();
    expect(frames[0].attitude).toBeNull();
  });

  it("不指定 type2Len 时能解析 168 字节固件（296B 帧，带加速度和姿态角）", () => {
    const frames = parse(makeStream(20, TYPE2_LEN_WITH_ACC), 4096);
    expect(frames).toHaveLength(20);
    expect(frames[0].quaternion).toEqual([0.5, 0.5, 0.5, 0.5]);
    expect(frames[0].acceleration).toEqual([1, 2, 3]);
    expect(frames[0].attitude).toEqual([4, 5, 6]);
  });

  it("按小块喂入（跨包边界切断）结果一致", () => {
    for (const chunk of [1, 7, 64, 149, 151]) {
      const frames = parse(makeStream(10, TYPE2_LEN_LEGACY), chunk);
      expect(frames, `chunk=${chunk}`).toHaveLength(10);
    }
  });

  it("开头有垃圾字节时能重新对齐", () => {
    const stream = makeStream(5, TYPE2_LEN_LEGACY);
    const dirty = new Uint8Array([0xff, 0x00, 0xaa, 0x55, 0x12, ...stream]);
    expect(parse(dirty, 4096)).toHaveLength(5);
  });

  it("显式写死不匹配的 type2Len 会丢掉几乎所有帧（这就是当初的 bug）", () => {
    // 真实抓包下是一帧都出不来；这里是均匀填充的合成流，错位时可能侥幸凑出个别帧，
    // 所以断言"绝大多数丢失"而不是恰好 0
    const frames = parse(
      makeStream(20, TYPE2_LEN_LEGACY),
      4096,
      TYPE2_LEN_WITH_ACC
    );
    expect(frames.length).toBeLessThan(3);
  });

  it("包2先到、没有对应包1时不会产出帧", () => {
    const orphan = new Uint8Array([
      ...packet(PACKET_TYPE_2, new Uint8Array(TYPE2_LEN_LEGACY)),
      ...HEADER,
    ]);
    expect(parse(orphan, 4096)).toHaveLength(0);
  });

  it("缓冲区不会随数据量无限增长", () => {
    const parser = new GloveParser({ onFrame: () => {} });
    const stream = makeStream(200, TYPE2_LEN_LEGACY);
    for (let i = 0; i < stream.length; i += 512)
      parser.push(stream.slice(i, i + 512));
    // 只应剩下末尾那个还没配上下一个帧头的帧头
    expect(parser["buffer"].length).toBeLessThan(
      HEADER_LEN + 2 + TYPE2_LEN_WITH_ACC
    );
  });
});
