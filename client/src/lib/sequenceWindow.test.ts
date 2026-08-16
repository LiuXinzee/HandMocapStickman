import { describe, it, expect } from "vitest";
import { SequenceWindowBuffer } from "./sequenceWindow";
import { SEQ_SENSOR_N, SEQ_IMU_N } from "./datasetStore";
import type { GloveFrame } from "./gloveProtocol";

function frame(t: number, value: number): GloveFrame {
  return {
    timestamp: t,
    sensor_type: 0x01,
    raw_data: [],
    mapped_data: new Array(SEQ_SENSOR_N).fill(value),
    quaternion: [1, 0, 0, 0],
    acceleration: [0, 0, 1],
    attitude: [0, 0, 0],
    handLabel: "Left",
  } as unknown as GloveFrame;
}

/** 以 100Hz 往缓冲里灌 ms 毫秒的数据，第 i 帧的传感值为 i */
function fill(buf: SequenceWindowBuffer, ms: number, startT = 1000): number {
  const step = 10;
  let t = startT;
  let i = 0;
  for (; t <= startT + ms; t += step, i++) {
    buf.push("left", frame(t, i % 256));
  }
  return t - step;
}

describe("SequenceWindowBuffer", () => {
  it("空缓冲不产生快照", () => {
    const buf = new SequenceWindowBuffer();
    expect(buf.latestTime()).toBeNull();
    expect(buf.snapshot(1500)).toBeNull();
  });

  it("数据不足一个完整窗口时拒绝出快照", () => {
    // 半个窗口若强行推理，前半段会被最近邻拉成直线 —— 一个假的"静止"动作
    const buf = new SequenceWindowBuffer();
    fill(buf, 700);
    expect(buf.snapshot(1500)).toBeNull();
  });

  it("攒够窗口后产出定长快照", () => {
    const buf = new SequenceWindowBuffer();
    fill(buf, 2000);
    const snap = buf.snapshot(1500)!;
    expect(snap).not.toBeNull();
    expect(snap.frameCount).toBe(75); // 1500ms @ 50Hz
    expect(snap.sourceFps).toBe(50);
    expect(snap.durationMs).toBe(1500);
    expect(snap.leftSensor!.length).toBe(75 * SEQ_SENSOR_N);
    expect(snap.leftImu!.length).toBe(75 * SEQ_IMU_N);
    expect(snap.timestamps[0]).toBe(0);
    expect(snap.timestamps[74]).toBeCloseTo(74 * 20, 4);
  });

  it("快照取的是最近的窗口而不是最早的数据", () => {
    const buf = new SequenceWindowBuffer();
    const last = fill(buf, 2500);
    const snap = buf.snapshot(1500)!;
    expect(buf.latestTime()).toBe(last);
    // 栅格覆盖 [end-1500, end-20]：最后一个栅格点在 (T-1)*20ms 处，
    // 即比最新帧早一个栅格间隔（20ms 延迟，可忽略）
    const lastIdx = (snap.frameCount - 1) * SEQ_SENSOR_N;
    const expected = ((last - 20 - 1000) / 10) % 256;
    expect(snap.leftSensor![lastIdx]).toBe(expected);
    // 窗口第一帧对应 1500ms 之前，明显不是最早灌进去的 0
    expect(snap.leftSensor![0]).toBeGreaterThan(0);
  });

  it("只有一只手时另一只手的列为 null（而不是全 0 的假数据）", () => {
    const buf = new SequenceWindowBuffer();
    fill(buf, 2000);
    const snap = buf.snapshot(1500)!;
    expect(snap.rightSensor).toBeNull();
    expect(snap.rightImu).toBeNull();
    expect(snap.leftLandmarks).toBeNull(); // 部署走纯触觉，不带视觉
  });

  it("缓冲长度受 bufferMs 约束，长时间运行不会无限增长", () => {
    const buf = new SequenceWindowBuffer({ bufferMs: 1000 });
    fill(buf, 30000); // 30 秒 @100Hz = 3000 帧
    const snap = buf.snapshot(900)!;
    expect(snap).not.toBeNull();
    // 缓冲里最多应该只剩约 1 秒的数据；用能否覆盖 2 秒窗口来间接验证
    expect(buf.snapshot(2000)).toBeNull();
  });

  it("clear 之后回到空状态", () => {
    const buf = new SequenceWindowBuffer();
    fill(buf, 2000);
    buf.clear();
    expect(buf.latestTime()).toBeNull();
    expect(buf.snapshot(1500)).toBeNull();
  });

  it("双手独立时间戳都能落到同一栅格上", () => {
    const buf = new SequenceWindowBuffer();
    // 两只手速率和相位都不同 —— 真实硬件就是两个独立 COM 口
    for (let t = 1000; t <= 3000; t += 10) buf.push("left", frame(t, 7));
    for (let t = 1003; t <= 3000; t += 13) buf.push("right", frame(t, 9));
    const snap = buf.snapshot(1500)!;
    expect(snap.leftSensor).not.toBeNull();
    expect(snap.rightSensor).not.toBeNull();
    expect(snap.leftSensor!.length).toBe(snap.rightSensor!.length);
    expect(snap.leftSensor![0]).toBe(7);
    expect(snap.rightSensor![0]).toBe(9);
  });
});
