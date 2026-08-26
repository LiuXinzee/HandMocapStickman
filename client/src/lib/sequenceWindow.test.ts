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

  describe("snapshotAll（整句捕获用）", () => {
    it("取缓冲里现有的全部，不要求攒满某个窗口", () => {
      // snapshot(windowMs) 要求 buf[0].t <= end-windowMs，整句捕获满足不了这个判据
      // （按下按钮后第一帧的时间戳必然晚于按下的时刻），所以必须有这条路
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const last = fill(buf, 700);
      const snap = buf.snapshotAll()!;
      expect(snap).not.toBeNull();
      expect(snap.frameCount).toBe(35); // 700ms @50Hz
      expect(buf.spanMs()).toBe(last - 1000);
    });

    it("maxMs 从尾部往前算（超长时丢的是最老的）", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      fill(buf, 5000);
      const snap = buf.snapshotAll(2000)!;
      expect(snap.frameCount).toBe(100);
      // 起点在 end-2000 = 4000ms 处，不是缓冲开头（值 = (t-1000)/10）
      expect(snap.leftSensor![0]).toBe(((4000 - 1000) / 10) % 256);
    });

    it("dropTailMs 砍的是尾巴，句子开头必须留着", () => {
      // 这是收句时的实际用法：整段 3000ms，末尾 800ms 是收句判据用掉的静止。
      // 砍错方向（靠调小 maxMs）会把开头那几个词丢掉，而尾部静止照样留着 ——
      // 表现是"每句话前面都少几个词"，非常难查
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      fill(buf, 3000);
      const snap = buf.snapshotAll(2200, "_sentence", 800)!;
      expect(snap.frameCount).toBe(110); // 2200ms @50Hz
      // 开头仍是缓冲的第一帧（值 0），不是被砍掉之后的某一帧
      expect(snap.leftSensor![0]).toBe(0);
      // 末尾停在 end-800 附近（值 = (t-1000)/10），而不是 end=4000 处的 300
      const lastIdx = (snap.frameCount - 1) * SEQ_SENSOR_N;
      expect(snap.leftSensor![lastIdx]).toBe(((3180 - 1000) / 10) % 256); // 218
      expect(snap.leftSensor![lastIdx]).toBeLessThan((3200 - 1000) / 10);
      expect(snap.leftSensor![lastIdx]).toBeGreaterThan((2000 - 1000) / 10);
    });

    it("尾巴砍过头（比整段还长）返回 null 而不是负长度的段", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      fill(buf, 500);
      expect(buf.snapshotAll(500, "_sentence", 800)).toBeNull();
    });

    it("起点取两手首帧的较晚者（较早者会把另一只手开头填成假静止）", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      for (let t = 1000; t <= 4000; t += 10) buf.push("left", frame(t, 7));
      for (let t = 2000; t <= 4000; t += 10) buf.push("right", frame(t, 9));
      const snap = buf.snapshotAll()!;
      // 起点 = 2000（右手首帧），所以整段 2000ms 而不是 3000ms
      expect(snap.frameCount).toBe(100);
      expect(buf.spanMs()).toBe(2000);
    });

    it("空缓冲返回 null", () => {
      expect(new SequenceWindowBuffer().snapshotAll()).toBeNull();
      expect(new SequenceWindowBuffer().spanMs()).toBeNull();
    });
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
