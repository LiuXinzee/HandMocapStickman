/*
 * 序列数据集二进制导出/导入的 round-trip 校验。
 *
 * 这条链路一旦错位，Python 侧读到的是"看起来合理但内容错乱"的张量 ——
 * 不会报错，只会训出一个莫名其妙的模型。所以要逐字节比。
 */
import { describe, it, expect } from "vitest";
import {
  encodeSequencesBinary,
  decodeSequencesBinary,
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "./datasetStore";

function makeSeq(
  label: string,
  T: number,
  opts: {
    withRight?: boolean;
    withVision?: boolean;
    origin?: "recorded" | "synthesized";
    seed?: number;
  } = {}
): SequenceSample {
  const { withRight = true, withVision = true, origin = "recorded" } = opts;
  let s = (opts.seed ?? 1) >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };

  const timestamps = new Float32Array(T);
  for (let t = 0; t < T; t++) timestamps[t] = t * 20;

  const sensor = () => {
    const a = new Uint8Array(T * SEQ_SENSOR_N);
    for (let i = 0; i < a.length; i++) a[i] = Math.floor(rnd() * 256);
    return a;
  };
  const imu = () => {
    const a = new Float32Array(T * SEQ_IMU_N);
    for (let i = 0; i < a.length; i++) a[i] = rnd() * 2 - 1;
    return a;
  };
  const lm = () => {
    const a = new Float32Array(T * SEQ_LANDMARK_N);
    for (let i = 0; i < a.length; i++) a[i] = rnd();
    return a;
  };

  return {
    segments: [{ label, startFrame: 0, endFrame: T }],
    primaryLabel: label,
    frameCount: T,
    timestamps,
    leftSensor: sensor(),
    rightSensor: withRight ? sensor() : null,
    leftImu: imu(),
    rightImu: withRight ? imu() : null,
    leftLandmarks: withVision ? lm() : null,
    rightLandmarks: withVision && withRight ? lm() : null,
    durationMs: T * 20,
    sourceFps: 50,
    origin,
    timestamp: 1700000000000,
  };
}

function expectSameArray(
  a: Uint8Array | Float32Array | null,
  b: Uint8Array | Float32Array | null
) {
  if (a === null || b === null) {
    expect(a).toBeNull();
    expect(b).toBeNull();
    return;
  }
  expect(b.length).toBe(a.length);
  expect(b.constructor.name).toBe(a.constructor.name);
  for (let i = 0; i < a.length; i++) {
    expect(b[i]).toBe(a[i]);
  }
}

describe("序列二进制导出 round-trip", () => {
  const seqs = [
    makeSeq("hello", 40, { seed: 1 }),
    makeSeq("_idle", 33, { withVision: false, seed: 2 }), // 无视觉
    makeSeq("go", 17, { withRight: false, seed: 3 }), // 单手
    makeSeq("num_3", 25, {
      withRight: false,
      withVision: false,
      origin: "synthesized",
      seed: 4,
    }),
  ];

  const { bin, manifest } = encodeSequencesBinary(seqs);
  const back = decodeSequencesBinary(bin, manifest);

  it("manifest 元信息正确", () => {
    expect(manifest.version).toBe("seq-1.0");
    expect(manifest.totalSequences).toBe(seqs.length);
    expect(manifest.sensorN).toBe(SEQ_SENSOR_N);
    expect(manifest.imuN).toBe(SEQ_IMU_N);
    expect(manifest.landmarkN).toBe(SEQ_LANDMARK_N);
    expect(manifest.labels).toEqual(["_idle", "go", "hello", "num_3"]);
    expect(manifest.sequences).toHaveLength(seqs.length);
  });

  it("所有 float32 段按 4 字节对齐（否则 numpy frombuffer 会炸）", () => {
    for (const e of manifest.sequences) {
      for (const [key, ref] of Object.entries(e.arrays)) {
        if (ref && ref.dtype === "float32") {
          expect(ref.offset % 4, `${key} 未对齐`).toBe(0);
        }
      }
    }
  });

  it("各段区间互不重叠且都落在 bin 内", () => {
    const spans: Array<[number, number]> = [];
    for (const e of manifest.sequences) {
      for (const ref of Object.values(e.arrays)) {
        if (!ref) continue;
        const bytes = ref.length * (ref.dtype === "uint8" ? 1 : 4);
        expect(ref.offset + bytes).toBeLessThanOrEqual(bin.byteLength);
        spans.push([ref.offset, ref.offset + bytes]);
      }
    }
    spans.sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < spans.length; i++) {
      expect(spans[i][0]).toBeGreaterThanOrEqual(spans[i - 1][1]);
    }
  });

  it("解码后逐字节等于原始数据", () => {
    expect(back).toHaveLength(seqs.length);
    for (let i = 0; i < seqs.length; i++) {
      const a = seqs[i];
      const b = back[i];
      expect(b.primaryLabel).toBe(a.primaryLabel);
      expect(b.frameCount).toBe(a.frameCount);
      expect(b.durationMs).toBe(a.durationMs);
      expect(b.sourceFps).toBe(a.sourceFps);
      expect(b.origin).toBe(a.origin);
      expect(b.timestamp).toBe(a.timestamp);
      expect(b.segments).toEqual(a.segments);
      expectSameArray(a.timestamps, b.timestamps);
      expectSameArray(a.leftSensor, b.leftSensor);
      expectSameArray(a.rightSensor, b.rightSensor);
      expectSameArray(a.leftImu, b.leftImu);
      expectSameArray(a.rightImu, b.rightImu);
      expectSameArray(a.leftLandmarks, b.leftLandmarks);
      expectSameArray(a.rightLandmarks, b.rightLandmarks);
    }
  });

  it("缺失的手/视觉在往返后仍是 null，而不是被补成全 0", () => {
    expect(back[1].leftLandmarks).toBeNull();
    expect(back[2].rightSensor).toBeNull();
    expect(back[2].rightImu).toBeNull();
    expect(back[3].rightLandmarks).toBeNull();
  });

  it("空数据集导出成 0 字节且不抛异常", () => {
    const empty = encodeSequencesBinary([]);
    expect(empty.bin.byteLength).toBe(0);
    expect(empty.manifest.totalSequences).toBe(0);
    expect(decodeSequencesBinary(empty.bin, empty.manifest)).toEqual([]);
  });
});
