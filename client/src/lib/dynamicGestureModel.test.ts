import * as tf from "@tensorflow/tfjs";
import { afterEach, describe, expect, it } from "vitest";
import type { DynamicGestureSequence } from "./datasetStore";
import {
  DYNAMIC_FEATURE_DIM,
  DYNAMIC_HAND_FEATURE_DIM,
  DYNAMIC_SEQUENCE_LENGTH,
  DYNAMIC_TCN_DILATIONS,
  buildDynamicTcnModel,
  loadDynamicGestureModelFromSaved,
  normalizeDynamicQuaternion,
  prepareDynamicHandFrames,
  preprocessDynamicGestureSequence,
  resampleDynamicHandFrames,
  serializeDynamicGestureModel,
  splitDynamicGestureSequences,
  trainDynamicGestureModel,
  type DynamicGestureFrameInput,
} from "./dynamicGestureModel";

function frame(
  relativeTimeMs: number,
  sensorValue: number,
  quaternion: readonly number[] = [1, 0, 0, 0]
): DynamicGestureFrameInput {
  return {
    relativeTimeMs,
    sensorData: new Array(137).fill(sensorValue),
    quaternion,
  };
}

function motionFrames(startValue: number, endValue: number) {
  return Array.from({ length: DYNAMIC_SEQUENCE_LENGTH }, (_, index) =>
    frame(
      index * 20,
      startValue +
        ((endValue - startValue) * index) / (DYNAMIC_SEQUENCE_LENGTH - 1)
    )
  );
}

function sequence(
  label: string,
  sessionId: string,
  leftFrames: DynamicGestureFrameInput[] = [frame(0, 0)],
  rightFrames: DynamicGestureFrameInput[] = []
): DynamicGestureSequence {
  return {
    schemaVersion: 1,
    label,
    sessionId,
    startedAt: 1,
    durationMs: 2560,
    targetDurationMs: 2560,
    leftFrames,
    rightFrames,
    visionFrames: [],
  } as DynamicGestureSequence;
}

afterEach(() => {
  tf.disposeVariables();
});

describe("dynamic gesture temporal preprocessing", () => {
  it("normalizes quaternions and keeps consecutive samples in one hemisphere", () => {
    expect(normalizeDynamicQuaternion([2, 0, 0, 0])).toEqual([1, 0, 0, 0]);
    const prepared = prepareDynamicHandFrames([
      frame(0, 0, [2, 0, 0, 0]),
      frame(20, 0, [-2, 0, 0, 0]),
    ]);
    expect(prepared[0].quaternion).toEqual([1, 0, 0, 0]);
    expect(prepared[1].quaternion).toEqual([1, 0, 0, 0]);
  });

  it("uses the final frame when relative timestamps are duplicated", () => {
    const prepared = prepareDynamicHandFrames([
      frame(20, 10),
      frame(0, 0),
      frame(20, 200),
    ]);
    expect(prepared).toHaveLength(2);
    expect(prepared[1].sensors[0]).toBeCloseTo(200 / 255, 6);
  });

  it("interpolates raw streams on relative time rather than array position", () => {
    const sampled = resampleDynamicHandFrames(
      [frame(0, 0), frame(100, 100), frame(50, 200)],
      { sequenceLength: 3, sampleRateHz: 20 }
    );
    expect(sampled[0][0]).toBe(0);
    expect(sampled[1][0]).toBeCloseTo(200 / 255, 6);
    expect(sampled[2][0]).toBeCloseTo(100 / 255, 6);
    expect(sampled.every(row => row[DYNAMIC_HAND_FEATURE_DIM - 1] === 1)).toBe(
      true
    );
  });

  it("marks out-of-range samples and long local gaps as missing", () => {
    const sampled = resampleDynamicHandFrames(
      [frame(100, 10), frame(500, 20)],
      { sequenceLength: 4, sampleRateHz: 10 }
    );
    expect(sampled[0][DYNAMIC_HAND_FEATURE_DIM - 1]).toBe(0);
    expect(sampled[1][DYNAMIC_HAND_FEATURE_DIM - 1]).toBe(1);
    expect(sampled[2][DYNAMIC_HAND_FEATURE_DIM - 1]).toBe(0);
    expect(sampled[3][DYNAMIC_HAND_FEATURE_DIM - 1]).toBe(0);
  });

  it("keeps the first real frame after a long local gap", () => {
    const sampled = resampleDynamicHandFrames([frame(0, 10), frame(400, 200)], {
      sequenceLength: 5,
      sampleRateHz: 10,
    });
    expect(sampled.map(row => row[DYNAMIC_HAND_FEATURE_DIM - 1])).toEqual([
      1, 0, 0, 0, 1,
    ]);
    expect(sampled[4][0]).toBeCloseTo(200 / 255, 6);
  });

  it("produces 128x284 and marks a missing hand with zero presence", () => {
    const features = preprocessDynamicGestureSequence(
      sequence("hello", "session-a")
    );
    expect(features).toHaveLength(DYNAMIC_SEQUENCE_LENGTH);
    expect(features[0]).toHaveLength(DYNAMIC_FEATURE_DIM);
    expect(features[0][DYNAMIC_HAND_FEATURE_DIM - 1]).toBe(1);
    expect(features[0][DYNAMIC_FEATURE_DIM - 1]).toBe(0);
    expect(features[0].slice(DYNAMIC_HAND_FEATURE_DIM)).toEqual(
      new Array(DYNAMIC_HAND_FEATURE_DIM).fill(0)
    );
  });

  it("normalizes a variable-duration clip onto the fixed TCN window", () => {
    const sourceFrames = Array.from({ length: 53 }, (_, index) => {
      const relativeTimeMs = Math.min(index * 100, 5120);
      return frame(relativeTimeMs, (relativeTimeMs / 5120) * 255);
    });
    const sample = sequence("hello", "session-a", sourceFrames);
    sample.durationMs = 5120;
    sample.targetDurationMs = 5120;

    const features = preprocessDynamicGestureSequence(sample);
    expect(features[64][0]).toBeCloseTo(0.5, 1);
    expect(features[127][0]).toBeGreaterThan(0.98);
  });
});

describe("dynamic dataset splitting", () => {
  it("never places the same session in both train and validation", () => {
    const samples = [
      sequence("a", "s1"),
      sequence("b", "s1"),
      sequence("a", "s2"),
      sequence("b", "s2"),
      sequence("a", "s3"),
      sequence("b", "s3"),
    ];
    const split = splitDynamicGestureSequences(samples, 0.34, 7);
    const trainSessions = new Set(split.train.map(item => item.sessionId));
    const validationSessions = new Set(
      split.validation.map(item => item.sessionId)
    );
    expect(split.validation.length).toBeGreaterThan(0);
    for (const id of validationSessions)
      expect(trainSessions.has(id)).toBe(false);
  });

  it("does not leak clips from a single session into validation", () => {
    const samples = [
      sequence("a", "only-session"),
      sequence("b", "only-session"),
      sequence("a", "only-session"),
      sequence("b", "only-session"),
    ];
    const split = splitDynamicGestureSequences(samples, 0.25, 7);
    expect(split.train).toHaveLength(samples.length);
    expect(split.validation).toHaveLength(0);
  });
});

describe("dynamic TCN", () => {
  it("has the fixed temporal input and requested residual dilations", () => {
    const model = buildDynamicTcnModel(3, { dropoutRate: 0 });
    expect(model.inputs[0].shape).toEqual([
      null,
      DYNAMIC_SEQUENCE_LENGTH,
      DYNAMIC_FEATURE_DIM,
    ]);
    expect(model.outputs[0].shape).toEqual([null, 3]);
    DYNAMIC_TCN_DILATIONS.forEach((dilationRate, index) => {
      expect(
        model.getLayer(`tcn_block_${index + 1}_conv_1_context_d${dilationRate}`)
      ).toBeTruthy();
    });
    model.dispose();
  });

  it("backpropagates through every dilated residual block", async () => {
    const samples = [
      sequence("__idle__", "s1", motionFrames(0, 2)),
      sequence("hello", "s1", motionFrames(80, 180)),
    ];
    expect(
      preprocessDynamicGestureSequence(samples[1]).every(
        row => row[DYNAMIC_HAND_FEATURE_DIM - 1] === 1
      )
    ).toBe(true);
    const result = await trainDynamicGestureModel(samples, {
      epochs: 1,
      batchSize: 2,
      validationSplit: 0,
      augmentNoise: 0,
      dropoutRate: 0,
    });

    expect(result.history).toHaveLength(1);
    expect(Number.isFinite(result.history[0].loss)).toBe(true);
    result.model.dispose();
  }, 30_000);

  it("serializes topology, ArrayBuffer weights, and fixed input metadata", async () => {
    const model = buildDynamicTcnModel(2, { dropoutRate: 0 });
    const saved = await serializeDynamicGestureModel(
      model,
      ["a", "b"],
      0.75,
      "test_tcn"
    );
    expect(saved.modelType).toBe("tcn");
    expect(saved.weightsData).toBeInstanceOf(ArrayBuffer);
    expect(saved.sequenceLength).toBe(DYNAMIC_SEQUENCE_LENGTH);
    expect(saved.featureDim).toBe(DYNAMIC_FEATURE_DIM);
    expect(JSON.parse(saved.modelJson).modelTopology).toBeTruthy();
    const loaded = await loadDynamicGestureModelFromSaved(saved, {
      activate: false,
    });
    const input = tf.zeros([1, DYNAMIC_SEQUENCE_LENGTH, DYNAMIC_FEATURE_DIM]);
    const prediction = loaded.predict(input) as tf.Tensor;
    const probabilities = Array.from(await prediction.data());
    expect(prediction.shape).toEqual([1, 2]);
    expect(probabilities.every(Number.isFinite)).toBe(true);
    expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(
      1,
      5
    );
    prediction.dispose();
    input.dispose();
    loaded.dispose();
    model.dispose();
  });
});
