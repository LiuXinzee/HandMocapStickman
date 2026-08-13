import * as tf from "@tensorflow/tfjs";
import type {
  DynamicGestureSequence,
  SavedDynamicModelRecord,
} from "./datasetStore";

export const DYNAMIC_SAMPLE_RATE_HZ = 50;
export const DYNAMIC_SEQUENCE_LENGTH = 128;
export const DYNAMIC_SENSOR_COUNT = 137;
export const DYNAMIC_HAND_FEATURE_DIM = DYNAMIC_SENSOR_COUNT + 4 + 1;
export const DYNAMIC_FEATURE_DIM = DYNAMIC_HAND_FEATURE_DIM * 2;
export const DYNAMIC_WINDOW_DURATION_MS =
  (DYNAMIC_SEQUENCE_LENGTH / DYNAMIC_SAMPLE_RATE_HZ) * 1000;

const TCN_FILTERS = 64;
export const DYNAMIC_TCN_DILATIONS = [1, 2, 4, 8, 16] as const;
const EPSILON = 1e-8;
const MAX_DYNAMIC_FRAME_GAP_MS = 200;

export interface DynamicGestureHandInput {
  sensor_data?: readonly number[];
  sensorData?: readonly number[];
  quaternion: readonly number[];
}

export interface DynamicGestureFrameInput extends DynamicGestureHandInput {
  relativeTimeMs: number;
}

export interface DynamicGestureSequenceInput {
  label?: string;
  sessionId?: string;
  durationMs?: number;
  targetDurationMs?: number;
  leftFrames: readonly DynamicGestureFrameInput[];
  rightFrames: readonly DynamicGestureFrameInput[];
}

export interface DynamicResampleOptions {
  sequenceLength: number;
  sampleRateHz: number;
}

export interface DynamicGestureTrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  augmentNoise: number;
  dropoutRate: number;
  randomSeed: number;
}

export type ResolvedDynamicGestureTrainingConfig =
  Readonly<DynamicGestureTrainingConfig>;

export interface DynamicGestureTrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

/** Short alias used by the dynamic training page. */
export type DynamicTrainingProgress = DynamicGestureTrainingProgress;

export interface DynamicGesturePrediction {
  label: string;
  confidence: number;
  allProbabilities: Array<{ label: string; probability: number }>;
}

export interface DynamicGestureTrainingResult {
  model: tf.LayersModel;
  labels: string[];
  accuracy: number;
  validationAccuracy: number;
  validationLoss: number;
  trainingSequenceCount: number;
  validationSequenceCount: number;
  history: DynamicGestureTrainingProgress[];
  config: ResolvedDynamicGestureTrainingConfig;
}

export interface DynamicGestureDatasetSplit {
  train: DynamicGestureSequence[];
  validation: DynamicGestureSequence[];
}

export interface DynamicGestureModelMetadata {
  sampleRateHz: number;
  sequenceLength: number;
  featureDim: number;
}

export const DEFAULT_DYNAMIC_GESTURE_TRAINING_CONFIG: ResolvedDynamicGestureTrainingConfig =
  Object.freeze({
    epochs: 60,
    batchSize: 8,
    learningRate: 0.001,
    validationSplit: 0.2,
    augmentNoise: 0.01,
    dropoutRate: 0.15,
    randomSeed: 42,
  });

const DEFAULT_MODEL_METADATA: DynamicGestureModelMetadata = Object.freeze({
  sampleRateHz: DYNAMIC_SAMPLE_RATE_HZ,
  sequenceLength: DYNAMIC_SEQUENCE_LENGTH,
  featureDim: DYNAMIC_FEATURE_DIM,
});

interface PreparedFrame {
  relativeTimeMs: number;
  sensors: number[];
  quaternion: [number, number, number, number];
}

interface SavedDynamicModelView {
  name: string;
  labels: string[];
  modelJson: string;
  weightsData: ArrayBuffer;
  sampleRateHz?: number;
  sequenceLength?: number;
  featureDim?: number;
}

function finiteOrZero(value: number | undefined): number {
  return Number.isFinite(value) ? (value as number) : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Normalize a quaternion, using identity for invalid or zero-length input. */
export function normalizeDynamicQuaternion(
  input: readonly number[]
): [number, number, number, number] {
  const q: [number, number, number, number] = [
    finiteOrZero(input[0]),
    finiteOrZero(input[1]),
    finiteOrZero(input[2]),
    finiteOrZero(input[3]),
  ];
  const magnitude = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(magnitude) || magnitude < EPSILON) {
    return [1, 0, 0, 0];
  }
  return [
    q[0] / magnitude,
    q[1] / magnitude,
    q[2] / magnitude,
    q[3] / magnitude,
  ];
}

function quaternionDot(
  first: readonly number[],
  second: readonly number[]
): number {
  return (
    first[0] * second[0] +
    first[1] * second[1] +
    first[2] * second[2] +
    first[3] * second[3]
  );
}

function negateQuaternion(
  quaternion: readonly number[]
): [number, number, number, number] {
  const negate = (value: number) => {
    const result = -value;
    return result === 0 ? 0 : result;
  };
  return quaternion.map(negate) as [number, number, number, number];
}

function sensorsFromFrame(frame: DynamicGestureHandInput): readonly number[] {
  return frame.sensorData ?? frame.sensor_data ?? [];
}

/**
 * Sort a raw hand stream and collapse duplicate relative timestamps. Map#set
 * deliberately keeps the last frame captured at each timestamp.
 */
export function prepareDynamicHandFrames(
  frames: readonly DynamicGestureFrameInput[]
): PreparedFrame[] {
  const byTime = new Map<number, DynamicGestureFrameInput>();
  for (const frame of frames) {
    if (!Number.isFinite(frame.relativeTimeMs)) continue;
    byTime.set(frame.relativeTimeMs, frame);
  }

  const ordered = Array.from(byTime.values()).sort(
    (first, second) => first.relativeTimeMs - second.relativeTimeMs
  );
  let previousQuaternion: [number, number, number, number] | null = null;

  return ordered.map(frame => {
    const sourceSensors = sensorsFromFrame(frame);
    const sensors = Array.from(
      { length: DYNAMIC_SENSOR_COUNT },
      (_, index) => clamp(finiteOrZero(sourceSensors[index]), 0, 255) / 255
    );
    let quaternion = normalizeDynamicQuaternion(frame.quaternion);
    if (
      previousQuaternion &&
      quaternionDot(previousQuaternion, quaternion) < 0
    ) {
      quaternion = negateQuaternion(quaternion);
    }
    previousQuaternion = quaternion;
    return { relativeTimeMs: frame.relativeTimeMs, sensors, quaternion };
  });
}

function interpolatePreparedFrame(
  first: PreparedFrame,
  second: PreparedFrame,
  ratio: number
): number[] {
  const amount = clamp(ratio, 0, 1);
  const sensors = first.sensors.map(
    (value, index) => value + (second.sensors[index] - value) * amount
  );
  const quaternion = normalizeDynamicQuaternion(
    first.quaternion.map(
      (value, index) => value + (second.quaternion[index] - value) * amount
    )
  );
  return [...sensors, ...quaternion, 1];
}

function encodePreparedFrame(frame: PreparedFrame): number[] {
  return [...frame.sensors, ...frame.quaternion, 1];
}

/** Resample one asynchronous glove stream onto a fixed monotonic time grid. */
export function resampleDynamicHandFrames(
  frames: readonly DynamicGestureFrameInput[],
  options: DynamicResampleOptions = {
    sequenceLength: DYNAMIC_SEQUENCE_LENGTH,
    sampleRateHz: DYNAMIC_SAMPLE_RATE_HZ,
  }
): number[][] {
  if (
    !Number.isInteger(options.sequenceLength) ||
    options.sequenceLength <= 0
  ) {
    throw new RangeError("sequenceLength must be a positive integer");
  }
  if (!Number.isFinite(options.sampleRateHz) || options.sampleRateHz <= 0) {
    throw new RangeError("sampleRateHz must be positive");
  }

  const prepared = prepareDynamicHandFrames(frames);
  const missingFrame = () => new Array(DYNAMIC_HAND_FEATURE_DIM).fill(0);
  if (prepared.length === 0) {
    return Array.from({ length: options.sequenceLength }, missingFrame);
  }

  const stepMs = 1000 / options.sampleRateHz;
  const result: number[][] = [];
  let upperIndex = 1;

  for (let index = 0; index < options.sequenceLength; index++) {
    const targetTime = index * stepMs;
    if (targetTime < prepared[0].relativeTimeMs) {
      result.push(missingFrame());
      continue;
    }
    if (targetTime === prepared[0].relativeTimeMs) {
      result.push(encodePreparedFrame(prepared[0]));
      continue;
    }
    if (targetTime > prepared[prepared.length - 1].relativeTimeMs) {
      result.push(missingFrame());
      continue;
    }
    if (targetTime === prepared[prepared.length - 1].relativeTimeMs) {
      result.push(encodePreparedFrame(prepared[prepared.length - 1]));
      continue;
    }

    while (
      upperIndex < prepared.length - 1 &&
      prepared[upperIndex].relativeTimeMs < targetTime
    ) {
      upperIndex++;
    }
    const lower = prepared[upperIndex - 1];
    const upper = prepared[upperIndex];
    if (targetTime === lower.relativeTimeMs) {
      result.push(encodePreparedFrame(lower));
      continue;
    }
    if (targetTime === upper.relativeTimeMs) {
      result.push(encodePreparedFrame(upper));
      continue;
    }
    const interval = upper.relativeTimeMs - lower.relativeTimeMs;
    if (interval > MAX_DYNAMIC_FRAME_GAP_MS) {
      result.push(missingFrame());
      continue;
    }
    const ratio =
      interval > 0 ? (targetTime - lower.relativeTimeMs) / interval : 1;
    result.push(interpolatePreparedFrame(lower, upper, ratio));
  }

  return result;
}

/** Build the canonical [128, 284] TCN input from independent glove streams. */
export function preprocessDynamicGestureSequence(
  sequence: DynamicGestureSequenceInput
): number[][] {
  const options: DynamicResampleOptions = {
    sequenceLength: DYNAMIC_SEQUENCE_LENGTH,
    sampleRateHz: DYNAMIC_SAMPLE_RATE_HZ,
  };
  const declaredDuration = sequence.durationMs ?? sequence.targetDurationMs;
  const observedDuration = Math.max(
    0,
    ...sequence.leftFrames.map(frame => frame.relativeTimeMs),
    ...sequence.rightFrames.map(frame => frame.relativeTimeMs)
  );
  const sourceDuration =
    Number.isFinite(declaredDuration) && (declaredDuration ?? 0) > 0
      ? Math.max(declaredDuration as number, observedDuration)
      : null;
  const scale = sourceDuration
    ? DYNAMIC_WINDOW_DURATION_MS / sourceDuration
    : 1;
  const scaleFrames = (frames: readonly DynamicGestureFrameInput[]) =>
    scale === 1
      ? frames
      : frames.map(frame => ({
          ...frame,
          relativeTimeMs: frame.relativeTimeMs * scale,
        }));
  const left = resampleDynamicHandFrames(
    scaleFrames(sequence.leftFrames),
    options
  );
  const right = resampleDynamicHandFrames(
    scaleFrames(sequence.rightFrames),
    options
  );
  return left.map((features, index) => [...features, ...right[index]]);
}

function applyLayer(
  layer: tf.layers.Layer,
  input: tf.SymbolicTensor | tf.SymbolicTensor[]
): tf.SymbolicTensor {
  return layer.apply(input) as tf.SymbolicTensor;
}

function shiftTemporalContext(
  input: tf.SymbolicTensor,
  dilationRate: number,
  direction: "past" | "future",
  name: string
): tf.SymbolicTensor {
  const crop =
    direction === "past"
      ? ([
          [0, dilationRate],
          [0, 0],
        ] as [[number, number], [number, number]])
      : ([
          [dilationRate, 0],
          [0, 0],
        ] as [[number, number], [number, number]]);
  const padding =
    direction === "past"
      ? ([
          [dilationRate, 0],
          [0, 0],
        ] as [[number, number], [number, number]])
      : ([
          [0, dilationRate],
          [0, 0],
        ] as [[number, number], [number, number]]);
  const cropped = applyLayer(
    tf.layers.cropping2D({ name: `${name}_crop`, cropping: crop }),
    input
  );
  return applyLayer(
    tf.layers.zeroPadding2d({ name: `${name}_pad`, padding }),
    cropped
  );
}

/**
 * A kernel-3 dilated convolution expressed as fixed temporal shifts followed
 * by a trainable 1x1 projection. TF.js cannot backpropagate through native
 * Conv1D dilation > 1, while this equivalent graph is fully trainable.
 */
function dilatedTemporalConv(
  input: tf.SymbolicTensor,
  dilationRate: number,
  name: string
): tf.SymbolicTensor {
  const past = shiftTemporalContext(
    input,
    dilationRate,
    "past",
    `${name}_past_d${dilationRate}`
  );
  const future = shiftTemporalContext(
    input,
    dilationRate,
    "future",
    `${name}_future_d${dilationRate}`
  );
  const context = applyLayer(
    tf.layers.concatenate({
      name: `${name}_context_d${dilationRate}`,
      axis: -1,
    }),
    [past, input, future]
  );
  return applyLayer(
    tf.layers.conv2d({
      name,
      filters: TCN_FILTERS,
      kernelSize: [1, 1],
      strides: [1, 1],
      padding: "same",
      useBias: false,
      kernelInitializer: "heNormal",
    }),
    context
  );
}

function residualTcnBlock(
  input: tf.SymbolicTensor,
  dilationRate: number,
  dropoutRate: number,
  blockIndex: number
): tf.SymbolicTensor {
  const prefix = `tcn_block_${blockIndex}`;
  let branch = dilatedTemporalConv(input, dilationRate, `${prefix}_conv_1`);
  branch = applyLayer(
    tf.layers.batchNormalization({ name: `${prefix}_bn_1` }),
    branch
  );
  branch = applyLayer(
    tf.layers.activation({ name: `${prefix}_relu_1`, activation: "relu" }),
    branch
  );
  branch = applyLayer(
    tf.layers.dropout({ name: `${prefix}_dropout`, rate: dropoutRate }),
    branch
  );
  branch = dilatedTemporalConv(branch, dilationRate, `${prefix}_conv_2`);
  branch = applyLayer(
    tf.layers.batchNormalization({ name: `${prefix}_bn_2` }),
    branch
  );

  let shortcut = input;
  if (input.shape[input.shape.length - 1] !== TCN_FILTERS) {
    shortcut = applyLayer(
      tf.layers.conv2d({
        name: `${prefix}_shortcut`,
        filters: TCN_FILTERS,
        kernelSize: [1, 1],
        padding: "same",
        useBias: false,
      }),
      input
    );
  }

  const added = applyLayer(tf.layers.add({ name: `${prefix}_residual` }), [
    shortcut,
    branch,
  ]);
  return applyLayer(
    tf.layers.activation({ name: `${prefix}_out`, activation: "relu" }),
    added
  );
}

export function resolveDynamicGestureTrainingConfig(
  config: Partial<DynamicGestureTrainingConfig> = {}
): ResolvedDynamicGestureTrainingConfig {
  const resolved = { ...DEFAULT_DYNAMIC_GESTURE_TRAINING_CONFIG, ...config };
  if (!Number.isInteger(resolved.epochs) || resolved.epochs <= 0) {
    throw new RangeError("epochs must be a positive integer");
  }
  if (!Number.isInteger(resolved.batchSize) || resolved.batchSize <= 0) {
    throw new RangeError("batchSize must be a positive integer");
  }
  if (!Number.isFinite(resolved.learningRate) || resolved.learningRate <= 0) {
    throw new RangeError("learningRate must be positive");
  }
  if (resolved.validationSplit < 0 || resolved.validationSplit >= 1) {
    throw new RangeError("validationSplit must be in [0, 1)");
  }
  if (resolved.augmentNoise < 0 || !Number.isFinite(resolved.augmentNoise)) {
    throw new RangeError("augmentNoise must be non-negative");
  }
  if (resolved.dropoutRate < 0 || resolved.dropoutRate >= 1) {
    throw new RangeError("dropoutRate must be in [0, 1)");
  }
  return Object.freeze(resolved);
}

/** Build a residual, dilated TCN with a 125-frame receptive field. */
export function buildDynamicTcnModel(
  numClasses: number,
  config: Partial<DynamicGestureTrainingConfig> = {}
): tf.LayersModel {
  if (!Number.isInteger(numClasses) || numClasses < 2) {
    throw new RangeError("numClasses must be at least 2");
  }
  const cfg = resolveDynamicGestureTrainingConfig(config);
  const input = tf.input({
    name: "dynamic_gesture_input",
    shape: [DYNAMIC_SEQUENCE_LENGTH, DYNAMIC_FEATURE_DIM],
  });
  let encoded = applyLayer(
    tf.layers.reshape({
      name: "tcn_add_spatial_axis",
      targetShape: [DYNAMIC_SEQUENCE_LENGTH, 1, DYNAMIC_FEATURE_DIM],
    }),
    input
  );
  encoded = applyLayer(
    tf.layers.conv2d({
      name: "tcn_input_projection",
      filters: TCN_FILTERS,
      kernelSize: [1, 1],
      padding: "same",
      useBias: false,
      kernelInitializer: "heNormal",
    }),
    encoded
  );
  encoded = applyLayer(
    tf.layers.batchNormalization({ name: "tcn_input_bn" }),
    encoded
  );
  encoded = applyLayer(
    tf.layers.activation({ name: "tcn_input_relu", activation: "relu" }),
    encoded
  );

  DYNAMIC_TCN_DILATIONS.forEach((dilationRate, index) => {
    encoded = residualTcnBlock(
      encoded,
      dilationRate,
      cfg.dropoutRate,
      index + 1
    );
  });

  let output = applyLayer(
    tf.layers.globalAveragePooling2d({ name: "temporal_average" }),
    encoded
  );
  output = applyLayer(
    tf.layers.dense({
      name: "gesture_embedding",
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal",
    }),
    output
  );
  output = applyLayer(
    tf.layers.dropout({ name: "classifier_dropout", rate: cfg.dropoutRate }),
    output
  );
  output = applyLayer(
    tf.layers.dense({
      name: "gesture_probabilities",
      units: numClasses,
      activation: "softmax",
    }),
    output
  );

  const model = tf.model({
    name: "dynamic_gesture_tcn",
    inputs: input,
    outputs: output,
  });
  model.compile({
    optimizer: tf.train.adam(cfg.learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace<T>(values: T[], random: () => number): void {
  for (let index = values.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
}

/**
 * Split whole sequences. When multiple sessions exist, an entire session is
 * assigned to one side so no person's recording session leaks into validation.
 */
export function splitDynamicGestureSequences(
  sequences: readonly DynamicGestureSequence[],
  validationSplit: number,
  randomSeed = DEFAULT_DYNAMIC_GESTURE_TRAINING_CONFIG.randomSeed
): DynamicGestureDatasetSplit {
  if (validationSplit <= 0 || sequences.length < 2) {
    return { train: [...sequences], validation: [] };
  }
  if (validationSplit >= 1) {
    throw new RangeError("validationSplit must be less than 1");
  }

  const random = seededRandom(randomSeed);
  const sessionIds = new Set(
    sequences.map(sequence => sequence.sessionId).filter(Boolean)
  );
  if (sessionIds.size <= 1) {
    return { train: [...sequences], validation: [] };
  }
  const groups = new Map<string, DynamicGestureSequence[]>();
  sequences.forEach((sequence, index) => {
    const key = sequence.sessionId || `__sequence_${index}`;
    const group = groups.get(key);
    if (group) group.push(sequence);
    else groups.set(key, [sequence]);
  });

  const shuffledGroups = Array.from(groups.values());
  shuffleInPlace(shuffledGroups, random);
  const targetValidationCount = Math.max(
    1,
    Math.round(sequences.length * validationSplit)
  );
  const totalByLabel = new Map<string, number>();
  for (const sequence of sequences) {
    totalByLabel.set(
      sequence.label,
      (totalByLabel.get(sequence.label) ?? 0) + 1
    );
  }
  const validationByLabel = new Map<string, number>();
  const validation: DynamicGestureSequence[] = [];
  const train: DynamicGestureSequence[] = [];

  for (const group of shuffledGroups) {
    const groupByLabel = new Map<string, number>();
    for (const sequence of group) {
      groupByLabel.set(
        sequence.label,
        (groupByLabel.get(sequence.label) ?? 0) + 1
      );
    }
    const keepsTrainingExample = Array.from(groupByLabel).every(
      ([label, count]) =>
        (totalByLabel.get(label) ?? 0) -
          (validationByLabel.get(label) ?? 0) -
          count >=
        1
    );
    const shouldValidate =
      keepsTrainingExample && validation.length < targetValidationCount;
    if (shouldValidate) {
      validation.push(...group);
      groupByLabel.forEach((count, label) => {
        validationByLabel.set(
          label,
          (validationByLabel.get(label) ?? 0) + count
        );
      });
    } else {
      train.push(...group);
    }
  }

  if (train.length === 0 && validation.length > 0) {
    train.push(validation.pop() as DynamicGestureSequence);
  }
  return { train, validation };
}

function augmentDynamicFeatureSequence(
  features: readonly (readonly number[])[],
  noise: number,
  random: () => number
): number[][] {
  return features.map(frame => {
    const augmented = [...frame];
    for (let index = 0; index < DYNAMIC_SENSOR_COUNT; index++) {
      augmented[index] = clamp(
        augmented[index] + (random() * 2 - 1) * noise,
        0,
        1
      );
      const rightIndex = DYNAMIC_HAND_FEATURE_DIM + index;
      augmented[rightIndex] = clamp(
        augmented[rightIndex] + (random() * 2 - 1) * noise,
        0,
        1
      );
    }
    return augmented;
  });
}

function oneHot(index: number, size: number): number[] {
  const target = new Array(size).fill(0);
  target[index] = 1;
  return target;
}

function metric(logs: tf.Logs | undefined, ...names: string[]): number {
  for (const name of names) {
    const value = logs?.[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return 0;
}

export async function trainDynamicGestureModel(
  sequences: readonly DynamicGestureSequence[],
  config: Partial<DynamicGestureTrainingConfig> = {},
  onProgress?: (progress: DynamicGestureTrainingProgress) => void,
  signal?: AbortSignal
): Promise<DynamicGestureTrainingResult> {
  const throwIfAborted = () => {
    if (!signal?.aborted) return;
    const error = new Error("动态训练已取消");
    error.name = "AbortError";
    throw error;
  };
  throwIfAborted();
  const cfg = resolveDynamicGestureTrainingConfig(config);
  const usable = sequences.filter(
    sequence =>
      sequence.label &&
      (sequence.leftFrames.length > 0 || sequence.rightFrames.length > 0)
  );
  const labels = Array.from(
    new Set(usable.map(sequence => sequence.label))
  ).sort();
  if (labels.length < 2) {
    throw new Error("至少需要 2 个动态手势类别才能训练 TCN");
  }

  const split = splitDynamicGestureSequences(
    usable,
    cfg.validationSplit,
    cfg.randomSeed
  );
  if (split.train.length === 0) {
    throw new Error("没有可用于训练的动态手势片段");
  }

  const labelToIndex = new Map(labels.map((label, index) => [label, index]));
  const random = seededRandom(cfg.randomSeed ^ 0xa5a5a5a5);
  const trainFeatures: number[][][] = [];
  const trainTargets: number[][] = [];

  for (const sequence of split.train) {
    throwIfAborted();
    const features = preprocessDynamicGestureSequence(sequence);
    const target = oneHot(
      labelToIndex.get(sequence.label) as number,
      labels.length
    );
    trainFeatures.push(features);
    trainTargets.push(target);
    if (cfg.augmentNoise > 0) {
      trainFeatures.push(
        augmentDynamicFeatureSequence(features, cfg.augmentNoise, random)
      );
      trainTargets.push([...target]);
    }
  }

  const validationFeatures = split.validation.map(sequence =>
    preprocessDynamicGestureSequence(sequence)
  );
  const validationTargets = split.validation.map(sequence =>
    oneHot(labelToIndex.get(sequence.label) as number, labels.length)
  );
  const trainXs = tf.tensor3d(trainFeatures, [
    trainFeatures.length,
    DYNAMIC_SEQUENCE_LENGTH,
    DYNAMIC_FEATURE_DIM,
  ]);
  const trainYs = tf.tensor2d(trainTargets, [
    trainTargets.length,
    labels.length,
  ]);
  const validationXs =
    validationFeatures.length > 0
      ? tf.tensor3d(validationFeatures, [
          validationFeatures.length,
          DYNAMIC_SEQUENCE_LENGTH,
          DYNAMIC_FEATURE_DIM,
        ])
      : null;
  const validationYs =
    validationTargets.length > 0
      ? tf.tensor2d(validationTargets, [
          validationTargets.length,
          labels.length,
        ])
      : null;
  const model = buildDynamicTcnModel(labels.length, cfg);
  const history: DynamicGestureTrainingProgress[] = [];
  const handleAbort = () => {
    model.stopTraining = true;
  };
  signal?.addEventListener("abort", handleAbort, { once: true });

  try {
    throwIfAborted();
    await model.fit(trainXs, trainYs, {
      epochs: cfg.epochs,
      batchSize: Math.min(cfg.batchSize, trainFeatures.length),
      shuffle: true,
      validationData:
        validationXs && validationYs ? [validationXs, validationYs] : undefined,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress: DynamicGestureTrainingProgress = {
            epoch: epoch + 1,
            totalEpochs: cfg.epochs,
            loss: metric(logs, "loss"),
            accuracy: metric(logs, "acc", "accuracy"),
            valLoss: metric(logs, "val_loss", "valLoss"),
            valAccuracy: metric(logs, "val_acc", "val_accuracy", "valAccuracy"),
          };
          history.push(progress);
          onProgress?.(progress);
          if (signal?.aborted) model.stopTraining = true;
          await tf.nextFrame();
        },
      },
    });
    throwIfAborted();
  } catch (error) {
    model.optimizer?.dispose();
    model.dispose();
    throw error;
  } finally {
    signal?.removeEventListener("abort", handleAbort);
    trainXs.dispose();
    trainYs.dispose();
    validationXs?.dispose();
    validationYs?.dispose();
  }

  const finalProgress = history[history.length - 1];
  const validationAccuracy = finalProgress?.valAccuracy ?? 0;
  return {
    model,
    labels,
    accuracy:
      split.validation.length > 0
        ? validationAccuracy
        : finalProgress?.accuracy || 0,
    validationAccuracy,
    validationLoss: finalProgress?.valLoss ?? 0,
    trainingSequenceCount: split.train.length,
    validationSequenceCount: split.validation.length,
    history,
    config: cfg,
  };
}

let activeDynamicGestureModel: tf.LayersModel | null = null;
let activeDynamicGestureLabels: string[] = [];
let activeDynamicGestureMetadata: DynamicGestureModelMetadata = {
  ...DEFAULT_MODEL_METADATA,
};

export function setActiveDynamicGestureModel(
  model: tf.LayersModel,
  labels: readonly string[],
  metadata:
    | Partial<DynamicGestureModelMetadata>
    | Partial<DynamicGestureTrainingConfig> = {}
): void {
  if (model.inputs[0]?.shape[1] !== DYNAMIC_SEQUENCE_LENGTH) {
    throw new Error(`动态模型序列长度必须为 ${DYNAMIC_SEQUENCE_LENGTH}`);
  }
  if (model.inputs[0]?.shape[2] !== DYNAMIC_FEATURE_DIM) {
    throw new Error(`动态模型每帧特征维度必须为 ${DYNAMIC_FEATURE_DIM}`);
  }
  const previousModel = activeDynamicGestureModel;
  activeDynamicGestureModel = model;
  if (previousModel && previousModel !== model) {
    previousModel.dispose();
  }
  activeDynamicGestureLabels = [...labels];
  const modelMetadata =
    "sampleRateHz" in metadata ||
    "sequenceLength" in metadata ||
    "featureDim" in metadata
      ? (metadata as Partial<DynamicGestureModelMetadata>)
      : {};
  activeDynamicGestureMetadata = {
    ...DEFAULT_MODEL_METADATA,
    ...modelMetadata,
  };
}

export function isDynamicGestureModelLoaded(): boolean {
  return (
    activeDynamicGestureModel !== null && activeDynamicGestureLabels.length > 0
  );
}

export function getLoadedDynamicGestureLabels(): string[] {
  return [...activeDynamicGestureLabels];
}

export function getActiveDynamicGestureModelMetadata(): DynamicGestureModelMetadata {
  return { ...activeDynamicGestureMetadata };
}

export function predictDynamicGesture(
  sequence: DynamicGestureSequenceInput | readonly (readonly number[])[]
): DynamicGesturePrediction | null {
  if (!activeDynamicGestureModel || activeDynamicGestureLabels.length === 0) {
    return null;
  }
  const features = Array.isArray(sequence)
    ? sequence.map(frame => [...frame])
    : preprocessDynamicGestureSequence(sequence as DynamicGestureSequenceInput);
  if (
    features.length !== DYNAMIC_SEQUENCE_LENGTH ||
    features.some(frame => frame.length !== DYNAMIC_FEATURE_DIM)
  ) {
    throw new Error(
      `动态模型输入必须为 [${DYNAMIC_SEQUENCE_LENGTH}, ${DYNAMIC_FEATURE_DIM}]`
    );
  }

  const probabilities = tf.tidy(() => {
    const input = tf.tensor3d(
      [features],
      [1, DYNAMIC_SEQUENCE_LENGTH, DYNAMIC_FEATURE_DIM]
    );
    const output = activeDynamicGestureModel?.predict(input);
    const tensor = Array.isArray(output) ? output[0] : output;
    if (!tensor) return [];
    return Array.from(tensor.dataSync());
  });
  if (probabilities.length === 0) return null;

  const allProbabilities = activeDynamicGestureLabels
    .map((label, index) => ({
      label,
      probability: probabilities[index] ?? 0,
    }))
    .sort((first, second) => second.probability - first.probability);
  const best = allProbabilities[0];
  return {
    label: best.label,
    confidence: best.probability,
    allProbabilities,
  };
}

async function getModelArtifacts(
  model: tf.LayersModel
): Promise<tf.io.ModelArtifacts> {
  return new Promise((resolve, reject) => {
    void model
      .save(
        tf.io.withSaveHandler(async artifacts => {
          resolve(artifacts);
          return {
            modelArtifactsInfo: {
              dateSaved: new Date(),
              modelTopologyType: "JSON",
            },
          };
        })
      )
      .catch(reject);
  });
}

function mergeWeightData(
  weightData: tf.io.WeightData | undefined
): ArrayBuffer {
  if (weightData instanceof ArrayBuffer) return weightData;
  if (!Array.isArray(weightData)) return new ArrayBuffer(0);
  const buffers = weightData as ArrayBuffer[];
  const totalLength = buffers.reduce(
    (sum, buffer) => sum + buffer.byteLength,
    0
  );
  const merged = new Uint8Array(totalLength);
  let offset = 0;
  for (const buffer of buffers) {
    merged.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return merged.buffer;
}

export async function serializeDynamicGestureModel(
  model: tf.LayersModel,
  labels: readonly string[],
  accuracy: number,
  name: string,
  metadataOrValidationLoss:
    | number
    | Partial<DynamicGestureTrainingConfig>
    | Partial<DynamicGestureModelMetadata> = 0
): Promise<SavedDynamicModelRecord> {
  const artifacts = await getModelArtifacts(model);
  const validationLoss =
    typeof metadataOrValidationLoss === "number" ? metadataOrValidationLoss : 0;
  return {
    name,
    createdAt: Date.now(),
    accuracy,
    labels: [...labels],
    modelJson: JSON.stringify({
      modelTopology: artifacts.modelTopology,
      weightSpecs: artifacts.weightSpecs,
    }),
    weightsData: mergeWeightData(artifacts.weightData),
    modelType: "tcn",
    sampleRateHz: DYNAMIC_SAMPLE_RATE_HZ,
    sequenceLength: DYNAMIC_SEQUENCE_LENGTH,
    featureDim: DYNAMIC_FEATURE_DIM,
    validationLoss,
  } as SavedDynamicModelRecord;
}

export async function loadDynamicGestureModelFromSaved(
  saved: SavedDynamicModelRecord,
  options: { activate?: boolean } = {}
): Promise<tf.LayersModel> {
  const view = saved as unknown as SavedDynamicModelView;
  const parsed = JSON.parse(view.modelJson) as {
    modelTopology: tf.io.ModelJSON;
    weightSpecs?: tf.io.WeightsManifestEntry[];
    weightsManifest?: tf.io.WeightsManifestEntry[];
  };
  const weightSpecs = parsed.weightSpecs ?? parsed.weightsManifest;
  if (!parsed.modelTopology || !weightSpecs) {
    throw new Error("动态模型制品缺少 topology 或 weight specs");
  }
  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: parsed.modelTopology,
      weightSpecs,
      weightData: view.weightsData,
    })
  );
  if (options.activate !== false) {
    setActiveDynamicGestureModel(model, view.labels, {
      sampleRateHz: view.sampleRateHz ?? DYNAMIC_SAMPLE_RATE_HZ,
      sequenceLength: view.sequenceLength ?? DYNAMIC_SEQUENCE_LENGTH,
      featureDim: view.featureDim ?? DYNAMIC_FEATURE_DIM,
    });
  }
  return model;
}
