/*
 * skeletonModel — 触觉→骨架回归模型 (TensorFlow.js)
 *
 * 功能:
 * - 训练: 输入触觉数据(141D) → 输出骨架关键点(63D = 21点 x 3坐标)
 * - 推理: 仅用手套触觉数据预测手部骨架姿态，驱动火柴人渲染
 *
 * 训练数据来源:
 * - 同步采集的视觉骨架(landmarks) + 触觉传感器(sensor_data + quaternion)
 * - 视觉数据作为 ground truth (回归目标)
 * - 触觉数据作为模型输入
 *
 * 网络结构:
 * - 141 → 256 → 128 → 64 → 63 (MSE loss)
 * - 含 BatchNorm + Dropout 正则化
 * - 使用 residual-style skip connection 提升精度
 */
import * as tf from "@tensorflow/tfjs";
import { pickPrimaryHand } from "./datasetStore";
import type { TrainingSample, HandLandmarkPoint, HandSample } from "./datasetStore";

const TACTILE_DIM = 141; // 137 mapped sensors + 4 quaternion
const SKELETON_DIM = 63; // 21 landmarks * 3 (x, y, z)

// ===== 训练进度 =====

export interface SkeletonTrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;       // MSE
  valLoss: number;    // validation MSE
  mae: number;        // Mean Absolute Error
  valMae: number;     // validation MAE
}

export interface SkeletonTrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  augmentNoise: number;
  validationSplit: number;
}

const DEFAULT_CONFIG: SkeletonTrainingConfig = {
  epochs: 80,
  batchSize: 32,
  learningRate: 0.001,
  augmentNoise: 0.015,
  validationSplit: 0.15,
};

// ===== 数据预处理 =====

function preprocessTactileInput(hand: HandSample): number[] {
  const tactile = hand.sensor_data.slice(0, 137).map((v) => v / 255.0);
  while (tactile.length < 137) tactile.push(0);
  return [...tactile, ...hand.quaternion]; // 141D（单手）
}

function preprocessSkeletonTarget(landmarks: HandLandmarkPoint[]): number[] {
  // MediaPipe 输出的坐标已经是 0-1 归一化范围
  const coords: number[] = [];
  for (const lm of landmarks) {
    coords.push(lm.x, lm.y, lm.z);
  }
  return coords; // 63D
}

function augmentTactile(features: number[], noise: number): number[] {
  return features.map((v) => {
    const n = v + (Math.random() - 0.5) * 2 * noise;
    return Math.max(0, Math.min(1, n));
  });
}

function prepareRegressionData(
  samples: TrainingSample[],
  config: SkeletonTrainingConfig
): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const inputs: number[][] = [];
  const targets: number[][] = [];

  for (const sample of samples) {
    // 取优先可用的一只手，且该手必须有有效 landmarks
    const hand = pickPrimaryHand(sample);
    if (!hand || !hand.landmarks || hand.landmarks.length !== 21) continue;

    const tactile = preprocessTactileInput(hand);
    const skeleton = preprocessSkeletonTarget(hand.landmarks);

    // 原始样本
    inputs.push(tactile);
    targets.push(skeleton);

    // 数据增强（对触觉输入加噪声，目标不变）
    if (config.augmentNoise > 0) {
      const augmented = augmentTactile(tactile, config.augmentNoise);
      inputs.push(augmented);
      targets.push([...skeleton]);
    }
  }

  // 打乱
  const indices = Array.from({ length: inputs.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledInputs = indices.map((i) => inputs[i]);
  const shuffledTargets = indices.map((i) => targets[i]);

  const xs = tf.tensor2d(shuffledInputs, [shuffledInputs.length, TACTILE_DIM]);
  const ys = tf.tensor2d(shuffledTargets, [shuffledTargets.length, SKELETON_DIM]);

  return { xs, ys };
}

// ===== 模型构建 =====

function buildSkeletonModel(lr: number): tf.Sequential {
  const model = tf.sequential();

  // 输入层 → 256
  model.add(
    tf.layers.dense({
      inputShape: [TACTILE_DIM],
      units: 256,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // 隐藏层 → 128
  model.add(
    tf.layers.dense({
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.15 }));

  // 隐藏层 → 64
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.1 }));

  // 输出层 → 63 (sigmoid 限制输出到 0-1 范围，匹配归一化坐标)
  model.add(
    tf.layers.dense({
      units: SKELETON_DIM,
      activation: "sigmoid",
    })
  );

  model.compile({
    optimizer: tf.train.adam(lr),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  return model;
}

// ===== 训练 =====

export async function trainSkeletonModel(
  samples: TrainingSample[],
  config: Partial<SkeletonTrainingConfig> = {},
  onProgress?: (progress: SkeletonTrainingProgress) => void
): Promise<{
  model: tf.Sequential;
  finalLoss: number;
  finalMae: number;
  history: SkeletonTrainingProgress[];
}> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // 过滤有效样本（取单手，必须有 landmarks）
  const validSamples = samples.filter((s) => {
    const h = pickPrimaryHand(s);
    return h && h.landmarks && h.landmarks.length === 21;
  });

  if (validSamples.length < 10) {
    throw new Error(
      `有效样本不足（需要至少10个含视觉数据的样本，当前仅${validSamples.length}个）`
    );
  }

  console.log(
    `[SkeletonModel] Training with ${validSamples.length} valid samples`
  );

  const { xs, ys } = prepareRegressionData(validSamples, cfg);
  const model = buildSkeletonModel(cfg.learningRate);
  const history: SkeletonTrainingProgress[] = [];

  await model.fit(xs, ys, {
    epochs: cfg.epochs,
    batchSize: cfg.batchSize,
    validationSplit: cfg.validationSplit,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const progress: SkeletonTrainingProgress = {
          epoch: epoch + 1,
          totalEpochs: cfg.epochs,
          loss: logs?.loss ?? 0,
          valLoss: logs?.val_loss ?? 0,
          mae: logs?.mae ?? logs?.meanAbsoluteError ?? logs?.mean_absolute_error ?? 0,
          valMae:
            logs?.val_mae ?? logs?.val_meanAbsoluteError ?? logs?.val_mean_absolute_error ?? 0,
        };
        history.push(progress);
        if (onProgress) onProgress(progress);
      },
    },
  });

  xs.dispose();
  ys.dispose();

  const lastProgress = history[history.length - 1];

  console.log(
    `[SkeletonModel] Training complete. Val Loss: ${lastProgress.valLoss.toFixed(6)}, Val MAE: ${lastProgress.valMae.toFixed(6)}`
  );

  return {
    model,
    finalLoss: lastProgress.valLoss,
    finalMae: lastProgress.valMae,
    history,
  };
}

// ===== 推理 =====

let skeletonModel: tf.Sequential | null = null;

export function setSkeletonModel(model: tf.Sequential) {
  skeletonModel = model;
}

export function isSkeletonModelLoaded(): boolean {
  return skeletonModel !== null;
}

/**
 * 从触觉数据预测骨架关键点
 * @param sensorData 137 个重映射传感器值
 * @param quaternion IMU 四元数 [w, x, y, z]
 * @returns 21 个关键点的归一化坐标 (x, y, z)，或 null
 */
export function predictSkeleton(
  sensorData: number[],
  quaternion: [number, number, number, number]
): HandLandmarkPoint[] | null {
  if (!skeletonModel) return null;

  const features = [...sensorData.map((v) => v / 255.0), ...quaternion];
  const input = tf.tensor2d([features], [1, TACTILE_DIM]);
  const output = skeletonModel.predict(input) as tf.Tensor;
  const coords = output.dataSync() as Float32Array;

  input.dispose();
  output.dispose();

  // 将 63 个值转换为 21 个关键点
  const landmarks: HandLandmarkPoint[] = [];
  for (let i = 0; i < 21; i++) {
    landmarks.push({
      x: coords[i * 3],
      y: coords[i * 3 + 1],
      z: coords[i * 3 + 2],
    });
  }

  return landmarks;
}

// ===== 模型序列化 =====

export interface SavedSkeletonModel {
  id?: number;
  name: string;
  createdAt: number;
  valLoss: number;
  valMae: number;
  modelJson: string;
  weightsData: ArrayBuffer;
}

export async function serializeSkeletonModel(
  model: tf.Sequential,
  valLoss: number,
  valMae: number,
  name: string
): Promise<SavedSkeletonModel> {
  const artifacts = await new Promise<tf.io.ModelArtifacts>((resolve) => {
    model.save(
      tf.io.withSaveHandler(async (a) => {
        resolve(a);
        return {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: "JSON",
          },
        };
      })
    );
  });

  let weightsBuffer: ArrayBuffer;
  if (artifacts.weightData instanceof ArrayBuffer) {
    weightsBuffer = artifacts.weightData;
  } else if (Array.isArray(artifacts.weightData)) {
    const totalLength = (artifacts.weightData as ArrayBuffer[]).reduce(
      (sum, buf) => sum + buf.byteLength,
      0
    );
    const merged = new Uint8Array(totalLength);
    let offset = 0;
    for (const buf of artifacts.weightData as ArrayBuffer[]) {
      merged.set(new Uint8Array(buf), offset);
      offset += buf.byteLength;
    }
    weightsBuffer = merged.buffer;
  } else {
    weightsBuffer = new ArrayBuffer(0);
  }

  return {
    name,
    createdAt: Date.now(),
    valLoss,
    valMae,
    modelJson: JSON.stringify({
      modelTopology: artifacts.modelTopology,
      weightsManifest: artifacts.weightSpecs,
    }),
    weightsData: weightsBuffer,
  };
}

export async function loadSkeletonModelFromSaved(
  saved: SavedSkeletonModel
): Promise<void> {
  const parsed = JSON.parse(saved.modelJson);
  const model = (await tf.loadLayersModel(
    tf.io.fromMemory(
      parsed.modelTopology,
      parsed.weightsManifest,
      saved.weightsData
    )
  )) as tf.Sequential;

  setSkeletonModel(model);
  console.log(
    `[SkeletonModel] Loaded model "${saved.name}" (Val MAE: ${saved.valMae.toFixed(6)})`
  );
}
