/*
 * signLanguageModel — TensorFlow.js 手语识别模型（双分支架构）
 *
 * 训练策略:
 * - 阶段1: 使用融合特征（视觉63D + 触觉137D + 四元数4D = 204D）训练教师模型
 * - 阶段2: 知识蒸馏，训练仅触觉分支（141D）的学生模型
 * - 最终部署: 仅使用触觉分支（141D输入）进行推理
 *
 * 网络结构:
 * - 教师模型: 204 → 256 → 128 → 64 → N (with dropout + BN)
 * - 学生模型: 141 → 128 → 64 → 32 → N (with dropout + BN)
 *   学生模型额外接收教师的 soft labels 作为蒸馏目标
 */
import * as tf from "@tensorflow/tfjs";
import { sampleHasVision } from "./datasetStore";
import type { TrainingSample, SavedModel, HandSample } from "./datasetStore";

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  augmentNoise: number;
  distillationTemp: number; // 蒸馏温度
  distillationAlpha: number; // 蒸馏损失权重 (0-1)
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
  phase: "teacher" | "student"; // 当前训练阶段
}

export interface PredictionResult {
  label: string;
  confidence: number;
  allProbabilities: Array<{ label: string; probability: number }>;
}

const DEFAULT_CONFIG: TrainingConfig = {
  epochs: 50,
  batchSize: 32,
  learningRate: 0.001,
  validationSplit: 0.2,
  augmentNoise: 0.02,
  distillationTemp: 3.0,
  distillationAlpha: 0.5,
};

const SENSOR_N = 137;
const HAND_TACTILE_DIM = SENSOR_N + 4; // 137 传感 + 4 四元数 = 141
const HAND_VISUAL_DIM = 63; // 21 landmarks * 3
const TACTILE_DIM = HAND_TACTILE_DIM * 2; // 双手 = 282
const VISUAL_DIM = HAND_VISUAL_DIM * 2; // 双手 = 126
const FUSED_DIM = TACTILE_DIM + VISUAL_DIM; // 408

// ===== 双手特征构建（缺失的手填 0） =====

/** 单手触觉：137 归一化传感 + 4 四元数 = 141（缺失填 0） */
function handTactile(
  h: { sensor_data: number[]; quaternion: [number, number, number, number] } | null
): number[] {
  if (!h || !h.sensor_data || h.sensor_data.length === 0) {
    return new Array(HAND_TACTILE_DIM).fill(0);
  }
  const s = h.sensor_data.slice(0, SENSOR_N).map((v) => v / 255.0);
  while (s.length < SENSOR_N) s.push(0);
  return [...s, ...h.quaternion];
}

/** 单手视觉：21 关键点 * 3 = 63（缺失或不足填 0） */
function handVisual(h: HandSample | null): number[] {
  if (!h || !h.landmarks || h.landmarks.length !== 21) {
    return new Array(HAND_VISUAL_DIM).fill(0);
  }
  const v: number[] = [];
  for (const lm of h.landmarks) v.push(lm.x, lm.y, lm.z);
  return v;
}

// ===== 数据预处理 =====

function preprocessFusedSample(sample: TrainingSample): number[] {
  return [
    ...handTactile(sample.left),
    ...handTactile(sample.right),
    ...handVisual(sample.left),
    ...handVisual(sample.right),
  ]; // 282 + 126 = 408
}

function preprocessTactileSample(sample: TrainingSample): number[] {
  return [...handTactile(sample.left), ...handTactile(sample.right)]; // 282
}

function augmentFeatures(features: number[], noise: number): number[] {
  return features.map((v) => {
    const n = v + (Math.random() - 0.5) * 2 * noise;
    return Math.max(0, Math.min(1, n));
  });
}

function prepareData(
  samples: TrainingSample[],
  labels: string[],
  config: TrainingConfig,
  mode: "fused" | "tactile"
): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const labelToIndex = new Map(labels.map((l, i) => [l, i]));
  const numClasses = labels.length;
  const dim = mode === "fused" ? FUSED_DIM : TACTILE_DIM;

  const features: number[][] = [];
  const targets: number[][] = [];

  for (const sample of samples) {
    const feat =
      mode === "fused"
        ? preprocessFusedSample(sample)
        : preprocessTactileSample(sample);
    const labelIdx = labelToIndex.get(sample.label);
    if (labelIdx === undefined) continue;

    // 原始样本
    features.push(feat);
    const oneHot = new Array(numClasses).fill(0);
    oneHot[labelIdx] = 1;
    targets.push(oneHot);

    // 数据增强
    if (config.augmentNoise > 0) {
      const augmented = augmentFeatures(feat, config.augmentNoise);
      features.push(augmented);
      targets.push([...oneHot]);
    }
  }

  // 打乱
  const indices = Array.from({ length: features.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledFeatures = indices.map((i) => features[i]);
  const shuffledTargets = indices.map((i) => targets[i]);

  const xs = tf.tensor2d(shuffledFeatures, [shuffledFeatures.length, dim]);
  const ys = tf.tensor2d(shuffledTargets, [shuffledTargets.length, numClasses]);

  return { xs, ys };
}

// ===== 模型构建 =====

function buildTeacherModel(numClasses: number, lr: number): tf.Sequential {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [FUSED_DIM],
      units: 256,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.dense({
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.1 }));

  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(lr),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

function buildStudentModel(numClasses: number, lr: number): tf.Sequential {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [TACTILE_DIM],
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.1 }));

  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(lr),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// ===== 知识蒸馏辅助 =====

function generateSoftLabels(
  teacherModel: tf.Sequential,
  samples: TrainingSample[],
  temperature: number
): tf.Tensor2D {
  // 获取教师模型对融合数据的预测（带温度缩放的 soft labels）
  const fusedFeatures: number[][] = [];
  for (const sample of samples) {
    fusedFeatures.push(preprocessFusedSample(sample));
  }

  const input = tf.tensor2d(fusedFeatures, [fusedFeatures.length, FUSED_DIM]);

  // 获取 logits (去掉最后的 softmax 层效果，用温度缩放)
  const predictions = teacherModel.predict(input) as tf.Tensor2D;

  // 温度缩放: softmax(logits / T)
  // 由于我们只有 softmax 输出，用 log → scale → softmax 近似
  const logProbs = predictions.log();
  const scaled = logProbs.div(tf.scalar(temperature));
  const softLabels = scaled.softmax() as tf.Tensor2D;

  input.dispose();
  predictions.dispose();
  logProbs.dispose();
  scaled.dispose();

  return softLabels;
}

// ===== 训练 =====

export async function trainModel(
  samples: TrainingSample[],
  config: Partial<TrainingConfig> = {},
  onProgress?: (progress: TrainingProgress) => void
): Promise<{
  teacherModel: tf.Sequential;
  studentModel: tf.Sequential;
  labels: string[];
  teacherAccuracy: number;
  studentAccuracy: number;
  history: TrainingProgress[];
}> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // 获取所有唯一标签
  const labelSet = new Set(samples.map((s) => s.label));
  const labels = Array.from(labelSet).sort();
  const numClasses = labels.length;

  if (numClasses < 2) {
    throw new Error("至少需要 2 个不同的手语词汇才能训练模型");
  }

  console.log(
    `[Model] Training with ${samples.length} samples, ${numClasses} classes`
  );
  console.log(`[Model] Labels: ${labels.join(", ")}`);

  const history: TrainingProgress[] = [];

  // ===== 阶段1: 训练教师模型（融合特征） =====
  console.log("[Model] Phase 1: Training teacher model (fused features)...");

  const { xs: fusedXs, ys: fusedYs } = prepareData(
    samples,
    labels,
    cfg,
    "fused"
  );
  const teacherModel = buildTeacherModel(numClasses, cfg.learningRate);

  const teacherEpochs = Math.ceil(cfg.epochs * 0.6); // 60% epochs 给教师

  await teacherModel.fit(fusedXs, fusedYs, {
    epochs: teacherEpochs,
    batchSize: cfg.batchSize,
    validationSplit: cfg.validationSplit,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const progress: TrainingProgress = {
          epoch: epoch + 1,
          totalEpochs: cfg.epochs,
          loss: logs?.loss ?? 0,
          accuracy: logs?.acc ?? 0,
          valLoss: logs?.val_loss ?? 0,
          valAccuracy: logs?.val_acc ?? 0,
          phase: "teacher",
        };
        history.push(progress);
        if (onProgress) onProgress(progress);
      },
    },
  });

  fusedXs.dispose();
  fusedYs.dispose();

  const teacherAccuracy =
    history.length > 0
      ? history.filter((h) => h.phase === "teacher").pop()?.valAccuracy ?? 0
      : 0;

  console.log(
    `[Model] Teacher model trained. Val accuracy: ${(teacherAccuracy * 100).toFixed(1)}%`
  );

  // ===== 阶段2: 知识蒸馏训练学生模型（仅触觉） =====
  console.log("[Model] Phase 2: Training student model (tactile only, with distillation)...");

  // 数据质量检查：统计有效视觉数据的比例
  const samplesWithLandmarks = samples.filter(sampleHasVision);
  const landmarkRatio = samplesWithLandmarks.length / samples.length;
  console.log(
    `[Model] Data quality: ${samplesWithLandmarks.length}/${samples.length} samples have valid landmarks (${(landmarkRatio * 100).toFixed(1)}%)`
  );

  // 准备学生训练数据（仅触觉）
  const labelToIndex = new Map(labels.map((l, i) => [l, i]));
  const tactileFeatures: number[][] = [];
  const hardTargets: number[][] = [];

  for (const sample of samples) {
    const feat = preprocessTactileSample(sample);
    const labelIdx = labelToIndex.get(sample.label);
    if (labelIdx === undefined) continue;
    tactileFeatures.push(feat);
    const oneHot = new Array(numClasses).fill(0);
    oneHot[labelIdx] = 1;
    hardTargets.push(oneHot);

    // 数据增强
    if (cfg.augmentNoise > 0) {
      const augmented = augmentFeatures(feat, cfg.augmentNoise);
      tactileFeatures.push(augmented);
      hardTargets.push([...oneHot]);
    }
  }

  // 打乱
  const studentIndices = Array.from({ length: tactileFeatures.length }, (_, i) => i);
  for (let i = studentIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [studentIndices[i], studentIndices[j]] = [studentIndices[j], studentIndices[i]];
  }
  const shuffledTactile = studentIndices.map((i) => tactileFeatures[i]);
  const shuffledHardTargets = studentIndices.map((i) => hardTargets[i]);

  const studentXs = tf.tensor2d(shuffledTactile, [shuffledTactile.length, TACTILE_DIM]);
  const hardYs = tf.tensor2d(shuffledHardTargets, [shuffledHardTargets.length, numClasses]);

  // 决定蒸馏策略：如果视觉数据质量高，使用蒸馏；否则直接用 hard labels
  let trainingYs: tf.Tensor2D;

  if (landmarkRatio >= 0.8) {
    // 视觉数据充足，使用知识蒸馏
    console.log("[Model] Using knowledge distillation (landmark data sufficient)");

    // 生成教师的 soft labels（基于原始样本顺序）
    const softLabels = generateSoftLabels(teacherModel, samples, cfg.distillationTemp);

    // 对 soft labels 做同样的增强和打乱（复制每个 soft label 两次，对应原始+增强）
    const softLabelsData = softLabels.arraySync() as number[][];
    const expandedSoftLabels: number[][] = [];
    for (const sl of softLabelsData) {
      expandedSoftLabels.push(sl);
      if (cfg.augmentNoise > 0) {
        expandedSoftLabels.push([...sl]); // 增强样本使用相同的 soft label
      }
    }
    softLabels.dispose();

    // 按相同顺序打乱
    const shuffledSoftLabels = studentIndices.map((i) => expandedSoftLabels[i]);
    const softYs = tf.tensor2d(shuffledSoftLabels, [shuffledSoftLabels.length, numClasses]);

    // 混合标签: alpha * soft + (1-alpha) * hard
    const alpha = cfg.distillationAlpha;
    trainingYs = softYs
      .mul(tf.scalar(alpha))
      .add(hardYs.mul(tf.scalar(1 - alpha))) as tf.Tensor2D;
    softYs.dispose();
  } else {
    // 视觉数据不足，跳过蒸馏，直接用 hard labels 训练学生
    console.warn(
      `[Model] Skipping distillation: only ${(landmarkRatio * 100).toFixed(1)}% samples have landmarks. Training student with hard labels only.`
    );
    trainingYs = hardYs;
  }

  const studentModel = buildStudentModel(numClasses, cfg.learningRate * 0.5); // 学生用更小的学习率
  const studentEpochs = Math.max(cfg.epochs - teacherEpochs, 40); // 学生至少训练 40 epochs

  await studentModel.fit(studentXs, trainingYs, {
    epochs: studentEpochs,
    batchSize: cfg.batchSize,
    validationSplit: cfg.validationSplit,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const progress: TrainingProgress = {
          epoch: teacherEpochs + epoch + 1,
          totalEpochs: teacherEpochs + studentEpochs,
          loss: logs?.loss ?? 0,
          accuracy: logs?.acc ?? 0,
          valLoss: logs?.val_loss ?? 0,
          valAccuracy: logs?.val_acc ?? 0,
          phase: "student",
        };
        history.push(progress);
        if (onProgress) onProgress(progress);
      },
    },
  });

  // 清理
  studentXs.dispose();
  hardYs.dispose();
  if (trainingYs !== hardYs) trainingYs.dispose();

  const studentAccuracy =
    history.filter((h) => h.phase === "student").pop()?.valAccuracy ?? 0;

  console.log(
    `[Model] Student model trained. Val accuracy: ${(studentAccuracy * 100).toFixed(1)}%`
  );

  return {
    teacherModel,
    studentModel,
    labels,
    teacherAccuracy,
    studentAccuracy,
    history,
  };
}

// ===== 推理（仅使用学生/触觉模型） =====

let loadedModel: tf.Sequential | null = null;
let loadedLabels: string[] = [];

export function setActiveModel(model: tf.Sequential, labels: string[]) {
  loadedModel = model;
  loadedLabels = labels;
}

/** 单手触觉输入（供实时推理传入） */
export interface HandTactileInput {
  sensor_data: number[]; // 137 mapped
  quaternion: [number, number, number, number];
}

export function predict(
  left: HandTactileInput | null,
  right: HandTactileInput | null
): PredictionResult | null {
  if (!loadedModel || loadedLabels.length === 0) return null;

  const features = [...handTactile(left), ...handTactile(right)]; // 282
  const input = tf.tensor2d([features], [1, TACTILE_DIM]);
  const output = loadedModel.predict(input) as tf.Tensor;
  const probabilities = output.dataSync() as Float32Array;

  input.dispose();
  output.dispose();

  let maxIdx = 0;
  let maxProb = 0;
  const allProbabilities: Array<{ label: string; probability: number }> = [];

  for (let i = 0; i < probabilities.length; i++) {
    allProbabilities.push({
      label: loadedLabels[i],
      probability: probabilities[i],
    });
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIdx = i;
    }
  }

  allProbabilities.sort((a, b) => b.probability - a.probability);

  return {
    label: loadedLabels[maxIdx],
    confidence: maxProb,
    allProbabilities,
  };
}

export function isModelLoaded(): boolean {
  return loadedModel !== null && loadedLabels.length > 0;
}

export function getLoadedLabels(): string[] {
  return loadedLabels;
}

// ===== 模型序列化 =====

export async function serializeModel(
  model: tf.Sequential,
  labels: string[],
  accuracy: number,
  name: string,
  modelType: "fused" | "tactile" = "tactile"
): Promise<SavedModel> {
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
    accuracy,
    labels,
    modelJson: JSON.stringify({
      modelTopology: artifacts.modelTopology,
      weightsManifest: artifacts.weightSpecs,
    }),
    weightsData: weightsBuffer,
    modelType,
  };
}

export async function loadModelFromSaved(saved: SavedModel): Promise<void> {
  const parsed = JSON.parse(saved.modelJson);

  const model = (await tf.loadLayersModel(
    tf.io.fromMemory(parsed.modelTopology, parsed.weightsManifest, saved.weightsData)
  )) as tf.Sequential;

  setActiveModel(model, saved.labels);
  console.log(
    `[Model] Loaded ${saved.modelType || "tactile"} model "${saved.name}" with ${saved.labels.length} classes`
  );
}

// ===== 模型评估 =====

export function evaluateOnSamples(samples: TrainingSample[]): {
  accuracy: number;
  confusionMatrix: Record<string, Record<string, number>>;
} {
  if (!loadedModel || loadedLabels.length === 0) {
    return { accuracy: 0, confusionMatrix: {} };
  }

  let correct = 0;
  const confusion: Record<string, Record<string, number>> = {};

  for (const label of loadedLabels) {
    confusion[label] = {};
    for (const l2 of loadedLabels) {
      confusion[label][l2] = 0;
    }
  }

  for (const sample of samples) {
    const result = predict(sample.left, sample.right);
    if (!result) continue;

    const trueLabel = sample.label;
    const predLabel = result.label;

    if (confusion[trueLabel]) {
      confusion[trueLabel][predLabel] =
        (confusion[trueLabel][predLabel] || 0) + 1;
    }

    if (predLabel === trueLabel) correct++;
  }

  return {
    accuracy: samples.length > 0 ? correct / samples.length : 0,
    confusionMatrix: confusion,
  };
}
