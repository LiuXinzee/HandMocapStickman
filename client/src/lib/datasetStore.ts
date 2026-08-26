/*
 * datasetStore — IndexedDB 数据集存储管理
 * 用于存储手语训练样本（视觉骨架 + 触觉传感器数据 + 标签）
 *
 * 数据结构:
 * - 每个样本 = 一次静态手势的多模态快照
 * - landmarks: 21 个手部关键点 (x, y, z) — 视觉骨架数据
 * - sensor_data: number[137] 有效传感点（物理顺序，经过索引重映射）
 * - quaternion: [w, x, y, z] IMU 四元数
 * - label: 手语词汇 ID
 * - timestamp: 采集时间
 *
 * 训练策略:
 * - 训练时使用融合特征: landmarks(63D) + sensor_data(137D) + quaternion(4D) = 204D
 * - 推理时仅使用触觉特征: sensor_data(137D) + quaternion(4D) = 141D
 * - 通过知识蒸馏让触觉分支学习视觉分支的表征
 */

import type { ImuHealthReport } from "./imuHealth";

const DB_NAME = "hand_mocap_dataset";
const DB_VERSION = 6; // v6: 新增 sequences store（时序样本）；旧 samples 保持不动，供合成迁移读取
const STORE_SAMPLES = "samples";
const STORE_MODELS = "models";
const STORE_SKELETON_MODELS = "skeleton_models";
const STORE_SKELETON_SAMPLES = "skeleton_samples"; // 骨架姿态采集样本（独立于手语样本）
const STORE_SEQUENCES = "sequences"; // 时序样本（动态手语词）

export interface HandLandmarkPoint {
  x: number;
  y: number;
  z: number;
}

/** 单只手的一次多模态快照 */
export interface HandSample {
  sensor_data: number[]; // 137 有效传感点（物理顺序，经过索引重映射）
  quaternion: [number, number, number, number]; // IMU 四元数
  landmarks: HandLandmarkPoint[]; // 21 个手部关键点（无视觉时为空数组）
}

/**
 * 训练样本（双手）。手语可能单手或双手完成：
 * - 缺失的那只手为 null，特征向量对应位置填 0
 * - 每只手含触觉(137传感+4四元数) + 视觉(21关键点)
 */
export interface TrainingSample {
  id?: number; // auto-increment
  label: string; // 手语词汇 ID
  left: HandSample | null;
  right: HandSample | null;
  timestamp: number; // 采集时间戳
}

/** 该样本是否含任一只手的有效视觉关键点（21点） */
export function sampleHasVision(s: TrainingSample): boolean {
  return (
    (s.left?.landmarks?.length === 21) ||
    (s.right?.landmarks?.length === 21)
  );
}

/** 取样本中优先可用的一只手（优先右手，其次左手），用于单手骨架回归 */
export function pickPrimaryHand(s: TrainingSample): HandSample | null {
  if (s.right && s.right.sensor_data?.length) return s.right;
  if (s.left && s.left.sensor_data?.length) return s.left;
  return null;
}

export interface SavedModel {
  id?: number;
  name: string;
  createdAt: number;
  accuracy: number;
  labels: string[]; // 支持的词汇 ID 列表
  modelJson: string; // TF.js model topology JSON
  weightsData: ArrayBuffer; // 模型权重
  /**
   * 模型类型标记。
   * - fused/tactile: 单帧静态 MLP（signLanguageModel.ts）
   * - seq_fused/seq_tactile: 时序模型（sequenceModel.ts）
   */
  modelType: "fused" | "tactile" | "seq_fused" | "seq_tactile";
  // ↓ 仅时序模型有值。推理时必须用与训练一致的 T / frameDim，否则输入形状对不上
  seqLen?: number; // 定长重采样帧数 T
  frameDim?: number; // 每帧特征维度
  backbone?: string; // "tcn" | "tcn_bigru"
}

/** 是否为时序模型 */
export function isSequenceModel(m: SavedModel): boolean {
  return m.modelType === "seq_fused" || m.modelType === "seq_tactile";
}

export interface DatasetStats {
  totalSamples: number;
  labelCounts: Record<string, number>;
  labels: string[];
}

// ===== 时序样本（动态手语词） =====

/**
 * 序列中一个词的边界。孤立词训练时每条样本只有一个 segment 覆盖整段；
 * 句子级（连续手语）时同一 schema 直接装多个 segment，无需数据库迁移。
 */
export interface SequenceSegment {
  label: string;
  startFrame: number; // inclusive
  endFrame: number; // exclusive
}

/**
 * 这条样本是不是**句子级**录制（一条录制里连着打了好几个词）。
 *
 * 判据就是 `segments.length > 1`。之所以要有这么一个函数、而不是各处各写一遍：
 * 句子样本的 `primaryLabel` 等于**第一个词**（见下面 SequenceSample 的字段注释），
 * 所以在任何"按 primaryLabel 数样本 / 推类别表"的地方，一条「我 名字 王」都会
 * 冒充一条 `我` 的孤立词。孤立词训练、词频表、每词条数全都会被它污染，
 * 而且污染的样子很像"数据没问题，就是某个词训不准"，极难追。
 *
 * 所以孤立词那条链路一律要用它把句子样本挡掉；句子（CTC）那条链路反过来只要
 * 它为真的样本。两条链路共用同一个判据，才不会出现"一条样本两边都算 / 两边都不算"。
 */
export function isSentenceSample(s: { segments: SequenceSegment[] }): boolean {
  return s.segments.length > 1;
}

/**
 * 一条时序样本 —— 列存（SoA）+ TypedArray。
 *
 * 为什么不用 `number[]`：一条 1.5s@100Hz 的双手序列用 JS number 存约 700KB，
 * 30 词 × 20 条就是 420MB，IndexedDB 会撑爆。Uint8（传感器本就是 0-255）
 * + Float32 + 50Hz 栅格后约 64KB/条。TypedArray 走 structured clone，
 * IndexedDB 原生支持，不需要额外序列化。
 *
 * 视觉缺失帧填 NaN 而非 0：0 是合法的关键点坐标，填 0 会与真实数据混淆，
 * NaN 让预处理阶段能自由选择填充策略（插值 / 置零 + mask）。
 */
export interface SequenceSample {
  id?: number;
  segments: SequenceSegment[];
  primaryLabel: string; // = segments[0].label，冗余字段供 IndexedDB index 使用
  frameCount: number; // T
  timestamps: Float32Array; // [T] 相对录制起点 ms
  leftSensor: Uint8Array | null; // [T*137] 0-255
  rightSensor: Uint8Array | null;
  leftImu: Float32Array | null; // [T*10] = quat(w,x,y,z) + acc(x,y,z) + att(yaw,roll,pitch)
  rightImu: Float32Array | null;
  leftLandmarks: Float32Array | null; // [T*63]，视觉缺失帧填 NaN
  rightLandmarks: Float32Array | null;
  durationMs: number;
  sourceFps: number; // 重采样栅格频率
  origin: "recorded" | "synthesized";
  timestamp: number; // 采集时间戳
  /**
   * 录制当时的 IMU 健康度（陀螺漂移检测结果），左右手各一份。
   * 旧样本与合成样本为 undefined —— 这属于**可选字段，不需要升 DB_VERSION**：
   * IndexedDB 存的是 structured clone 的对象，给已有 store 的对象加可选属性
   * 不涉及 schema 迁移，而 `sequences` 现有索引只有 primaryLabel/timestamp/origin，
   * 没有一个落在这个字段上。谁以后看到这里别以为漏了迁移。
   *
   * 存量样本也不必重录：报告能由 leftImu/rightImu 用 analyzeSequenceImu 事后重算。
   */
  imuHealth?: { left: ImuHealthReport | null; right: ImuHealthReport | null };
}

/** 每手每帧的传感点数 / IMU 维度 / 关键点维度 —— 与 sensorMapping、gloveProtocol 对齐 */
export const SEQ_SENSOR_N = 137;
export const SEQ_IMU_N = 10; // quat4 + acc3 + att3
export const SEQ_LANDMARK_N = 63; // 21 点 × 3

export interface SequenceStats {
  totalSequences: number;
  recordedCount: number;
  synthesizedCount: number;
  /**
   * 句子级样本条数（`segments.length > 1`，见 `isSentenceSample`）。
   *
   * **不进 `labelCounts`**：句子的 `primaryLabel` 是它的第一个词，混进去就等于给
   * 那个词虚增条数，采集页会显示"这个词已经够了"而其实一条没多。单列成一个数，
   * 才能在页面上如实说"另有 N 条句子样本，不参与孤立词训练"。
   */
  sentenceCount: number;
  /** 每个标签下 [真实, 合成] 条数。**只统计孤立词样本**，句子样本走 `sentenceCount` */
  labelCounts: Record<string, { recorded: number; synthesized: number }>;
  labels: string[];
  avgDurationMs: number;
  estimatedBytes: number;
  /**
   * 按**戴了哪只手套**分的条数。
   *
   * 为什么单列：特征层给两只手各留一段独立槽位（缺手那段全 0），且 137 维指序左右相反，
   * 所以「全部用右手采」训出来的模型在左手输入上**从没见过任何信号**，输出会塌到
   * 某一个固定的词上。这件事在页面上一直看不见 —— 总条数、每词条数、覆盖率都正常，
   * 只有分手别计数能暴露它。推理端已有镜像归一化兜底（`handMirror.ts`），
   * 但那只救单手词；双手词只能靠这里发现数据偏了再补采。
   */
  handCounts: { leftOnly: number; rightOnly: number; both: number; neither: number };
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      const oldVersion = (event as IDBVersionChangeEvent).oldVersion;
      console.log("[DatasetStore] Upgrading DB from version", oldVersion, "to", DB_VERSION);
      // 处理版本升级 — 每个 store 只在不存在时创建
      if (!db.objectStoreNames.contains(STORE_SAMPLES)) {
        const store = db.createObjectStore(STORE_SAMPLES, {
          keyPath: "id",
          autoIncrement: true,
        });
        store.createIndex("label", "label", { unique: false });
        store.createIndex("timestamp", "timestamp", { unique: false });
      } else if (oldVersion < 5) {
        // v5: 样本结构从单手改为双手，旧样本不兼容，清空
        const tx = (event.target as IDBOpenDBRequest).transaction;
        tx?.objectStore(STORE_SAMPLES).clear();
        console.warn("[DatasetStore] v5 升级：已清空旧的单手样本，请重新采集双手数据");
      }
      if (!db.objectStoreNames.contains(STORE_MODELS)) {
        db.createObjectStore(STORE_MODELS, {
          keyPath: "id",
          autoIncrement: true,
        });
      }
      if (!db.objectStoreNames.contains(STORE_SKELETON_MODELS)) {
        db.createObjectStore(STORE_SKELETON_MODELS, {
          keyPath: "id",
          autoIncrement: true,
        });
      }
      // v4: 骨架姿态采集独立 store
      if (!db.objectStoreNames.contains(STORE_SKELETON_SAMPLES)) {
        const skStore = db.createObjectStore(STORE_SKELETON_SAMPLES, {
          keyPath: "id",
          autoIncrement: true,
        });
        skStore.createIndex("gesture", "gesture", { unique: false });
        skStore.createIndex("timestamp", "timestamp", { unique: false });
      }
      // v6: 时序样本 store（动态手语词）
      if (!db.objectStoreNames.contains(STORE_SEQUENCES)) {
        const seqStore = db.createObjectStore(STORE_SEQUENCES, {
          keyPath: "id",
          autoIncrement: true,
        });
        seqStore.createIndex("primaryLabel", "primaryLabel", { unique: false });
        seqStore.createIndex("timestamp", "timestamp", { unique: false });
        seqStore.createIndex("origin", "origin", { unique: false });
      }
    };
    request.onsuccess = () => {
      console.log("[DatasetStore] DB opened successfully, version:", request.result.version);
      resolve(request.result);
    };
    request.onerror = () => {
      console.error("[DatasetStore] DB open error:", request.error);
      reject(request.error);
    };
  });
}

// ===== 样本操作 =====

export async function addSample(sample: TrainingSample): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readwrite");
    const store = tx.objectStore(STORE_SAMPLES);
    const request = store.add(sample);
    request.onsuccess = () => resolve(request.result as number);
    request.onerror = () => reject(request.error);
  });
}

export async function addSamples(samples: TrainingSample[]): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readwrite");
    const store = tx.objectStore(STORE_SAMPLES);
    for (const sample of samples) {
      store.add(sample);
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getSamplesByLabel(
  label: string
): Promise<TrainingSample[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readonly");
    const store = tx.objectStore(STORE_SAMPLES);
    const index = store.index("label");
    const request = index.getAll(label);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getAllSamples(): Promise<TrainingSample[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readonly");
    const store = tx.objectStore(STORE_SAMPLES);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function deleteSamplesByLabel(label: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readwrite");
    const store = tx.objectStore(STORE_SAMPLES);
    const index = store.index("label");
    const request = index.openCursor(label);
    request.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearAllSamples(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readwrite");
    const store = tx.objectStore(STORE_SAMPLES);
    const request = store.clear();
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const samples = await getAllSamples();
  const labelCounts: Record<string, number> = {};
  for (const s of samples) {
    labelCounts[s.label] = (labelCounts[s.label] || 0) + 1;
  }
  return {
    totalSamples: samples.length,
    labelCounts,
    labels: Object.keys(labelCounts),
  };
}

export async function getSampleCount(): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SAMPLES, "readonly");
    const store = tx.objectStore(STORE_SAMPLES);
    const request = store.count();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// ===== 时序样本操作 =====

export async function addSequence(seq: SequenceSample): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readwrite");
    const request = tx.objectStore(STORE_SEQUENCES).add(seq);
    request.onsuccess = () => resolve(request.result as number);
    request.onerror = () => reject(request.error);
  });
}

export async function addSequences(seqs: SequenceSample[]): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readwrite");
    const store = tx.objectStore(STORE_SEQUENCES);
    for (const s of seqs) store.add(s);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getAllSequences(): Promise<SequenceSample[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readonly");
    const request = tx.objectStore(STORE_SEQUENCES).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getSequencesByLabel(
  label: string
): Promise<SequenceSample[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readonly");
    const index = tx.objectStore(STORE_SEQUENCES).index("primaryLabel");
    const request = index.getAll(label);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * 删除单条序列。序列样本比静态样本贵得多（录一条要摆位+做动作），
 * 录废了必须能单独剔掉而不是整个标签重录。
 */
export async function deleteSequence(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readwrite");
    const request = tx.objectStore(STORE_SEQUENCES).delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function deleteSequencesByLabel(label: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readwrite");
    const index = tx.objectStore(STORE_SEQUENCES).index("primaryLabel");
    const request = index.openCursor(label);
    request.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** 只删合成样本（重录真实静态词序列后做校准替换时用） */
export async function deleteSynthesizedSequences(): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    let deleted = 0;
    const tx = db.transaction(STORE_SEQUENCES, "readwrite");
    const index = tx.objectStore(STORE_SEQUENCES).index("origin");
    const request = index.openCursor("synthesized");
    request.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor) {
        cursor.delete();
        deleted++;
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve(deleted);
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearAllSequences(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readwrite");
    const request = tx.objectStore(STORE_SEQUENCES).clear();
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

function sequenceByteSize(s: SequenceSample): number {
  const arrays = [
    s.timestamps,
    s.leftSensor,
    s.rightSensor,
    s.leftImu,
    s.rightImu,
    s.leftLandmarks,
    s.rightLandmarks,
  ];
  let total = 0;
  for (const a of arrays) if (a) total += a.byteLength;
  return total;
}

export async function getSequenceStats(): Promise<SequenceStats> {
  const seqs = await getAllSequences();
  const labelCounts: SequenceStats["labelCounts"] = {};
  let recordedCount = 0;
  let synthesizedCount = 0;
  let totalDuration = 0;
  let estimatedBytes = 0;
  let sentenceCount = 0;
  const handCounts = { leftOnly: 0, rightOnly: 0, both: 0, neither: 0 };

  for (const s of seqs) {
    // 判据用 sensor/imu 是否为 null，与特征层"缺手那段保持全 0"完全同一口径
    const hasL = !!(s.leftSensor || s.leftImu);
    const hasR = !!(s.rightSensor || s.rightImu);
    if (hasL && hasR) handCounts.both++;
    else if (hasL) handCounts.leftOnly++;
    else if (hasR) handCounts.rightOnly++;
    else handCounts.neither++;
    if (s.origin === "synthesized") synthesizedCount++;
    else recordedCount++;
    totalDuration += s.durationMs;
    estimatedBytes += sequenceByteSize(s);

    // 只有孤立词进每词计数。句子的 primaryLabel 是它的第一个词，算进去就是给那个词
    // 虚增条数 —— 采集页会据此显示"这个词够了"，而其实一条孤立词都没多
    if (isSentenceSample(s)) {
      sentenceCount++;
      continue;
    }
    const entry = (labelCounts[s.primaryLabel] ??= {
      recorded: 0,
      synthesized: 0,
    });
    if (s.origin === "synthesized") entry.synthesized++;
    else entry.recorded++;
  }

  return {
    // 总数/真实合成/分手别/平均时长都算**全部**样本（句子录制也是一条真录制）；
    // 只有 labelCounts 把句子排除掉。所以 labelCounts 之和 = 总数 − sentenceCount
    totalSequences: seqs.length,
    recordedCount,
    synthesizedCount,
    sentenceCount,
    labelCounts,
    labels: Object.keys(labelCounts),
    avgDurationMs: seqs.length ? totalDuration / seqs.length : 0,
    estimatedBytes,
    handCounts,
  };
}

export async function getSequenceCount(): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SEQUENCES, "readonly");
    const request = tx.objectStore(STORE_SEQUENCES).count();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// ===== 模型操作 =====

export async function saveModel(model: SavedModel): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_MODELS, "readwrite");
    const store = tx.objectStore(STORE_MODELS);
    const request = store.add(model);
    request.onsuccess = () => resolve(request.result as number);
    request.onerror = () => reject(request.error);
  });
}

export async function getAllModels(): Promise<SavedModel[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_MODELS, "readonly");
    const store = tx.objectStore(STORE_MODELS);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getLatestModel(): Promise<SavedModel | null> {
  // 语义不变：只返回单帧静态模型，供 Translate 的旧路径使用。
  // v6 起同一个 store 里也存时序模型，这里必须把它们排除掉，
  // 否则回退分支会把序列模型喂给静态推理路径（输入形状对不上）。
  const models = (await getAllModels()).filter((m) => !isSequenceModel(m));
  // 优先返回 tactile 类型的模型（用于推理）
  const tactileModels = models.filter((m) => m.modelType === "tactile");
  if (tactileModels.length > 0) {
    return tactileModels.sort((a, b) => b.createdAt - a.createdAt)[0];
  }
  // 兼容旧模型
  if (models.length === 0) return null;
  return models.sort((a, b) => b.createdAt - a.createdAt)[0];
}

/** 最新的时序模型：优先 seq_tactile（部署用学生），无则退回 seq_fused */
export async function getLatestSequenceModel(): Promise<SavedModel | null> {
  const models = await getAllModels();
  const students = models.filter((m) => m.modelType === "seq_tactile");
  if (students.length > 0) {
    return students.sort((a, b) => b.createdAt - a.createdAt)[0];
  }
  const teachers = models.filter((m) => m.modelType === "seq_fused");
  if (teachers.length === 0) return null;
  return teachers.sort((a, b) => b.createdAt - a.createdAt)[0];
}

export async function getAllSequenceModels(): Promise<SavedModel[]> {
  const models = await getAllModels();
  return models
    .filter(isSequenceModel)
    .sort((a, b) => b.createdAt - a.createdAt);
}

export async function deleteModel(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_MODELS, "readwrite");
    const store = tx.objectStore(STORE_MODELS);
    const request = store.delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// ===== 导入/导出 =====

export async function exportDatasetJSON(): Promise<string> {
  const samples = await getAllSamples();
  return JSON.stringify({
    version: "3.0",
    exportedAt: new Date().toISOString(),
    totalSamples: samples.length,
    schema: {
      left: "左手 { sensor_data:137, quaternion:[w,x,y,z], landmarks:21点 } 或 null",
      right: "右手 { sensor_data:137, quaternion:[w,x,y,z], landmarks:21点 } 或 null",
      note: "缺失的手为 null；特征向量对应位置填 0",
    },
    samples: samples.map(({ id, ...rest }) => rest),
  }, null, 2);
}

export async function importDatasetJSON(jsonStr: string): Promise<number> {
  const data = JSON.parse(jsonStr);
  if (!data.samples || !Array.isArray(data.samples)) {
    throw new Error("Invalid dataset format");
  }
  await addSamples(data.samples);
  return data.samples.length;
}

// ===== 序列数据集导出（给 Python 训练用） =====

/** manifest 里一个 TypedArray 的定位信息 */
export interface SeqArrayRef {
  offset: number; // 字节偏移（相对 dataset.bin 起点）
  length: number; // 元素个数（不是字节数）
  dtype: "uint8" | "float32";
}

export interface SeqManifestEntry {
  segments: SequenceSegment[];
  primaryLabel: string;
  frameCount: number;
  durationMs: number;
  sourceFps: number;
  origin: "recorded" | "synthesized";
  timestamp: number;
  arrays: Record<string, SeqArrayRef | null>;
}

export interface SeqManifest {
  version: string;
  exportedAt: string;
  totalSequences: number;
  sensorN: number;
  imuN: number;
  landmarkN: number;
  labels: string[];
  /** 各数组的 per-frame 宽度，Python 侧 reshape 用 */
  arrayLayout: Record<string, { perFrame: number; dtype: string }>;
  sequences: SeqManifestEntry[];
}

const SEQ_ARRAY_KEYS = [
  "timestamps",
  "leftSensor",
  "rightSensor",
  "leftImu",
  "rightImu",
  "leftLandmarks",
  "rightLandmarks",
] as const;

type SeqArrayKey = (typeof SEQ_ARRAY_KEYS)[number];

const SEQ_ARRAY_LAYOUT: Record<
  SeqArrayKey,
  { perFrame: number; dtype: "uint8" | "float32" }
> = {
  timestamps: { perFrame: 1, dtype: "float32" },
  leftSensor: { perFrame: SEQ_SENSOR_N, dtype: "uint8" },
  rightSensor: { perFrame: SEQ_SENSOR_N, dtype: "uint8" },
  leftImu: { perFrame: SEQ_IMU_N, dtype: "float32" },
  rightImu: { perFrame: SEQ_IMU_N, dtype: "float32" },
  leftLandmarks: { perFrame: SEQ_LANDMARK_N, dtype: "float32" },
  rightLandmarks: { perFrame: SEQ_LANDMARK_N, dtype: "float32" },
};

/**
 * 导出为 dataset.bin + dataset.json 两个文件。
 *
 * 不导出成纯 JSON 数组：600 条 × 64KB 二进制展开成 JSON 文本约 150MB，
 * 生成和解析都不可用。Python 侧 np.frombuffer(buf, dtype, count, offset)
 * 按 offset 切片即可，零拷贝。
 *
 * Float32 段按 4 字节对齐（在 uint8 段之后补 padding），
 * 否则 numpy 在部分平台上 frombuffer 会因未对齐而报错。
 */
export async function exportSequencesBinary(): Promise<{
  bin: ArrayBuffer;
  manifest: SeqManifest;
}> {
  return encodeSequencesBinary(await getAllSequences());
}

/** exportSequencesBinary 的纯函数内核（不碰 IndexedDB，可单测 round-trip） */
export function encodeSequencesBinary(seqs: SequenceSample[]): {
  bin: ArrayBuffer;
  manifest: SeqManifest;
} {
  // 第一遍：算偏移与总长
  let cursor = 0;
  const entries: SeqManifestEntry[] = [];
  const plan: Array<Array<{ offset: number; array: ArrayBufferView }>> = [];

  for (const s of seqs) {
    const arrays: Record<string, SeqArrayRef | null> = {};
    const items: Array<{ offset: number; array: ArrayBufferView }> = [];
    for (const key of SEQ_ARRAY_KEYS) {
      const arr = s[key] as Uint8Array | Float32Array | null;
      if (!arr) {
        arrays[key] = null;
        continue;
      }
      const dtype = SEQ_ARRAY_LAYOUT[key].dtype;
      if (dtype === "float32" && cursor % 4 !== 0) {
        cursor += 4 - (cursor % 4); // 对齐 padding
      }
      arrays[key] = { offset: cursor, length: arr.length, dtype };
      items.push({ offset: cursor, array: arr });
      cursor += arr.byteLength;
    }
    plan.push(items);
    entries.push({
      segments: s.segments,
      primaryLabel: s.primaryLabel,
      frameCount: s.frameCount,
      durationMs: s.durationMs,
      sourceFps: s.sourceFps,
      origin: s.origin,
      timestamp: s.timestamp,
      arrays,
    });
  }

  // 第二遍：实际写入
  const bin = new ArrayBuffer(cursor);
  const view = new Uint8Array(bin);
  for (const items of plan) {
    for (const { offset, array } of items) {
      view.set(
        new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
        offset
      );
    }
  }

  const labels = Array.from(new Set(seqs.map((s) => s.primaryLabel))).sort();

  return {
    bin,
    manifest: {
      version: "seq-1.0",
      exportedAt: new Date().toISOString(),
      totalSequences: seqs.length,
      sensorN: SEQ_SENSOR_N,
      imuN: SEQ_IMU_N,
      landmarkN: SEQ_LANDMARK_N,
      labels,
      arrayLayout: SEQ_ARRAY_LAYOUT,
      sequences: entries,
    },
  };
}

/** 二进制导出的逆操作，用于 round-trip 校验与跨机器迁移 */
export function decodeSequencesBinary(
  bin: ArrayBuffer,
  manifest: SeqManifest
): SequenceSample[] {
  return manifest.sequences.map((e) => {
    const read = (key: SeqArrayKey) => {
      const ref = e.arrays[key];
      if (!ref) return null;
      // slice 而非视图：IndexedDB structured clone 会连带整个大 buffer
      return ref.dtype === "uint8"
        ? new Uint8Array(bin.slice(ref.offset, ref.offset + ref.length))
        : new Float32Array(bin.slice(ref.offset, ref.offset + ref.length * 4));
    };
    return {
      segments: e.segments,
      primaryLabel: e.primaryLabel,
      frameCount: e.frameCount,
      timestamps: read("timestamps") as Float32Array,
      leftSensor: read("leftSensor") as Uint8Array | null,
      rightSensor: read("rightSensor") as Uint8Array | null,
      leftImu: read("leftImu") as Float32Array | null,
      rightImu: read("rightImu") as Float32Array | null,
      leftLandmarks: read("leftLandmarks") as Float32Array | null,
      rightLandmarks: read("rightLandmarks") as Float32Array | null,
      durationMs: e.durationMs,
      sourceFps: e.sourceFps,
      origin: e.origin,
      timestamp: e.timestamp,
    };
  });
}

// ===== 骨架回归模型操作 =====

export interface SavedSkeletonModelRecord {
  id?: number;
  name: string;
  createdAt: number;
  valLoss: number;
  valMae: number;
  modelJson: string;
  weightsData: ArrayBuffer;
}

export async function saveSkeletonModel(
  model: SavedSkeletonModelRecord
): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SKELETON_MODELS, "readwrite");
    const store = tx.objectStore(STORE_SKELETON_MODELS);
    const request = store.add(model);
    request.onsuccess = () => resolve(request.result as number);
    request.onerror = () => reject(request.error);
  });
}

export async function getAllSkeletonModels(): Promise<
  SavedSkeletonModelRecord[]
> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SKELETON_MODELS, "readonly");
    const store = tx.objectStore(STORE_SKELETON_MODELS);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getLatestSkeletonModel(): Promise<SavedSkeletonModelRecord | null> {
  const models = await getAllSkeletonModels();
  if (models.length === 0) return null;
  return models.sort((a, b) => b.createdAt - a.createdAt)[0];
}

export async function deleteSkeletonModel(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_SKELETON_MODELS, "readwrite");
    const store = tx.objectStore(STORE_SKELETON_MODELS);
    const request = store.delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}
