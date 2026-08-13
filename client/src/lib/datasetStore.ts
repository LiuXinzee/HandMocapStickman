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

const DB_NAME = "hand_mocap_dataset";
const DB_VERSION = 6; // v6: 新增动态手势序列/模型 store；保留 v5 静态数据
const STORE_SAMPLES = "samples";
const STORE_MODELS = "models";
const STORE_SKELETON_MODELS = "skeleton_models";
const STORE_SKELETON_SAMPLES = "skeleton_samples"; // 骨架姿态采集样本（独立于手语样本）
const STORE_DYNAMIC_SEQUENCES = "dynamic_sequences";
const STORE_DYNAMIC_MODELS = "dynamic_models";

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
  return s.left?.landmarks?.length === 21 || s.right?.landmarks?.length === 21;
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
  modelType: "fused" | "tactile"; // 模型类型标记
}

export interface DatasetStats {
  totalSamples: number;
  labelCounts: Record<string, number>;
  labels: string[];
}

/** 动态序列中的单帧手套数据，时间均基于同一 performance.now() 时钟。 */
export interface DynamicGloveFrameRecord {
  relativeTimeMs: number;
  timestamp: number;
  frameId: number;
  hand: number;
  sensorData: number[];
  quaternion: [number, number, number, number];
  acceleration: [number, number, number] | null;
  attitude: [number, number, number] | null;
}

/** 动态序列中的视觉骨架帧。 */
export interface DynamicVisionFrameRecord {
  relativeTimeMs: number;
  timestamp: number;
  landmarks: HandLandmarkPoint[][] | null;
  handedness: string[];
  trackingIds: string[];
  surfaces: Array<"palm" | "back" | "unknown">;
  surfaceConfidences: number[];
  gloveConfidences: number[];
  detectionSources: Array<"standard" | "glove-enhanced">;
}

/** 训练预处理阶段可生成的统一时间轴帧；原始存储仍保留三路独立流。 */
export interface DynamicSequenceFrame {
  relativeTimeMs: number;
  left: DynamicGloveFrameRecord | null;
  right: DynamicGloveFrameRecord | null;
  vision: DynamicVisionFrameRecord | null;
  leftValid: boolean;
  rightValid: boolean;
  visionValid: boolean;
}

/** 一次完整动态词语动作；三路原始流在训练前再统一重采样。 */
export interface DynamicGestureSequence {
  id?: number;
  schemaVersion: 1;
  label: string;
  sessionId: string;
  startedAt: number;
  durationMs: number;
  targetDurationMs: number;
  leftFrames: DynamicGloveFrameRecord[];
  rightFrames: DynamicGloveFrameRecord[];
  visionFrames: DynamicVisionFrameRecord[];
}

export interface DynamicDatasetStats {
  totalSequences: number;
  totalFrames: number;
  totalDurationMs: number;
  labelCounts: Record<string, number>;
  labels: string[];
}

/** 序列 TCN 模型的 IndexedDB 存储记录。 */
export interface SavedDynamicModelRecord {
  id?: number;
  name: string;
  createdAt: number;
  accuracy: number;
  labels: string[];
  modelJson: string;
  weightsData: ArrayBuffer;
  modelType: "tcn";
  sampleRateHz?: number;
  sequenceLength?: number;
  featureDim?: number;
  validationLoss?: number;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = event => {
      const db = (event.target as IDBOpenDBRequest).result;
      const oldVersion = (event as IDBVersionChangeEvent).oldVersion;
      console.log(
        "[DatasetStore] Upgrading DB from version",
        oldVersion,
        "to",
        DB_VERSION
      );
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
        console.warn(
          "[DatasetStore] v5 升级：已清空旧的单手样本，请重新采集双手数据"
        );
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
      // v6: 动态词语以完整序列保存，不与静态单帧样本混用。
      if (!db.objectStoreNames.contains(STORE_DYNAMIC_SEQUENCES)) {
        const dynamicStore = db.createObjectStore(STORE_DYNAMIC_SEQUENCES, {
          keyPath: "id",
          autoIncrement: true,
        });
        dynamicStore.createIndex("label", "label", { unique: false });
        dynamicStore.createIndex("startedAt", "startedAt", { unique: false });
        dynamicStore.createIndex("sessionId", "sessionId", { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_DYNAMIC_MODELS)) {
        const dynamicModelStore = db.createObjectStore(STORE_DYNAMIC_MODELS, {
          keyPath: "id",
          autoIncrement: true,
        });
        dynamicModelStore.createIndex("createdAt", "createdAt", {
          unique: false,
        });
      }
    };
    request.onsuccess = () => {
      console.log(
        "[DatasetStore] DB opened successfully, version:",
        request.result.version
      );
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
    request.onerror = () => {
      console.error("[DatasetStore] DB open error:", request.error);
      reject(request.error);
    };
    request.onblocked = () => {
      console.warn("[DatasetStore] DB upgrade blocked by another open tab");
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
    request.onsuccess = event => {
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

// ===== 动态序列操作 =====

export async function addDynamicSequence(
  sequence: DynamicGestureSequence
): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readwrite");
    const request = tx.objectStore(STORE_DYNAMIC_SEQUENCES).add(sequence);
    let insertedId: number | null = null;
    request.onsuccess = () => {
      insertedId = request.result as number;
    };
    tx.oncomplete = () => {
      if (insertedId === null) {
        reject(
          new Error("Dynamic sequence transaction completed without an id")
        );
        return;
      }
      resolve(insertedId);
    };
    tx.onerror = () => reject(tx.error ?? request.error);
    tx.onabort = () =>
      reject(tx.error ?? new Error("Dynamic sequence save aborted"));
  });
}

export async function addDynamicSequences(
  sequences: DynamicGestureSequence[]
): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readwrite");
    const store = tx.objectStore(STORE_DYNAMIC_SEQUENCES);
    for (const sequence of sequences) store.add(sequence);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getAllDynamicSequences(): Promise<
  DynamicGestureSequence[]
> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readonly");
    const request = tx.objectStore(STORE_DYNAMIC_SEQUENCES).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getDynamicSequencesByLabel(
  label: string
): Promise<DynamicGestureSequence[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readonly");
    const index = tx.objectStore(STORE_DYNAMIC_SEQUENCES).index("label");
    const request = index.getAll(label);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function deleteDynamicSequence(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readwrite");
    const request = tx.objectStore(STORE_DYNAMIC_SEQUENCES).delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function deleteDynamicSequencesByLabel(
  label: string
): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readwrite");
    const store = tx.objectStore(STORE_DYNAMIC_SEQUENCES);
    const request = store.index("label").openCursor(label);
    request.onsuccess = event => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (!cursor) return;
      cursor.delete();
      cursor.continue();
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearAllDynamicSequences(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_SEQUENCES, "readwrite");
    const request = tx.objectStore(STORE_DYNAMIC_SEQUENCES).clear();
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function getDynamicDatasetStats(): Promise<DynamicDatasetStats> {
  const sequences = await getAllDynamicSequences();
  const labelCounts: Record<string, number> = {};
  let totalFrames = 0;
  let totalDurationMs = 0;

  for (const sequence of sequences) {
    labelCounts[sequence.label] = (labelCounts[sequence.label] || 0) + 1;
    totalFrames +=
      sequence.leftFrames.length +
      sequence.rightFrames.length +
      sequence.visionFrames.length;
    totalDurationMs += sequence.durationMs;
  }

  return {
    totalSequences: sequences.length,
    totalFrames,
    totalDurationMs,
    labelCounts,
    labels: Object.keys(labelCounts),
  };
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
  const models = await getAllModels();
  // 优先返回 tactile 类型的模型（用于推理）
  const tactileModels = models.filter(m => m.modelType === "tactile");
  if (tactileModels.length > 0) {
    return tactileModels.sort((a, b) => b.createdAt - a.createdAt)[0];
  }
  // 兼容旧模型
  if (models.length === 0) return null;
  return models.sort((a, b) => b.createdAt - a.createdAt)[0];
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

// ===== 动态 TCN 模型操作 =====

export async function saveDynamicModel(
  model: SavedDynamicModelRecord
): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_MODELS, "readwrite");
    const request = tx.objectStore(STORE_DYNAMIC_MODELS).add(model);
    let insertedId: number | null = null;
    request.onsuccess = () => {
      insertedId = request.result as number;
    };
    tx.oncomplete = () => {
      if (insertedId === null) {
        reject(new Error("Dynamic model transaction completed without an id"));
        return;
      }
      resolve(insertedId);
    };
    tx.onerror = () => reject(tx.error ?? request.error);
    tx.onabort = () =>
      reject(tx.error ?? new Error("Dynamic model save aborted"));
  });
}

export async function getAllDynamicModels(): Promise<
  SavedDynamicModelRecord[]
> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_MODELS, "readonly");
    const request = tx.objectStore(STORE_DYNAMIC_MODELS).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getLatestDynamicModel(): Promise<SavedDynamicModelRecord | null> {
  const models = await getAllDynamicModels();
  if (models.length === 0) return null;
  return models.sort((a, b) => b.createdAt - a.createdAt)[0];
}

export async function deleteDynamicModel(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DYNAMIC_MODELS, "readwrite");
    const request = tx.objectStore(STORE_DYNAMIC_MODELS).delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// ===== 导入/导出 =====

export async function exportDatasetJSON(): Promise<string> {
  const samples = await getAllSamples();
  return JSON.stringify(
    {
      version: "3.0",
      exportedAt: new Date().toISOString(),
      totalSamples: samples.length,
      schema: {
        left: "左手 { sensor_data:137, quaternion:[w,x,y,z], landmarks:21点 } 或 null",
        right:
          "右手 { sensor_data:137, quaternion:[w,x,y,z], landmarks:21点 } 或 null",
        note: "缺失的手为 null；特征向量对应位置填 0",
      },
      samples: samples.map(({ id, ...rest }) => rest),
    },
    null,
    2
  );
}

export async function importDatasetJSON(jsonStr: string): Promise<number> {
  const data = JSON.parse(jsonStr);
  if (!data.samples || !Array.isArray(data.samples)) {
    throw new Error("Invalid dataset format");
  }
  await addSamples(data.samples);
  return data.samples.length;
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
