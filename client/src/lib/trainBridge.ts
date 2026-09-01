/**
 * 训练桥的浏览器端。对面是 `vite-plugin-train-bridge.ts`（仓库根）。
 *
 * 桥**只在 `npm run dev` 存在**。生产构建里插件不挂载（`apply: "serve"`），
 * 所以本模块的每个入口都要先过 `bridgeConfig()`：拿不到就是没有桥，
 * 界面必须显式禁用并写明原因，而不是让按钮点了没反应。
 */
import type { SeqManifest } from "./datasetStore";

interface BridgeConfig {
  token: string;
  base: string;
}

declare global {
  interface Window {
    __TRAIN_BRIDGE__?: BridgeConfig;
  }
}

/** 没有桥时返回 null。调用方**必须**处理这一支 */
export function bridgeConfig(): BridgeConfig | null {
  const c = typeof window !== "undefined" ? window.__TRAIN_BRIDGE__ : undefined;
  return c && typeof c.token === "string" && typeof c.base === "string" ? c : null;
}

export function bridgeAvailable(): boolean {
  return bridgeConfig() !== null;
}

/** 桥不在时统一的说法，界面直接显示这句 */
export const BRIDGE_ABSENT_REASON =
  "训练桥不可用。它只在 `npm run dev` 的 Vite dev server 里挂载 —— " +
  "生产构建里没有这些路由（「收文件 + 起进程」不该出现在部署产物里）。" +
  "改用下载按钮，手动拷进 python_train/data/。";

class BridgeError extends Error {}

async function call(
  path: string,
  init: RequestInit & { raw?: BodyInit } = {}
): Promise<Response> {
  const cfg = bridgeConfig();
  if (!cfg) throw new BridgeError(BRIDGE_ABSENT_REASON);
  const res = await fetch(cfg.base + path, {
    ...init,
    headers: { ...init.headers, "X-Train-Token": cfg.token },
  });
  if (!res.ok) {
    // 桥的错误体一律是 { error }。读不出来就退回状态码 —— 别让一个 403
    // 变成一句 "Unexpected token < in JSON"
    let msg = `HTTP ${res.status}`;
    try {
      const j = (await res.json()) as { error?: string };
      if (j?.error) msg = j.error;
    } catch {
      /* 保留状态码 */
    }
    throw new BridgeError(msg);
  }
  return res;
}

// ===== 数据集直送 =====

/**
 * 把 manifest + bin 一次传过去，桥直接落到 `python_train/data/`。
 *
 * 帧格式 `[4 字节小端 json 长度][json][bin]`：比 multipart 简单得多，而且两个文件
 * 在同一个请求里 —— 半个 json 配旧 bin 这种组合根本构造不出来。
 */
export async function pushDataset(
  manifest: SeqManifest,
  bin: ArrayBuffer
): Promise<{ binBytes: number; jsonBytes: number; dataDir: string }> {
  const json = new TextEncoder().encode(JSON.stringify(manifest, null, 2));
  const head = new Uint8Array(4);
  new DataView(head.buffer).setUint32(0, json.byteLength, true);
  const body = new Blob([head, json, bin]);
  const res = await call("/dataset", { method: "POST", body });
  return (await res.json()) as { binBytes: number; jsonBytes: number; dataDir: string };
}

// ===== 起 / 停 / 状态 =====

export interface RunOptions {
  target: "ctc" | "distill" | "export";
  epochs?: number;
  noSynth?: boolean;
  synthPerTemplate?: number;
  noTrim?: boolean;
}

export async function startTraining(
  opts: RunOptions
): Promise<{ runId: string; argv: string[] }> {
  const res = await call("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  return (await res.json()) as { runId: string; argv: string[] };
}

export async function stopTraining(): Promise<void> {
  await call("/stop", { method: "POST" });
}

// ===== 导出到浏览器 =====

/**
 * 让桥去跑 `export_weights.py`，把 `out/sentence_student.keras` 的权重导到
 * `client/public/models/seq_sentence/` —— 也就是 `sentenceModel.ts` 读的那个目录。
 *
 * 权重**不进 IndexedDB**。浏览器自己训的孤立词模型才在 IndexedDB 里；Python 训的
 * 是构建产物（见 sentenceModel.ts 顶部）。导到 public/ 的另一个好处是这份产物
 * 同时就是要提交/部署的那份，不存在"页面上能用但部署出去没有"。
 *
 * ⚠ **导完 Vite 会整页刷新**。publicDir 在 watch 范围里，写文件会触发 full-reload。
 * 这不是出错 —— 日志靠 `getStatus()` 接得回来。
 *
 * 与训练共用同一个进程槽位：训练还在往 .keras 写的时候导出会读到半个文件，
 * 桥那边直接 409 挡掉。
 */
export async function exportWeights(): Promise<{ runId: string; argv: string[] }> {
  const res = await call("/export", { method: "POST" });
  return (await res.json()) as { runId: string; argv: string[] };
}

export interface DeployedModel {
  exists: boolean;
  modelDir: string;
  /** weights.json 的 mtime */
  deployedAt?: number;
  bytes?: number;
  /** out/sentence_student.keras 的 mtime。比 deployedAt 新 = 训完还没导 */
  trainedAt?: number | null;
  meta?: {
    labels?: string[];
    valWer?: number;
    numReal?: number;
    blankIndex?: number;
    synthOnly?: boolean;
  } | null;
}

export async function getDeployedModel(): Promise<DeployedModel | null> {
  if (!bridgeAvailable()) return null;
  try {
    return (await (await call("/model")).json()) as DeployedModel;
  } catch {
    return null;
  }
}

export interface BridgeStatus {
  running: boolean;
  runId: string | null;
  target: string | null;
  argv: string[] | null;
  startedAt: number | null;
  exitCode: number | null;
  error: string | null;
  /** 到目前为止的全部日志。SSE 自己不带历史，页面刷新后靠这个接回来 */
  lines: string[];
}

export async function getStatus(): Promise<BridgeStatus> {
  return (await call("/status")).json() as Promise<BridgeStatus>;
}

export async function ping(): Promise<{
  pythonReady: boolean;
  pythonError: string | null;
  dataDir: string;
}> {
  return (await call("/ping")).json() as Promise<{
    pythonReady: boolean;
    pythonError: string | null;
    dataDir: string;
  }>;
}

/**
 * 订阅实时日志。返回退订函数。
 *
 * `EventSource` 带不了自定义头，所以 token 走查询串。桥两边都收（头或查询串）——
 * 回环校验那道防护与 token 无关，仍然生效。
 */
export function subscribeLog(
  onLine: (line: string) => void,
  onDone: (info: { exitCode: number | null; error: string | null }) => void
): () => void {
  const cfg = bridgeConfig();
  if (!cfg) return () => {};
  const es = new EventSource(
    `${cfg.base}/stream?token=${encodeURIComponent(cfg.token)}`
  );
  es.addEventListener("line", (e) => onLine(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("done", (e) => onDone(JSON.parse((e as MessageEvent).data)));
  return () => es.close();
}

// ===== 词骨干（CTC 的前置步骤）=====

/** 桥回的原始事实。判断在 `backboneVerdict()` 里做 */
export interface BackboneInfo {
  /** `python_train/out/student.keras` 在不在 */
  exists: boolean;
  trainedAt: number | null;
  /** `python_train/data/dataset.json` 的 mtime */
  datasetAt: number | null;
  /** 词模型训过的类别（原始标签，未合并）。查不到时 null */
  backboneLabels: string[] | null;
  /** 直送过去的数据集里的标签。还没直送过时 null */
  datasetLabels: string[] | null;
  seqLen: number | null;
}

export async function getBackbone(): Promise<BackboneInfo | null> {
  if (!bridgeAvailable()) return null;
  try {
    return (await (await call("/backbone")).json()) as BackboneInfo;
  } catch {
    return null;
  }
}

export type BackboneLevel = "ok" | "stale" | "missing" | "unknown";

export interface BackboneVerdict {
  level: BackboneLevel;
  /** 数据集里有、骨干没训过的词。level 为 "stale" 时非空 */
  missing: string[];
  /** 一句话说明，直接显示 */
  detail: string;
}

/**
 * 判定词骨干够不够用。
 *
 * **判据是标签差集，不是时间戳。** mtime 只能说"旧了"，而旧可能只是同一批词
 * 多录了几条（那种情况迁移仍然有效）；标签差集直接给出词的名字，
 * 也就是「骨干没见过漂亮」这句话里的那个词。
 *
 * `datasetAt > trainedAt` 只降级成附注、不单独构成 stale：直送完还没重训骨干
 * 是常态，多数时候无害。
 *
 * `UNTRAINED_WORDS` 必须减掉 —— 那几个词在数据集里但**故意**不训（见
 * sentenceTemplates.ts），不减的话页面上会永久挂着一条假告警。
 */
export function backboneVerdict(
  info: BackboneInfo | null,
  untrainedWords: readonly string[]
): BackboneVerdict {
  if (!info) {
    return { level: "unknown", missing: [], detail: "查不到骨干状态（桥不可用）。" };
  }
  if (!info.exists) {
    return {
      level: "missing",
      missing: [],
      detail:
        "找不到 out/student.keras。CTC 会从随机初始化起跑 —— " +
        "几百条句子基本训不动，表现是 loss 半天不降而不报错。",
    };
  }
  if (!info.backboneLabels || !info.datasetLabels) {
    return {
      level: "unknown",
      missing: [],
      detail: !info.datasetLabels
        ? "还没直送过数据集，比不出骨干缺哪些词。先去 /train-seq 点「直送」。"
        : "读不到 out/student_meta.json，比不出骨干缺哪些词。",
    };
  }
  const drop = new Set(untrainedWords);
  const trained = new Set(info.backboneLabels);
  const missing = info.datasetLabels
    .filter((l) => !drop.has(l) && !trained.has(l))
    .sort();
  if (missing.length) {
    return {
      level: "stale",
      missing,
      detail:
        `骨干没见过：${missing.join("、")}。conv/bn 的形状不随类别数变，` +
        "所以迁移照样会打印「成功」—— 这些词只会表现为 WER 特别差。",
    };
  }
  const behind =
    info.datasetAt !== null && info.trainedAt !== null && info.datasetAt > info.trainedAt;
  return {
    level: "ok",
    missing: [],
    detail: behind
      ? `${info.backboneLabels.length} 类全都训过。（数据集比骨干新 —— ` +
        "同一批词多录了几条，迁移仍然有效。)"
      : `${info.backboneLabels.length} 类全都训过。`,
  };
}

// ===== 结构化事件 =====

export interface DataEvent {
  kind: "data";
  classes: string[];
  blankIndex: number;
  orphans: string[];
  seqLen: number;
  outputFrames: number;
  numReal: number;
  numRealTrain: number;
  numRealVal: number;
  numSynthTrain: number;
  numSynthVal: number;
  numTrain: number;
  numVal: number;
  featShape: number[];
  maxLabelLen: number;
  trimmed: boolean;
  epochs: number;
}

export interface EpochEvent {
  kind: "epoch";
  epoch: number;
  total: number;
  loss: number;
  /**
   * 验证集为空时是 null。
   *
   * ⚠ **可以大于 1。** WER 的分母是参考词数，插词能让它超过 100%（实测未收敛时
   * 到过 248%）。图表的 WER 轴不能写死 0~1，否则前几个 epoch 会齐平顶在天花板上，
   * 看着像"一开始就 100% 全错然后突然好了"
   */
  valWer: number | null;
  saved: boolean;
  bestWer: number | null;
}

export interface ErrorsEvent {
  kind: "errors";
  nBad: number;
  nTotal: number;
  /** 错在第 1 个词的条数。句首错是时间包络问题，句中错是切词问题 */
  headBad: number;
  examples: Array<{ ref: string[]; hyp: string[] }>;
}

export interface DoneEvent {
  kind: "done";
  meta: Record<string, unknown>;
}

export type TrainEvent = DataEvent | EpochEvent | ErrorsEvent | DoneEvent;

export interface ParsedEvents {
  data: DataEvent | null;
  epochs: EpochEvent[];
  errors: ErrorsEvent | null;
  done: DoneEvent | null;
  /** 认不出来的行数。>0 说明 Python 那边加了新 kind 而这里还没跟上 */
  skipped: number;
}

/**
 * 解析 events JSONL。
 *
 * **三件事一件都不能抛**，因为三件事都会真的发生：
 *
 * 1. **最后一行写了一半。** 训练还在跑，我们边写边读。JSON.parse 会抛，
 *    而抛出去的后果是整张图表白屏 —— 只因为读得太快了半毫秒。
 * 2. **认不出的 `kind`。** Python 那边加事件不该让旧页面崩。
 * 3. **空文件。** 训练刚起来、还没写第一行。
 *
 * 所以：逐行 try/catch，不认识的记进 `skipped`，绝不 throw。
 */
export function parseEvents(text: string): ParsedEvents {
  const out: ParsedEvents = {
    data: null,
    epochs: [],
    errors: null,
    done: null,
    skipped: 0,
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    let ev: TrainEvent;
    try {
      ev = JSON.parse(line) as TrainEvent;
    } catch {
      out.skipped++; // 半行。下一次轮询它就完整了
      continue;
    }
    if (!ev || typeof ev !== "object") {
      out.skipped++;
      continue;
    }
    switch (ev.kind) {
      case "data":
        out.data = ev;
        break;
      case "epoch":
        out.epochs.push(ev);
        break;
      case "errors":
        out.errors = ev;
        break;
      case "done":
        out.done = ev;
        break;
      default:
        out.skipped++;
    }
  }
  return out;
}

/** 拉一次 events 文件并解析。没有桥或文件不存在时给一份空的 */
export async function fetchEvents(): Promise<ParsedEvents> {
  if (!bridgeAvailable()) return parseEvents("");
  try {
    const res = await call("/events");
    const ct = res.headers.get("Content-Type") ?? "";
    // 文件还不存在时桥回的是 JSON `{events:[]}`，不是 JSONL
    if (ct.includes("application/json")) return parseEvents("");
    return parseEvents(await res.text());
  } catch {
    return parseEvents("");
  }
}
