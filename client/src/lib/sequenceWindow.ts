/*
 * sequenceWindow — 实时滑窗推理用的环形缓冲
 *
 * 与 useSequenceRecorder 的区别：录制是"先攒完整段再重采样一次"，
 * 这里是"永远只保留最近几秒，每 100ms 切一段出来推理"，所以：
 *   - 缓冲有上限，超时的帧要丢掉，否则长时间开着翻译会一直涨内存
 *   - 触觉是必须的，视觉**默认关**（见下）
 *
 * 切片时重采样到与录制时相同的 50Hz 栅格。虽然 buildSequenceFeatures 之后
 * 还会再重采样到 T=32，但中间这一步不能省：左右手是两个独立 COM 口，
 * 各自的时间戳对不齐，必须先落到公共栅格上，两只手的第 t 帧才是"同一时刻"。
 *
 * ===== 视觉通道：默认关，只有句子**采集**页开 =====
 *
 * 部署时没有摄像头，所以推理端（`/translate`）不采视觉，`vision` 保持 false，
 * 切出来的样本 `leftLandmarks/rightLandmarks` 恒为 null —— 与这个开关加进来之前
 * 逐位相同。
 *
 * 句子采集页（`/collect-sentence`）打开它，但**视觉不会变成特征**：
 * `buildSequenceFeatures` 的 `includeVision` 默认 false，`synth_sentences.py` 合成时
 * 一律把视觉丢掉。关键点的用处只有两个，都是训练期的工具：
 *   1. `sequenceTrim` 的可见性判据（没有视觉时那一层直接判 `no_vision`）
 *   2. `signTransitions` 量真实句子里的词间过渡时长（合成端 `overlap_ms` 的实测依据）
 *
 * 视觉是**被动通道**：它不参与 `snapshotAll` 的区间计算，也不影响返回 null 的条件。
 * 摄像头没开、或整段都没检出手，取出来的样本与不开视觉时只差 landmarks 全是 NaN。
 */
import type { GloveFrame } from "./gloveProtocol";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "./datasetStore";

interface WindowEntry {
  t: number;
  sensor: number[];
  quat: [number, number, number, number];
  acc: [number, number, number] | null;
  att: [number, number, number] | null;
}

interface VisionEntry {
  t: number;
  left: Float32Array | null; // 63 维，见 visionLandmarks.ts
  right: Float32Array | null;
}

/**
 * 视觉最近邻的最大时间差，与 `useSequenceRecorder` 的 `visionMaxGapMs` 同值同理由：
 * 30Hz 视觉的帧间隔 33ms，取 50ms 容一帧抖动但不容跨两帧硬凑。
 * 超过就判这个栅格点没有视觉、填 NaN。
 */
const VISION_MAX_GAP_MS = 50;

export interface SequenceWindowOptions {
  /** 缓冲保留时长，需要 >= 推理窗口长度 */
  bufferMs?: number;
  /** 公共重采样栅格，必须与采集端一致 */
  gridFps?: number;
  /**
   * 是否留视觉通道。**默认 false** —— 推理端不采视觉，这个默认值保证
   * `/translate` 的行为与视觉通道加进来之前逐位相同。
   */
  vision?: boolean;
}

export class SequenceWindowBuffer {
  private left: WindowEntry[] = [];
  private right: WindowEntry[] = [];
  private visionBuf: VisionEntry[] = [];
  private readonly bufferMs: number;
  private readonly gridFps: number;
  private readonly vision: boolean;

  constructor(options: SequenceWindowOptions = {}) {
    this.bufferMs = options.bufferMs ?? 3000;
    this.gridFps = options.gridFps ?? 50;
    this.vision = options.vision ?? false;
  }

  push(hand: "left" | "right", frame: GloveFrame): void {
    const buf = hand === "left" ? this.left : this.right;
    buf.push({
      t: frame.timestamp,
      sensor: frame.mapped_data,
      quat: frame.quaternion,
      acc: frame.acceleration,
      att: frame.attitude,
    });
    this.trim(buf, frame.timestamp);
  }

  /**
   * 压一帧视觉。`vision: false` 时**直接丢掉** —— 误调不会悄悄给推理端的样本
   * 塞进 landmarks（那会让人以为特征维度变了去找 bug，而实际上 `includeVision`
   * 默认 false 根本不读它）。
   *
   * `t` 必须与手套帧同一个时钟（`performance.now()`）：重采样是按时间最近邻对齐的。
   * 这只手没检出时传 null，写栅格时填 NaN。
   */
  pushVision(t: number, left: Float32Array | null, right: Float32Array | null): void {
    if (!this.vision) return;
    this.visionBuf.push({ t, left, right });
    // 视觉与手套各自 trim：句子录到 12s 时视觉有约 360 帧，不 trim 会跟着涨
    const cutoff = t - this.bufferMs;
    let drop = 0;
    while (drop < this.visionBuf.length && this.visionBuf[drop].t < cutoff) drop++;
    if (drop > 0) this.visionBuf.splice(0, drop);
  }

  /** 缓冲里现有的视觉帧数。采集页用它显示"这一句到底有没有视觉" */
  visionFrames(): number {
    return this.visionBuf.length;
  }

  clear(): void {
    this.left = [];
    this.right = [];
    this.visionBuf = [];
  }

  /** 缓冲里最新一帧的时间戳，两手取较大者；空缓冲返回 null */
  latestTime(): number | null {
    const lt = this.left.length ? this.left[this.left.length - 1].t : -Infinity;
    const rt = this.right.length
      ? this.right[this.right.length - 1].t
      : -Infinity;
    const m = Math.max(lt, rt);
    return Number.isFinite(m) ? m : null;
  }

  /**
   * 取最近 windowMs 的一段，重采样成 SequenceSample。
   * 返回 null 表示这一刻数据不足以推理（刚开始翻译、或手套刚掉线）。
   */
  snapshot(windowMs: number, label = "_live"): SequenceSample | null {
    const end = this.latestTime();
    if (end === null) return null;
    // 缓冲还没攒够一整个窗口就别推理 —— 前半段会被最近邻拉成一条直线，
    // 那是个"静止"的假动作，模型多半会输出错词
    const start = end - windowMs;
    const covers = (buf: WindowEntry[]) => buf.length > 0 && buf[0].t <= start;
    if (!covers(this.left) && !covers(this.right)) return null;
    // 显式传 windowMs 而不是让 resample 用 end-start 还原：浮点相减可能差 1 ULP，
    // 而 T = floor(windowMs/dt) 在恰好整除时（2000/20=100）会因此掉一帧
    return this.resample(
      start,
      windowMs,
      covers(this.left),
      covers(this.right),
      label
    );
  }

  /**
   * 取**缓冲里现有的全部**一段（可选上限 maxMs、可选砍掉尾部 dropTailMs），
   * 给整句捕获用。
   *
   * 为什么不能用 `snapshot(now - 按下的时刻)`：`snapshot` 要求
   * `buf[0].t <= end - windowMs`，而按下之后第一帧的时间戳必然**略大于**按下的时刻，
   * 这个判据会不成立，整句捕获永远返回 null。调用方靠减一点余量去凑是撞浮点，
   * 不同帧率下时好时坏 —— 所以区间边界由缓冲自己算。
   *
   * **只收时长，不收时刻。** 帧时间戳是 `performance.now()`，调用方的状态机可能跑在
   * 另一个时钟上（`Date.now()`），绝对时刻不可比；而"多长""砍掉多长"是时长，跨时钟成立。
   *
   * 起点取两只手首帧的**较晚者**（不是较早者）：较早者会让另一只手开头那段被最近邻
   * 填成一条直线，那是假的"静止"。少取几十毫秒换全段两手都是真数据。
   *
   * @param maxMs 最长取多少（从**尾部**往前算）；不传 = 缓冲里有多少取多少
   * @param dropTailMs 从尾部砍掉多少。收句判据用掉的那段静止要从**末尾**砍，
   *        不能靠调小 maxMs —— 那样砍掉的是句子的**开头**（区间是贴着尾部对齐的），
   *        表现为"每句话前几个词都丢了"
   */
  snapshotAll(maxMs?: number, label = "_live", dropTailMs = 0): SequenceSample | null {
    const latest = this.latestTime();
    if (latest === null) return null;
    const end = latest - Math.max(0, dropTailMs);
    const firsts: number[] = [];
    if (this.left.length) firsts.push(this.left[0].t);
    if (this.right.length) firsts.push(this.right[0].t);
    if (!firsts.length) return null;
    let start = Math.max(...firsts);
    if (maxMs !== undefined) start = Math.max(start, end - maxMs);
    // 砍过头（尾部静止比整段还长）：没有可用区间，返回 null 而不是负长度的段
    if (end - start < 1) return null;
    // 两只手都从 start 起有真数据（start 是首帧较晚的那只手的首帧），
    // 所以只要那只手有数据就算 covers
    return this.resample(
      start,
      end - start,
      this.left.length > 0,
      this.right.length > 0,
      label
    );
  }

  /** 缓冲里两只手**共同**覆盖的时长（ms）；不足以推理时返回 null */
  spanMs(): number | null {
    const end = this.latestTime();
    if (end === null) return null;
    const firsts: number[] = [];
    if (this.left.length) firsts.push(this.left[0].t);
    if (this.right.length) firsts.push(this.right[0].t);
    return firsts.length ? end - Math.max(...firsts) : null;
  }

  private resample(
    start: number,
    windowMs: number,
    hasLeft: boolean,
    hasRight: boolean,
    label: string
  ): SequenceSample | null {
    if (!hasLeft && !hasRight) return null;
    const dt = 1000 / this.gridFps;
    const T = Math.floor(windowMs / dt);
    if (T < 4) return null;

    const timestamps = new Float32Array(T);
    const leftSensor = hasLeft ? new Uint8Array(T * SEQ_SENSOR_N) : null;
    const rightSensor = hasRight ? new Uint8Array(T * SEQ_SENSOR_N) : null;
    const leftImu = hasLeft ? new Float32Array(T * SEQ_IMU_N) : null;
    const rightImu = hasRight ? new Float32Array(T * SEQ_IMU_N) : null;
    // 只要这只手有手套数据就分配视觉列（缺帧填 NaN），与 `useSequenceRecorder`
    // 同一条口径：这样下游能区分"这只手没戴手套"和"这只手视觉丢了"。
    // 视觉缓冲为空时不分配 —— 摄像头没开的那些条与不开视觉时完全一样
    const hasVision = this.vision && this.visionBuf.length > 0;
    const leftLandmarks =
      hasLeft && hasVision ? new Float32Array(T * SEQ_LANDMARK_N) : null;
    const rightLandmarks =
      hasRight && hasVision ? new Float32Array(T * SEQ_LANDMARK_N) : null;

    let li = 0;
    let ri = 0;
    let vi = 0;
    for (let t = 0; t < T; t++) {
      const target = start + t * dt;
      timestamps[t] = t * dt;
      if (leftSensor && leftImu) {
        const r = nearest(this.left, target, li);
        li = r.idx;
        write(r.entry, leftSensor, leftImu, t);
      }
      if (rightSensor && rightImu) {
        const r = nearest(this.right, target, ri);
        ri = r.idx;
        write(r.entry, rightSensor, rightImu, t);
      }
      if (leftLandmarks || rightLandmarks) {
        const r = nearest(this.visionBuf, target, vi);
        vi = r.idx;
        const inGap =
          r.entry !== null && Math.abs(r.entry.t - target) <= VISION_MAX_GAP_MS;
        if (leftLandmarks) writeVision(inGap ? r.entry!.left : null, leftLandmarks, t);
        if (rightLandmarks) writeVision(inGap ? r.entry!.right : null, rightLandmarks, t);
      }
    }

    return {
      segments: [{ label, startFrame: 0, endFrame: T }],
      primaryLabel: label,
      frameCount: T,
      timestamps,
      leftSensor,
      rightSensor,
      leftImu,
      rightImu,
      leftLandmarks,
      rightLandmarks,
      durationMs: windowMs,
      sourceFps: this.gridFps,
      origin: "recorded",
      timestamp: Date.now(),
    };
  }

  private trim(buf: WindowEntry[], now: number): void {
    const cutoff = now - this.bufferMs;
    let drop = 0;
    while (drop < buf.length && buf[drop].t < cutoff) drop++;
    if (drop > 0) buf.splice(0, drop);
  }
}

function nearest<T extends { t: number }>(
  buf: T[],
  target: number,
  startIdx: number
): { entry: T | null; idx: number } {
  if (buf.length === 0) return { entry: null, idx: 0 };
  let i = Math.max(0, Math.min(startIdx, buf.length - 1));
  while (i + 1 < buf.length && buf[i + 1].t <= target) i++;
  let best = i;
  if (
    i + 1 < buf.length &&
    Math.abs(buf[i + 1].t - target) < Math.abs(buf[i].t - target)
  ) {
    best = i + 1;
  }
  return { entry: buf[best], idx: i };
}

function write(
  entry: WindowEntry | null,
  sensor: Uint8Array,
  imu: Float32Array,
  t: number
): void {
  const so = t * SEQ_SENSOR_N;
  const io = t * SEQ_IMU_N;
  if (!entry) {
    imu[io] = 1; // 单位四元数，避免零向量归一化时退化
    return;
  }
  const n = Math.min(SEQ_SENSOR_N, entry.sensor.length);
  for (let c = 0; c < n; c++) {
    const v = entry.sensor[c];
    sensor[so + c] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  imu[io] = entry.quat[0];
  imu[io + 1] = entry.quat[1];
  imu[io + 2] = entry.quat[2];
  imu[io + 3] = entry.quat[3];
  if (entry.acc) {
    imu[io + 4] = entry.acc[0];
    imu[io + 5] = entry.acc[1];
    imu[io + 6] = entry.acc[2];
  }
  if (entry.att) {
    imu[io + 7] = entry.att[0];
    imu[io + 8] = entry.att[1];
    imu[io + 9] = entry.att[2];
  }
}

/**
 * 写一帧关键点。缺手时填 **NaN 而不是 0** —— 0 是一个合法坐标（画面左上角），
 * `sequenceTrim.handVisibleAt` 靠"首尾两个数是否有限"判可见，填 0 等于宣称
 * 整段都看得见手，可见性那一层会彻底失效。
 */
function writeVision(lm: Float32Array | null, dst: Float32Array, t: number): void {
  const o = t * SEQ_LANDMARK_N;
  if (!lm) {
    dst.fill(NaN, o, o + SEQ_LANDMARK_N);
    return;
  }
  dst.set(lm, o);
}
