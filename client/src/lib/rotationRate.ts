/*
 * rotationRate —— 用四元数的**累计路径转角速率**衡量"这只手在转没在转"
 *
 * ===== 为什么需要它，以及为什么不是"首帧起的最大转角" =====
 *
 * 谢谢 / 难过 这一刀原来只靠拇指压力判（`fistGate.ts`）。使用者 2026-08-30 指出
 * 一个更根本的物理差别：
 *
 *   谢谢  = 手基本不动，只有大拇指下压 → 手腕朝向几乎不变
 *   难过  = 掌心朝胸口**画一个圈** → 朝向一路在变
 *
 * 手套的 packet-2 IMU 只有四元数可用（加速度/姿态角那 6 个槽位实测全零，见
 * `glove-packet2-length` 那条记录），所以"转了多少"只能从四元数算。
 *
 * 关键是**算法选哪一种**。两种都试过，2000ms 窗、348 个窗：
 *
 *   首帧起的最大转角   最佳阈值 69.2° → 分对 71.0%
 *   累计路径转角       最佳阈值 112.6° → 分对 97.7%
 *
 * 差这么多的原因就是"圈"这个字：难过画完一圈**手回到了原位**，所以它对首帧的
 * 偏离并不大；但一路走过的弧长很长。所以这里算的是逐帧相对旋转角之**和**，
 * 不是端点之间的夹角。
 *
 * ===== 为什么是速率（度/秒）而不是总度数 =====
 *
 * 逐词滑窗的窗固定 2000ms，总度数可以直接比。但句子路径每个词占的 CTC 区间
 * 长短不一（~200ms 到 3s），总度数天然随区间长度缩放 —— 短词会全被判成"静止"。
 * 除以时长就没这个问题。
 *
 * ===== ⚠ 这个判据有一个硬性的适用下限：区间必须够长 =====
 *
 * 2026-08-30 实测（各窗长，逐窗最佳速率阈值 vs 现役拇指闸门）：
 *
 *    窗长     窗数    拇指闸门   速率(最佳)   速率三段式 50/70
 *    300ms   1138     71.4%      72.7%          72.2%
 *    500ms    798     78.6%      75.1%          75.2%
 *    800ms    708     84.0%      77.8%          79.7%
 *   1000ms    648     87.5%      82.6%          82.6%
 *   1200ms    588     88.3%      85.9%          85.9%
 *   1500ms    498     88.0%      94.8%          94.8%
 *   2000ms    348     88.2%      97.7%          97.4%
 *   2500ms    198     87.4%      99.5%          99.5%
 *
 * **短区间下速率判据比拇指还差。** 原因是谢谢的转动是**阵发**的 —— 抬手到位、
 * 收手离场各有一下急动：
 *
 *   谢谢的速率  300ms 窗 p50=16 p90=102 max=253 °/s
 *              2000ms 窗 p50=33 p90= 52 max= 57 °/s
 *   难过的速率  300ms 窗 p50=77          2000ms 窗 p50=90 °/s  ← 一直在转
 *
 * 短窗会正好切在那一下急动上，读出跟画圈一样的速率。窗够长时那一下被平摊掉。
 * 所以本模块导出 `ROT_MIN_SPAN_MS`，**区间短于它就必须放弃这个判据**，
 * 退回拇指闸门。交叉点实测在 1200~1500ms 之间，取 1500 留余量。
 *
 * ===== 阈值 50 / 70 是量出来的，但样本量要打折看 =====
 *
 * 2000ms 窗下两侧的极值：谢谢 max 56.5 °/s、难过 min 46.2 °/s ——
 * 两个阈值都落在这段重叠区里，所以：
 *   - 速率 ≥ 70 → 谢谢一个都没有（0/193）**这一侧完全干净**
 *   - 速率 < 50 → 难过有 22.4% 也在这儿，所以这一侧不干净
 * 于是做成三段：外面两段由速率定，中间 [50,70) 交给拇指仲裁（实测只占 16%）。
 *
 * ⚠ 这 348 个窗来自**30 条录制**，而且窗步进 100ms、窗长 2000ms，相邻窗重叠 95%
 * —— 有效样本数接近 30 而不是 348。所以这两个数是**暂定值**，要靠翻译页上的
 * 实时读数校准，不要拿"97.4%"当作已经验证过的实时表现。
 *
 * 选 50/70 而不是分对率更高的 40/70（98.0%）的理由：40/70 有 28% 的窗落进中间带，
 * 也就是 28% 的判断仍然押在拇指上 —— 而拇指恰恰是被怀疑的那个通道。
 * 假设实时拇指读数全坏（一律读成 <5）时，2000ms 窗：
 *   三段式 50/70  分对 92.0%  谢谢判对 168/193
 *   三段式 40/80  分对 81.0%  谢谢判对 127/193
 *   只看拇指      分对 44.5%  谢谢判对   0/193
 */
import { SEQ_IMU_N } from "./datasetStore";
import type { SequenceSample } from "./datasetStore";
import { quatConj, quatMul, quatNormalize, type Quat } from "./sequenceFeatures";

/**
 * 速率判据的适用下限（毫秒）。区间短于它，**读数作废**（`usable: false`），
 * 调用方必须退回拇指闸门。见文件头那张各窗长对照表 —— 短窗下这个判据比拇指还差。
 *
 * 逐词滑窗的窗是 2000ms，永远满足；句子路径的词区间大多短于它，
 * 也就是说这个改动实际上**只对逐词档生效**，句子档仍由拇指闸门管。
 */
export const ROT_MIN_SPAN_MS = 1500;

/** 速率低于它 → 判"没在转"（= 谢谢）。这一侧不干净：22% 的难过窗也在这儿 */
export const ROT_RATE_LO = 50;

/** 速率不低于它 → 判"在转"（= 难过）。这一侧干净：2000ms 窗下 0/193 个谢谢窗到过这里 */
export const ROT_RATE_HI = 70;

export interface RotationReading {
  /** 累计路径转角速率，度/秒。`usable` 为 false 时这个数没有意义 */
  ratePerSec: number;
  /** 累计路径转角（度）—— 界面读数用，判断一律用 `ratePerSec` */
  totalDeg: number;
  /** 区间实际时长（毫秒） */
  spanMs: number;
  /**
   * 这个读数能不能用来判断。false 的原因有三种，**都不该被当成"没在转"**：
   * 没有右手 IMU、区间短于 `ROT_MIN_SPAN_MS`、四元数全零或有 NaN。
   */
  usable: boolean;
}

const UNUSABLE: RotationReading = {
  ratePerSec: -1,
  totalDeg: -1,
  spanMs: 0,
  usable: false,
};

function quatAt(imu: Float32Array, frame: number): Quat {
  const o = frame * SEQ_IMU_N;
  return [imu[o], imu[o + 1], imu[o + 2], imu[o + 3]];
}

/** 四元数是全零（没数据）还是有 NaN —— 归一化之前先挡掉，否则会算出 NaN 速率 */
function badQuat(q: Quat): boolean {
  let sum = 0;
  for (const v of q) {
    if (!Number.isFinite(v)) return true;
    sum += v * v;
  }
  return sum < 1e-12;
}

/**
 * 样本里 `[from01, to01)` 这一段的右手累计路径转角速率。
 *
 * @param sample 应当是**归一化到右手口径之后**的样本。转角**大小**本身是镜像不变的
 *               （镜像是反射，保持角度绝对值），但左手数据在镜像前放的是 `leftImu`
 *               槽位，这里只读 `rightImu` —— 不归一化就读不到东西，返回 `usable: false`。
 * @param from01 区间起点，归一化位置（0 = 首帧）
 * @param to01   区间终点（1 = 末帧）
 *
 * 与 `thumbPeakSpan` 一致：区间越界时夹进合法范围。但**不**把退化区间夹到一帧 ——
 * 一帧算不出转角，那种情况返回 `usable: false`，让调用方退回拇指闸门。
 */
export function rotationRateSpan(
  sample: SequenceSample,
  from01: number,
  to01: number
): RotationReading {
  const imu = sample.rightImu;
  const n = sample.frameCount;
  if (!imu || n < 2) return UNUSABLE;

  let a = Math.floor(Math.max(0, Math.min(1, from01)) * n);
  let b = Math.ceil(Math.max(0, Math.min(1, to01)) * n);
  a = Math.max(0, Math.min(a, n - 1));
  b = Math.max(a, Math.min(b, n));
  if (b - a < 2) return UNUSABLE;

  const ts = sample.timestamps;
  const spanMs =
    ts && ts.length >= b
      ? ts[b - 1] - ts[a]
      : (sample.durationMs * (b - 1 - a)) / Math.max(1, n - 1);
  if (!(spanMs > 0)) return UNUSABLE;

  let prev = quatAt(imu, a);
  if (badQuat(prev)) return UNUSABLE;
  prev = quatNormalize(prev);
  let total = 0;
  for (let t = a + 1; t < b; t++) {
    const cur = quatAt(imu, t);
    if (badQuat(cur)) return UNUSABLE;
    const q = quatNormalize(cur);
    /*
     * 逐帧相对旋转角：|w| 而不是 w —— q 与 −q 是同一个旋转，IMU 输出会随机跳符号。
     * 不取绝对值的话每次跳符号都会记一个 180°，静止的手也能读出几千度。
     */
    const rel = quatMul(quatConj(prev), q);
    total += (2 * Math.acos(Math.min(1, Math.abs(rel[0]))) * 180) / Math.PI;
    prev = q;
  }
  if (!Number.isFinite(total)) return UNUSABLE;

  return {
    ratePerSec: (total * 1000) / spanMs,
    totalDeg: total,
    spanMs,
    usable: spanMs >= ROT_MIN_SPAN_MS,
  };
}

/** 整条样本的转角速率（逐词滑窗那条路上，一个窗口就是一条样本） */
export function rotationRate(sample: SequenceSample): RotationReading {
  return rotationRateSpan(sample, 0, 1);
}

/** 速率落在哪一段。`"unknown"` = 读数不可用或落在中间带，都得靠别的判据 */
export type RotationBand = "still" | "rotating" | "unknown";

export function rotationBand(
  rot: RotationReading | null | undefined,
  lo: number = ROT_RATE_LO,
  hi: number = ROT_RATE_HI
): RotationBand {
  if (!rot?.usable) return "unknown";
  if (rot.ratePerSec >= hi) return "rotating";
  if (rot.ratePerSec < lo) return "still";
  return "unknown";
}
