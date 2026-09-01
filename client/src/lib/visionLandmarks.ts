/*
 * visionLandmarks —— 把 MediaPipe 的检测结果压成 63 维一帧的关键点数组。
 *
 * 这个函数原来是 `useSequenceRecorder.ts` 里的私有函数。搬出来的理由：
 * **孤立词采集（`/collect-seq`）和句子采集（`/collect-sentence`）必须产出同一种排布**。
 * 两边各写一份的话，21 点的顺序、缺手时填什么、左右手怎么认，任何一处不一致都会让
 * `sequenceTrim` 的可见性判据在其中一条链路上悄悄失效 —— 而症状只是"某一批数据裁得
 * 比另一批少"，没有任何线索指回这里。
 *
 * 排布必须与 `datasetStore.SEQ_LANDMARK_N`(63 = 21 点 × xyz) 一致，
 * 也就是 `[p0.x, p0.y, p0.z, p1.x, ...]`。`sequenceTrim.handVisibleAt` 靠
 * 首尾两个数是否有限来判"这一帧看得见手"，所以**缺手的帧必须填 NaN 而不是 0** ——
 * 0 是一个合法坐标（画面左上角），填 0 等于宣称手在画面角上。
 */
import { SEQ_LANDMARK_N } from "./datasetStore";

/**
 * MediaPipe 结果里本函数用到的那部分。
 *
 * 这里刻意**不** import `hooks/useHandTracking` 的 `HandResult`：lib 不该依赖 hooks，
 * 而且结构类型已经够 —— 传进来的对象少一个字段或类型不对，编译期就会拦住。
 */
export interface LandmarkFrameSource {
  landmarks: { x: number; y: number; z: number }[][];
  handedness: string[];
}

/**
 * 从一帧检测结果里抽出指定手的 63 维关键点；这只手没被检出时返回 null。
 *
 * `which` 是 MediaPipe 的口径（`"Left"`/`"Right"`，镜像与否由 `useHandTracking`
 * 决定），调用方直接照抄它给的 handedness 字符串，别在这里做左右翻转 ——
 * 数据集里存的必须是原始录制，镜像发生在建特征那一层（见 `handMirror.ts`）。
 */
export function extractHandLandmarks(
  result: LandmarkFrameSource | null,
  which: "Left" | "Right"
): Float32Array | null {
  if (!result?.landmarks?.length) return null;
  const idx = result.handedness.findIndex((h) => h === which);
  if (idx < 0) return null;
  const lms = result.landmarks[idx];
  if (!lms || lms.length !== 21) return null;
  const out = new Float32Array(SEQ_LANDMARK_N);
  for (let i = 0; i < 21; i++) {
    out[i * 3] = lms[i].x;
    out[i * 3 + 1] = lms[i].y;
    out[i * 3 + 2] = lms[i].z;
  }
  return out;
}
