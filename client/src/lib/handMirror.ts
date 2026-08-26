/*
 * handMirror.ts —— 把一只手的数据镜像成另一只手的口径。
 *
 * 为什么需要：特征层给两只手各留一段独立槽位（`sequenceFeatures.ts` 里左手写
 * `[0, HAND_TACTILE_FRAME_DIM)`、右手写 `[HAND_TACTILE_FRAME_DIM, …)`，缺手那段保持全 0），
 * 而 137 维的槽位顺序**左右手是镜像的**（`sensorMapping.ts`：`LEFT_HAND_INDEX_MAP` 是
 * 小指→拇指，`RIGHT_HAND_INDEX_MAP` 是拇指→小指）。两件事叠起来的后果是：
 *
 *   用右手采的数据训出来的模型，**从没在左手那段槽位上见过任何信号**；
 *   就算见过，同一根物理手指在两只手里落在不同下标上。
 *
 * 所以戴左手套做一个用右手训过的单手词，模型看到的是"右手槽位全 0 + 左手槽位有信号"，
 * 这个输入在训练集里根本不存在，输出会塌到某一个固定的类上。这不是 bug，是结构缺口。
 *
 * 手语上单手词换手不改变词义（换手只是惯用手不同），所以正确做法是把左手数据
 * **归一化到右手口径**再喂模型，而不是给每个词都再采一遍左手。
 *
 * ===== 137 维置换表怎么来的 =====
 *
 * 两个独立来源互相印证，不是推测：
 *
 * 1. 规格书 V1.1 p10「16x16 点阵的 256 压力数组解析方案」给出字节到点阵的关系
 *    `byte = (row−1)×16 + col`。把两只手的 137 个点铺到这张网格上，两只手是一个
 *    精确的 180° 旋转，即 **1-based `byte n ↔ 257−n`**（0-based `b ↔ 255−b`）。
 * 2. 同款手套的另一份第三方实现（`3_lnr/laonianren/.../get_adc_form_csv.py` 的
 *    `LeftHand` / `RightHand` 两个类）把线序写成**解剖学网格**：每指 5 行 × 3 列
 *    （第 5 行是弯折传感器）、手掌 5 行 × 15 列、虎口 3 格为空。在那套坐标里同一个映射
 *    表现为「只反转列、行不动」的左右镜像 —— 行序（指尖→指根）左右一致，只有横跨
 *    手指宽度的 3 列换边，这正是解剖学上应有的样子。137 点逐点比对 0 处不一致，
 *    3 个空格的位置也互相落上。
 *
 * 由此推出的 137 维置换（下面的 `MIRROR_PERM_137`）已核对：双射、**自逆**
 * （`perm[perm[k]] === k`，所以左→右和右→左是同一张表）、手指块仍落在手指块、
 * 手掌块仍落在手掌块、弯折 60~64 → [64,63,62,61,60]，且每一槽都落在厂家网格里
 * 解剖学镜像的那一格上。
 *
 * ===== IMU 怎么镜像 =====
 *
 * 镜像一个旋转得到的是**反射共轭** `R' = M R M`（`M = diag(−1,1,1)`，镜面法向沿 x）。
 * 对四元数就是 `(w,x,y,z) → (w,x,−y,−z)`；对真矢量（加速度、重力）是 `(x,y,z) → (−x,y,z)`。
 *
 * 只镜像**每帧原始四元数**就够了，特征层的两个派生量会自动跟着对：
 *   - 相对四元数 `q₀⁻¹⊗qₜ`：`(M R₀ M)⁻¹(M Rₜ M) = M (R₀⁻¹Rₜ) M`，正是相对旋转的镜像；
 *   - 重力在手系的投影 `gravityInHandFrame(q)`：世界重力没有 x 分量，于是
 *     `g' = M R⁻¹ M g = M g`，正是重力矢量的镜像。
 *
 * **未经硬件验证的一点**：镜面法向取 x 轴，是从 `orientationCalib.ts` 的
 * `MODEL_MOTION_AXES` 反推的 —— 俯仰轴左右手同为 `[-1,0,0]`，偏摆轴左手 `[0,1,0]`
 * 右手 `[0,-1,0]`，y 翻号说明 x 垂直于镜面。这条推理成立与否要戴两只手套做同一个
 * 单手词、看镜像后的朝向通道是否重合才能定论。压力/弯折那 137 维不受它影响。
 */
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "@/lib/datasetStore";

/**
 * 137 维槽位的左右镜像置换：`dst[MIRROR_PERM_137[k]] = src[k]`。
 *
 * **自逆**，所以左→右与右→左用同一张表，不需要两份。
 * 生成方式见文件头；`handMirror.test.ts` 会把这张表的所有不变量再锁一遍
 * （双射、自逆、分块不越界、以及直接从 `sensorMapping.ts` 的两张索引表重算一致）。
 */
export const MIRROR_PERM_137: readonly number[] = [
   50,  49,  48,  53,  52,  51,  56,  55,  54,  59,  58,  57,
   38,  37,  36,  41,  40,  39,  44,  43,  42,  47,  46,  45,
   26,  25,  24,  29,  28,  27,  32,  31,  30,  35,  34,  33,
   14,  13,  12,  17,  16,  15,  20,  19,  18,  23,  22,  21,
    2,   1,   0,   5,   4,   3,   8,   7,   6,  11,  10,   9,
   64,  63,  62,  61,  60,  76,  75,  74,  73,  72,  71,  70,
   69,  68,  67,  66,  65,  91,  90,  89,  88,  87,  86,  85,
   84,  83,  82,  81,  80,  79,  78,  77, 106, 105, 104, 103,
  102, 101, 100,  99,  98,  97,  96,  95,  94,  93,  92, 121,
  120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109,
  108, 107, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127,
  126, 125, 124, 123, 122,
];

if (MIRROR_PERM_137.length !== SEQ_SENSOR_N) {
  // 表是按 137 维写死的。SEQ_SENSOR_N 若被改动，静默错位比崩掉危险得多
  throw new Error(
    `MIRROR_PERM_137 长度 ${MIRROR_PERM_137.length} 与 SEQ_SENSOR_N=${SEQ_SENSOR_N} 不符`
  );
}

/** 单帧 137 维镜像。`dst` 与 `src` 不能是同一段内存（置换不是原地可做的） */
export function mirrorFrame137(
  src: ArrayLike<number>,
  srcOffset: number,
  dst: { [i: number]: number },
  dstOffset: number
): void {
  for (let k = 0; k < SEQ_SENSOR_N; k++) {
    dst[dstOffset + MIRROR_PERM_137[k]] = src[srcOffset + k];
  }
}

/** 整条 `[T*137]` 传感器序列镜像，返回新数组 */
export function mirrorSensorSeries(
  src: Uint8Array,
  frameCount: number
): Uint8Array {
  const out = new Uint8Array(frameCount * SEQ_SENSOR_N);
  for (let t = 0; t < frameCount; t++) {
    const o = t * SEQ_SENSOR_N;
    if (o + SEQ_SENSOR_N > src.length) break;
    mirrorFrame137(src, o, out, o);
  }
  return out;
}

/** 四元数 `[w,x,y,z]` 的镜像（反射共轭，镜面法向沿 x） */
export function mirrorQuat(
  q: readonly [number, number, number, number]
): [number, number, number, number] {
  return [q[0], q[1], -q[2], -q[3]];
}

/** 真矢量（加速度 / 重力）的镜像 */
export function mirrorVec3(
  v: readonly [number, number, number]
): [number, number, number] {
  return [-v[0], v[1], v[2]];
}

/**
 * 整条 `[T*10]` IMU 序列镜像（quat4 + acc3 + att3）。
 *
 * att（姿态角 roll/pitch/yaw）按同一反射规则取 `(roll, −pitch, −yaw)`：绕 x 的
 * 转动在这个镜像下不变号，绕 y / z 的变号。这三路**目前特征层完全没用到**
 * （`writeHandTactile` 只取 quat 与 acc），所以它不承重；写上是为了让镜像后的
 * `SequenceSample` 整体自洽，别让以后启用 att 的人拿到没镜像的值。
 */
export function mirrorImuSeries(
  src: Float32Array,
  frameCount: number
): Float32Array {
  const out = new Float32Array(frameCount * SEQ_IMU_N);
  for (let t = 0; t < frameCount; t++) {
    const o = t * SEQ_IMU_N;
    if (o + SEQ_IMU_N > src.length) break;
    out[o] = src[o];
    out[o + 1] = src[o + 1];
    out[o + 2] = -src[o + 2];
    out[o + 3] = -src[o + 3];
    out[o + 4] = -src[o + 4];
    out[o + 5] = src[o + 5];
    out[o + 6] = src[o + 6];
    out[o + 7] = src[o + 7];
    out[o + 8] = -src[o + 8];
    out[o + 9] = -src[o + 9];
  }
  return out;
}

/**
 * 整条 `[T*63]` 关键点序列的镜像：21 个点逐点 `x → 1 − x`，y / z 不动。
 *
 * 为什么是 `1 − x` 而不是 `−x`：MediaPipe 的关键点是**归一化图像坐标**，x ∈ [0,1]
 * 且向右为正，所以"把画面左右翻过来"就是绕画面竖直中线反射，即 `1 − x`。
 * z 是相对手腕的深度，绕竖直平面反射不改变它；y 同理。
 *
 * 与 `normalizeLandmarkFrame` 的配合是自动的：那里手腕点保留绝对坐标、其余 20 点取
 * 相对手腕的差再除手长，而 `(1−x_p) − (1−x_w) = −(x_p − x_w)`，相对量正好跟着变号。
 * 所以镜像放在**原始关键点**上就够了，不需要在特征层再补一次。
 *
 * 缺失帧是整帧 NaN（`datasetStore.ts` 的约定），`1 − NaN = NaN`，可见性判据
 * （`sequenceTrim.ts` 的 `handVisibleAt`）不受影响。
 *
 * ⚠ **一个近似**：绕**画面**中线反射，不是绕人体中线。人站得偏离画面中心时，
 * 这一步除了镜像还附带一个横向平移，而手腕绝对 x 是承载轨迹的特征之一
 * （`sequenceFeatures.ts` 的注释 3），于是那一路会带上这个偏移。绕人体中线才是对的，
 * 但手部关键点里没有躯干信息，单手录制更是连中线都估不出来。
 * 触觉那 137+10 维完全不受这条影响；视觉只在教师（`FUSED_FRAME_DIM`）里用到，
 * 视觉覆盖不足 80% 时教师根本不训（`trainSequenceModel` 的 `useDistillation`），
 * 那时这个近似不参与任何计算。
 */
export function mirrorLandmarkSeries(
  src: Float32Array,
  frameCount: number
): Float32Array {
  const out = new Float32Array(frameCount * SEQ_LANDMARK_N);
  for (let t = 0; t < frameCount; t++) {
    const o = t * SEQ_LANDMARK_N;
    if (o + SEQ_LANDMARK_N > src.length) break;
    for (let p = 0; p < SEQ_LANDMARK_N; p += 3) {
      out[o + p] = 1 - src[o + p];
      out[o + p + 1] = src[o + p + 1];
      out[o + p + 2] = src[o + p + 2];
    }
  }
  return out;
}

/**
 * 单帧输入的镜像 —— 静态模型（`signLanguageModel.predict`）那条路用。
 *
 * 静态模型的输入是 `[...handTactile(left), ...handTactile(right)]`，
 * 和时序模型一样是**两段独立槽位**，所以同一个换手缺口在那边一模一样地存在。
 */
export interface MirrorableHandInput {
  sensor_data: number[];
  quaternion: [number, number, number, number];
}

export function mirrorHandInput(
  input: MirrorableHandInput
): MirrorableHandInput {
  const out = new Array<number>(SEQ_SENSOR_N).fill(0);
  const n = Math.min(SEQ_SENSOR_N, input.sensor_data.length);
  for (let k = 0; k < n; k++) out[MIRROR_PERM_137[k]] = input.sensor_data[k];
  return { sensor_data: out, quaternion: mirrorQuat(input.quaternion) };
}

/**
 * 静态单帧那条路的整体镜像：**两只手一起镜像 + 槽位互换**。
 *
 * 为什么是互换而不是"把左手搬进右手槽、右手槽清零"：训练时两只手套是**都连着**的
 * （右手做动作、左手闲着），所以模型见过的输入是「左槽=静止基线，右槽=有动作」。
 * 把闲着那只手清零并不等于静止基线 —— 全 0 同样是训练集里不存在的输入。
 * 互换之后左槽拿到的是镜像后的静止手、右槽拿到的是镜像后的动作手，
 * 与训练口径结构一致。
 *
 * 空的一路保持空：只连了一只手套时，互换退化成上面那个"搬过去"的行为。
 */
export function mirrorStaticInputs(
  left: MirrorableHandInput | null,
  right: MirrorableHandInput | null
): { left: MirrorableHandInput | null; right: MirrorableHandInput | null } {
  return {
    left: right ? mirrorHandInput(right) : null,
    right: left ? mirrorHandInput(left) : null,
  };
}

/**
 * 整条样本的镜像：**两只手一起镜像 + 槽位互换**（`left' = M(right)`、`right' = M(left)`）。
 *
 * 这是"换惯用手"这件事唯一自洽的操作，对单手词与双手词都成立 ——
 * 双手词整体镜像之后就是同一个词由另一只手主导的版本。**自逆**。
 *
 * 空的一路保持空，所以只连一只手套时它退化成"把那只手搬到对面槽位"。
 *
 * 视觉通道（`leftLandmarks` / `rightLandmarks`）**一起镜像互换**。
 * 推理滑窗那条路两边本来都是 null（`sequenceWindow.ts`），这一步是空操作；
 * 但训练那条路的样本**是有视觉的**，教师模型吃的是 420 维触觉+视觉拼接
 * （`sequenceFeatures.ts` 的 `FUSED_FRAME_DIM`）。只换触觉不换视觉的话，
 * 左手那半段会变成"触觉是镜像后的右手、视觉是原封不动的左手"，
 * 两路指的不是同一只手，教师直接被喂脏，蒸出来的学生跟着脏。
 * `imuHealth` 也跟着互换 —— 特征层没用到它，但留着不换会是个以后踩的坑。
 */
export function mirrorSample(sample: SequenceSample): SequenceSample {
  const T = sample.frameCount;
  const health = sample.imuHealth;
  return {
    ...sample,
    leftSensor: sample.rightSensor
      ? mirrorSensorSeries(sample.rightSensor, T)
      : null,
    rightSensor: sample.leftSensor
      ? mirrorSensorSeries(sample.leftSensor, T)
      : null,
    leftImu: sample.rightImu ? mirrorImuSeries(sample.rightImu, T) : null,
    rightImu: sample.leftImu ? mirrorImuSeries(sample.leftImu, T) : null,
    leftLandmarks: sample.rightLandmarks
      ? mirrorLandmarkSeries(sample.rightLandmarks, T)
      : null,
    rightLandmarks: sample.leftLandmarks
      ? mirrorLandmarkSeries(sample.leftLandmarks, T)
      : null,
    ...(health
      ? { imuHealth: { left: health.right, right: health.left } }
      : {}),
  };
}

/**
 * **已判定**的主手（做手语的那只手）。`both` = 两只手活动量接近，按双手词处理。
 *
 * 这里刻意没有 `auto`：判定是 `dominantHand.ts` 的事，它从滑窗里量两只手的运动能量
 * 得出结论，本模块只负责按结论做变换。以前这个类型带 `auto`、由用户在界面上手选，
 * 理由是"闲着那只手也在出静止数据，分不出没戴和戴着不动"—— 那句话没错，但要判的
 * 本来就不是哪只手戴了手套，而是哪只手在动，那是量得出来的。详见 `dominantHand.ts`。
 */
export type Dominance = "left" | "right" | "both";

export interface NormalizeResult {
  sample: SequenceSample;
  /** 是否真的做了镜像 */
  mirrored: boolean;
  /** 判定结果，可直接显示 */
  reason: "mirrored" | "already_right" | "both_hands" | "no_hand";
}

/**
 * 推理前归一化到**右手口径**（模型是用右手采的数据训的）。
 *
 * - `dominant = "left"` → 整体镜像并互换槽位。单手词换手不改词义，
 *   双手词整体镜像后也是同一个词由另一只手主导的版本。
 * - `dominant = "right"` → 本来就是训练口径，原样返回。
 * - `dominant = "both"` → 双手词、两只手都在动，谁也不是"闲着的那只"，
 *   镜像无从下手，原样返回并给出 `both_hands`。
 *
 * 注意 `both` 只在两只手都有数据时才有意义；只有一只手有数据却传了 `both`
 * （判定器不会这么给）时按原样返回，宁可不变换也不要猜。
 */
export function normalizeHandedness(
  sample: SequenceSample,
  dominant: Dominance
): NormalizeResult {
  const hasLeft = !!(sample.leftSensor || sample.leftImu);
  const hasRight = !!(sample.rightSensor || sample.rightImu);
  if (!hasLeft && !hasRight)
    return { sample, mirrored: false, reason: "no_hand" };

  if (dominant === "left")
    return { sample: mirrorSample(sample), mirrored: true, reason: "mirrored" };
  if (dominant === "right")
    return { sample, mirrored: false, reason: "already_right" };
  return { sample, mirrored: false, reason: "both_hands" };
}
