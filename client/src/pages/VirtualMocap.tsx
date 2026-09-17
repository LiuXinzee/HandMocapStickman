/*
 * VirtualMocap — 仅触觉驱动的虚拟动捕 + 手套体检页面
 * DESIGN: Cyberpunk HUD 风格
 *
 * 无需摄像头，仅靠手套即可驱动。页面上跑**两条互相独立**的链路：
 *
 *  BEND + IMU — 5 路弯折 ADC 经两点标定 → 指节弯曲角，IMU 四元数 → 手腕朝向
 *      → GLB 3D 手模。**不依赖任何模型**，连上手套就能动。左右手各一块。
 *
 *  SKELETON REGRESSION — 骨架回归模型 predictSkeleton(137 传感 + 四元数) → 21 关键点
 *      → 2D 火柴人。依赖已训练模型，没模型时显示等待动画。
 *      模型是用 pickPrimaryHand 训练的、对左右手不敏感，所以只画一副，
 *      数据源优先取右手、右手没连才用左手。
 *
 * 两条链路共用同一份手套数据但算法完全不同，所以并排看就是一个免费的交叉校验：
 * 手型明显不一致 = 回归模型跑偏（或弯折通道标定失效）。
 *
 * **本页是双手套页面**（走 useGloves()，与 /collect、/collect-seq、/translate 同一份连接）。
 * 早先只连一只手套，体检得"做完左手→断开→插右手"，而漏掉一只手的代价并不对称：
 *   - 陀螺漂移漏了还能补救：/collect-seq 录每条时会对左右手各算一次 imuHealth
 *   - 弯折标定漏了不影响数据：它只喂本页的 3D 手模，不进数据集也不进任何模型
 *   - **死通道漏了没救**：一路弯折传感器坏掉会往 137 维里灌一个常数，每条样本都带着它，
 *     训练不报错、能量曲线正常，只表现为"某几个词就是学不好"。而查它靠的正是
 *     两点标定的跨度（weakChannels），所以两只手都必须标定过一次。
 * 这就是这一页改成双手的唯一理由。
 *
 * 注意：陀螺校准本身是**硬件动作**（短按每只手套主控按键），软件只能检测并指出该按哪只。
 *
 * **四步校准向导**（平铺→竖立→手心相对→握拳，移植自 cc_part2）一趟采出两份标定：
 *   - 弯折两点（张开/握拳）→ 手指能弯到多少。这一份也是死通道的唯一探测手段。
 *   - 朝向零位 + 轴向映射（见 orientationCalib.ts）→ 手模朝向跟不跟得上戴着的手。
 * 两份互不依赖，缺一份另一份照样生效；朝向那份只影响本页显示、不进数据集。
 *
 * 四步采样的累计姿态变化只作筛查，不能直接称为陀螺零偏。独立诊断提供
 * 稳定后 10 秒静置记录、三轴对照、多姿态轴向拟合，以及复用轴向的日常归零。
 *
 * **本页是第 1 步"准备"**，也是全流程里唯一的手套连接入口 —— 串口连接住在
 * App 层的 GloveProvider 里（一个 COM 口同一时刻只能被一个持有者打开），
 * 第 2~4 步只显示状态。所以这里连一次、标定一次，后面三步全程复用。
 */
import type { HandChannel } from "@/hooks/useDualGloveSerial";
import { useGloves } from "@/contexts/GloveContext";
import StepNav from "@/components/StepNav";
import type { GloveFrame } from "@/lib/gloveProtocol";
import {
  predictSkeleton,
  isSkeletonModelLoaded,
  loadSkeletonModelFromSaved,
} from "@/lib/skeletonModel";
import { getLatestSkeletonModel } from "@/lib/datasetStore";
import { FINGER_CONNECTION_GROUPS } from "@/hooks/useHandTracking";
import HandModel, {
  makeHandDrive,
  type HandDrive,
} from "@/components/HandModel";
import MotionDiagnostics from "@/components/MotionDiagnostics";
import { MANUAL_IMPORT_READY, pendingManualImport } from "@/lib/calibrationHistory";
import {
  checkPoseHold,
  validPose,
  type PoseSample,
} from "@/lib/motionCalibration";
import {
  averageBends,
  bendRatios,
  canonicalBendRaw,
  channelSpans,
  FINGER_NAMES,
  clearBendRange,
  isCalibrated,
  loadBendRange,
  MIN_USEFUL_SPAN,
  saveBendRange,
  weakChannels,
  type BendRange,
  type HandKey,
} from "@/lib/bendRange";
import {
  applyOrientationCalib,
  assembleOrientationCalib,
  averageQuaternions,
  axisMapFailReason,
  axisQualityWarning,
  clearOrientationCalib,
  loadOrientationCalib,
  normalizeQuat,
  MIN_MOTION_DEG,
  relativeAxisAngle,
  saveOrientationCalib,
  TARGET_MOTION_DEG,
  type OrientationCalib,
  type Quat,
  type Vec3,
} from "@/lib/orientationCalib";
import {
  analyzeImuHealth,
  worstReport,
  type ImuHealthReport,
} from "@/lib/imuHealth";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Bone,
  Hand,
  Wifi,
  WifiOff,
  Activity,
  Compass,
  Ruler,
  AlertTriangle,
  CheckCircle2,
  Wand2,
  X,
} from "lucide-react";

/*
 * 火柴人的配色。
 *
 * ⚠ **这两张表不能改用 `var(--hud-*)`**，尽管页面上别处都改了。两个原因，
 * 每一个都足以让它失效：
 *  1. 它们喂的是 canvas 2D 的 `fillStyle/strokeStyle` —— 那是自己解析色值的，
 *     不走 CSS，`var()` 会被静默丢弃（画出来是上一笔的残留色）。
 *  2. 下面还有 `color + "50"` 这种**拼十六进制透明度**的用法，
 *     值必须是 6 位十六进制字面量才拼得出合法颜色。
 * 所以这里写死浅色版的值，改配色时和 index.css 那几个 --hud-* 一起改。
 */
const FINGER_COLORS: Record<string, string> = {
  thumb: "#1677ff",
  index: "#16a34a",
  middle: "#7c3aed",
  ring: "#d97706",
  pinky: "#e11d48",
  palm: "#1677ff",
};

const FINGER_GLOW_COLORS: Record<string, string> = {
  thumb: "rgba(22, 119, 255, 0.28)",
  index: "rgba(22, 163, 74, 0.28)",
  middle: "rgba(124, 58, 237, 0.28)",
  ring: "rgba(217, 119, 6, 0.28)",
  pinky: "rgba(225, 29, 72, 0.28)",
  palm: "rgba(22, 119, 255, 0.16)",
};

const CANVAS_W = 640;
const CANVAS_H = 480;

/** 标定按钮取多少帧做平均（单帧受规格书标称的 ±8% 重复性影响太大） */
const CALIB_AVG_FRAMES = 8;

const HAND_KEYS: HandKey[] = ["LH", "RH"];

// ===== 四步校准向导 =====

/** 向导每采到一帧新数据就往采样槽里推一条 */
interface CalibSample {
  /** canonical 拇指→小指 的 5 路弯折 ADC */
  bend: number[];
  /** **原始** IMU 四元数（标定要反解零位，绝不能用已标定过的值） */
  quat: Quat;
  /** 加速度三轴；旧手套（272B 帧）没有这个字段，为 null。供陀螺漂移自检用 */
  acc: [number, number, number] | null;
  /** performance.now()。缺了它 imuHealth 算不出 stillRotationDegPerMin */
  t: number;
}
type CalibSink = (hand: HandKey, sample: CalibSample) => void;

type StepKey = "open" | "zero" | "palms" | "fist";

interface WizardStep {
  key: StepKey;
  short: string;
  title: string;
  instruction: string;
  /** 示意手模的弯曲度 0~1 */
  curl: number;
  /** 示意手模的目标姿态（与实际 IMU 无关，只是给用户照着摆） */
  poseQuat: Record<HandKey, Quat>;
}

const IDENTITY_QUAT: Quat = [1, 0, 0, 0];
const quatAboutX = (deg: number): Quat => {
  const half = (deg * Math.PI) / 360;
  return [Math.cos(half), Math.sin(half), 0, 0];
};
const quatAboutY = (deg: number): Quat => {
  const half = (deg * Math.PI) / 360;
  return [Math.cos(half), 0, Math.sin(half), 0];
};
/** 平铺示意：手心朝上、指尖朝画面深处 */
const FLAT_POSE = quatAboutX(-90);

/**
 * 四步照搬 cc_part2：一趟同时采出**四样**东西 ——
 * ① 弯折张开基线 + 俯仰参考姿态、② 陀螺零位、③ 偏摆参考姿态、④ 弯折握拳量程。
 * 顺序不能改：②③ 的相对旋转都是相对 ② 的零位算的，而 ①④ 必须一头一尾各取一次极值。
 */
const WIZARD_STEPS: WizardStep[] = [
  {
    key: "open",
    short: "① 平铺",
    title: "水平平铺 · 弯曲基线 + 俯仰参考",
    instruction:
      "双手水平放平、手心朝上、手指完全伸直（照着示意手模摆）并保持静止。这一步同时记录各弯折通道的伸直基线，和「竖立→平铺」这个俯仰动作的参考姿态。",
    curl: 0,
    poseQuat: { LH: FLAT_POSE, RH: FLAT_POSE },
  },
  {
    key: "zero",
    short: "② 竖立",
    title: "抬起竖立 · 陀螺仪零位",
    /*
     * ⚠ "手心朝自己"不能改成"手心朝屏幕"。这两句差 180°，而 `MODEL_MOTION_AXES`
     * 那张表只在"手心朝自己"下成立：换成朝屏幕，① 平铺就从绕 −X 的 90° 变成
     * 一个 180° 复合旋转，轴系整个解歪。理由写在 orientationCalib.ts 文件头。
     *
     * 这里曾经写的就是"手心正对屏幕" —— 同一句话里又要求"摆成与示意手模一致"，
     * 而示意手模（poseQuat 为单位四元数）是手心朝相机、也就是手心朝自己。
     * 两条指示互相矛盾，照文字做的人会得到一份 180° 歪掉的标定。
     */
    instruction:
      "手指保持伸直，前臂向上抬起 90°，手心朝自己（手背对着屏幕），摆成与示意手模一致的竖立姿态并保持静止。这一步记录姿态零位 —— 之后所有帧都按「相对这个姿态转了多少」来显示。",
    curl: 0,
    poseQuat: { LH: IDENTITY_QUAT, RH: IDENTITY_QUAT },
  },
  {
    key: "palms",
    short: "③ 相对",
    title: "双手手心相对 · 翻腕参考",
    /*
     * ⚠ "向内翻"必须写明是哪一边。翻反方向解出来的矩阵是**合法的**，
     * 三个质量数（俯仰/偏摆/轴分离）全部正常，界面只显示一个✓ ——
     * 而用户看到的是绕 Y/Z 的旋转全部反号（摆拇指朝上，手模拇指朝下）。
     * 现在有重力核对能自动纠正（见 orientationCalib.ts 的
     * `validateAxisMapWithGravity`），但那一关要靠 ② 站直才有定论，
     * 所以这里仍然要把方向说清楚，别指望下游兜底。
     */
    instruction:
      "手肘不动，双手向内翻腕，让左右手心相对（照着示意手模的朝向）并保持静止。「向内」= 左手掌心转向你的右边、右手掌心转向你的左边，两掌心最后面对面。转到约 90° 就停，别转过头 —— 超过 165° 转轴的正负号就不稳了。这一步给出偏摆轴，和 ① 的俯仰轴一起反解出轴向映射矩阵。",
    curl: 0,
    poseQuat: { LH: quatAboutY(90), RH: quatAboutY(-90) },
  },
  {
    key: "fist",
    short: "④ 握拳",
    title: "用力握拳 · 弯曲量程",
    instruction:
      "照着示意手模用力握紧拳头并保持静止。这一步定义弯折满量程 —— 这里握得松，之后实机握拳就永远显得不够弯。",
    curl: 1,
    poseQuat: { LH: IDENTITY_QUAT, RH: IDENTITY_QUAT },
  },
];

/**
 * 一步之内的平均加速度（机体系），供重力核对用。全程没有 acc 的旧款手套返回 null。
 *
 * 直接求平均、不剔野值：这一步是静止 3 秒，平均本来就以重力为主；而重力核对只分辨
 * "正着"与"整体翻转 180°"，容差 60°（`GRAVITY_DECISIVE`），远大于抖动带来的偏差。
 * 真要剔野值该用 `imuHealth.ts` 那套按模长筛的做法，不该在这里再写一份。
 */
function averageAcc(samples: CalibSample[]): Vec3 | null {
  const valid = samples.filter(s => s.acc);
  if (!valid.length) return null;
  const sum: Vec3 = [0, 0, 0];
  for (const s of valid) for (let i = 0; i < 3; i++) sum[i] += s.acc![i];
  return [sum[0] / valid.length, sum[1] / valid.length, sum[2] / valid.length];
}

const WIZARD_COUNTDOWN_S = 3;
const WIZARD_SAMPLE_MS = 3000;
/** 一只手这一步至少要收到这么多帧才算有效（30Hz 下 3 秒约 90 帧，5 是很松的下限） */
const WIZARD_MIN_FRAMES = 5;

/*
 * ===== 实时角度引导（②③ 两步） =====
 *
 * 轴向矩阵靠 ①→② 与 ②→③ 两个 90° 动作反解，可实测里出现过 9°、15° 这种
 * 根本没做到位的采样 —— 用户在采样结束前**没有任何读数**知道自己转没转够，
 * 门限（MIN_MOTION_DEG = 40°）不过就只写零位，"朝向对不上"的抱怨多半源于此。
 *
 * 引导用的参照就是上一步刚采完的平均四元数（② 对 ①、③ 对 ②），和汇算里
 * `relativeAxisAngle` 用的是同一份数学，所以引导读数和最终质量数必然一致。
 * ① 是链条的锚点本身、④ 只看弯折，都没有角度可引导。
 *
 * 参与向导的手必须持续稳定，②③ 实测角度在 70–110° 才采样。
 * 等待超时进入重试，不采入明显偏离目标的数据；采样结束再核对稳定性。
 */
const GATE_MIN_DEG = 70;
const GATE_MAX_DEG = 110;
const GATE_TIMEOUT_MS = 30000;

/** ②③ 两步的角度参照取自哪一步；其余步骤无角度引导 */
const STEP_ANCHOR: Partial<Record<StepKey, StepKey>> = {
  zero: "open",
  palms: "zero",
};

type WizardPhase = "countdown" | "sampling" | "blocked" | "done";

/**
 * 单只手的体检状态：弯折两点标定 + 朝向标定 + IMU 漂移自检 + 3D 手模驱动。
 * 左右手各调用一次，两份状态完全独立（标定按 handKey 分别存 localStorage，
 * 所以左手标完不会被右手覆盖，反之亦然）。
 *
 * @param sinkRef 向导采样槽；非 null 时 tick 会把每帧新数据推进去（原始四元数）
 */
function useHandCheck(
  channel: HandChannel,
  handKey: HandKey,
  sinkRef: RefObject<CalibSink | null>
) {
  const frameRef = channel.latestFrameRef; // 全速更新的 ref

  // 3D 手模驱动：手套 100Hz，用 state 传值会把整页每秒重渲染上百次；
  // 这里走 ref，由页面已有的 rAF 循环顺手填上，HandModel 在自己的 useFrame 里读。
  const driveRef = useRef(makeHandDrive());
  /** 上一次推给向导的帧对象，用于去重（rAF 比手套帧率快一倍） */
  const lastSinkFrameRef = useRef<GloveFrame | null>(null);

  // handKey 在一个 hook 实例里是常量（useDualGloveSerial 把 handType 写死成 0x01/0x02），
  // 所以标定只需惰性读一次，不用像单手套版那样监听换手再重载。
  const [bendRange, setBendRange] = useState<BendRange | null>(() =>
    loadBendRange(handKey)
  );
  const bendRangeRef = useRef<BendRange | null>(bendRange);
  /** 最近若干帧的 canonical 弯折 ADC，供"捕捉张开/握拳"取平均 */
  const bendHistoryRef = useRef<number[][]>([]);
  /** 展示用的 5 指比例，节流到 ~12Hz 以免频繁重渲染 */
  const [bendUi, setBendUi] = useState<number[]>([0, 0, 0, 0, 0]);
  const bendUiTickRef = useRef(0);
  const [calibMsg, setCalibMsg] = useState("");

  const captureBendPose = useCallback(
    (pose: "open" | "fist") => {
      const avg = averageBends(bendHistoryRef.current.slice(-CALIB_AVG_FRAMES));
      if (!avg) {
        setCalibMsg("没有数据，先连接这只手的手套");
        return;
      }
      const next: BendRange = {
        open: pose === "open" ? avg : (bendRange?.open ?? []),
        fist: pose === "fist" ? avg : (bendRange?.fist ?? []),
      };
      setBendRange(next);
      bendRangeRef.current = next;
      if (isCalibrated(next)) {
        saveBendRange(handKey, next);
        const w = weakChannels(next);
        setCalibMsg(
          w.length
            ? `标定完成，但 ${w.map(i => FINGER_NAMES[i]).join("/")} 跨度不足 ${MIN_USEFUL_SPAN}，请检查这几路弯折传感器`
            : "两点标定完成，已保存"
        );
      } else {
        setCalibMsg(
          pose === "open" ? "已记录张开，再捕捉握拳" : "已记录握拳，再捕捉张开"
        );
      }
    },
    [bendRange, handKey]
  );

  const resetBendRange = useCallback(() => {
    clearBendRange(handKey);
    setBendRange(null);
    bendRangeRef.current = null;
    setCalibMsg("已清除标定，当前为未标定预览");
  }, [handKey]);

  const weak = useMemo(() => weakChannels(bendRange), [bendRange]);
  const calibrated = isCalibrated(bendRange);
  const spans = useMemo(
    () => (bendRange && calibrated ? channelSpans(bendRange) : null),
    [bendRange, calibrated]
  );
  /** 原始 ADC，纯诊断用：手模看着不对时，先看这一路到底动不动 */
  const [rawUi, setRawUi] = useState<number[]>([0, 0, 0, 0, 0]);

  // ===== 朝向标定（零位 + 轴向映射）=====
  const [orientCalib, setOrientCalib] = useState<OrientationCalib | null>(() =>
    loadOrientationCalib(handKey)
  );
  const orientRef = useRef<OrientationCalib | null>(orientCalib);

  const resetOrientCalib = useCallback(() => {
    clearOrientationCalib(handKey);
    setOrientCalib(null);
    orientRef.current = null;
  }, [handKey]);

  const applyOrientation = useCallback(
    (calib: OrientationCalib) => {
      setOrientCalib(calib);
      orientRef.current = calib;
      saveOrientationCalib(handKey, calib);
    },
    [handKey]
  );

  // ===== IMU 漂移自检 =====
  // 没有独立的自检按钮：结论由四步向导汇算时给出（向导每步都是静止 3 秒）。
  // 只存内存、不落盘 —— 陀螺零偏是掉电即变的硬件状态，存下来隔天就是假消息。
  const [imuReport, setImuReport] = useState<ImuHealthReport | null>(null);

  /** 向导算完后回写这只手的两份标定 + 陀螺结论（都可为 null，表示这一份没采成） */
  const applyWizardResult = useCallback(
    (
      range: BendRange | null,
      orient: OrientationCalib | null,
      imu: ImuHealthReport | null
    ) => {
      if (range && isCalibrated(range)) {
        setBendRange(range);
        bendRangeRef.current = range;
        saveBendRange(handKey, range);
        const w = weakChannels(range);
        setCalibMsg(
          w.length
            ? `向导已写入，但 ${w.map(i => FINGER_NAMES[i]).join("/")} 跨度不足 ${MIN_USEFUL_SPAN}`
            : "向导已写入弯折两点标定"
        );
      }
      if (orient) {
        setOrientCalib(orient);
        orientRef.current = orient;
        saveOrientationCalib(handKey, orient);
      }
      setImuReport(imu);
    },
    [handKey]
  );

  /**
   * 由页面的 rAF 循环每帧调用：填 3D 驱动 + 累积标定历史 + 节流刷新比例条。
   * 只碰 ref 和 setState，所以这个回调是稳定的，可以安全地从循环里经 ref 调用。
   */
  const tick = useCallback(() => {
    const frame = frameRef.current;
    if (
      !channel.isConnected ||
      !frame ||
      performance.now() - frame.timestamp > 250
    ) {
      driveRef.current.hasData = false;
      return;
    }
    const raw = canonicalBendRaw(frame.sensor_data, frame.hand);
    const ratios = bendRatios(
      frame.sensor_data,
      frame.hand,
      bendRangeRef.current
    );
    // 朝向标定：ref⁻¹⊗q（+ 轴向重映射）。没标定时 applyOrientationCalib 原样返回。
    if (frame.quaternionValid !== false && validPose(frame.quaternion)) {
      driveRef.current.quaternion = applyOrientationCalib(
        normalizeQuat(frame.quaternion),
        orientRef.current
      );
    }
    driveRef.current.curl = ratios;
    driveRef.current.hasData = true;

    // 标定用的历史帧（canonical 拇指→小指），只保留够平均用的量
    const hist = bendHistoryRef.current;
    hist.push(raw);
    if (hist.length > CALIB_AVG_FRAMES * 2)
      hist.splice(0, hist.length - CALIB_AVG_FRAMES * 2);

    // 向导采样：手套 30Hz、rAF 60Hz，靠帧对象身份去重，否则每帧会被记两遍。
    // 推的是**原始**四元数 —— 零位是相对旋转，用标定后的值会算出单位旋转。
    // acc/t 顺带一起推：陀螺自检要的数据本来就在帧里，不必另起一条采样链。
    const sink = sinkRef.current;
    if (sink && frame !== lastSinkFrameRef.current) {
      lastSinkFrameRef.current = frame;
      sink(handKey, {
        bend: raw,
        quat:
          frame.quaternionValid === false ? [NaN, 0, 0, 0] : frame.quaternion,
        acc: frame.acceleration,
        t: frame.timestamp,
      });
    }

    // 比例条节流到 ~12Hz：这是唯一会触发 React 重渲染的一路
    bendUiTickRef.current += 1;
    if (bendUiTickRef.current >= 5) {
      bendUiTickRef.current = 0;
      setBendUi(ratios);
      setRawUi(raw);
    }
  }, [frameRef, handKey, sinkRef, channel.isConnected]);

  return {
    handKey,
    channel,
    driveRef,
    bendUi,
    rawUi,
    spans,
    calibMsg,
    captureBendPose,
    resetBendRange,
    weak,
    calibrated,
    orientCalib,
    resetOrientCalib,
    applyOrientation,
    applyWizardResult,
    imuReport,
    tick,
  };
}

type HandCheck = ReturnType<typeof useHandCheck>;

/**
 * 四步校准向导：全程自动连播（每步蒙板倒数 3 秒 → 静止采样 3 秒 → 自动进下一步），
 * 双手同时采 —— 只连了一只手也能跑，另一只手到汇算时因帧数不足自动跳过。
 *
 * 采样不自己起循环：`sinkRef` 交给 useHandCheck 的 tick，页面那个已有的 rAF
 * 循环顺手把帧推进来。这样采样率与手模刷新完全同源，也不会多一条监听链。
 */
function useCalibWizard(
  sinkRef: RefObject<CalibSink | null>,
  onApply: (
    hand: HandKey,
    range: BendRange | null,
    orient: OrientationCalib | null,
    imu: ImuHealthReport | null
  ) => void,
  /** 双手的实时帧（原始四元数），②③ 两步的实时角度引导与角度门都从这里读 */
  frameRefs: Record<HandKey, RefObject<GloveFrame | null>>
) {
  const [open, setOpen] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [phase, setPhase] = useState<WizardPhase>("countdown");
  const [countdown, setCountdown] = useState(WIZARD_COUNTDOWN_S);
  const [progress, setProgress] = useState(0);
  const [counts, setCounts] = useState<Record<HandKey, number>>({
    LH: 0,
    RH: 0,
  });
  const [errorMsg, setErrorMsg] = useState("");
  const [summary, setSummary] = useState<string[]>([]);
  const participants = useRef<HandKey[]>([]);
  const recent = useRef<Record<HandKey, PoseSample[]>>({ LH: [], RH: [] });
  const [attempt, setAttempt] = useState(0);

  /** 本步的采样缓冲 */
  const bufRef = useRef<Record<HandKey, CalibSample[]>>({ LH: [], RH: [] });
  /** 已完成各步的采样 */
  const stepsRef = useRef<
    Partial<Record<StepKey, Record<HandKey, CalibSample[]>>>
  >({});
  // onApply 每次渲染都是新函数，走 ref 才能让下面的回调保持稳定
  const applyRef = useRef(onApply);
  applyRef.current = onApply;

  /** 示意手模的驱动：每步换一次目标姿态，让用户照着摆 */
  const demoLRef = useRef(makeHandDrive());
  const demoRRef = useRef(makeHandDrive());
  useEffect(() => {
    const step = WIZARD_STEPS[stepIndex];
    if (!step) return;
    const pairs: [HandKey, typeof demoLRef][] = [
      ["LH", demoLRef],
      ["RH", demoRRef],
    ];
    for (const [key, ref] of pairs) {
      ref.current.quaternion = step.poseQuat[key];
      ref.current.curl = [0, 1, 2, 3, 4].map(() => step.curl);
      ref.current.hasData = true;
    }
  }, [stepIndex]);

  const push = useCallback<CalibSink>((hand, sample) => {
    bufRef.current[hand].push(sample);
  }, []);

  /** 倒数已到 0、在等角度门放行（界面据此把倒数文案换成"转到位"提示） */
  const [gateWaiting, setGateWaiting] = useState(false);

  /**
   * 某步采完的平均四元数，作实时角度的参照。每次现算：一步约 90 帧的
   * 符号对齐求和，10Hz 的引导轮询下开销可忽略，比维护一份缓存省心。
   */
  const anchorQuat = useCallback((key: StepKey, hand: HandKey): Quat | null => {
    const seg = stepsRef.current[key]?.[hand];
    if (!seg || seg.length < WIZARD_MIN_FRAMES) return null;
    return averageQuaternions(seg.map(s => s.quat));
  }, []);

  /**
   * 这只手此刻相对本步参照转了多少度；null = 本步无引导 / 锚点缺失 / 没有帧。
   * 与汇算共用 `relativeAxisAngle`，读数和最终质量数出自同一份数学。
   */
  const liveAngleDeg = useCallback(
    (hand: HandKey): number | null => {
      const anchorKey = STEP_ANCHOR[WIZARD_STEPS[stepIndex]?.key];
      if (!anchorKey) return null;
      const anchor = anchorQuat(anchorKey, hand);
      const frame = frameRefs[hand].current;
      if (
        !anchor ||
        !frame ||
        performance.now() - frame.timestamp > 250 ||
        frame.quaternionValid === false ||
        !validPose(frame.quaternion)
      )
        return null;
      // 两姿态几乎重合时转轴无定义，对引导来说就是"还没开始转"
      return relativeAxisAngle(anchor, frame.quaternion)?.angleDeg ?? 0;
    },
    [anchorQuat, frameRefs, stepIndex]
  );

  const startWizard = useCallback(() => {
    participants.current = HAND_KEYS.filter(hand => {
      const frame = frameRefs[hand].current;
      return frame && performance.now() - frame.timestamp < 250;
    });
    recent.current = { LH: [], RH: [] };
    stepsRef.current = {};
    bufRef.current = { LH: [], RH: [] };
    setSummary([]);
    setErrorMsg("");
    setCounts({ LH: 0, RH: 0 });
    setStepIndex(0);
    setPhase(participants.current.length ? "countdown" : "blocked");
    if (!participants.current.length)
      setErrorMsg("没有收到新鲜数据。请确认连接后重开向导。");
    setOpen(true);
    setAttempt(n => n + 1);
  }, [frameRefs]);

  const retryStep = useCallback(
    (previous = false) => {
      sinkRef.current = null;
      const index = previous ? Math.max(0, stepIndex - 1) : stepIndex;
      WIZARD_STEPS.slice(index).forEach(s => {
        delete stepsRef.current[s.key];
      });
      recent.current = { LH: [], RH: [] };
      bufRef.current = { LH: [], RH: [] };
      setErrorMsg("");
      setStepIndex(index);
      setPhase("countdown");
      setAttempt(n => n + 1);
    },
    [sinkRef, stepIndex]
  );

  const closeWizard = useCallback(() => {
    sinkRef.current = null;
    setOpen(false);
  }, [sinkRef]);

  /** 汇算：两只手各自独立，一只手数据不全只跳过它、不影响另一只 */
  const assemble = useCallback(() => {
    const steps = stepsRef.current;
    const lines: string[] = [];
    for (const hand of participants.current) {
      const of = (k: StepKey) => steps[k]?.[hand] ?? [];
      const openS = of("open");
      const zeroS = of("zero");
      const palmsS = of("palms");
      const fistS = of("fist");
      const name = handName(hand);
      if (
        openS.length < WIZARD_MIN_FRAMES ||
        zeroS.length < WIZARD_MIN_FRAMES ||
        fistS.length < WIZARD_MIN_FRAMES
      ) {
        lines.push(`${name}：采样帧不足，未写入（这只手没连就属正常）`);
        // 清掉这只手上一轮的陀螺结论：这一轮没测，留着旧的就是假消息
        applyRef.current(hand, null, null, null);
        continue;
      }
      const range: BendRange = {
        open: averageBends(openS.map(s => s.bend)) ?? [],
        fist: averageBends(fistS.map(s => s.bend)) ?? [],
      };
      const ok = isCalibrated(range);
      let orient = assembleOrientationCalib(
        hand,
        averageQuaternions(zeroS.map(s => s.quat)),
        averageQuaternions(openS.map(s => s.quat)),
        palmsS.length >= WIZARD_MIN_FRAMES
          ? averageQuaternions(palmsS.map(s => s.quat))
          : null,
        // 零位那步的平均加速度：用来核对轴向矩阵有没有整体翻转
        averageAcc(zeroS)
      );
      const quality = orient?.axisQuality;
      if (
        !orient?.axisMap ||
        !quality ||
        quality.separationDeg < 70 ||
        quality.pitchDeg < GATE_MIN_DEG ||
        quality.pitchDeg > GATE_MAX_DEG ||
        quality.swingDeg < GATE_MIN_DEG ||
        quality.swingDeg > GATE_MAX_DEG
      ) {
        lines.push(
          `${name}：朝向参考不一致，本轮未覆盖已有朝向。请用三轴验证检查原始角度。`
        );
        orient = null;
      } else {
        orient.updatedAt = Date.now();
        orient.method = "four-step";
      }

      // 分步计算，防止把步骤间的主动转动混入；累计角度仍不能视为零偏。
      const imu = worstReport(
        WIZARD_STEPS.map(s => {
          const seg = of(s.key);
          if (seg.length < WIZARD_MIN_FRAMES) return null;
          return analyzeImuHealth(
            seg.map(x => ({ q: x.quat, acc: x.acc, t: x.t }))
          );
        })
      );

      applyRef.current(hand, ok ? range : null, orient, imu);

      if (ok) {
        const spans = channelSpans(range).map(s => Math.round(s));
        const weak = weakChannels(range);
        lines.push(
          `${name} 弯折跨度：${spans.join(" / ")}${
            weak.length
              ? ` —— ${weak.map(i => FINGER_NAMES[i]).join("、")} 不足 ${MIN_USEFUL_SPAN}，这几路传感器要查`
              : "（拇指→小指，都够用）"
          }`
        );
      } else {
        lines.push(`${name} 弯折：取值异常，未写入`);
      }
      if (!orient) {
        lines.push(`${name} 朝向：本轮未通过检查，原有标定保持不变`);
      } else if (orient.axisMap) {
        const q = orient.axisQuality!;
        // 三个数后面都跟上目标值：只报实测值时 40° 和 160° 看起来一样正常
        lines.push(
          `${name} 朝向：零位 + 轴向映射已写入（俯仰 ${q.pitchDeg.toFixed(0)}° · 偏摆 ${q.swingDeg.toFixed(0)}° · 轴分离 ${q.separationDeg.toFixed(0)}°，三项都应接近 ${TARGET_MOTION_DEG}°）`
        );
        /*
         * 重力核对的结论。**必须单独报一行**：三个质量数对"③ 翻反方向"是全绿的，
         * 唯一能说明这件事的就是这一关。三种口径分开说，因为该做的事不一样：
         *   纠正了 → 已经好了，但下次那一步该往对的方向翻
         *   通过了 → 真的没问题
         *   没做   → 旧款手套不上报加速度，这一关缺席，用户得自己看手模
         * （"没定论"那种由 axisQualityWarning 报，不在这里重复。）
         */
        const g = orient.gravityCheck;
        if (g?.corrected) {
          lines.push(
            `${name} 朝向：重力核对发现轴向矩阵整体翻转了 180°，**已自动纠正**` +
              `（零位测到的"上"换算成 ${g.upY.toFixed(2)}，应为 +1）—— ` +
              `原因是 ③ 那步翻反了方向：${handName(hand)}"往内"是把掌心转向${
                hand === "LH" ? "你的右边" : "你的左边"
              }。这次已经修好，下次照对的方向做就不必纠正`
          );
        } else if (g) {
          lines.push(`${name} 朝向：重力核对通过（上 = ${g.upY.toFixed(2)}）`);
        } else {
          lines.push(
            `${name} 朝向：这只手套不上报加速度，无法做重力核对 —— ` +
              `"③ 翻反方向"这类整体翻转拦不住，请摆个拇指朝上确认手模跟着朝上`
          );
        }
        // 过了门限不等于好用 —— 这条不报的话，一份拧歪的矩阵只会显示成一个✓
        const warn = axisQualityWarning(orient);
        if (warn) lines.push(`${name} 朝向 WARN：${warn}`);
      } else {
        lines.push(
          `${name} 朝向：只写入零位 —— ${axisMapFailReason(orient)}。整体朝向已经对齐，但翻腕方向可能仍会串轴`
        );
      }
      if (imu)
        lines.push(
          `${name} 采样姿态变化：${imu.stillRotationDegPerMin?.toFixed(0) ?? "—"}°/分钟（累计值，包含抖动，不等于零偏）。有疑问请做独立静置检测。`
        );
    }
    setSummary(lines);
    setPhase("done");
  }, []);

  const finishSampling = useCallback(() => {
    const step = WIZARD_STEPS[stepIndex];
    const buf = bufRef.current;
    for (const hand of participants.current) {
      const hold = checkPoseHold(
        buf[hand].map(s => ({ q: s.quat, t: s.t })),
        2700,
        performance.now()
      );
      if (!hold.ok) {
        sinkRef.current = null;
        setErrorMsg(
          `${handName(hand)}：${hold.reason}，本步未保存。摆稳后重试。`
        );
        setPhase("blocked");
        return;
      }
    }
    if (
      buf.LH.length < WIZARD_MIN_FRAMES &&
      buf.RH.length < WIZARD_MIN_FRAMES
    ) {
      // 两只手都没数据 = 手套断流，重做这一步而不是往下走（否则最后汇算全空）
      setErrorMsg(`「${step.short}」没有足够数据，请确认连接后重试。`);
      setPhase("blocked");
      return;
    }
    stepsRef.current[step.key] = { LH: [...buf.LH], RH: [...buf.RH] };
    setErrorMsg("");
    if (stepIndex >= WIZARD_STEPS.length - 1) assemble();
    else {
      setStepIndex(stepIndex + 1);
      setPhase("countdown");
    }
  }, [stepIndex, assemble, sinkRef]);

  const startSampling = useCallback(() => {
    bufRef.current = { LH: [], RH: [] };
    setCounts({ LH: 0, RH: 0 });
    setProgress(0);
    setGateWaiting(false);
    sinkRef.current = push; // 从这一刻起 tick 开始往缓冲里推帧
    setPhase("sampling");
  }, [push, sinkRef]);

  // 每步开场倒数，数到 0 自动开始采样，全程无需点击。
  // ②③ 两步多一道角度门：倒数完还没转够 GATE_MIN_DEG 就先不采，
  // 稳定且角度在范围内才放行，超时提供重试，不强行采样。
  useEffect(() => {
    if (!open || phase !== "countdown") return;
    setCountdown(WIZARD_COUNTDOWN_S);
    setGateWaiting(false);
    const anchorKey = STEP_ANCHOR[WIZARD_STEPS[stepIndex]?.key];
    recent.current = { LH: [], RH: [] };
    const t0 = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = Date.now() - t0;
      const now = performance.now();
      for (const hand of participants.current) {
        const frame = frameRefs[hand].current;
        const list = recent.current[hand];
        if (frame && frame.timestamp !== list.at(-1)?.t) {
          list.push({
            q:
              frame.quaternionValid === false
                ? [NaN, 0, 0, 0]
                : frame.quaternion,
            t: frame.timestamp,
          });
        }
        recent.current[hand] = list.filter(s => now - s.t <= 1400);
      }
      const left = WIZARD_COUNTDOWN_S - Math.floor(elapsed / 1000);
      if (left > 0) {
        setCountdown(left);
        return;
      }
      setCountdown(0);
      const pending =
        !participants.current.length ||
        participants.current.some(hand => {
          if (!checkPoseHold(recent.current[hand], 900, now).ok) return true;
          if (!anchorKey) return false;
          if (!anchorQuat(anchorKey, hand)) return true;
          const deg = liveAngleDeg(hand);
          return deg === null || deg < GATE_MIN_DEG || deg > GATE_MAX_DEG;
        });
      if (pending) {
        setGateWaiting(true);
        if (elapsed >= WIZARD_COUNTDOWN_S * 1000 + GATE_TIMEOUT_MS) {
          window.clearInterval(timer);
          setErrorMsg(
            "未取得稳定且接近目标的姿态，本步未采样。请重试或返回上一步检查参考；持续偏差大时使用朝向诊断。"
          );
          setPhase("blocked");
        }
        return;
      }
      window.clearInterval(timer);
      startSampling();
    }, 100);
    return () => window.clearInterval(timer);
  }, [
    open,
    phase,
    stepIndex,
    startSampling,
    anchorQuat,
    liveAngleDeg,
    frameRefs,
    attempt,
  ]);

  // 采样计时；cleanup 里摘掉 sink，所以中途关窗/卸载都不会继续往缓冲里灌帧
  useEffect(() => {
    if (!open || phase !== "sampling") return;
    const t0 = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = Date.now() - t0;
      setProgress(Math.min(100, (elapsed / WIZARD_SAMPLE_MS) * 100));
      setCounts({
        LH: bufRef.current.LH.length,
        RH: bufRef.current.RH.length,
      });
      if (elapsed >= WIZARD_SAMPLE_MS) {
        window.clearInterval(timer);
        finishSampling();
      }
    }, 80);
    return () => {
      window.clearInterval(timer);
      sinkRef.current = null;
    };
  }, [open, phase, stepIndex, finishSampling, sinkRef]);

  return {
    open,
    stepIndex,
    phase,
    countdown,
    progress,
    counts,
    errorMsg,
    summary,
    demoLRef,
    demoRRef,
    gateWaiting,
    liveAngleDeg,
    retryStep,
    startWizard,
    closeWizard,
    restart: startWizard,
  };
}

type CalibWizard = ReturnType<typeof useCalibWizard>;

export default function VirtualMocap() {
  // 连接住在 App 层的 GloveProvider 里，全应用只开一次串口。
  // 本页是唯一放"连接/断开"按钮的地方（第 1 步），第 2~4 步只读状态。
  const {
    left,
    right,
    isSupported: gloveSupported,
    anyConnected,
    bothConnected,
  } = useGloves();

  // 向导采样槽：向导开始采样时挂上，采完摘掉。两只手的 tick 都往这一个槽里推。
  const sinkRef = useRef<CalibSink | null>(null);

  const [diagnosticsOpen, setDiagnosticsOpen] = useState<"zero" | "fit" | null>(
    null
  );
  useEffect(() => {
    const imported = () => { if (pendingManualImport()) setDiagnosticsOpen("fit"); };
    imported();
    window.addEventListener(MANUAL_IMPORT_READY, imported);
    return () => window.removeEventListener(MANUAL_IMPORT_READY, imported);
  }, []);
  const L = useHandCheck(left, "LH", sinkRef);
  const R = useHandCheck(right, "RH", sinkRef);
  // rAF 循环通过 ref 读这两个对象，避免把每次渲染新建的 L/R 放进 effect 依赖
  const checksRef = useRef<HandCheck[]>([]);
  checksRef.current = [L, R];

  // 必须 memo：向导拿它进了 effect 依赖链，每次渲染新建对象会让倒数计时器不停重置
  const wizardFrameRefs = useMemo(
    () => ({ LH: left.latestFrameRef, RH: right.latestFrameRef }),
    [left.latestFrameRef, right.latestFrameRef]
  );
  const wizard = useCalibWizard(
    sinkRef,
    (hand, range, orient, imu) => {
      const hc = checksRef.current.find(c => c.handKey === hand);
      hc?.applyWizardResult(range, orient, imu);
    },
    wizardFrameRefs
  );

  /**
   * 骨架回归只画一副：模型是用 pickPrimaryHand 训练的、对左右手不敏感。
   * 优先取右手，右手没连才退到左手。
   */
  const skeletonSide: HandKey | null = right.isConnected
    ? "RH"
    : left.isConnected
      ? "LH"
      : null;

  const [modelReady, setModelReady] = useState(isSkeletonModelLoaded());
  const [modelName, setModelName] = useState("");
  const [loadMsg, setLoadMsg] = useState("");
  const [predFps, setPredFps] = useState(0);
  const [confidence, setConfidence] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** 画布底色的**实际值**（从 --hud-stage 解析出来的）。见下面 render 里那段注释 */
  const stageColorRef = useRef("#eaeff6");
  const animRef = useRef<number>(0);
  const prevLandmarksRef = useRef<{ x: number; y: number; z: number }[] | null>(
    null
  );
  const fpsCountRef = useRef({ count: 0, lastTime: performance.now() });

  // 自动加载最新骨架模型
  useEffect(() => {
    if (!isSkeletonModelLoaded()) {
      (async () => {
        setLoadMsg("正在加载骨架模型...");
        const saved = await getLatestSkeletonModel();
        if (saved) {
          try {
            await loadSkeletonModelFromSaved(saved);
            setModelReady(true);
            setModelName(saved.name);
            setLoadMsg("");
          } catch (e: any) {
            setLoadMsg(`模型加载失败: ${e.message}`);
          }
        } else {
          setLoadMsg("未找到骨架模型，请先训练");
        }
      })();
    } else {
      setModelReady(true);
    }
  }, []);

  // 平滑关键点
  const smoothLandmarks = useCallback(
    (current: { x: number; y: number; z: number }[]) => {
      const prev = prevLandmarksRef.current;
      if (!prev || prev.length !== current.length) {
        prevLandmarksRef.current = current;
        return current;
      }
      const alpha = 0.45; // 稍微更平滑
      const smoothed = current.map((lm, i) => ({
        x: prev[i].x + alpha * (lm.x - prev[i].x),
        y: prev[i].y + alpha * (lm.y - prev[i].y),
        z: prev[i].z + alpha * (lm.z - prev[i].z),
      }));
      prevLandmarksRef.current = smoothed;
      return smoothed;
    },
    []
  );

  // Canvas 渲染循环
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;

    // 解析一次就够：换肤不会在运行时发生（皮肤是 CSS 里的常量，不是可切的开关）
    stageColorRef.current =
      getComputedStyle(canvas).getPropertyValue("--hud-stage").trim() ||
      "#eaeff6";

    const render = () => {
      const time = Date.now() / 1000;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

      /* 背景。**canvas 2D 里不能写 var(--hud-page)** —— fillStyle 是 CSS 色值
         的解析，不走自定义属性；给它 var() 会被静默丢弃（画出来是上一帧残留）。
         所以这里从元素的计算样式里把变量读成实际颜色再喂进去，
         页面换肤时这块底也跟着变。 */
      ctx.fillStyle = stageColorRef.current;
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      drawGrid(ctx, CANVAS_W, CANVAS_H);
      drawScanLines(ctx, CANVAS_W, CANVAS_H);

      // --- 两块 3D 手模的驱动数据：与骨架回归完全无关，没有模型也照样填。
      //     经 ref 拿 useHandCheck，避免把每次渲染新建的 L/R 放进本 effect 的依赖里。 ---
      const checks = checksRef.current;
      for (const hc of checks) hc.tick();

      // 骨架回归只画一副：模型对左右手不敏感，优先用右手（checks[1]）的帧
      const frame =
        checks[1]?.channel.latestFrameRef.current ??
        checks[0]?.channel.latestFrameRef.current ??
        null;
      let predicted = false;

      if (frame && modelReady && isSkeletonModelLoaded()) {
        const landmarks = predictSkeleton(frame.mapped_data, frame.quaternion);

        if (landmarks && landmarks.length === 21) {
          const smoothed = smoothLandmarks(landmarks);
          drawSkeleton(ctx, smoothed, CANVAS_W, CANVAS_H);
          drawJoints(ctx, smoothed, CANVAS_W, CANVAS_H);
          drawPalmCrosshair(ctx, smoothed, CANVAS_W, CANVAS_H);
          predicted = true;

          // 计算置信度（基于关键点分布的合理性）
          const spread = computeSpread(smoothed);
          setConfidence(Math.min(100, Math.max(0, spread)));

          // FPS 计算
          fpsCountRef.current.count++;
          const now = performance.now();
          if (now - fpsCountRef.current.lastTime > 1000) {
            setPredFps(fpsCountRef.current.count);
            fpsCountRef.current.count = 0;
            fpsCountRef.current.lastTime = now;
          }
        }
      }

      if (!predicted) {
        drawWaitingState(ctx, CANVAS_W, CANVAS_H, time);
      }

      drawHUDCorners(ctx, CANVAS_W, CANVAS_H);

      // HUD 标签
      ctx.save();
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = "rgba(245, 158, 11, 0.6)";
      ctx.fillText("VIRTUAL MOCAP — TACTILE ONLY", 12, 18);
      if (predicted) {
        ctx.fillStyle = "rgba(0, 229, 160, 0.5)";
        ctx.fillText(`PRED FPS: ${predFps}`, CANVAS_W - 100, 18);
      }
      ctx.restore();

      animRef.current = requestAnimationFrame(render);
    };

    render();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [modelReady, smoothLandmarks, predFps]);

  return (
    // h-screen 而不是 min-h-screen：主区那个网格要按"剩下多少高度"分配三块视口，
    // 页面可以整体变高的话它就没有确定高度，会被撑到一屏放不下（侧栏内容很长）。
    // 侧栏自己 overflow-y-auto 内部滚动，主区永远是一屏。首页 HUD 也是这个写法。
    <div
      className="h-screen flex flex-col overflow-hidden"
      style={{ backgroundColor: "var(--hud-page)" }}
    >
      {/* 顶部导航 */}
      <header className="h-12 flex items-center justify-between px-4 border-b border-[#1677ff]/15 shrink-0">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[#1677ff]/20" />
          <span className="text-xs font-bold tracking-widest text-[var(--hud-warn)] font-mono">
            VIRTUAL MOCAP
          </span>
          <span className="text-[9px] text-[var(--hud-dim)] font-mono ml-2">
            TACTILE → SKELETON / 3D HAND（两条独立链路 · 双手体检）
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          {modelReady && (
            <span className="text-[var(--hud-warn)] flex items-center gap-1">
              <Bone className="w-3 h-3" />
              MODEL: {modelName || "LOADED"}
            </span>
          )}
          {/* LH/RH 状态点也在 StepNav 里，不再单独挂 ConnChip */}
          <StepNav />
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/*
         * 主区排布：2×2 四个格子 —— 上排左右手 3D，下排骨架回归（严格 4:3）+ 传感器热力图。
         * 用**填满高度的网格**而不是 flex-wrap：几块 4:3 的视口按自然尺寸排会一屏放不下，
         * 换行后又被垂直居中，就成了"上面一大片空白、骨架那块掉到屏幕外"。
         * 热力图升级成同级格子（不再是塞在骨架旁边的窄条）：它本来挤在 288px 的侧栏里、
         * 还得滚到底才看得见，而这是判断"某一路传感器有没有反应"最直观的一张图。
         *
         * 窄屏（<xl）退成一列四行：此时给 min-h-[900px]，主区自己纵向滚，
         * 让每块至少 215px 高 —— 硬塞进一屏只会四块全变成没法看的扁条。
         */}
        <div className="flex-1 min-w-0 p-3 overflow-y-auto">
          <div className="h-full grid gap-3 min-h-[900px] grid-cols-1 grid-rows-[repeat(4,minmax(215px,1fr))] xl:min-h-[560px] xl:grid-cols-2 xl:grid-rows-[1.15fr_1fr]">
            {/* 弯折 + IMU → 3D 手模，左右各一块。不依赖任何已训练模型 */}
            <HandViewport hc={L} />
            <HandViewport hc={R} />

            {/* 骨架回归 → 2D 火柴人。只画一副，数据源见右侧 PREDICTION 的 SOURCE */}
            <Viewport
              boxClass="flex-1 min-h-0 aspect-[4/3] w-auto mx-auto"
              label="SKELETON REGRESSION"
              hint={
                skeletonSide
                  ? `模型推理 · 21 关键点 · 源=${skeletonSide === "LH" ? "左手" : "右手"}`
                  : "模型推理 · 21 关键点"
              }
              badge={
                <>
                  <Hand className="w-3 h-3 inline mr-1" />
                  GLOVE-ONLY MODE
                </>
              }
            >
              {/* canvas 内部分辨率固定 640×480，容器严格 4:3，不会把人拉变形 */}
              <canvas
                ref={canvasRef}
                className="block w-full h-full"
                style={{ imageRendering: "auto" }}
              />
            </Viewport>

            <SensorPanel checks={[L, R]} />
          </div>
        </div>

        {/* 右侧面板 */}
        <div className="w-72 border-l border-[#1677ff]/15 overflow-y-auto p-3 space-y-4 shrink-0">
          {/*
            体检汇总：两只手全过才给绿灯，否则直接点名缺哪只手的哪一项。

            两只手都要过。弯折标定本身只喂本页的 3D 手模、不进数据集，但**死通道只能
            靠它的跨度查出来** —— 一路弯折坏掉会往 137 维里灌一个常数，训练不报错、
            能量曲线也正常，只表现为某几个词永远学不好。所以这一项不是"手模好看"的
            可选项，别因为"标定不进数据集"就把它从 checklist 里摘掉。
          */}
          <Section title="CHECKLIST">
            {[L, R].map(hc => {
              const issues = handIssues(hc);
              const ok = issues.length === 0;
              return (
                <div key={hc.handKey} className="flex items-start gap-1.5">
                  {ok ? (
                    <CheckCircle2 className="w-3 h-3 shrink-0 mt-px text-[var(--hud-ok)]" />
                  ) : (
                    <AlertTriangle className="w-3 h-3 shrink-0 mt-px text-[var(--hud-warn)]" />
                  )}
                  <span className="text-[10px] font-mono w-5 shrink-0 text-[var(--hud-soft)]">
                    {hc.handKey}
                  </span>
                  <span
                    className="flex-1 text-[9px] font-mono leading-relaxed"
                    style={{ color: ok ? "var(--hud-ok)" : "var(--hud-warn)" }}
                  >
                    {ok ? "全部通过" : issues.join(" · ")}
                  </span>
                </div>
              );
            })}
          </Section>

          {/* 首次完整校准；日常归零复用安装方向，诊断按需运行。 */}
          <Section title="CALIBRATION WIZARD">
            <button
              onClick={wizard.startWizard}
              disabled={!anyConnected || Boolean(diagnosticsOpen)}
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5 disabled:opacity-40"
              style={{ borderColor: "rgba(168,85,247,0.4)" }}
            >
              <Wand2 className="w-3 h-3" />
              首次 / 重做四步校准
            </button>
            <button
              onClick={() => setDiagnosticsOpen("fit")}
              disabled={wizard.open}
              className="w-full cyber-btn px-3 py-2 rounded-sm text-xs border-violet-500 disabled:opacity-40"
            >
              手动角度校准（无需摄像头）
            </button>
            <button
              onClick={() => setDiagnosticsOpen("zero")}
              disabled={wizard.open}
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] disabled:opacity-40"
            >
              朝向检查 / 日常归零
            </button>
            <a href="/calibration-history" target="_blank" rel="noopener noreferrer" className="w-full cyber-btn block text-center px-3 py-2 rounded-sm text-xs">
              历史数据 / 同姿势对比 ↗
            </a>
            <p className="text-[9px] text-[var(--hud-soft)] leading-relaxed">
              标定自动保留。每次上电先快速归零；方向偏差用手动角度校准，检测与验证按需使用。
            </p>
            <OrientBlock hc={L} />
            <OrientBlock hc={R} />
            <div className="pt-1.5 border-t border-[#7c3aed]/10 space-y-1">
              <ImuVerdictBlock hc={L} />
              <ImuVerdictBlock hc={R} />
            </div>
          </Section>

          {/* 手套连接：两只手分别一个 COM 口，各连一次 */}
          <Section title="GLOVE CONNECTION">
            {!gloveSupported && (
              <p className="text-[8px] text-[var(--hud-err)] leading-relaxed">
                当前浏览器不支持 Web Serial，请用 Chrome / Edge。
              </p>
            )}
            <ConnRow hc={L} />
            <ConnRow hc={R} />
            <p className="text-[8px] text-[var(--hud-faint)] leading-relaxed">
              {bothConnected
                ? "双手已连接。"
                : anyConnected
                  ? "还差一只手 —— 点上面的按钮再选另一个 CH343 口。"
                  : "两只手在不同 COM 口，需分别连接，各弹一次串口选择窗。"}
            </p>
          </Section>

          {/* 推理状态 */}
          <Section title="PREDICTION">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[var(--hud-dim)]">SOURCE</span>
              <span className="text-[var(--hud-soft)]">
                {skeletonSide ? (skeletonSide === "LH" ? "左手" : "右手") : "—"}
              </span>
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[var(--hud-dim)]">MODEL</span>
              <span
                className={
                  modelReady ? "text-[var(--hud-ok)]" : "text-[var(--hud-err)]"
                }
              >
                {modelReady ? "READY" : "NOT LOADED"}
              </span>
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[var(--hud-dim)]">PRED FPS</span>
              <span className="text-[var(--hud-warn)]">{predFps}</span>
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[var(--hud-dim)]">CONFIDENCE</span>
              <span className="text-[var(--hud-wrist)]">
                {confidence.toFixed(0)}%
              </span>
            </div>
            {/* 置信度条 */}
            <div className="h-1.5 bg-[var(--hud-track)] rounded-full overflow-hidden border border-[#d97706]/10">
              <div
                className="h-full rounded-full transition-all duration-200"
                style={{
                  width: `${confidence}%`,
                  background:
                    confidence > 70
                      ? "var(--hud-ok)"
                      : confidence > 40
                        ? "var(--hud-warn)"
                        : "var(--hud-err)",
                  boxShadow: `0 0 6px ${
                    confidence > 70
                      ? "rgba(0,229,160,0.5)"
                      : confidence > 40
                        ? "rgba(245,158,11,0.5)"
                        : "rgba(255,45,123,0.5)"
                  }`,
                }}
              />
            </div>
            {loadMsg && (
              <p className="text-[8px] text-[var(--hud-warn)]">{loadMsg}</p>
            )}
          </Section>

          {/*
            弯折两点标定 —— 3D 手模的手指驱动源，左右手各一份（分别存 localStorage）。
            通道输出是 8 位 ADC 且**极性未知**，所以必须张开/握拳各捕捉一次才能换算成
            角度；未标定时手模只是柔和预览（满量程只到 42%，握拳不会成形）。

            这里的两个按钮是单点补录用的，主路径是上面的四步向导。
            每行右侧两个小数字 = 当前原始 ADC · 标定跨度：纹丝不动 = 那一路传感器没反应，
            在动但跨度小 = 标定时没做到极值。这两个数字是排查的入口，别当装饰删掉。
          */}
          <Section title="BEND CALIBRATION">
            <BendCalibBlock hc={L} />
            <BendCalibBlock hc={R} />
          </Section>

          {/*
            这里曾有一块「NEXT · 去采集」，把三条采集入口（静态/时序/句子）平铺成
            三个按钮。删掉了 —— header 右侧的 StepNav 在 /mocap 上已经给出同样的
            三个出口，两处并列只是同一件事写两遍。
            ⚠ 三条链路必须都能进得去（原来这页只有一个指向 /collect 的「数据采集」，
            另外两条根本进不去）。要动 StepNav 的 `/mocap` 那条时记住这一点。
          */}

          {/* 快捷导航 */}
          <Section title="NAVIGATION">
            <Link
              href="/train-skeleton"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{ borderColor: "rgba(245,158,11,0.3)" }}
            >
              <Activity className="w-3 h-3" />
              骨架训练
            </Link>
            <Link
              href="/translate"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5 mt-1.5"
            >
              手语翻译
            </Link>
          </Section>
        </div>
      </div>

      {diagnosticsOpen && (
        <MotionDiagnostics
          initialMode={diagnosticsOpen}
          channels={{ LH: left, RH: right }}
          calibrations={{ LH: L.orientCalib, RH: R.orientCalib }}
          drives={{ LH: L.driveRef, RH: R.driveRef }}
          onApply={(hand, calib) =>
            checksRef.current
              .find(hc => hc.handKey === hand)
              ?.applyOrientation(calib)
          }
          onClose={() => setDiagnosticsOpen(null)}
        />
      )}
      {wizard.open && (
        <CalibWizardModal
          wz={wizard}
          connected={{ LH: left.isConnected, RH: right.isConnected }}
        />
      )}
    </div>
  );
}

// ===== 辅助组件 =====

function verdictColor(verdict: ImuHealthReport["verdict"]): string {
  switch (verdict) {
    case "ok":
      return "var(--hud-ok)";
    case "warn":
      return "var(--hud-warn)";
    case "bad":
      return "var(--hud-err)";
    default:
      return "var(--hud-soft)";
  }
}

/** 中文手别名，UI 里到处要用 */
function handName(handKey: HandKey): string {
  return handKey === "LH" ? "左手" : "右手";
}

/**
 * 这只手还缺哪一项体检。空数组 = 过了。
 * 注意标定状态来自 localStorage，所以手套已拔掉也算"标定过"——
 * 这正是要的语义：记录的是"这只手查过一次通道跨度"。
 */
function handIssues(hc: HandCheck): string[] {
  const out: string[] = [];
  if (!hc.channel.isConnected) out.push("未连接");
  if (!hc.calibrated) out.push("弯折未标定");
  else if (hc.weak.length)
    out.push(`${hc.weak.map(i => FINGER_NAMES[i]).join("/")} 通道跨度不足`);
  // 只查"有没有零位"。轴向映射缺失不算不合格 —— 它取决于两个动作做得够不够开，
  // 而且朝向标定纯粹是显示层的事、不进数据集，缺了它的原因在向导那一栏写着。
  if (!hc.orientCalib) out.push("朝向未标定");
  if (hc.imuReport && ["warn", "bad"].includes(hc.imuReport.verdict))
    out.push("采样姿态变化偏大，可用独立静置检测复核（不等于零偏）");
  return out;
}

/** 一只手的连接状态 + 连/断按钮 */
function ConnRow({ hc }: { hc: HandCheck }) {
  const ch = hc.channel;
  const name = handName(hc.handKey);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] font-mono">
        <span className="text-[var(--hud-dim)]">{name}</span>
        <span
          className={
            ch.isConnected ? "text-[var(--hud-ok)]" : "text-[var(--hud-err)]"
          }
        >
          {ch.isConnected
            ? `${ch.gloveFps} FPS · ${ch.gloveFrameCount}`
            : ch.isConnecting
              ? "CONNECTING..."
              : "DISCONNECTED"}
        </span>
      </div>
      {ch.error && (
        <p className="text-[8px] text-[var(--hud-err)]">{ch.error}</p>
      )}
      <button
        onClick={ch.isConnected ? ch.disconnect : ch.connect}
        disabled={ch.isConnecting}
        className="w-full cyber-btn px-3 py-1 rounded-sm text-[9px] flex items-center justify-center gap-1.5"
        style={{
          borderColor: ch.isConnected
            ? "rgba(255,45,123,0.3)"
            : "rgba(0,229,160,0.3)",
        }}
      >
        {ch.isConnected ? (
          <>
            <WifiOff className="w-3 h-3" /> 断开{name}
          </>
        ) : (
          <>
            <Wifi className="w-3 h-3" /> 连接{name}
          </>
        )}
      </button>
    </div>
  );
}

/** 一只手的朝向标定状态（零位 / 轴向映射各自独立） */
function OrientBlock({ hc }: { hc: HandCheck }) {
  const calib = hc.orientCalib;
  const fail = calib ? axisMapFailReason(calib) : null;
  // 写入了矩阵但质量不佳：状态灯仍是绿的（矩阵确实在用），但要把话说出来
  const warn = calib ? axisQualityWarning(calib) : null;
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-[10px] font-mono">
        <span className="text-[var(--hud-soft)]">
          {handName(hc.handKey)} 朝向
        </span>
        <span
          style={{
            color: !calib
              ? "var(--hud-warn)"
              : calib.axisMap
                ? "var(--hud-ok)"
                : "var(--hud-accent)",
          }}
        >
          {!calib ? "未标定" : calib.axisMap ? "零位 + 轴向" : "仅零位"}
        </span>
      </div>
      {calib?.method === "multi-pose" && (
        <p className="text-[8px] text-[var(--hud-dim)]">
          多姿态拟合 · 轴误差 {calib.fitErrorDeg?.toFixed(1) ?? "—"}°
        </p>
      )}
      {calib?.updatedAt && (
        <p className="text-[8px] text-[var(--hud-faint)]">
          最近更新 {new Date(calib.updatedAt).toLocaleString()}
        </p>
      )}
      {calib?.axisQuality && (
        <div className="text-[8px] font-mono text-[var(--hud-dim)]">
          俯仰 {calib.axisQuality.pitchDeg.toFixed(0)}° · 偏摆{" "}
          {calib.axisQuality.swingDeg.toFixed(0)}° · 分离{" "}
          {calib.axisQuality.separationDeg.toFixed(0)}°
          <span className="text-[var(--hud-faint)]">
            {" "}
            / 目标 {TARGET_MOTION_DEG}°
          </span>
        </div>
      )}
      {calib && fail && (
        <p className="text-[8px] text-[var(--hud-accent)] leading-relaxed">
          {fail}
        </p>
      )}
      {calib && warn && (
        <p className="text-[8px] text-[var(--hud-warn)] leading-relaxed">
          {warn}
        </p>
      )}
      {calib && (
        <button
          onClick={hc.resetOrientCalib}
          className="text-[8px] font-mono text-[#e11d48]/70 hover:text-[var(--hud-err)] underline"
        >
          清除{handName(hc.handKey)}朝向标定
        </button>
      )}
    </div>
  );
}

/** 四步校准向导弹窗：示意手模 + 倒计时 / 采样进度 / 汇总 */
function CalibWizardModal({
  wz,
  connected,
}: {
  wz: CalibWizard;
  connected: Record<HandKey, boolean>;
}) {
  const step = WIZARD_STEPS[wz.stepIndex];
  const done = wz.phase === "done";
  // ②③ 两步的实时角度引导：~8Hz 轮询够了，读数只是给人看的
  const hasAngleGuide = !done && Boolean(step && STEP_ANCHOR[step.key]);
  const [liveDeg, setLiveDeg] = useState<Record<HandKey, number | null>>({
    LH: null,
    RH: null,
  });
  useEffect(() => {
    if (!hasAngleGuide) return;
    const timer = window.setInterval(() => {
      setLiveDeg({
        LH: wz.liveAngleDeg("LH"),
        RH: wz.liveAngleDeg("RH"),
      });
    }, 120);
    return () => window.clearInterval(timer);
  }, [hasAngleGuide, wz.liveAngleDeg]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--hud-scrim)] p-4">
      <div
        className="w-full max-w-3xl max-h-full overflow-y-auto rounded-sm border p-4 space-y-3"
        style={{
          backgroundColor: "var(--hud-page)",
          borderColor: "rgba(168,85,247,0.35)",
          boxShadow: "0 0 40px rgba(168,85,247,0.12)",
        }}
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold tracking-widest font-mono text-[var(--hud-violet)]">
            GLOVE CALIBRATION · 四步
          </span>
          <button
            onClick={wz.closeWizard}
            className="cyber-btn px-2 py-1 rounded-sm text-[9px] flex items-center gap-1"
          >
            <X className="w-3 h-3" />
            {done ? "关闭" : "中止"}
          </button>
        </div>

        {/* 步骤指示 */}
        <div className="flex gap-1.5">
          {WIZARD_STEPS.map((s, i) => {
            const state =
              done || i < wz.stepIndex
                ? "done"
                : i === wz.stepIndex
                  ? "active"
                  : "todo";
            return (
              <div
                key={s.key}
                className="flex-1 px-2 py-1 rounded-sm border text-[9px] font-mono text-center"
                style={{
                  borderColor:
                    state === "active"
                      ? "rgba(168,85,247,0.6)"
                      : state === "done"
                        ? "rgba(0,229,160,0.35)"
                        : "rgba(85,102,119,0.25)",
                  color:
                    state === "active"
                      ? "var(--hud-violet)"
                      : state === "done"
                        ? "var(--hud-ok)"
                        : "var(--hud-dim)",
                }}
              >
                {s.short}
              </div>
            );
          })}
        </div>

        {done ? (
          <div className="space-y-2">
            <p className="text-[11px] font-mono text-[var(--hud-ok)]">
              本轮检查结束；各项是否保存，请看下方结果。左右手分别保存在当前浏览器。
            </p>
            <div className="space-y-1">
              {wz.summary.map((line, i) => (
                <p
                  key={i}
                  className="text-[9px] font-mono leading-relaxed text-[var(--hud-soft)]"
                >
                  {line}
                </p>
              ))}
            </div>
            <p className="text-[8px] text-[var(--hud-faint)] leading-relaxed">
              回到主界面握一下拳看手模：满量程按解剖学行程分配（掌指 85° / 近端
              100° / 远端
              70°，指尖骨不转），指尖应该落在掌面上。若仍显得不够弯，
              说明④那步握得不够紧或该路弯折跨度太小 —— 看上面的跨度数。
            </p>
            <div className="flex gap-2">
              <button
                onClick={wz.restart}
                className="flex-1 cyber-btn px-3 py-1.5 rounded-sm text-[10px]"
                style={{ borderColor: "rgba(168,85,247,0.4)" }}
              >
                重做一遍
              </button>
              <button
                onClick={wz.closeWizard}
                className="flex-1 cyber-btn px-3 py-1.5 rounded-sm text-[10px]"
                style={{ borderColor: "rgba(0,229,160,0.4)" }}
              >
                完成
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div>
              <p className="text-[11px] font-mono text-[var(--hud-warn)]">
                {step.title}
              </p>
              <p className="text-[9px] font-mono text-[var(--hud-soft)] leading-relaxed mt-1">
                {step.instruction}
              </p>
            </div>

            {/* 示意手模：照着摆。这里的姿态是写死的目标姿态，与实际 IMU 无关 */}
            <div className="flex flex-wrap gap-3">
              <DemoViewport
                label="左手示意"
                driveRef={wz.demoLRef}
                side="left"
                connected={connected.LH}
                frames={wz.counts.LH}
                sampling={wz.phase === "sampling"}
              />
              <DemoViewport
                label="右手示意"
                driveRef={wz.demoRRef}
                side="right"
                connected={connected.RH}
                frames={wz.counts.RH}
                sampling={wz.phase === "sampling"}
              />
            </div>

            {/* ②③ 两步的实时角度读数：转没转够在采样前就能看见，
                别等汇算才发现只转了 15°（那时门限不过、矩阵不写入） */}
            {hasAngleGuide && (
              <div className="flex gap-2">
                {HAND_KEYS.map(hand => {
                  const deg = liveDeg[hand];
                  const color =
                    deg === null
                      ? "var(--hud-dim)"
                      : deg >= GATE_MIN_DEG && deg <= GATE_MAX_DEG
                        ? "var(--hud-ok)"
                        : deg >= MIN_MOTION_DEG
                          ? "var(--hud-warn)"
                          : "var(--hud-err)";
                  return (
                    <div
                      key={hand}
                      className="flex-1 px-2 py-1.5 rounded-sm border text-[10px] font-mono flex items-baseline justify-between"
                      style={{ borderColor: "rgba(85,102,119,0.25)" }}
                    >
                      <span className="text-[var(--hud-soft)]">
                        {handName(hand)} 已转
                      </span>
                      <span style={{ color }}>
                        {deg === null ? "—" : `${deg.toFixed(0)}°`}
                        <span className="text-[var(--hud-faint)]">
                          {" "}
                          / 目标 {TARGET_MOTION_DEG}°
                        </span>
                      </span>
                    </div>
                  );
                })}
              </div>
            )}

            {wz.phase === "blocked" ? (
              <div className="flex flex-wrap gap-2">
                <button
                  className="cyber-btn px-3 py-2 text-xs"
                  onClick={() => wz.retryStep()}
                >
                  重试当前步骤
                </button>
                {wz.stepIndex > 0 && (
                  <button
                    className="cyber-btn px-3 py-2 text-xs"
                    onClick={() => wz.retryStep(true)}
                  >
                    返回上一步重采
                  </button>
                )}
              </div>
            ) : wz.phase === "countdown" ? (
              <div className="flex items-center gap-3">
                <span className="text-2xl font-mono font-bold text-[var(--hud-violet)] w-8 text-center">
                  {wz.countdown}
                </span>
                <span
                  className="text-[9px] font-mono"
                  style={{
                    color: wz.gateWaiting
                      ? "var(--hud-warn)"
                      : "var(--hud-dim)",
                  }}
                >
                  {wz.gateWaiting
                    ? hasAngleGuide
                      ? `等待稳定约 1 秒，角度需在 ${GATE_MIN_DEG}–${GATE_MAX_DEG}°（目标 ${TARGET_MOTION_DEG}°）`
                      : "等待连续稳定数据，保持不动约 1 秒"
                    : hasAngleGuide
                      ? `照示意转到目标附近，稳定后自动采样 3 秒（接受 ${GATE_MIN_DEG}–${GATE_MAX_DEG}°）`
                      : "摆好姿势并保持静止，倒数结束后自动采样 3 秒"}
                </span>
              </div>
            ) : (
              <div className="space-y-1">
                <div className="h-1.5 bg-[var(--hud-track)] rounded-full overflow-hidden border border-[#7c3aed]/20">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${wz.progress}%`,
                      background: "var(--hud-violet)",
                      boxShadow: "0 0 6px rgba(168,85,247,0.5)",
                    }}
                  />
                </div>
                <div className="text-[9px] font-mono text-[var(--hud-violet)]">
                  采样中… 保持静止（左 {wz.counts.LH} 帧 / 右 {wz.counts.RH}{" "}
                  帧）
                </div>
              </div>
            )}

            {wz.errorMsg && (
              <p className="text-[9px] font-mono text-[var(--hud-err)] leading-relaxed">
                {wz.errorMsg}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** 向导里的示意手模：小一号的视口 */
function DemoViewport({
  label,
  driveRef,
  side,
  connected,
  frames,
  sampling,
}: {
  label: string;
  driveRef: RefObject<HandDrive>;
  side: "left" | "right";
  connected: boolean;
  frames: number;
  sampling: boolean;
}) {
  return (
    <div className="basis-[240px] grow shrink min-w-0 space-y-1">
      <div className="flex items-baseline justify-between px-1">
        <span className="text-[9px] font-mono text-[var(--hud-violet)]">
          {label}
        </span>
        <span
          className="text-[8px] font-mono"
          style={{ color: connected ? "var(--hud-ok)" : "var(--hud-dim)" }}
        >
          {connected ? (sampling ? `${frames} 帧` : "已连接") : "未连接 · 跳过"}
        </span>
      </div>
      <div
        className="relative w-full aspect-[4/3] border rounded-sm overflow-hidden"
        style={{
          backgroundColor: "var(--hud-page)",
          borderColor: connected
            ? "rgba(168,85,247,0.3)"
            : "rgba(85,102,119,0.2)",
        }}
      >
        {/* armStyle="stub"：示意姿态里有手臂正对镜头的档（① 平铺），完整前臂会把
            这个小视口整个填满，只剩画面角上几个指尖。示意手模不承担朝向线索，
            腕下留一截收圆的臂桩就够了 */}
        <HandModel driveRef={driveRef} side={side} armStyle="stub" />
      </div>
    </div>
  );
}

/**
 * 两只手的 137 点传感器热力图，和骨架视口同占主区下排的一个格子。
 * 放在主区而不是侧栏：判断"某一路弯折/压力有没有反应"就看这张图，
 * 挤在 288px 侧栏里还要滚动才看得到，等于没有。
 *
 * 是网格项而不是 flex 项（早期版本把它和骨架塞进同一个格子里横排，
 * 窄屏没有 xl 两列时会被压成 ~55px，图例一行一个字）。
 */
function SensorPanel({ checks }: { checks: HandCheck[] }) {
  return (
    <div className="min-w-0 min-h-0 flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2 px-1 shrink-0">
        <span className="text-[10px] font-bold tracking-widest font-mono text-[var(--hud-accent)]">
          SENSOR OVERVIEW
        </span>
        <span className="text-[8px] font-mono text-[var(--hud-faint)]">
          137 mapped sensors · 每只手一张
        </span>
      </div>
      <div
        className="flex-1 min-h-0 border rounded-sm p-2 flex gap-3 overflow-hidden"
        style={{
          backgroundColor: "var(--hud-page)",
          borderColor: "rgba(0,240,255,0.2)",
        }}
      >
        {checks.map(hc => {
          const frame = hc.channel.latestFrame;
          return (
            <div key={hc.handKey} className="flex-1 min-w-0 space-y-1">
              <div className="text-[8px] font-mono text-[var(--hud-dim)]">
                {handName(hc.handKey)}
                {frame ? ` · ${hc.channel.gloveFps} FPS` : " · 未连接"}
              </div>
              {frame ? (
                <MiniHeatmap data={frame.mapped_data} />
              ) : (
                <div className="h-full min-h-[80px] flex items-center justify-center text-[9px] font-mono text-[var(--hud-faint)]">
                  没有数据
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** 一只手的 3D 手模视口 */
function HandViewport({ hc }: { hc: HandCheck }) {
  const side = hc.handKey === "LH" ? "left" : "right";
  const orient = hc.orientCalib;
  const orientHint = !orient
    ? "朝向未标定"
    : orient.axisMap
      ? "朝向零位+轴向"
      : "朝向仅零位";
  return (
    <Viewport
      label={`BEND + IMU · ${hc.handKey}`}
      hint={`${hc.calibrated ? "弯折两点标定" : "弯折未标定"} · ${orientHint}`}
      accent={hc.calibrated && orient ? "var(--hud-ok)" : "var(--hud-dim)"}
      badge={
        <>
          <Ruler className="w-3 h-3 inline mr-1" />
          {hc.handKey === "LH" ? "LEFT HAND" : "RIGHT HAND"}
        </>
      }
    >
      <HandModel driveRef={hc.driveRef} side={side} />
      {!hc.channel.isConnected && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="text-[10px] font-mono text-[var(--hud-dim)]">
            连接{handName(hc.handKey)}手套后即可驱动（无需模型）
          </span>
        </div>
      )}
      {/* 未标定时必须写在画面里：这个状态下握拳只弯到约一半，
          很容易被当成"手模坏了"或"模型不准"，而其实只是没标定。 */}
      {hc.channel.isConnected && !hc.calibrated && (
        <div className="absolute bottom-2 left-2 right-2 px-2 py-1 rounded-sm pointer-events-none bg-[#d97706]/12 border border-[#d97706]/35">
          <span className="text-[9px] font-mono text-[var(--hud-warn)] leading-relaxed">
            未标定 ·
            只有柔和预览：满量程也只弯约一半，握拳不会成形。跑一次四步校准。
          </span>
        </div>
      )}
    </Viewport>
  );
}

/** 一只手的弯折两点标定：状态 + 五指比例条（含原始 ADC / 跨度）+ 三个按钮 */
function BendCalibBlock({ hc }: { hc: HandCheck }) {
  const { calibrated, weak, bendUi, rawUi, spans, calibMsg, handKey } = hc;
  const connected = hc.channel.isConnected;
  return (
    <div className="space-y-1.5 pb-2 border-b border-[#d97706]/10 last:border-b-0">
      <div className="flex items-center justify-between text-[10px] font-mono">
        <span className="text-[var(--hud-soft)]">{handName(handKey)}</span>
        <span
          className={
            calibrated ? "text-[var(--hud-ok)]" : "text-[var(--hud-warn)]"
          }
        >
          {calibrated ? "CALIBRATED" : "UNCALIBRATED"}
        </span>
      </div>

      {/* 五指比例条 */}
      <div className="space-y-1">
        {FINGER_NAMES.map((name, i) => {
          const pct = Math.round((bendUi[i] ?? 0) * 100);
          const isWeak = weak.includes(i);
          return (
            <div key={name} className="flex items-center gap-1.5">
              <span
                className={`text-[8px] font-mono w-8 shrink-0 ${
                  isWeak ? "text-[var(--hud-err)]" : "text-[var(--hud-dim)]"
                }`}
              >
                {name}
              </span>
              <div className="flex-1 h-1.5 bg-[var(--hud-track)] rounded-full overflow-hidden border border-[#d97706]/10">
                <div
                  className="h-full rounded-full transition-all duration-100"
                  style={{
                    width: `${pct}%`,
                    background: isWeak
                      ? "var(--hud-err)"
                      : calibrated
                        ? "var(--hud-ok)"
                        : "var(--hud-dim)",
                  }}
                />
              </div>
              <span className="text-[8px] font-mono text-[var(--hud-soft)] w-7 text-right shrink-0">
                {pct}%
              </span>
              {/* 原始 ADC / 标定跨度：手模看着不对时，先看这一路到底动不动。
                  百分比是算出来的，只有这两个数能区分"传感器没反应"和"标定不对"。 */}
              <span
                className="text-[8px] font-mono w-14 text-right shrink-0"
                style={{
                  color: isWeak ? "var(--hud-err)" : "var(--hud-faint)",
                }}
                title="当前原始 ADC / 标定跨度"
              >
                {Math.round(rawUi[i] ?? 0)}
                {spans ? `·${Math.round(spans[i] ?? 0)}` : ""}
              </span>
            </div>
          );
        })}
      </div>

      <div className="flex gap-1.5">
        <button
          onClick={() => hc.captureBendPose("open")}
          disabled={!connected}
          className="flex-1 cyber-btn px-2 py-1 rounded-sm text-[9px] disabled:opacity-40"
        >
          捕捉张开
        </button>
        <button
          onClick={() => hc.captureBendPose("fist")}
          disabled={!connected}
          className="flex-1 cyber-btn px-2 py-1 rounded-sm text-[9px] disabled:opacity-40"
        >
          捕捉握拳
        </button>
        <button
          onClick={hc.resetBendRange}
          className="cyber-btn px-2 py-1 rounded-sm text-[9px]"
          style={{ borderColor: "rgba(255,45,123,0.3)" }}
        >
          清除
        </button>
      </div>

      {weak.length > 0 && (
        <p className="text-[8px] text-[var(--hud-err)] leading-relaxed flex gap-1">
          <AlertTriangle className="w-3 h-3 shrink-0 mt-px" />
          <span>
            {handName(handKey)}的 {weak.map(i => FINGER_NAMES[i]).join("、")}{" "}
            标定跨度不足 {MIN_USEFUL_SPAN}{" "}
            ADC，这几路弯折传感器可能没贴好或已损坏 —— 别急着录数据。
          </span>
        </p>
      )}
      {calibMsg && (
        <p className="text-[8px] text-[var(--hud-warn)] leading-relaxed">
          {calibMsg}
        </p>
      )}
    </div>
  );
}

/** 四步采样的参考指标，不把累计姿态变化解释为零偏。 */
function ImuVerdictBlock({ hc }: { hc: HandCheck }) {
  const report = hc.imuReport;
  return (
    <div className="space-y-1 text-[9px] text-[var(--hud-soft)]">
      <p>
        {handName(hc.handKey)} 采样检查：
        {report ? "已有本轮数据" : "本次未测（按需检测）"}
      </p>
      {report && (
        <>
          <p>
            姿态累计变化 {report.stillRotationDegPerMin?.toFixed(0) ?? "—"}
            °/分钟（非零偏）
          </p>
          <p>
            {report.usableFrames > 0
              ? `倾角一致性偏差 ${report.tiltInconsistencyDeg.toFixed(1)}° · P95 ${report.p95TiltDeg.toFixed(1)}° · ${report.usableFrames} 帧`
              : "未取得可用加速度，无法核对重力方向。"}
          </p>
          <p className="text-[var(--hud-faint)]">
            累计值包含抖动；请用“静置检测”区分首尾偏转、波动和突跳。
          </p>
        </>
      )}
    </div>
  );
}

/**
 * 一块画布容器。**尺寸由外面的网格给**（自己不定宽高），所以几块能一屏排完、
 * 上下不留空白 —— 之前是 basis-[400px] + aspect-[4/3] 自己撑，一屏放不下就换行，
 * 结果整块内容被垂直居中挤成"上面一大片黑、骨架那块掉到屏幕外"。
 *
 * boxClass 决定画面区怎么占格子：
 *  - 默认 `flex-1 min-h-0 w-full` = 填满格子。3D 手模用这个（fiber 的 Canvas
 *    会跟着容器重算相机 aspect，任意长宽比都不变形）。
 *  - 骨架那块传 `aspect-[4/3] w-auto mx-auto`：canvas 内部分辨率是写死的 640×480，
 *    容器必须严格 4:3，否则火柴人会被拉扁；宽度由高度反推，剩下的富余居中留白。
 */
function Viewport({
  label,
  hint,
  badge,
  accent = "var(--hud-warn)",
  boxClass = "flex-1 min-h-0 w-full",
  children,
}: {
  label: string;
  hint?: string;
  badge?: React.ReactNode;
  accent?: string;
  boxClass?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5 min-h-0 min-w-0">
      <div className="flex items-baseline gap-2 px-1 shrink-0">
        <span
          className="text-[10px] font-bold tracking-widest font-mono"
          style={{ color: accent }}
        >
          {label}
        </span>
        {hint && (
          <span className="text-[8px] font-mono text-[var(--hud-faint)]">
            {hint}
          </span>
        )}
      </div>
      <div
        className={`relative border rounded-sm overflow-hidden ${boxClass}`}
        style={{
          backgroundColor: "var(--hud-page)",
          borderColor: `${accent}33`,
          boxShadow:
            "0 0 30px rgba(245,158,11,0.08), inset 0 0 30px rgba(10,14,26,0.5)",
        }}
      >
        {children}
        {badge && (
          <div
            className="absolute top-2 left-2 px-2 py-0.5 rounded-sm pointer-events-none"
            style={{
              backgroundColor: `${accent}26`,
              border: `1px solid ${accent}4d`,
            }}
          >
            <span className="text-[9px] font-mono" style={{ color: accent }}>
              {badge}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 pb-1 border-b border-[#d97706]/15">
        <div className="w-1 h-3 bg-[var(--hud-warn)] rounded-full shadow-[0_0_4px_rgba(245,158,11,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[var(--hud-warn)] font-mono">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function MiniHeatmap({ data }: { data: number[] }) {
  const cols = 14;
  const rows = Math.ceil(data.length / cols);
  return (
    <div
      className="grid gap-px"
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
    >
      {data.slice(0, rows * cols).map((v, i) => {
        const norm = Math.min(v / 200, 1);
        const r = Math.round(norm * 245);
        const g = Math.round((1 - norm) * 158 + norm * 45);
        const b = Math.round((1 - norm) * 11 + norm * 123);
        return (
          <div
            key={i}
            className="aspect-square rounded-[1px]"
            style={{
              backgroundColor: `rgb(${r}, ${g}, ${b})`,
              opacity: 0.3 + norm * 0.7,
            }}
          />
        );
      })}
    </div>
  );
}

// ===== Canvas 绘制函数 =====

function computeSpread(
  landmarks: { x: number; y: number; z: number }[]
): number {
  // 基于关键点分布范围估算置信度
  let minX = 1,
    maxX = 0,
    minY = 1,
    maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  // 合理的手部范围大约 0.1-0.5
  const spread = (rangeX + rangeY) / 2;
  if (spread < 0.02) return 10; // 太集中，可能不准
  if (spread > 0.8) return 20; // 太分散，可能不准
  return Math.min(100, spread * 300);
}

function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const step = 50;
  ctx.strokeStyle = "rgba(245, 158, 11, 0.03)";
  ctx.lineWidth = 0.5;
  for (let x = 0; x < w; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(245, 158, 11, 0.06)";
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}

function drawScanLines(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = "rgba(245, 158, 11, 0.008)";
  for (let y = 0; y < h; y += 3) {
    ctx.fillRect(0, y, w, 1);
  }
}

function drawWaitingState(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  time: number
) {
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.12;

  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(245, 158, 11, 0.5)";
  ctx.lineWidth = 2;
  const angle1 = time * 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, angle1, angle1 + 1.2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(255, 45, 123, 0.4)";
  const angle2 = -time * 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.6, angle2, angle2 + 0.8);
  ctx.stroke();

  ctx.fillStyle = `rgba(245, 158, 11, ${0.4 + 0.2 * Math.sin(time * 2)})`;
  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.textAlign = "center";
  ctx.fillText("WAITING FOR GLOVE DATA...", cx, cy + radius + 35);

  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillStyle = "rgba(85, 102, 119, 0.6)";
  ctx.fillText("请连接手套并确保骨架模型已加载", cx, cy + radius + 55);

  ctx.restore();
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  w: number,
  h: number
) {
  Object.entries(FINGER_CONNECTION_GROUPS).forEach(([finger, connections]) => {
    const color = FINGER_COLORS[finger];
    const glowColor = FINGER_GLOW_COLORS[finger];

    connections.forEach(([from, to]) => {
      const x1 = landmarks[from].x * w;
      const y1 = landmarks[from].y * h;
      const x2 = landmarks[to].x * w;
      const y2 = landmarks[to].y * h;

      ctx.save();
      ctx.strokeStyle = glowColor;
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.restore();
    });
  });
}

function drawJoints(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  w: number,
  h: number
) {
  landmarks.forEach((lm, idx) => {
    const x = lm.x * w;
    const y = lm.y * h;

    let color = "var(--hud-warn)";
    if (idx <= 4) color = FINGER_COLORS.thumb;
    else if (idx <= 8) color = FINGER_COLORS.index;
    else if (idx <= 12) color = FINGER_COLORS.middle;
    else if (idx <= 16) color = FINGER_COLORS.ring;
    else color = FINGER_COLORS.pinky;

    const isTip = [4, 8, 12, 16, 20].includes(idx);
    const isWrist = idx === 0;
    const radius = isWrist ? 7 : isTip ? 5.5 : 3.5;

    ctx.save();
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 4);
    gradient.addColorStop(0, color + "50");
    gradient.addColorStop(0.5, color + "15");
    gradient.addColorStop(1, color + "00");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius * 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.beginPath();
    ctx.arc(x - radius * 0.25, y - radius * 0.25, radius * 0.3, 0, Math.PI * 2);
    ctx.fill();

    if (isTip) {
      ctx.strokeStyle = color + "60";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  });
}

function drawPalmCrosshair(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  w: number,
  h: number
) {
  const palmIndices = [0, 5, 9, 13, 17];
  const cx =
    (palmIndices.reduce((sum, i) => sum + landmarks[i].x, 0) /
      palmIndices.length) *
    w;
  const cy =
    (palmIndices.reduce((sum, i) => sum + landmarks[i].y, 0) /
      palmIndices.length) *
    h;

  const size = 18;
  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.35)";
  ctx.lineWidth = 0.8;

  ctx.beginPath();
  ctx.moveTo(cx - size, cy);
  ctx.lineTo(cx - 5, cy);
  ctx.moveTo(cx + 5, cy);
  ctx.lineTo(cx + size, cy);
  ctx.moveTo(cx, cy - size);
  ctx.lineTo(cx, cy - 5);
  ctx.moveTo(cx, cy + 5);
  ctx.lineTo(cx, cy + size);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 2, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(245, 158, 11, 0.15)";
  ctx.beginPath();
  ctx.arc(cx, cy, size + 5, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

function drawHUDCorners(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const cornerSize = 25;
  const offset = 6;
  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.3)";
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.moveTo(offset, offset + cornerSize);
  ctx.lineTo(offset, offset);
  ctx.lineTo(offset + cornerSize, offset);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(w - offset - cornerSize, offset);
  ctx.lineTo(w - offset, offset);
  ctx.lineTo(w - offset, offset + cornerSize);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(offset, h - offset - cornerSize);
  ctx.lineTo(offset, h - offset);
  ctx.lineTo(offset + cornerSize, h - offset);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(w - offset - cornerSize, h - offset);
  ctx.lineTo(w - offset, h - offset);
  ctx.lineTo(w - offset, h - offset - cornerSize);
  ctx.stroke();

  ctx.restore();
}
