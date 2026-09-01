/*
 * Translate — 实时手语翻译页面
 * DESIGN: Cyberpunk HUD 风格
 *
 * 功能:
 * 1. 连接手套后实时推理
 * 2. 显示识别结果（大字体 + 置信度）
 * 3. 历史翻译记录（句子拼接）
 * 4. Top-K 候选词显示
 *
 * 三种推理模式，可在右侧面板切换：
 * - static：旧的单帧 MLP，每 100ms 独立推理一帧（原有逻辑，一行没动）
 * - sequence：TCN 时序模型，手套帧全速写进环形缓冲，每 100ms 取最近 `WINDOW_MS`
 *   窗口推理一次。动态词（再见、来、工作…）只有这条路能识别。
 * - sentence：CTC 句子模型，**不做滑窗推理**。整句连着打完 → 一次性解码出词序列。
 *
 * 前两条路共用下游的置信度阈值 / 平滑窗口 / 2 秒去重逻辑 —— 那套逻辑与模型
 * 无关，只跟"一串预测结果如何变成一句话"有关，不该为时序模型重写一份。
 *
 * 第三条路**故意不共用**那套逻辑：「同一个词至少间隔 2 秒」的去重让「我 爱 我」
 * 打不出来，而 CTC 本来就靠 blank 区分重复词。它只用那个 100ms 循环做一件事 ——
 * 判断"这一句什么时候结束"（`sentenceCapture.ts` 的状态机），推理是收句时跑一次。
 *
 * 底部是**一手一个视口**（`HandModel` × 2，与第 1 步 /mocap 的自检同一个组件、
 * 同一份标定、同一套驱动，见 useHandModelDrive）。
 *
 * ⚠ 这里**曾经**是一块两手合一的「手语舞台」（`SigningStage` + 躯干剪影），
 * 已经撤掉了。撤的理由不是观感，是它承诺了一个做不到的东西：
 * 手语的**位置**是语言学通道（额头 / 下巴 / 胸前是不同的词），而手套只有
 * 弯折 + 压力 + IMU —— IMU 给朝向不给位置，六轴还观测不到绝对 yaw。
 * 所以把两只手摆进同一个人形场景里，看着像"在打手语"，实际上仍然展示不出
 * 一个完整的手语词，只是把"位置是编的"这件事藏得更深。
 * 分成两格反而诚实：一格 = 一只手的手型 + 朝向，正好是手套真的测到的那些通道。
 * **别再合回去**，除非哪天真加了位置观测（光学 / 超宽带之类）。
 *
 * 它不参与识别，是给"为什么不识别"提供第一手判据的：手模不动 = 手套没在出数据；
 * 手模动了但手型不像 = 弯折标定或某一路传感器的问题，跟模型无关。
 * 没有它的时候，这两种硬件问题在这一页表现为"模型不准"，会把人引向重训模型。
 */
import { useGloveFrames, useGloves } from "@/contexts/GloveContext";
import StepNav from "@/components/StepNav";
import HandModel, { type HandDrive } from "@/components/HandModel";
import { useHandModelDrive } from "@/hooks/useHandModelDrive";
import type { HandChannel } from "@/hooks/useDualGloveSerial";
import type { HandKey } from "@/lib/bendRange";
import {
  predict,
  isModelLoaded,
  getLoadedLabels,
  loadModelFromSaved,
} from "@/lib/signLanguageModel";
import {
  predictSequence,
  isSequenceModelLoaded,
  getLoadedSequenceLabels,
  loadSequenceModelFromSaved,
} from "@/lib/sequenceModel";
import {
  SENTENCE_MODEL_DIR,
  loadSentenceModel,
  predictSentence,
  sentenceModelAvailable,
  type LoadedSentenceModel,
} from "@/lib/sentenceModel";
import { fetchWordModelMeta, loadDeployedWordModel } from "@/lib/wordModel";
import { ModelMissingError } from "@/lib/modelWeights";
import {
  CONTINUOUS_SETTLE_MS,
  MAX_UTTERANCE_MS,
  SETTLE_MS,
  type CaptureAction,
  type CaptureStatus,
} from "@/lib/sentenceCapture";
import { SentenceEnvelope } from "@/lib/sentenceEnvelope";
import SentencePanel from "@/components/SentencePanel";
import { speakChinese, stopSpeaking, warmUpVoices } from "@/lib/speech";
import { resolveSentence } from "@/lib/sentenceGrammar";
import { SequenceWindowBuffer } from "@/lib/sequenceWindow";
import { judgeWindowMotion } from "@/lib/motionGate";
import { mirrorStaticInputs, normalizeHandedness } from "@/lib/handMirror";
import {
  DominanceTracker,
  describeDominance,
  type DominanceVerdict,
} from "@/lib/dominantHand";
import { loadBendRange, type BendRange } from "@/lib/bendRange";
import type { GloveFrame } from "@/lib/gloveProtocol";
import { getLatestModel, getLatestSequenceModel } from "@/lib/datasetStore";
import {
  getWordById,
  getCategoryColor,
  getDisplayLabel,
  getTranslationLabel,
  resolveToMember,
  IDLE_LABEL,
} from "@/lib/signLanguageVocab";
import { matchCompound } from "@/lib/compoundWords";
import { isSuppressed } from "@/lib/suppressedWords";
import { postprocessSentence } from "@/lib/sentencePostprocess";
import { applyFistGate, thumbPeak, THUMB_PEAK_GATE } from "@/lib/fistGate";
import {
  rotationRate,
  ROT_RATE_HI,
  ROT_RATE_LO,
  type RotationReading,
} from "@/lib/rotationRate";
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
  Brain,
  Hand,
  MessageSquare,
  Trash2,
  Volume2,
  Zap,
  Eye,
  EyeOff,
} from "lucide-react";

interface TranslationEntry {
  word: string;
  label: string;
  confidence: number;
  timestamp: number;
}

type ModelMode = "static" | "sequence" | "sentence";

/**
 * URL 里带过来的初始 MODE（`/translate?mode=sentence`）。
 *
 * 为什么需要它：下面那个自动加载 effect 会**按模型新旧**选 MODE，而它只认识
 * 静态和时序两条 —— 句子模型不在 IndexedDB 里（它是 Python 导出的构建产物，
 * 见 sentenceModel.ts），`getLatestModel` 系列查不到它。于是从 /train-sentence
 * 点「去使用」过来，落地永远是时序滑窗：刚训完句子模型的人看到的是另一条链路，
 * 而页面上没有任何提示说"你要的那一档在第三个按钮里"。
 *
 * 所以让链接自己带上意图。读不出来就返回 null，交给按新旧自动选的老逻辑。
 */
function modeFromUrl(): ModelMode | null {
  const m = new URLSearchParams(window.location.search).get("mode");
  return m === "static" || m === "sequence" || m === "sentence" ? m : null;
}

/**
 * 滑窗长度。
 *
 * 这个值**必须对齐训练侧的时间口径**，否则模型在推理时看到的是被时间缩放过的手势。
 * 训练是把每条录制**裁剪后的那一段**重采样到 `SEQ_LEN` 帧的，所以要对齐的不是
 * 录制总时长，而是裁剪后时长 —— /train-seq 的「数据体检」报的
 * 「裁剪后中位」就是这个数：实测 2031ms（08-13 批 2184 / 08-17 批 1882）。
 *
 * 原值 1500 是照"常见孤立词 1~2s"估的，比实测口径短约 1.35 倍。窗口偏短时，
 * 一个 2000ms 的词被切出 1500ms 再铺满 32 帧，等于喂给模型一个"1.35 倍速的残词" ——
 * 那种输入训练集里根本不存在，模型只能退回类先验，表现就是"打什么都输出同一个词"。
 *
 * 上限不再受环形缓冲约束（缓冲为了句子模式已经放到 12s），但仍然不该往大调：
 * `snapshot` 攒不满整窗直接返回 null，窗口越长，开始翻译后等第一次判定越久。
 */
const WINDOW_MS = 2000;

/**
 * 主手判定用的窗口长度。比推理窗口短：判"哪只手在动"不需要装下整个词，
 * 而窗口越短，刚开始翻译时越早能给出判定（`snapshot` 攒不满整窗会返回 null）。
 */
const DOMINANCE_WINDOW_MS = 1000;

/** 主手判定的重算间隔（ms）。10Hz 推理里每次都重算是白费，运动能量不会那么快变 */
const DOMINANCE_RECHECK_MS = 300;

export default function Translate() {
  /*
   * URL 指定了 MODE 就用它，并且**锁住**自动选择 —— 不然自动加载 effect 跑完
   * （异步，晚于首帧）会把它改掉，用户看到的是先闪一下句子档再跳去时序。
   */
  const urlMode = useRef<ModelMode | null>(null);
  if (urlMode.current === null) urlMode.current = modeFromUrl();
  const [modelMode, setModelMode] = useState<ModelMode>(
    urlMode.current ?? "static"
  );
  const windowBufRef = useRef<SequenceWindowBuffer | null>(null);
  if (windowBufRef.current === null) {
    // 12s **无条件**给到，不按模式分配：切 MODE 时重新 new 一个会把缓冲里的数据丢掉，
    // 而句子模式下那正好是用户刚打完的半句话。12s @100Hz 双手 ≈ 2400 条，内存无所谓
    windowBufRef.current = new SequenceWindowBuffer({ bufferMs: MAX_UTTERANCE_MS });
  }

  // 序列模式下手套帧必须走全速回调写进环形缓冲。
  // 不能轮询 latestFrameRef —— 那个 ref 按 targetFps 节流，轮询会丢帧，
  // 而动态词的轨迹细节正好丢在这里。
  const onLeftFrame = useCallback((f: GloveFrame) => {
    windowBufRef.current?.push("left", f);
  }, []);
  const onRightFrame = useCallback((f: GloveFrame) => {
    windowBufRef.current?.push("right", f);
  }, []);

  /**
   * 是否正在把左手数据归一化到右手口径。
   *
   * 模型是**用右手采的数据训的**，而特征层给两只手各留一段独立槽位、137 维指序还左右
   * 相反，所以只戴左手套时不归一化的话输入落在训练时永远全 0 的那半边，输出会塌到一个
   * 固定的词上。单手词换手不改词义，所以镜像是正确解，见 `handMirror.ts`。
   */
  const [mirrored, setMirrored] = useState(false);
  const mirroredRef = useRef(false);

  /**
   * 动作闸门是否正拦着（窗口里没人在动，见 `motionGate.ts`）。
   *
   * 这个状态**必须显示出来**：闸门关着的时候大字区是空的，而"空着"在界面上和
   * "模型挂了 / 手套断了 / 没加载模型"长得一模一样。不写明"在等你起手"，
   * 这个改动会把一个错词问题变成一个"点了没反应"问题。
   */
  const [gated, setGated] = useState(false);
  const gatedRef = useRef(false);

  /**
   * 主手（做手语的那只手）**从数据判**，不再让用户选。判定器量两只手在滑窗里的
   * 运动能量，带滞回防止一个词做到一半翻转，详见 `dominantHand.ts`。
   *
   * 判定结果同时进 ref 和 state：ref 给 10Hz 的推理循环读（不能等 React 重渲染），
   * state 只为了把能量条画出来 —— 判错时用户得能看见是哪只手能量高，
   * 而不是对着一个错词猜。所以 state 是**节流**写的，见 `DOMINANCE_RECHECK_MS`。
   */
  const dominanceRef = useRef<DominanceTracker | null>(null);
  if (dominanceRef.current === null) {
    dominanceRef.current = new DominanceTracker();
  }
  const [dominance, setDominance] = useState<DominanceVerdict | null>(null);

  /**
   * 弯折两点标定 —— 判定器用它把两只手的弯折量程归一化（实测左右量程差得很多）。
   * 惰性读一次即可：标定入口只在第 1 步，改完回到本页时组件已重新挂载。
   */
  const bendRangesRef = useRef<{ LH: BendRange | null; RH: BendRange | null } | null>(
    null
  );
  if (bendRangesRef.current === null) {
    bendRangesRef.current = { LH: loadBendRange("LH"), RH: loadBendRange("RH") };
  }

  // 连接在第 1 步（/mocap）建立、住在 GloveProvider 里，本页只订阅帧、不碰串口
  const { left: gloveLeft, right: gloveRight, anyConnected } = useGloves();
  useGloveFrames(onLeftFrame, onRightFrame);
  const isConnected = anyConnected;
  const bothConnected = gloveLeft.isConnected && gloveRight.isConnected;
  const gloveError = gloveLeft.error || gloveRight.error;

  /*
   * 底部两个手模视口的驱动。
   *
   * hook 留在页面这一层调（而不是塞进 `HandViewport` 里），是因为上方那一行
   * 连接/标定汇总也要读 `bendCalibrated` / `orientCalibrated`。它自带 rAF、
   * 写 ref 不触发 React 重渲染，100Hz 的手套数据不会打到这个页面组件上。
   */
  const leftHand = useHandModelDrive(gloveLeft, "LH");
  const rightHand = useHandModelDrive(gloveRight, "RH");

  // 状态
  const [staticReady, setStaticReady] = useState(isModelLoaded());
  const [seqReady, setSeqReady] = useState(isSequenceModelLoaded());

  /*
   * ===== 句子模式的状态 =====
   *
   * 句子模型和另外两个**不是一路来的**：它是 python_train 训完导出的静态文件
   * （`/models/seq_sentence/`），不进 IndexedDB，所以没法用 `getLatestSequenceModel`
   * 那套查。这里分两步：挂载时探一次在不在（只读 weights.json 的响应码，不下权重），
   * 真正切到这一档时才加载。
   *
   * 「没部署」和「部署了但坏了」必须分开：前者是正常状态，要给"先去跑 python_train"
   * 的引导；后者是 bug，要把错误原文显示出来。混成一句"不可用"会让人去重训一个
   * 其实已经训好的模型。
   */
  const sentenceModelRef = useRef<LoadedSentenceModel | null>(null);
  /**
   * 收句状态机 + 取数，**和采集页 `/collect-sentence` 共用同一个类**
   * （`sentenceEnvelope.ts`）。两边必须产生同一种时间包络：采集时多录进去的一段静止
   * 会在重采样到定长 T 时把每个词在归一化时间轴上整体挪位，而这一类偏差合成 val
   * 和真实 val 都看不出来，只有戴上手套才发现打什么都不准。
   * 所以这里**不要**把 tick/snapshotAll 拆开重写一份 —— 那就退回"靠人记得对齐"了。
   *
   * 缓冲由本页持有（三档共用一个），envelope 只借用。
   */
  const sentenceEnvRef = useRef<SentenceEnvelope | null>(null);
  if (sentenceEnvRef.current === null) {
    sentenceEnvRef.current = new SentenceEnvelope(windowBufRef.current);
  }
  /** null = 还在探测 / 探测本身失败（看 sentenceError） */
  const [sentenceAvailable, setSentenceAvailable] = useState<boolean | null>(null);
  const [sentenceLoaded, setSentenceLoaded] = useState(false);
  const [sentenceError, setSentenceError] = useState<string | null>(null);
  /** 模型解出的词序列（原始类别 id）；null = 这一档还没解过 */
  const [sentenceWords, setSentenceWords] = useState<string[] | null>(null);
  const [pronOverrides, setPronOverrides] = useState<Record<number, string>>({});
  const [grammarOn, setGrammarOn] = useState(true);
  /** 收句后自动把顺句结果念出来。默认开 —— 这一档的用途就是"打完让对面听见" */
  const [autoSpeak, setAutoSpeak] = useState(true);
  /**
   * 连续模式：收句后自动成句 + 自动重新等下一句，全程不用碰鼠标。默认开。
   *
   * 关掉就退回"一句一次"（每句都要点「开始一句」），采集页不受影响。
   */
  const [continuousMode, setContinuousMode] = useState(true);
  const [sentenceHistory, setSentenceHistory] = useState<string[]>([]);
  /**
   * 历史里最后一条是不是"当前这一句"（自动成句进去的、还能改）。
   *
   * 需要它是因为自动成句发生在解码那一刻，而用户**之后**还会点代词、删词。
   * 没有这个标记就没法把编辑同步回历史 —— 屏幕上的大字变了、历史里留着错的那个，
   * 界面上看不出来。手动「成句」= 定版，把它置 false。
   */
  const [lastLive, setLastLive] = useState(false);
  const [sentenceNote, setSentenceNote] = useState<string | null>(null);
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus | null>(null);

  /*
   * 自动朗读要读的两个开关走 **ref**，不直接读 state。
   *
   * 念这一步发生在 `decodeUtterance` 里，而那个 callback 在推理循环那个 effect 的
   * 依赖里。把 `autoSpeak` / `grammarOn` 写进它的 deps，勾一下 checkbox 就会重建
   * 100ms 定时器 —— 正好会打断当前这一句的收句计时。ref 没这个副作用。
   */
  const autoSpeakRef = useRef(autoSpeak);
  autoSpeakRef.current = autoSpeak;
  const grammarOnRef = useRef(grammarOn);
  grammarOnRef.current = grammarOn;
  /** 同上：`tickSentence` 在定时器里读它决定要不要自动接下一句 */
  const continuousRef = useRef(continuousMode);
  continuousRef.current = continuousMode;

  const modelReady =
    modelMode === "static"
      ? staticReady
      : modelMode === "sequence"
      ? seqReady
      : sentenceLoaded;
  const [currentPrediction, setCurrentPrediction] = useState<{
    label: string;
    word: string;
    confidence: number;
    allProbabilities: Array<{ label: string; probability: number }>;
    /**
     * 本窗口右手拇指压力单点峰值（0-255），-1 = 没有右手数据。
     *
     * 显示出来是为了让 `fistGate.THUMB_PEAK_GATE` 这个阈值**能当场校准** ——
     * 闸门的所有阈值都是从录制数据量的，而模型对录制数据本来就全对，所以录制
     * 数据证明不了阈值在实时下也对。有这个读数就不用为了调阈值去录一批。
     */
    thumbPeak: number;
    /**
     * 本窗口右手四元数累计路径转角速率（度/秒），null = 量不出（没有 IMU）。
     *
     * 这是 谢谢/难过 那一刀现在的**主判据**（见 rotationRate.ts）。显示出来的
     * 理由和拇指峰值一样、而且更迫切：50 / 70 这两个阈值是从 30 条录制里量的，
     * 相邻滑窗重叠 95%，有效样本数接近 30 —— 必须靠这个读数在实时下校准。
     * 打「谢谢」时这个数该在 30 上下，画圈打「难过」时该到 90 上下。
     */
    rotRate: number | null;
    /** 闸门有没有改掉模型的输出 —— 界面上标一下，免得把闸门的行为当成模型的行为 */
    gated: boolean;
    /** 闸门这一次是靠哪个判据下的结论 —— 界面上要说清是转角判的还是拇指判的 */
    gateReason: string;
  } | null>(null);
  const [history, setHistory] = useState<TranslationEntry[]>([]);
  const [isTranslating, setIsTranslating] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [smoothingWindow, setSmoothingWindow] = useState(5); // 平滑窗口
  const [message, setMessage] = useState("");
  /** 底部手模条开关。默认开；两个 WebGL 画面与 tfjs 推理共用 GPU，卡就关掉 */
  const [showHands, setShowHands] = useState(true);
  /** 两条链路各自实际加载的模型名，显示在 MODEL INFO —— 用来回答"现在到底在用哪个模型" */
  const [loadedNames, setLoadedNames] = useState<{
    static: string | null;
    sequence: string | null;
  }>({ static: null, sequence: null });

  // 平滑缓冲区
  const predictionBufferRef = useRef<string[]>([]);
  const lastAddedWordRef = useRef<string>("");
  const lastAddedTimeRef = useRef<number>(0);
  /** 上一个确认词的置信度。复合词合成后取两段里较小的那个，所以要留着 */
  const lastAddedConfRef = useRef<number>(0);
  const translateIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const confidenceThresholdRef = useRef(confidenceThreshold);
  const smoothingWindowRef = useRef(smoothingWindow);

  // 保持 ref 与 state 同步
  useEffect(() => {
    confidenceThresholdRef.current = confidenceThreshold;
  }, [confidenceThreshold]);
  useEffect(() => {
    smoothingWindowRef.current = smoothingWindow;
  }, [smoothingWindow]);

  /*
   * 自动加载最新模型 —— 两条链路各自独立加载，互不影响。
   *
   * 这里以前只播报静态那一条（"✓ 已自动加载静态模型 xxx"），时序模型是**默默**加载的，
   * 而 MODE 初值又固定是静态。于是刚在 /train-seq 训完时序模型的人一进来，
   * 看到的是"已加载静态模型"、用的也是静态模型，会以为自己训的模型没生效。
   * 现在：两条都播报（各自写清是哪一条），并且**默认停在更新的那一个模型上**
   * ——刚训完哪条就用哪条。用户手动点 MODE 之后不再自动改（这是挂载时跑一次的 effect）。
   *
   * 顺带一句会救人的对照：静态模型叫 `student_...`、时序模型叫 `seq_student_...`
   * （`Train.tsx:136` / `TrainSequence.tsx:211`），名字里没有 seq_ 前缀的一定是静态模型。
   */
  useEffect(() => {
    let cancelled = false;
    const loaded: string[] = [];
    /** 部署的词模型读取出错（不是"没部署"）。必须显示出来，见下 */
    let seqWarn: string | null = null;

    // 两条都**先查后判**：已在内存里的模型也要把名字查出来显示在 MODEL INFO 里，
    // 否则"到底在用哪个模型"这个问题在页面上无处可查
    const staticJob = getLatestModel().then(async (model) => {
      if (!model) return null;
      if (!isModelLoaded()) {
        await loadModelFromSaved(model);
        loaded.push(`静态单帧 "${model.name}"`);
      }
      return model;
    });

    /*
     * 时序（逐词）这一档有**两个来源**：IndexedDB 里页面内训出来的，
     * 和 Python 训好、随代码部署在 `/models/seq_student/` 的。
     *
     * 为什么必须接上部署这一条：08-28/29 那批词（我/像/笑/太阳/好看/名字）
     * 只在 Python 侧训过。以前浏览器只认 IndexedDB，用的是更早训的模型，
     * 那些词**不在标签表里** —— 表现不是"没认出来"，而是稳定输出别的词
     * （模型对任意输入都会给出一个已知类）。
     *
     * 选型沿用本文件既有的"谁更新用谁"：`SavedModel.createdAt` 与
     * `WordModelMeta.exportedAt` 都是 epoch 毫秒，可以直接比。
     * 老产物没有 exportedAt（undefined）→ 当 0，让页面内训的赢；
     * 反过来会让一个不知道多老的产物永久压住用户刚训完的模型。
     * 但**没有**页面内模型时，部署的照样加载（savedAt = -1）。
     */
    const seqJob = (async () => {
      const [saved, deployed] = await Promise.all([
        getLatestSequenceModel(),
        fetchWordModelMeta().catch((e) => {
          // 404 = 还没跑 export_weights.py。这是正常状态，不打扰
          if (e instanceof ModelMissingError) return null;
          // 别的错（500、CORS、代理插一脚）要让人看见 ——
          // 静默的表现是"我明明导出了，怎么没生效"，无处可查
          seqWarn = `部署的词模型读不出来（不是"没部署"）：${String(e)}`;
          return null;
        }),
      ]);

      const deployedAt = deployed?.exportedAt ?? 0;
      if (deployed && deployedAt >= (saved?.createdAt ?? -1)) {
        if (!isSequenceModelLoaded()) {
          await loadDeployedWordModel();
          loaded.push(`时序滑窗 "seq_student(部署)" · ${deployed.labels.length} 词`);
        }
        return { name: "seq_student(部署)", createdAt: deployedAt };
      }
      if (!saved) return null;
      if (!isSequenceModelLoaded()) {
        await loadSequenceModelFromSaved(saved);
        loaded.push(`时序滑窗 "${saved.name}"`);
      }
      return { name: saved.name, createdAt: saved.createdAt };
    })();

    Promise.all([staticJob, seqJob]).then(([staticModel, seqModel]) => {
      if (cancelled) return;
      if (staticModel) setStaticReady(true);
      if (seqModel) setSeqReady(true);
      setLoadedNames({
        static: staticModel?.name ?? null,
        sequence: seqModel?.name ?? null,
      });
      // URL 明确指定了 MODE 就不抢方向盘。这里的比较只认识静态/时序两条，
      // 句子模型不在 IndexedDB 里，让它插手会把 ?mode=sentence 覆盖掉
      if (urlMode.current === null) {
        // 谁的 createdAt 更新就切到谁；只有一个就用那一个
        if (seqModel && (!staticModel || seqModel.createdAt >= staticModel.createdAt)) {
          setModelMode("sequence");
        } else if (staticModel) {
          setModelMode("static");
        }
      }
      const msgs: string[] = [];
      if (loaded.length) msgs.push(`✓ 已自动加载：${loaded.join(" · ")}`);
      if (seqWarn) msgs.push(`⚠ ${seqWarn}`);
      if (msgs.length) setMessage(msgs.join("　|　"));
    });

    return () => {
      cancelled = true;
    };
  }, []);

  /*
   * 句子模型探测 —— 只看 weights.json 在不在，**不下权重**（638KB，没进这一档就别下）。
   *
   * 探测失败（500、CORS、代理插一脚）不能当成"没部署"：那会让 UI 显示
   * "先去跑 python_train"，而用户明明已经跑过了。所以只有 404 才算没部署，
   * 别的错误保留 `sentenceAvailable === null` 并把原文显示出来。
   */
  useEffect(() => {
    let cancelled = false;
    sentenceModelAvailable()
      .then((ok) => {
        if (!cancelled) setSentenceAvailable(ok);
      })
      .catch((e) => {
        if (cancelled) return;
        setSentenceError(`探测句子模型失败（不是"没部署"）：${String(e)}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /*
   * 切到句子档时才真正加载。加载失败**不做降级**：带着半个模型跑下去会输出
   * 看起来像话的乱句，比"这一档不可用"难查得多（见 sentenceModel.ts 顶部）。
   */
  useEffect(() => {
    if (modelMode !== "sentence") return;
    if (sentenceModelRef.current) return;
    let cancelled = false;
    setSentenceError(null);
    setSentenceNote("正在加载句子模型…");
    loadSentenceModel()
      .then((m) => {
        if (cancelled) {
          m.dispose();
          return;
        }
        sentenceModelRef.current = m;
        setSentenceAvailable(true);
        setSentenceLoaded(true);
        /*
         * 趁这里把 TTS 的 voice 列表预热掉。收句后是**自动**朗读，没有"用户点按钮"
         * 那几秒缓冲，第一句正好会撞在 `getVoices()` 还是空数组的冷启动上，
         * 退到系统默认（英文）引擎念汉字 → 一串字母音。见 speech.warmUpVoices。
         */
        warmUpVoices();
        // synthOnly 一定要说出来：合成句子里没有协同发音，那个 WER 不代表真实表现
        setSentenceNote(
          `✓ 句子模型已加载（${m.meta.labels.length} 类 · T=${m.meta.seqLen}` +
            (m.meta.valWer !== null
              ? ` · val WER ${(m.meta.valWer * 100).toFixed(1)}%`
              : "") +
            (m.meta.synthOnly ? " · 只用合成句训练，真实表现会更差" : "") +
            "）"
        );
      })
      .catch((e) => {
        if (cancelled) return;
        setSentenceLoaded(false);
        setSentenceNote(null);
        setSentenceError(String(e instanceof Error ? e.message : e));
      });
    return () => {
      cancelled = true;
    };
  }, [modelMode]);

  /*
   * 连续模式的收句门限。`SentenceEnvelope` 在本页只 new 一次，而这是个 checkbox，
   * 所以要在运行时改（`setSettle` 只影响下一跳，不打断当前这一句）。
   *
   * 一句一次模式一律回到 `SETTLE_MS` + 关自适应 —— 那是采集页的口径，必须逐位一致。
   */
  useEffect(() => {
    sentenceEnvRef.current?.setSettle(
      continuousMode ? CONTINUOUS_SETTLE_MS : SETTLE_MS,
      continuousMode
    );
  }, [continuousMode]);

  // 卸载时释放句子模型的显存。tfjs 的张量不归 GC 管，不 dispose 就一直占着
  useEffect(() => {
    return () => {
      sentenceModelRef.current?.dispose();
      sentenceModelRef.current = null;
      stopSpeaking();
    };
  }, []);

  /*
   * 下面三个上报函数都是**节流写 state** 的。推理循环 10Hz，无条件 setState
   * 会把整页每秒重渲染十次。
   *
   * 它们在组件作用域而不是 effect 里，是因为句子档的「结束」按钮和推理循环
   * 要走同一条解码路径（见 `decodeUtterance`），抄两份很快会走样。
   */
  // 归一化状态只在**翻转的那一刻**进 state；ref 记住上一次的值即可
  const reportMirrored = useCallback((v: boolean) => {
    if (mirroredRef.current === v) return;
    mirroredRef.current = v;
    setMirrored(v);
  }, []);
  // 同理：闸门状态每 100ms 变一次，只在翻转时进 state
  const reportGated = useCallback((v: boolean) => {
    if (gatedRef.current === v) return;
    gatedRef.current = v;
    setGated(v);
  }, []);
  /*
   * 捕获状态没法"只在翻转时写"—— 计时是连续变化的，进度条要跟着走。
   * 所以按**桶**发布：状态变了、或已录时长跨过 0.5s、或静止时长跨过 0.2s
   * 才写一次，约 2~5Hz。
   */
  const lastStatusKeyRef = useRef("");
  const publishCapture = useCallback((s: CaptureStatus) => {
    const key = `${s.state}:${Math.round(s.elapsedMs / 500)}:${Math.round(
      s.stillMs / 200
    )}`;
    if (key === lastStatusKeyRef.current) return;
    lastStatusKeyRef.current = key;
    setCaptureStatus(s);
  }, []);

  /**
   * 连续模式：这一句完事了，立刻重新等下一句。
   *
   * 自动收句和手动点「结束」**都要走这一个** —— 抄两份很快会走样（比如只在一条路上
   * 记得"要在解码之后调"）。
   *
   * ⚠ **必须在 `decodeUtterance` 之后调**：`rearm` 会清缓冲，先调就把这一句的数据
   * 清没了（解码读的是同一个缓冲）。
   */
  const rearmIfContinuous = useCallback(() => {
    if (!continuousRef.current) return;
    const env = sentenceEnvRef.current;
    if (!env) return;
    env.rearm(performance.now());
    publishCapture(env.status());
  }, [publishCapture]);

  /**
   * 收句 → 出词序列。推理循环和「结束」按钮**共用这一条路径**。
   *
   * 取多长的一段由 `env.take` 决定（span 与 dropTail 在那里成对算出，这里没有机会
   * 把它们拆开 —— 只缩 span 砍掉的是句子**开头**，见 sentenceEnvelope.ts）。
   */
  const decodeUtterance = useCallback(
    (act: CaptureAction) => {
      const env = sentenceEnvRef.current;
      if (!env || act.kind !== "decode") return;

      const raw = env.take(act, "_sentence");
      if (!raw) {
        setSentenceNote("这一段太短，取不出可用的数据段。");
        return;
      }
      const loadedSent = sentenceModelRef.current;
      if (!loadedSent) {
        setSentenceNote("句子模型还没加载好，这一句没解。");
        return;
      }
      // 与时序档同一套归一化：模型是用右手口径的数据训的（见 handMirror.ts）
      const dominant = dominanceRef.current?.current() ?? "right";
      const norm = normalizeHandedness(raw, dominant);
      reportMirrored(norm.mirrored);
      try {
        const pred = predictSentence(loadedSent, norm.sample);
        /*
         * 两道后处理（拇指闸门 + 复合词回收）—— 词路径上一直有，句子路径以前
         * 一道都没走，所以打「你好」会出「你 难过」。见 sentencePostprocess.ts。
         * **必须用 norm.sample**：闸门读右手 0..12 通道，镜像之前那是小拇指。
         */
        const post = postprocessSentence(pred, norm.sample, loadedSent.meta.labels);
        setSentenceWords(post.words);
        // 上一句的代词选择不能留到这一句：下标对不上，会把这句的某个词改成别的人称
        setPronOverrides({});
        const base =
          act.reason === "maxLength"
            ? `到了 ${MAX_UTTERANCE_MS / 1000}s 上限自动收句 —— 超出的部分没进模型。`
            : act.reason === "manual"
            ? `手动收句 · ${(raw.durationMs / 1000).toFixed(1)}s`
            : `停手收句 · ${(raw.durationMs / 1000).toFixed(1)}s（末尾静止已掐掉）`;
        /*
         * 后处理改了什么要**说出来**。这两道规则都会把模型的输出改掉，
         * 不显示的话线上表现是"模型好像认错了/少认了一个词"，而真正动手的是规则。
         * 拇指峰值一并显示：闸门判错时那个数就是唯一能改的旋钮（THUMB_PEAK_GATE）。
         */
        const fixes = [
          ...post.gated.map((g) => {
            // 判据要分开说：转角和拇指是两个不同的旋钮，混着写就不知道该调哪个
            const why = g.reason.startsWith("thumb")
              ? `拇指峰值 ${g.peak}，阈值 ${THUMB_PEAK_GATE}`
              : `转角速率判定，阈值 ${ROT_RATE_LO}/${ROT_RATE_HI}°/s`;
            return `闸门：${getTranslationLabel(g.from)}→${getTranslationLabel(g.to)}（${why}）`;
          }),
          ...post.merged.map((w) => `合成：${getTranslationLabel(w)}`),
          /*
           * 屏蔽也要报。**这一行是唯一能看出"少了一个词不是模型的锅"的地方** ——
           * `SUPPRESSED_WORDS` 是权宜表（见 sentencePostprocess.ts 的代价说明），
           * 不报的话下次没人记得它开着，会去查模型。
           */
          ...post.suppressed.map((w) => `已屏蔽：${getTranslationLabel(w)}`),
        ];
        /*
         * ===== 收句后自动朗读 =====
         *
         * 时机就是**这一刻**（刚解出来），不是"顺句结果变了就念"。后者会在用户
         * 点代词、删词、勾顺句规则时每改一下念一遍 —— 吵，而且盖掉他正在读的字。
         * 要重念有「朗读」按钮。
         *
         * 三条不能省：
         *  1. `post.words.length > 0`。全 blank 时 `speakChinese("")` 返回
         *     `reason:"没有内容可朗读"`，那句话会挤进下面的提示栏，盖掉
         *     "这一段没解出任何词"这个真正有用的信息。
         *  2. overrides 传 `{}` —— `setPronOverrides({})` 就在上面几行，此刻确实是空的。
         *  3. 失败/降级原因必须并进 `fixes` 显示。静默失败等于让人以为音箱坏了
         *     （`speakChinese` 返回 `{ok, reason}` 而不抛，就是为了这个）。
         */
        if (post.words.length > 0) {
          const text = resolveSentence(post.words, {}, grammarOnRef.current).text;
          if (autoSpeakRef.current) {
            const r = speakChinese(text);
            if (r.reason) fixes.push(`朗读：${r.reason}`);
          }
          /*
           * ===== 自动成句 =====
           *
           * 连续模式下这一步是**必需的**，不是方便功能：不进历史的话，下一句解出来
           * 会直接 `setSentenceWords` 覆盖，上一句没点过「成句」就永久没了。
           *
           * `lastLive` 标记让之后的编辑（改代词/删词）能同步回历史最后一条 ——
           * 见下面那个 effect。全 blank（`length === 0`）不进历史：一条空记录
           * 没有信息，只会把历史刷满。
           */
          if (continuousRef.current) {
            setSentenceHistory((prev) => [...prev, text]);
            setLastLive(true);
          }
        }
        setSentenceNote(fixes.length ? `${base} · ${fixes.join("；")}` : base);
      } catch (e) {
        setSentenceNote(`解码失败：${String(e instanceof Error ? e.message : e)}`);
      }
    },
    [reportMirrored]
  );

  /*
   * 自动成句之后的编辑要**同步回历史最后一条**。
   *
   * 少了这个 effect 的表现：连续模式下打完一句自动进历史，你再把某个「你」改成
   * 「他」—— 屏幕中间的大字跟着变了，历史里留的还是「你」。两处不一致，而界面上
   * 完全看不出哪一处才是最终结果。
   *
   * `prev[last] === text` 的相等判断是**防死循环**的：这个 effect 自己会
   * `setSentenceHistory`，不比较就会无限重渲染。
   */
  useEffect(() => {
    if (!lastLive || !sentenceWords) return;
    if (sentenceWords.length === 0) {
      // 词被删空了 —— 历史里那条也该撤掉，不能留一句已经不存在的话
      setSentenceHistory((prev) => prev.slice(0, -1));
      setLastLive(false);
      return;
    }
    const text = resolveSentence(sentenceWords, pronOverrides, grammarOn).text;
    setSentenceHistory((prev) =>
      prev.length === 0 || prev[prev.length - 1] === text
        ? prev
        : [...prev.slice(0, -1), text]
    );
  }, [lastLive, sentenceWords, pronOverrides, grammarOn]);

  // 推理循环 — 使用 ref 读取最新帧，避免闭包陷阱
  useEffect(() => {
    if (!isTranslating || !isConnected || !modelReady) return;

    console.log("[Translate] Starting inference loop...");

    /*
     * 主手判定：每 DOMINANCE_RECHECK_MS 重算一次，结果留在判定器里，
     * 推理循环每一跳都读它最新的结论（`tracker.current()`）。
     *
     * 为什么判定用自己的短窗口、而不是复用推理那个 `WINDOW_MS` 窗口：
     * 静态模式根本不取窗口（它只读最新一帧），但一样需要主手判定；
     * 而且推理窗口攒满要整整 `WINDOW_MS`，那段时间里主手是空的。两条路共用同一个
     * 判定器与同一个短窗口，行为才一致 —— 切 MODE 不该改变主手判定。
     */
    let lastDominanceAt = 0;
    const refreshDominance = () => {
      const now = Date.now();
      if (now - lastDominanceAt < DOMINANCE_RECHECK_MS) return;
      lastDominanceAt = now;
      const tracker = dominanceRef.current;
      if (!tracker) return;
      const snap =
        windowBufRef.current?.snapshot(DOMINANCE_WINDOW_MS, "_dom") ?? null;
      const v = tracker.update(snap, bendRangesRef.current ?? {}, {
        left: gloveLeft.isConnected,
        right: gloveRight.isConnected,
      });
      setDominance(v);
    };

    /**
     * 句子档的一跳：只推进收句状态机，拿到 `decode` 才跑一次推理。
     *
     * 与另外两条路的根本区别是**不做滑窗**：这里 100ms 一次的只有"手还在动吗"
     * 这个物理判据，模型一句只跑一次。
     */
    const tickSentence = () => {
      const env = sentenceEnvRef.current;
      if (!env) return;

      // 这里**不调 reportGated**：`gated` 那行提示只画在逐词档，
      // 句子档的"手停住了"是由捕获面板的静止进度条说的。多写一份 state
      // 只会每次起手/停手都触发一次整页重渲染，而且没人读。
      //
      // 量程传**镜像之前**的（`bendRangesRef` 就是原始的两只手量程）：镜像后左手数据
      // 配的是右手量程，动作能量的分母就错了。镜像发生在 decode 那一步、tick 之后
      const act = env.tick(performance.now(), bendRangesRef.current ?? {});
      publishCapture(env.status());
      if (act.kind === "none") return;
      if (act.kind === "abort") {
        /*
         * 连续模式下"等太久"不是错误 —— 你只是还没开始打下一句。
         * `keepWaiting` 不清缓冲（见 sentenceEnvelope）：清了会每 8 秒造一个
         * 600ms 探测盲区，正好在那时起手就吃掉句首。
         */
        if (continuousRef.current && act.reason === "armTimeout") {
          env.keepWaiting(performance.now());
          publishCapture(env.status());
          return;
        }
        setSentenceNote(
          act.reason === "armTimeout"
            ? "等了 8 秒没见到动作，这一句取消了。再点「开始一句」。"
            : "这一段没录到任何动作，没有送去解码（空段只会解出乱句）。"
        );
        return;
      }
      decodeUtterance(act);
      /*
       * 接成环。`rearmIfContinuous` 刻意**不调** `stopSpeaking()` /
       * `dominanceRef.reset()`：手动 `armSentence` 那条路有这两行，抄过来的后果
       * 分别是"每句话刚念一个字就被自己掐掉"和"左手用户每句句首按右手口径归一化"。
       */
      rearmIfContinuous();
    };

    translateIntervalRef.current = setInterval(() => {
      refreshDominance();
      const dominant = dominanceRef.current?.current() ?? "right";

      if (modelMode === "sentence") {
        tickSentence();
        return;
      }

      let result:
        | {
            label: string;
            confidence: number;
            allProbabilities: Array<{ label: string; probability: number }>;
          }
        | null = null;
      /**
       * 本窗口的右手拇指压力单点峰值，给 `fistGate` 用。
       * -1 = 还没量（静态档不走滑窗，没有窗口可量）—— 闸门收到 -1 会自己不介入。
       */
      let thumbPeakValue = -1;
      /**
       * 本窗口的右手转角速率读数，给 `fistGate` 当主判据用。
       * null = 静态档（不走滑窗，量不到）—— 闸门收到 null 会退回纯拇指行为。
       */
      let rotReading: RotationReading | null = null;
      let gatedThisTick = false;
      let gateReasonThisTick = "";

      if (modelMode === "static") {
        // 从左右手全速 ref 读取，构建双手触觉输入
        const lf = gloveLeft.latestFrameRef.current;
        const rf = gloveRight.latestFrameRef.current;
        if (!lf && !rf) {
          return;
        }
        let leftInput = lf
          ? { sensor_data: lf.mapped_data, quaternion: lf.quaternion }
          : null;
        let rightInput = rf
          ? { sensor_data: rf.mapped_data, quaternion: rf.quaternion }
          : null;
        // 归一化到右手口径（模型是用右手采的数据训的），理由见 handMirror.ts。
        // 判定与序列那条路**同一个判定器**：主手判成左手就镜像，
        // 判成 both（双手词）或右手都原样喂
        const doMirror = dominant === "left";
        if (doMirror) {
          const m = mirrorStaticInputs(leftInput, rightInput);
          leftInput = m.left;
          rightInput = m.right;
        }
        reportMirrored(doMirror);
        result = predict(leftInput, rightInput);
      } else {
        // 环形缓冲攒够一整个窗口前 snapshot 返回 null，此时安静地跳过。
        // 不要 fallback 到"用现有的半个窗口凑合推理" —— 半个窗口会被最近邻
        // 拉成一条直线，那是个假的"静止"动作，输出的词是错的。
        const raw = windowBufRef.current?.snapshot(WINDOW_MS) ?? null;
        if (!raw) return;

        /*
         * 动作闸门：窗口里没人在动就**不推理**，见 `motionGate.ts`。
         *
         * 必须在归一化之前判：能量要除以各自那只手的标定量程，镜像之后左手数据
         * 配的是右手量程，分母就错了。
         *
         * 走的是下面 `_idle` 那条已有的路（清空平滑缓冲 + 不出词）——
         * 那本该是 `_idle` 伪类的活，但 `_idle` 只有 4 条样本、几乎不可能被预测到，
         * 所以在模型之前用物理量把这些窗口挡掉。
         */
        const motion = judgeWindowMotion(raw, bendRangesRef.current ?? {});
        reportGated(!motion.moving);
        if (!motion.moving) {
          setCurrentPrediction(null);
          predictionBufferRef.current = [];
          return;
        }
        // 归一化到右手口径：整条样本镜像 + 左右槽位互换。模型的两只手是两段独立槽位、
        // 且 137 维指序左右相反，不归一化的话左手做的动作落在训练时只见过静止基线的
        // 那半边，输出会塌到某一个固定的词上（见 handMirror.ts 顶部）
        const norm = normalizeHandedness(raw, dominant);
        reportMirrored(norm.mirrored);
        // **必须用归一化之后的样本**：左手的 137 维指序与右手相反，镜像之前
        // 读 0-11 会读到小拇指的压力（见 fistGate.thumbPeak 的注释）
        thumbPeakValue = thumbPeak(norm.sample);
        /*
         * 转角速率同样**必须用归一化之后的样本** —— 不是因为镜像会改变转角大小
         * （反射保持角度绝对值，不会），而是因为左手数据镜像前放在 `leftImu`
         * 槽位里，`rotationRate` 只读 `rightImu`，不归一化直接读不到东西。
         */
        rotReading = rotationRate(norm.sample);
        result = predictSequence(norm.sample);
      }

      if (!result) {
        console.warn("[Translate] predict() returned null");
        return;
      }

      /*
       * 谢谢 / 难过 这一刀不让网络判，用物理量判（见 fistGate.ts）。
       *
       * **主判据是四元数累计路径转角速率**：谢谢整只手不动（2000ms 窗速率中位
       * 33°/s、最大 57），难过掌心朝胸口画圈（中位 90）。实测分对 97.7%。
       * 速率落在中间带 [50,70) 时才问拇指 —— 谢谢要「大拇指下压一次」，拇指必然
       * 吃力（单点峰值中位 12，208 个窗没有一个低于 5）；难过是**虚握**拳，
       * 拇指全程不吃力（中位 0）。拇指单独用只有 88.2%。
       *
       * 两条都只在模型自己的 top-2 恰好是这一对时介入。
       *
       * **必须放在这里** —— 在 setCurrentPrediction 和平滑缓冲之前。放在后面的话
       * 平滑缓冲攒的是未修正的标签，确认出来的词还是错的；而且复合词规则
       * （你+谢谢→你好）读的也是修正后的标签，靠它把「你好」的第二段扳回谢谢。
       */
      const gate = applyFistGate(
        result.label,
        result.allProbabilities[1]?.label,
        thumbPeakValue,
        THUMB_PEAK_GATE,
        rotReading
      );
      gateReasonThisTick = gate.reason;
      if (gate.changed) {
        gatedThisTick = true;
        result = {
          label: gate.label,
          /*
           * 置信度**沿用原来第一名的值**，不换成被改判那个词自己的概率。
           * 换了的话（比如谢谢 0.9 / 难过 0.08）置信度阈值会把这个词卡掉，
           * 结果是"不再认错，但也什么都不出" —— 那不是修好。
           * 这个数的正确读法是"模型有多确定它是这一对里的某一个",
           * 由闸门决定是哪一个 —— 闸门在这一刀上比模型更可靠（实测 0/208 误伤）。
           */
          confidence: result.confidence,
          // 只重排、不改数值：界面上会看到「难过 8%」排在「谢谢 90%」前面，
          // 看着别扭但是真话，而且一眼就能看出闸门介入了
          allProbabilities: [
            ...result.allProbabilities.filter((p) => p.label === gate.label),
            ...result.allProbabilities.filter((p) => p.label !== gate.label),
          ],
        };
      }

      // 收敛成 const，下面的闭包（.every）里才能保持非空收窄
      const pred = result;

      // idle 是滑窗推理必需的伪类：模型对任意窗口都会输出某个词，
      // 没有它，手放松/动作过渡期间会持续乱吐词。命中 idle 就清空平滑缓冲
      // （相当于打断当前这个词的确认过程），且绝不进历史。
      if (pred.label === IDLE_LABEL) {
        setCurrentPrediction(null);
        predictionBufferRef.current = [];
        return;
      }

      /*
       * 屏蔽表命中 → 和 `IDLE_LABEL` 同一种处理：整个丢掉这一跳，
       * 既不显示大字、也不进历史。
       *
       * 为什么不是"显示但不进历史"：那样屏幕上照样在吐「吃」，
       * 而用户要的就是不看见它（`suppressedWords.ts` 里有代价说明）。
       * 清平滑缓冲的理由同 idle —— 留着的话下一跳的多数投票还带着这个词。
       */
      if (isSuppressed(pred.label)) {
        setCurrentPrediction(null);
        predictionBufferRef.current = [];
        return;
      }

      // 合并类（merged_pron_sg）不在词表里，`getWordById` 查不到 —— 直接用
      // pred.label 兜底会把 `merged_pron_sg` 这个原始 id 露到界面上
      setCurrentPrediction({
        label: pred.label,
        word: getTranslationLabel(pred.label),
        confidence: pred.confidence,
        allProbabilities: pred.allProbabilities.slice(0, 5),
        thumbPeak: thumbPeakValue,
        // 速率算不出来时给 null，**不要给 0** —— 0 的含义是"手完全没动"，
        // 会让人以为闸门读到了"静止"，而实际上是根本没量到
        rotRate: rotReading ? rotReading.ratePerSec : null,
        gated: gatedThisTick,
        gateReason: gateReasonThisTick,
      });

      // 平滑处理：连续 N 帧相同结果才确认
      const window = smoothingWindowRef.current;
      predictionBufferRef.current.push(pred.label);
      if (predictionBufferRef.current.length > window) {
        predictionBufferRef.current.shift();
      }

      // 检查是否稳定
      if (predictionBufferRef.current.length >= window) {
        const allSame = predictionBufferRef.current.every(
          (l) => l === pred.label
        );
        const now = Date.now();
        const threshold = confidenceThresholdRef.current;
        if (
          allSame &&
          pred.confidence >= threshold &&
          (pred.label !== lastAddedWordRef.current ||
            now - lastAddedTimeRef.current > 2000) // 同一个词至少间隔2秒
        ) {
          /*
           * 复合词回收：你 → 谢谢 实际上是「你好」的两段（第二段的竖大拇指在纯
           * 触觉特征上与「谢谢」同型，模型没有通道能分开 —— 见 compoundWords.ts）。
           *
           * 命中时**替换历史里最后一条**，而不是延迟出词：延迟会让所有词都慢半拍，
           * 而这里只有两条规则。代价是界面上先出「你」再变成「你好」，闪一下。
           */
          const compound = matchCompound(
            lastAddedWordRef.current,
            lastAddedTimeRef.current,
            pred.label,
            now
          );
          if (compound) {
            const merged: TranslationEntry = {
              word: getTranslationLabel(compound.word),
              label: compound.word,
              // 两段各自的置信度都不代表整体，取较小的那个（更保守的读数）
              confidence: Math.min(pred.confidence, lastAddedConfRef.current),
              timestamp: now,
            };
            // 只替换最后一条。历史为空时（理论上不该发生：能命中说明刚加过一条）
            // 退化成追加，不静默丢词
            setHistory((prev) =>
              prev.length ? [...prev.slice(0, -1), merged] : [merged]
            );
            lastAddedWordRef.current = compound.word;
            lastAddedTimeRef.current = now;
            lastAddedConfRef.current = merged.confidence;
            predictionBufferRef.current = [];
            return;
          }

          // 确认识别结果
          const entry: TranslationEntry = {
            word: getTranslationLabel(pred.label),
            label: pred.label,
            confidence: pred.confidence,
            timestamp: now,
          };
          setHistory((prev) => [...prev, entry]);
          lastAddedWordRef.current = pred.label;
          lastAddedTimeRef.current = now;
          lastAddedConfRef.current = pred.confidence;
          predictionBufferRef.current = [];
        }
      }
    }, 100); // 10Hz 推理频率

    return () => {
      console.log("[Translate] Stopping inference loop.");
      if (translateIntervalRef.current) {
        clearInterval(translateIntervalRef.current);
        translateIntervalRef.current = null;
      }
      // 停下来之后这两行提示就不再反映任何正在发生的事，留着是假信息
      reportMirrored(false);
      reportGated(false);
    };
    // 不再依赖 latestFrame/confidenceThreshold/smoothingWindow（那些高频变）。
    // 两只手套的连接状态**必须**在依赖里：主手判定在窗口没攒满时靠它兜底，
    // 闭包里留一份陈旧的连接状态会让掉线后的预热窗口判错手
  }, [
    isTranslating,
    isConnected,
    modelReady,
    modelMode,
    gloveLeft.isConnected,
    gloveRight.isConnected,
  ]);

  // 开始/停止翻译
  const toggleTranslation = useCallback(() => {
    if (isTranslating) {
      setIsTranslating(false);
      setCurrentPrediction(null);
      predictionBufferRef.current = [];
    } else {
      // 开始前清掉环形缓冲：里面可能是上次停止翻译前留下的旧动作，
      // 不清会在刚开始的一个 `WINDOW_MS` 内拿陈旧数据推理。
      // 主手判定跟着一起清：上次的判定是上次那段动作得出的，这次可能换了人／换了手
      windowBufRef.current?.clear();
      dominanceRef.current?.reset();
      setDominance(null);
      setIsTranslating(true);
    }
  }, [isTranslating]);

  const switchMode = useCallback((mode: ModelMode) => {
    setModelMode(mode);
    setCurrentPrediction(null);
    predictionBufferRef.current = [];
    windowBufRef.current?.clear();
    // 缓冲清空后主手判定的依据就没了，留着上次的结论会在预热期误导人
    dominanceRef.current?.reset();
    setDominance(null);
    // 切档时正在录的那一句作废：缓冲刚被清掉，它的数据已经不在了。
    // 不 cancel 的话状态机还停在 capturing，切回来会拿一段跨越切档的残缺数据去解码
    sentenceEnvRef.current?.cancel();
    setCaptureStatus(null);
    lastStatusKeyRef.current = "";
    stopSpeaking();
    // 历史里那条"当前这一句"就此定版：切档之后 sentenceWords 还在，
    // 但已经不该再跟着编辑同步了（这一句的采集已经作废）
    setLastLive(false);
    /*
     * 切档一律停掉推理循环。
     *
     * 句子档是被「开始一句」隐式打开的（没有独立的「开始翻译」按钮），
     * 不在这里停的话，切到逐词档会**直接开始出词** —— 用户没点过「开始翻译」，
     * 看起来像是页面自己在乱吐词。反过来切进句子档也一样：循环空转着，
     * 而面板上写的是"未开始"。
     */
    setIsTranslating(false);
  }, []);

  /*
   * ===== 句子档的交互 =====
   *
   * 「开始一句」同时负责把推理循环开起来（`isTranslating`）—— 这一档没有单独的
   * 「开始翻译」按钮。两个开关会让人点了一个不管用，而这一档本来就是"一句一次"的。
   */
  const armSentence = useCallback(() => {
    // `env.arm` 里会清缓冲（里面可能有上一句的尾巴，`snapshotAll` 是"有多少取多少"）。
    // 缓冲清空后主手判定的依据也就没了，留着上次的结论会在预热期误导人
    sentenceEnvRef.current?.arm(performance.now());
    dominanceRef.current?.reset();
    setDominance(null);
    setCaptureStatus(sentenceEnvRef.current?.status() ?? null);
    setSentenceWords(null);
    setPronOverrides({});
    setSentenceNote(null);
    stopSpeaking();
    // 上一句在历史里就此定版：words 已经清了，同步 effect 再跑一次会把
    // 历史最后一条重写成空的当前句
    setLastLive(false);
    setIsTranslating(true);
  }, []);

  const finishSentence = useCallback(() => {
    const env = sentenceEnvRef.current;
    if (!env) return;
    const act = env.finish();
    setCaptureStatus(env.status());
    lastStatusKeyRef.current = "";
    if (act.kind === "abort") {
      setSentenceNote("还没录到动作就结束了，没有送去解码。");
      // 手动「结束」在连续模式下也要接回环：没录到动作时状态机已经被
      // finish() 打回 idle，不 rearm 的话面板停在"未开始"，再起手不会开始
      rearmIfContinuous();
      return;
    }
    // 不能"等下一跳 tick 去解码"：finish() 已经把状态机打回 idle，
    // 下一跳只会返回 none，这一句就永远不解了
    decodeUtterance(act);
    // 顺序不能反：rearm 会清缓冲，先 rearm 就把这一句的数据清没了
    rearmIfContinuous();
  }, [decodeUtterance, rearmIfContinuous]);

  /**
   * 连续模式的「停止」。
   *
   * `cancel()` 而不是 `finish()`：停止的意思是"别再录了"，不是"把手头这半句解出来"。
   * 正在录到一半时按停止，那半句本来就不完整，解出来只会往历史里塞一条乱句。
   *
   * **不清 `sentenceWords`** —— 停下来之后用户往往还要改代词、点朗读。
   * 但 `lastLive` 要清：采集已经结束，历史里那条不该再跟着编辑动。
   */
  const stopSentence = useCallback(() => {
    sentenceEnvRef.current?.cancel();
    setCaptureStatus(sentenceEnvRef.current?.status() ?? null);
    lastStatusKeyRef.current = "";
    setLastLive(false);
    setIsTranslating(false);
    setSentenceNote("已停止。点「开始」继续。");
  }, []);

  /**
   * 清历史。**必须一起清 `lastLive`** —— 不清的话同步 effect 下一跳会看到
   * "历史为空但 lastLive 为真"，把当前这句又写回一条空历史里
   * （effect 里对 `prev.length === 0` 有兜底，但语义上这一句已经不在历史里了）
   */
  const clearSentenceHistory = useCallback(() => {
    setSentenceHistory([]);
    setLastLive(false);
  }, []);

  const setOverride = useCallback((index: number, member: string) => {
    setPronOverrides((prev) => ({ ...prev, [index]: member }));
  }, []);

  /**
   * 删掉一个词。**overrides 必须跟着重排下标** —— 它是按词序下标存的，
   * 不重排的话删掉第 1 个词之后，本来给第 2 个词选的「你」会落到第 3 个词上，
   * 用户看到的是"我改了一个词，另一个词自己变了"。
   */
  const deleteSentenceWord = useCallback((index: number) => {
    setSentenceWords((prev) => {
      if (!prev || index < 0 || index >= prev.length) return prev;
      return prev.filter((_, i) => i !== index);
    });
    setPronOverrides((prev) => {
      const next: Record<number, string> = {};
      for (const [k, v] of Object.entries(prev)) {
        const i = Number(k);
        if (i === index) continue;
        next[i > index ? i - 1 : i] = v;
      }
      return next;
    });
  }, []);

  /**
   * 「成句」按钮。**两档语义不同。**
   *
   * 一句一次模式：把这一句加进历史（用户没点就不进）。
   *
   * 连续模式：这一句在解出来的那一刻就已经自动进历史了（否则下一句的
   * `setSentenceWords` 会把它覆盖掉、永久丢失）。所以这里**不能再追加** ——
   * 那会让同一句进两遍。它的含义变成**定版**：解除"历史最后一条跟着编辑同步"，
   * 之后改代词/删词不再影响已经记下的那条。
   */
  const commitSentence = useCallback((text: string) => {
    if (continuousRef.current) {
      setLastLive(false);
      setSentenceWords(null);
      setPronOverrides({});
      setSentenceNote("已定版。接着打下一句就行，不用点按钮。");
      return;
    }
    setSentenceHistory((prev) => [...prev, text]);
    setSentenceWords(null);
    setPronOverrides({});
    setSentenceNote("已成句。再点「开始一句」打下一句。");
  }, []);

  const activeLabels = useMemo(
    () => (modelMode === "static" ? getLoadedLabels() : getLoadedSequenceLabels()),
    // labels 存在模块级变量里，React 看不到它变化，只能靠这两个信号触发重算
    [modelMode, modelReady]
  );

  // 清除历史
  const clearHistory = useCallback(() => {
    setHistory([]);
    lastAddedWordRef.current = "";
  }, []);

  // 组合翻译文本
  const translatedText = history.map((h) => h.word).join(" ");

  return (
    /* h-screen + overflow-hidden：底部手模条要按"剩下多少高度"占位，
       父级高度必须是确定值；min-h-screen 下长历史会把手模条顶到屏幕外 */
    <div
      className="h-screen flex flex-col overflow-hidden"
      style={{ backgroundColor: "#0a0e1a" }}
    >
      {/* 顶部导航 */}
      <header className="h-12 flex items-center justify-between px-4 border-b border-[#00f0ff]/15 shrink-0">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[#00f0ff]/20" />
          <span className="text-xs font-bold tracking-widest text-[#00f0ff] font-mono">
            SIGN LANGUAGE TRANSLATOR
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          {modelReady && (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <Brain className="w-3 h-3" />
              MODEL
            </span>
          )}
          {isTranslating && (
            <span className="text-[#ff2d7b] flex items-center gap-1 animate-pulse">
              <Volume2 className="w-3 h-3" />
              LIVE
            </span>
          )}
          {/* 手套状态（左右手分开）+ 下一步 */}
          <StepNav />
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 主内容区：上＝翻译输出，下＝双手 3D 手模条 */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          {/*
           * 翻译输出区 —— 滚动容器与居中容器**必须分成两层**。
           *
           * 原来是一层：`flex-1 min-h-0 overflow-y-auto flex flex-col justify-center`。
           * `justify-center` 和 `overflow-y-auto` 凑在一起是 flexbox 的经典坑：内容超高时
           * 会同时朝两头溢出，而**朝顶部溢出的那部分滚不到**（scrollTop 到 0 就停了）。
           * 底下的手模条一占走 300px，这里剩的高度就装不下"大字＋置信度条＋按钮＋历史面板"，
           * 于是最先看不见的恰好是排在最上面的识别结果大字 —— 表现和"输出窗口被删了"一样。
           *
           * 现在外层只管滚动、内层用 `min-h-full` 管居中：内容矮时内层撑满外层、正常居中；
           * 内容高时内层自然长过外层，从顶部开始滚，一行都不会丢。
           */}
          <div className="flex-1 min-h-0 overflow-y-auto">
            <div className="min-h-full flex flex-col items-center justify-center p-6 space-y-6">
              {/* 模型未加载 —— 句子档单独一套文案：它的模型不在 IndexedDB 里，
                  指向 /train-seq 会让人训出一个这一档根本不会加载的模型 */}
              {!modelReady &&
                (modelMode === "sentence" ? (
                  <SentenceModelMissing
                    available={sentenceAvailable}
                    error={sentenceError}
                  />
                ) : (
                  <div className="text-center space-y-3">
                    <Brain className="w-16 h-16 mx-auto text-[#334455]" />
                    <p className="text-sm text-[#556677]">
                      {modelMode === "static"
                        ? "请先训练并加载静态模型"
                        : "请先训练并加载时序模型"}
                    </p>
                    <Link
                      href={modelMode === "static" ? "/train" : "/train-seq"}
                      className="cyber-btn px-4 py-2 rounded-sm text-xs inline-flex items-center gap-2"
                    >
                      {modelMode === "static" ? "前往静态训练 →" : "前往时序训练 →"}
                    </Link>
                  </div>
                ))}

              {/* 手套未连接 */}
              {modelReady && !isConnected && (
                <div className="text-center space-y-4">
                  <Hand className="w-16 h-16 mx-auto text-[#556677]" />
                  <p className="text-sm text-[#8899aa]">手套未连接</p>
                  {gloveError && (
                    <p className="text-[10px] text-[#ff2d7b]">{gloveError}</p>
                  )}
                  {/* 连接入口只有第 1 步一处 —— 那里同时做零位与弯折标定，
                      在别处另开一次串口只会把那些标定作废 */}
                  <Link
                    href="/mocap"
                    className="cyber-btn px-5 py-2.5 rounded-sm text-xs inline-flex items-center gap-2"
                  >
                    <Zap className="w-4 h-4" />
                    去第 1 步连接手套
                  </Link>
                  <p className="text-[10px] text-[#556677]">连接任一只手即可翻译；双手手语请两只都连</p>
                </div>
              )}

              {/* 句子档就绪 —— 整块换成 SentencePanel。
                  逐词那套（大字 + 置信度条 + 开始翻译按钮 + 词历史）在这一档全都
                  不适用：没有"当前这个词"，也没有逐词置信度，句子的开始/结束
                  由面板自己的按钮驱动 */}
              {modelReady && isConnected && modelMode === "sentence" && (
                <SentencePanel
                  words={sentenceWords}
                  overrides={pronOverrides}
                  onOverride={setOverride}
                  onDeleteWord={deleteSentenceWord}
                  grammarOn={grammarOn}
                  onToggleGrammar={setGrammarOn}
                  autoSpeak={autoSpeak}
                  onToggleAutoSpeak={setAutoSpeak}
                  continuous={continuousMode}
                  onToggleContinuous={setContinuousMode}
                  status={captureStatus}
                  note={sentenceNote}
                  onArm={armSentence}
                  onStop={stopSentence}
                  onFinish={finishSentence}
                  onCommit={commitSentence}
                  history={sentenceHistory}
                  onClearHistory={clearSentenceHistory}
                  running={isTranslating}
                  disabled={!modelReady || !isConnected}
                />
              )}

              {/* 就绪状态 - 可以翻译（逐词两档） */}
              {modelReady && isConnected && modelMode !== "sentence" && (
                <>
                  {/* 当前识别结果 */}
                  <div className="text-center space-y-4">
                    {currentPrediction ? (
                      <>
                        <div
                          className="text-7xl font-bold transition-all duration-300"
                          style={{
                            fontFamily: "'Space Grotesk', sans-serif",
                            color:
                              currentPrediction.confidence >= confidenceThreshold
                                ? getCategoryColor(
                                    getWordById(resolveToMember(currentPrediction.label))?.category ?? ""
                                  )
                                : "#556677",
                            textShadow:
                              currentPrediction.confidence >= confidenceThreshold
                                ? `0 0 30px ${getCategoryColor(
                                    getWordById(resolveToMember(currentPrediction.label))?.category ?? ""
                                  )}40`
                                : "none",
                            opacity:
                              currentPrediction.confidence >= 0.5
                                ? 1
                                : 0.4,
                          }}
                        >
                          {currentPrediction.word}
                        </div>
                        {/* 置信度条 */}
                        <div className="w-48 mx-auto space-y-1">
                          <div className="h-2 bg-[#1a2030] rounded-full overflow-hidden border border-[#00f0ff]/20">
                            <div
                              className="h-full rounded-full transition-all duration-200"
                              style={{
                                width: `${currentPrediction.confidence * 100}%`,
                                backgroundColor:
                                  currentPrediction.confidence >= confidenceThreshold
                                    ? "#00e5a0"
                                    : currentPrediction.confidence >= 0.5
                                    ? "#f59e0b"
                                    : "#ff2d7b",
                              }}
                            />
                          </div>
                          <div className="text-[10px] font-mono text-[#556677] text-center">
                            置信度: {(currentPrediction.confidence * 100).toFixed(1)}%
                          </div>
                          {/*
                            转角速率读数 —— 谢谢/难过 那条闸门现在的**主判据**
                            （见 rotationRate.ts）。50 / 70 这两个阈值是从 30 条录制
                            量的，相邻滑窗重叠 95%，有效样本数接近 30 —— 所以这个
                            读数不是"顺便显示一下"，是校准这两个常量的唯一手段。
                            打「谢谢」时该在 30 上下，画圈打「难过」时该到 90 上下。
                          */}
                          {currentPrediction.rotRate !== null && (
                            <div className="text-[10px] font-mono text-center text-[#556677]">
                              转角速率{" "}
                              <span
                                className={
                                  currentPrediction.rotRate >= ROT_RATE_HI
                                    ? "text-[#f59e0b]"
                                    : currentPrediction.rotRate < ROT_RATE_LO
                                      ? "text-[#00e5a0]"
                                      : "text-[#7788aa]"
                                }
                              >
                                {currentPrediction.rotRate.toFixed(0)}
                              </span>
                              <span className="text-[#334455]">
                                {" "}
                                °/s · 静止&lt;{ROT_RATE_LO} / 画圈≥{ROT_RATE_HI}
                              </span>
                              {/* 中间带要明说，否则读数在 50~70 之间时看不出是谁在做决定 */}
                              {currentPrediction.rotRate >= ROT_RATE_LO &&
                                currentPrediction.rotRate < ROT_RATE_HI && (
                                  <span className="ml-1 text-[#7788aa]">
                                    中间带→看拇指
                                  </span>
                                )}
                            </div>
                          )}
                          {/*
                            拇指压力读数 —— 原来是这条闸门的唯一判据，现在降级成
                            中间带的仲裁 + 短区间（句子路径）的唯一判据。
                            显示出来同样是为了让阈值能当场校准：闸门的阈值全是从录制数据
                            量的，而模型对录制数据本来就全对，所以录制数据证明不了阈值
                            在实时下也对。握拳时这个数该接近 0，点拇指时该跳到 12 上下。
                            只在逐词档显示（静态档不走滑窗，量不到）。
                          */}
                          {currentPrediction.thumbPeak >= 0 && (
                            <div className="text-[10px] font-mono text-center text-[#556677]">
                              拇指压力峰值{" "}
                              <span
                                className={
                                  currentPrediction.thumbPeak >= THUMB_PEAK_GATE
                                    ? "text-[#00e5a0]"
                                    : "text-[#334455]"
                                }
                              >
                                {currentPrediction.thumbPeak}
                              </span>
                              <span className="text-[#334455]">
                                {" "}
                                / 闸门 {THUMB_PEAK_GATE}
                              </span>
                              {/* 改判了要说清是**哪个判据**改的，否则调阈值时不知道该调哪一个 */}
                              {currentPrediction.gated && (
                                <span className="ml-1 text-[#f59e0b]">
                                  已改判（
                                  {currentPrediction.gateReason.startsWith("thumb")
                                    ? "拇指"
                                    : "转角"}
                                  ）
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </>

                    ) : isTranslating ? (
                      /* 这两种"空着"要分开说：闸门拦着 = 一切正常、在等你起手；
                         没拦着还空着 = 窗口在攒 或 模型没给出稳定结论。
                         合成一句"等待手势输入"的话，用户没法判断该不该去查手套 */
                      <div className="space-y-2">
                        <div className="text-4xl text-[#334455] animate-pulse">
                          {gated ? "—" : "..."}
                        </div>
                        <p className="text-[10px] text-[#556677] font-mono">
                          {gated
                            ? "静止中 · 等你起手（手没动时不出词）"
                            : "等待手势输入"}
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <MessageSquare className="w-12 h-12 mx-auto text-[#334455]" />
                        <p className="text-sm text-[#556677]">
                          点击"开始翻译"进入实时识别模式
                        </p>
                      </div>
                    )}
                  </div>

                  {/* 控制按钮 */}
                  <div className="flex items-center gap-3">
                    <button
                      onClick={toggleTranslation}
                      className={`cyber-btn px-6 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
                        isTranslating ? "cyber-btn-accent" : ""
                      }`}
                    >
                      {isTranslating ? (
                        <>
                          <div className="w-2 h-2 rounded-full bg-[#ff2d7b] animate-pulse" />
                          停止翻译
                        </>
                      ) : (
                        <>
                          <Volume2 className="w-4 h-4" />
                          开始翻译
                        </>
                      )}
                    </button>
                  </div>

                  {/* 翻译历史文本 */}
                  {history.length > 0 && (
                    <div className="w-full max-w-2xl">
                      <div className="cyber-panel p-4 rounded-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
                            Translation Output
                          </span>
                          <button
                            onClick={clearHistory}
                            className="text-[#556677] hover:text-[#ff2d7b] transition-colors"
                            title="清除"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                        <p
                          className="text-lg leading-relaxed"
                          style={{
                            fontFamily: "'Space Grotesk', sans-serif",
                            color: "#ccd6e0",
                          }}
                        >
                          {translatedText}
                        </p>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* 换手归一化提示 —— 静默做这件事很危险：识别结果对不上时，
                  用户没法分辨是"手语做错了"还是"软件把左手当右手在算" */}
              {mirrored && (
                <div className="text-[10px] font-mono text-[#a855f7]">
                  ⇄ 已按镜像归一化到右手口径推理（
                  {dominance ? describeDominance(dominance) : "主手＝左手"}
                  ，换手不改词义）
                </div>
              )}

              {/* 判定还在滞回确认期间就说明主手可能要换了。不提示的话，这几百毫秒里
                  喂给模型的是旧口径，用户只会看到"刚换手那一下识别不准" */}
              {isTranslating && dominance?.reason === "pending_flip" && (
                <div className="text-[10px] font-mono text-[#f59e0b]">
                  ⚠ 检测到主手可能换了，正在确认（切换要连续几个窗口一致，
                  避免一个词做到一半翻转口径）
                </div>
              )}

              {/* 消息 */}
              {message && (
                <div className="text-[10px] font-mono text-[#00e5a0]">
                  {message}
                </div>
              )}
            </div>
          </div>

          {/* 手模区：一手一格，与识别链路完全无关，只反映手套原始数据。
              高度按视口比例给、并封顶：写死像素值时矮屏幕上会把上面的识别结果挤没。

              比例比舞台版**放大了**（32vh/300px → 46vh/460px）：手模的框取景是
              竖直 FOV 34° @ 距离 17，半高 5.20 而手从腕到指尖 5.51 —— 手是**撑满
              竖直方向**的，所以这一块有多高就直接决定手有多大，横向加宽不起作用。
              下限也一起抬（170 → 240），否则矮屏上两格并排会各自缩成一小块。 */}
          <div
            className={`shrink-0 border-t border-[#00f0ff]/15 px-3 pt-1.5 pb-2 flex flex-col ${
              showHands ? "h-[46vh] min-h-[240px] max-h-[460px]" : ""
            }`}
          >
            <div className="flex items-center justify-between shrink-0 pb-1">
              <span className="text-[9px] font-mono tracking-widest text-[#556677]">
                LIVE HAND · BEND + IMU（不参与识别）
              </span>
              <button
                onClick={() => setShowHands((v) => !v)}
                className="text-[9px] font-mono text-[#556677] hover:text-[#00f0ff] flex items-center gap-1 transition-colors"
                title={
                  showHands
                    ? "隐藏手模（3D 画面和推理抢同一块 GPU，机器吃力时可以关掉）"
                    : "显示手模"
                }
              >
                {showHands ? (
                  <>
                    <EyeOff className="w-3 h-3" />
                    隐藏
                  </>
                ) : (
                  <>
                    <Eye className="w-3 h-3" />
                    显示手模
                  </>
                )}
              </button>
            </div>
            {/* 关掉时是真卸载 Canvas，不是 hidden —— 隐藏的 WebGL 画面照样在渲染 */}
            {showHands && (
              /* 左手在左、右手在右 —— 第一人称（照镜子）。和第 1 步自检的四格布局
                 同一个顺序，两页对照时不用在脑子里翻一次。
                 各占一半宽：单手视口的取景是竖直方向撑满的（见上面那段注释），
                 横向多出来的空间本来就是白送的。 */
              <div className="flex-1 min-h-0 flex gap-2">
                <HandViewport
                  label="LH · 左手"
                  side="left"
                  channel={gloveLeft}
                  driveRef={leftHand.driveRef}
                  bendCalibrated={leftHand.bendCalibrated}
                  orientCalibrated={leftHand.orientCalibrated}
                />
                <HandViewport
                  label="RH · 右手"
                  side="right"
                  channel={gloveRight}
                  driveRef={rightHand.driveRef}
                  bendCalibrated={rightHand.bendCalibrated}
                  orientCalibrated={rightHand.orientCalibrated}
                />
              </div>
            )}
          </div>
        </div>

        {/* 右侧面板 */}
        <div className="w-60 border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-4 shrink-0">
          {/* 推理模式切换 */}
          <Section title="MODE">
            {/* 三档配色与下面的导航分组、以及 SentencePanel 里的橙色一致：
                青＝静态单帧、紫＝时序滑窗、橙＝连续句子。类名必须写字面量，
                Tailwind 是编译期扫源码，拼出来的类名不会被生成 */}
            <div className="grid grid-cols-3 gap-1">
              <button
                onClick={() => switchMode("static")}
                className={`px-1.5 py-1.5 rounded-sm text-[10px] font-mono border transition-colors ${
                  modelMode === "static"
                    ? "border-[#00f0ff]/60 text-[#00f0ff] bg-[#00f0ff]/10"
                    : "border-[#00f0ff]/15 text-[#556677]"
                }`}
              >
                静态单帧
              </button>
              <button
                onClick={() => switchMode("sequence")}
                className={`px-1.5 py-1.5 rounded-sm text-[10px] font-mono border transition-colors ${
                  modelMode === "sequence"
                    ? "border-[#a855f7]/60 text-[#a855f7] bg-[#a855f7]/10"
                    : "border-[#00f0ff]/15 text-[#556677]"
                }`}
              >
                时序滑窗
              </button>
              <button
                onClick={() => switchMode("sentence")}
                className={`px-1.5 py-1.5 rounded-sm text-[10px] font-mono border transition-colors ${
                  modelMode === "sentence"
                    ? "border-[#f59e0b]/60 text-[#f59e0b] bg-[#f59e0b]/10"
                    : "border-[#00f0ff]/15 text-[#556677]"
                }`}
              >
                连续句子
              </button>
            </div>
            <div className="text-[9px] font-mono text-[#556677] leading-relaxed">
              {modelMode === "static" ? (
                <>每 100ms 推理单帧。看不到运动轨迹，「再见」「来」这类动态词识别不了。</>
              ) : modelMode === "sequence" ? (
                <>
                  每 100ms 取最近 {WINDOW_MS}ms 窗口推理，能识别动态词。
                  {!seqReady && (
                    <span className="text-[#ff2d7b]">
                      {" "}
                      当前没有时序模型，先去 /train-seq 训练。
                    </span>
                  )}
                </>
              ) : (
                <>
                  整句连着打完再一次性解码（CTC），中间不用停。
                  <span className="text-[#f59e0b]">
                    {" "}
                    这一档不做滑窗，也没有「同词间隔 2 秒」的去重 —— 重复词打得出来。
                  </span>
                </>
              )}
            </div>
          </Section>

          {/* 主手 —— 归一化的唯一输入，**从数据自动判**，没有手选项。
              两条能量条是判定的依据本身：判错时要能一眼看出是哪只手能量高，
              否则用户只能对着一个错词猜 */}
          <Section title="DOMINANT HAND">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[#556677]">自动判定</span>
              <span
                className={
                  dominance?.dominant === "left"
                    ? "text-[#a855f7]"
                    : dominance?.dominant === "both"
                    ? "text-[#f59e0b]"
                    : "text-[#00e5a0]"
                }
              >
                {dominance
                  ? dominance.dominant === "left"
                    ? "左手（已镜像）"
                    : dominance.dominant === "both"
                    ? "双手词"
                    : "右手"
                  : "待翻译时判定"}
              </span>
            </div>
            <EnergyBar label="LH" energy={dominance?.left?.total ?? null} />
            <EnergyBar label="RH" energy={dominance?.right?.total ?? null} />
            <div className="text-[9px] font-mono text-[#556677] leading-relaxed">
              {dominance ? (
                describeDominance(dominance)
              ) : (
                <>按两只手在最近 1 秒里的活动量判：动得明显多的那只是主手。</>
              )}
            </div>
            {/* 没标定就只能用兜底量程，左右量程差异（实测 40~176）没被抵消 */}
            {dominance && !dominance.calibrated && bothConnected && (
              <div className="text-[9px] font-mono text-[#f59e0b] leading-relaxed">
                有手未做弯折两点标定，活动量用的是兜底量程。回第 1 步补标定能让主手判得更稳。
              </div>
            )}
          </Section>

          {/* Top-K 候选 */}
          {currentPrediction && isTranslating && (
            <Section title="CANDIDATES">
              <div className="space-y-1">
                {currentPrediction.allProbabilities.map((p, i) => {
                  const word = getTranslationLabel(p.label);
                  return (
                    <div
                      key={p.label}
                      className="flex items-center justify-between text-[10px] font-mono"
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-[#556677] w-3">{i + 1}.</span>
                        <span
                          style={{
                            color:
                              p.probability >= confidenceThreshold
                                ? "#ccd6e0"
                                : "#556677",
                          }}
                        >
                          {word}
                        </span>
                      </div>
                      <span
                        style={{
                          color:
                            p.probability >= confidenceThreshold
                              ? "#00e5a0"
                              : p.probability >= 0.3
                              ? "#f59e0b"
                              : "#334455",
                        }}
                      >
                        {(p.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {/* 参数设置 */}
          <Section title="SETTINGS">
            <div className="space-y-2">
              <div className="space-y-1">
                <div className="flex justify-between text-[9px] font-mono text-[#556677]">
                  <span>置信度阈值</span>
                  <span className="text-[#00f0ff]">
                    {(confidenceThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="0.95"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) =>
                    setConfidenceThreshold(parseFloat(e.target.value))
                  }
                  className="w-full h-1 bg-[#1a2030] rounded-full appearance-none cursor-pointer"
                  style={{ accentColor: "#00f0ff" }}
                />
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-[9px] font-mono text-[#556677]">
                  <span>平滑窗口</span>
                  <span className="text-[#00f0ff]">{smoothingWindow} 帧</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="15"
                  step="1"
                  value={smoothingWindow}
                  onChange={(e) =>
                    setSmoothingWindow(parseInt(e.target.value))
                  }
                  className="w-full h-1 bg-[#1a2030] rounded-full appearance-none cursor-pointer"
                  style={{ accentColor: "#00f0ff" }}
                />
              </div>
            </div>
          </Section>

          {/* 模型信息 */}
          {modelReady && (
            <Section title="MODEL INFO">
              <div className="text-[9px] font-mono text-[#556677] space-y-0.5">
                <p>
                  Type:{" "}
                  <span
                    className={
                      modelMode === "static"
                        ? "text-[#00f0ff]"
                        : "text-[#a855f7]"
                    }
                  >
                    {modelMode === "static" ? "MLP (静态单帧)" : "TCN (时序滑窗)"}
                  </span>
                </p>
                {/* 模型名是唯一能确认"用的不是另一条链路的模型"的东西：
                    seq_ 前缀 = 时序，没有 = 静态 */}
                <p className="truncate">
                  Name:{" "}
                  <span className="text-[#8899aa]">
                    {(modelMode === "static"
                      ? loadedNames.static
                      : loadedNames.sequence) ?? "—"}
                  </span>
                </p>
                <p>
                  Classes:{" "}
                  <span className="text-[#00f0ff]">{activeLabels.length}</span>
                </p>
                <p>
                  Vocab:{" "}
                  <span className="text-[#8899aa]">
                    {activeLabels
                      .map(getDisplayLabel)
                      .slice(0, 8)
                      .join(", ")}
                    {activeLabels.length > 8 ? "..." : ""}
                  </span>
                </p>
              </div>
            </Section>
          )}

          {/* 翻译历史 */}
          {history.length > 0 && (
            <Section title="HISTORY">
              <div className="space-y-0.5 max-h-40 overflow-y-auto">
                {history
                  .slice(-20)
                  .reverse()
                  .map((entry, i) => (
                    <div
                      key={i}
                      className="flex justify-between text-[9px] font-mono"
                    >
                      <span className="text-[#8899aa]">{entry.word}</span>
                      <span className="text-[#556677]">
                        {(entry.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          {/* 导航 —— 按静态/时序**分成两组**。
              这四个链接原来叫「采集数据 / 训练模型 / 序列采集 / 时序训练」，
              光看名字分不出哪两个属于静态单帧、哪两个属于时序滑窗；
              走错一条就是在给另一条链路喂数据或训练另一个模型，
              而两个模型各有各的数据集和 localStorage 键，出了错要很久才发现。
              这里用的词和配色与上面的 MODE 开关完全一致（青＝静态、紫＝时序），
              当前模式那一组高亮，另一组压暗。 */}
          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-2">
            <NavGroup tone="static" active={modelMode === "static"} />
            <NavGroup tone="sequence" active={modelMode === "sequence"} />
          </div>

          {/* 手套状态（只读；连接/断开都在第 1 步） */}
          <div className="flex items-center justify-center gap-2 text-[10px] font-mono pt-2">
            <span
              className={
                gloveLeft.isConnected ? "text-[#00e5a0]" : "text-[#556677]"
              }
            >
              左手{" "}
              {gloveLeft.isConnected ? `✓${gloveLeft.gloveFps}Hz` : "○未连接"}
            </span>
            <span className="text-[#334455]">|</span>
            <span
              className={
                gloveRight.isConnected ? "text-[#00e5a0]" : "text-[#556677]"
              }
            >
              右手{" "}
              {gloveRight.isConnected ? `✓${gloveRight.gloveFps}Hz` : "○未连接"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ===== 辅助组件 =====

/**
 * 句子档没模型时的引导。
 *
 * 这一档最容易出的误会是**去 /train-seq 训一个**——那条路训的是逐词滑窗模型、
 * 存进 IndexedDB，这一档根本不会去读它。所以这里必须把真正的四步写出来，
 * 并且明说"不是 /train-seq"。
 *
 * 三种状态分开说：还在探测 / 确认没部署（404）/ 探测或加载本身出错。
 * 合并成一句"不可用"会让人去重训一个其实已经训好的模型。
 */
function SentenceModelMissing({
  available,
  error,
}: {
  available: boolean | null;
  error: string | null;
}) {
  return (
    <div className="text-center space-y-3 max-w-lg">
      <Brain className="w-16 h-16 mx-auto text-[#334455]" />
      {error ? (
        <>
          <p className="text-sm text-[#ff2d7b]">句子模型加载失败</p>
          {/* 原文照抄出来：blankIndex 不符、frameDim 不符这类错误是"模型要重训"的
              明确信号，藏起来就只剩"不可用" */}
          <p className="text-[10px] font-mono text-[#ff2d7b] leading-relaxed break-all">
            {error}
          </p>
        </>
      ) : available === null ? (
        <p className="text-sm text-[#556677]">正在检查有没有句子模型…</p>
      ) : (
        <>
          <p className="text-sm text-[#556677]">还没有句子模型</p>
          <div className="text-[10px] font-mono text-[#556677] leading-relaxed text-left inline-block space-y-1">
            {/* 这条警告要留着：两个模型长得像但不通用，指错一次就是白训一轮 */}
            <p className="text-[#f59e0b]">
              注意：这一档要的<b>不是</b> /train-seq 训的那个模型（那是逐词滑窗、存在
              IndexedDB 里）。句子模型在本机 Python 里训，是一份静态文件。
            </p>
            <p className="text-[#334455]">
              缺的文件是 {SENTENCE_MODEL_DIR}/weights.json
            </p>
          </div>
          {/*
            原来这里列的是四条手敲命令（导数据集 → synth_sentences → train_seq --ctc
            → export_weights）。/train-sentence 现在把这一整圈做成了按钮，
            照抄命令行会把人引去做已经不必要的事。
            命令行仍然可用，所以下面那句保留 —— 训练桥只在 npm run dev 存在
          */}
          <Link
            href="/train-sentence"
            className="cyber-btn px-4 py-2 rounded-sm text-xs inline-flex items-center gap-2"
            style={{ borderColor: "rgba(168,85,247,0.5)", color: "#a855f7" }}
          >
            前往句子训练 →
          </Link>
          <p className="text-[9px] font-mono text-[#334455] leading-relaxed">
            那页负责送数据、开训、看曲线、把权重导到这里。
            它要本机的 python_train/.venv，且只在 npm run dev 下可用；
            部署环境里仍然只能靠命令行跑 train_seq.py --ctc + export_weights.py。
          </p>
        </>
      )}
    </div>
  );
}

/**
 * 一只手的手模视口：标题行（FPS + 标定状态）+ 一个 `HandModel` Canvas。
 *
 * 与第 1 步 /mocap 自检里的那格**刻意长得一样** —— 同一个 `HandModel`、
 * 同一个相机、同一份标定。两页看到的手必须是同一只手，否则用户没法拿自检那页
 * 当基准来判断"这一页的手模是不是不对"。
 *
 * 标定是**只读**的：连接与标定的唯一入口在第 1 步，这里改标定只会让两页说法不
 * 一致。所以未标定时不给按钮，只给提示 —— 未标定的手模最多弯到 0.42（柔和预览），
 * 拿它判断手型会得出错误结论，必须标死在画面里。
 */
function HandViewport({
  label,
  side,
  channel,
  driveRef,
  bendCalibrated,
  orientCalibrated,
}: {
  label: string;
  side: "left" | "right";
  channel: HandChannel;
  driveRef: RefObject<HandDrive>;
  bendCalibrated: boolean;
  orientCalibrated: boolean;
}) {
  const connected = channel.isConnected;
  return (
    <div className="flex-1 min-w-0 min-h-0 flex flex-col gap-1">
      <div className="shrink-0 flex items-center gap-1.5 px-0.5 text-[9px] font-mono">
        <span className="tracking-widest text-[#f59e0b]">{label}</span>
        <span className={connected ? "text-[#00e5a0]" : "text-[#556677]"}>
          {connected ? `${channel.gloveFps.toFixed(0)} FPS` : "未连接"}
        </span>
        {connected && bendCalibrated && !orientCalibrated && (
          <span className="text-[#556677]">朝向未标定</span>
        )}
      </div>
      <div
        className="relative flex-1 min-h-0 rounded-sm border overflow-hidden bg-[#070a13]"
        style={{
          borderColor: connected
            ? "rgba(0,240,255,0.15)"
            : "rgba(85,102,119,0.2)",
        }}
      >
        <HandModel driveRef={driveRef} side={side} />
        {!connected && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#070a13]/70 pointer-events-none">
            <span className="px-2 py-1 rounded-sm bg-[#0a0e1a]/90 border border-[#00f0ff]/15 text-[10px] font-mono text-[#8899aa]">
              这只手套没连接，去第 1 步
            </span>
          </div>
        )}
        {/* 未标定这条必须压在画面里，不能只写在标题行：这个状态下握拳只弯到约
            42%，看起来就是"手模坏了"或"模型不准"，而其实只是没跑标定 */}
        {connected && !bendCalibrated && (
          <div className="absolute bottom-1.5 left-1.5 right-1.5 px-2 py-1 rounded-sm pointer-events-none bg-[#f59e0b]/12 border border-[#f59e0b]/35">
            <span className="text-[9px] font-mono text-[#f59e0b] leading-relaxed">
              未标定弯折 · 满量程也只弯约 42%，握拳不会成形。去第 1 步跑向导
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/*
 * 两条链路各自的「采集 → 训练」入口。
 *
 * 配色与 MODE 开关同源：青 `#00f0ff` = 静态单帧（MLP），紫 `#a855f7` = 时序滑窗（TCN）。
 * 类名写成两份**字面量**而不是拼 `border-[${color}]/50`——Tailwind 是编译期扫源码，
 * 拼出来的类名不会被生成，运行时就是没有边框。
 */
const NAV_TONES = {
  static: {
    group: "静态单帧 · MLP",
    title: "text-[#00f0ff]",
    on: "border-[#00f0ff]/50 text-[#00f0ff] hover:bg-[#00f0ff]/10",
    off: "border-[#00f0ff]/15 text-[#556677] hover:text-[#00f0ff]",
    links: [
      { href: "/collect", label: "静态采集" },
      { href: "/train", label: "静态训练" },
    ],
  },
  sequence: {
    group: "时序滑窗 · TCN",
    title: "text-[#a855f7]",
    on: "border-[#a855f7]/50 text-[#a855f7] hover:bg-[#a855f7]/10",
    off: "border-[#a855f7]/15 text-[#556677] hover:text-[#a855f7]",
    links: [
      { href: "/collect-seq", label: "时序采集" },
      { href: "/train-seq", label: "时序训练" },
    ],
  },
} as const;

function NavGroup({
  tone,
  active,
}: {
  tone: keyof typeof NAV_TONES;
  active: boolean;
}) {
  const t = NAV_TONES[tone];
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-[9px] font-mono leading-none">
        <span className={active ? t.title : "text-[#556677]"}>{t.group}</span>
        {active && <span className="text-[#556677]">· 当前模式</span>}
      </div>
      <div className="grid grid-cols-2 gap-1">
        {t.links.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={`px-2 py-1.5 rounded-sm border font-mono text-[10px] text-center transition-colors ${
              active ? t.on : t.off
            }`}
          >
            ← {l.label}
          </Link>
        ))}
      </div>
    </div>
  );
}

/**
 * 一只手的运动能量条。能量单位是"份"（1.0 ≈ 明显是个有意动作，见 dominantHand.ts），
 * 条宽按 2 份满格 —— 上限不是硬边界，做大幅动作时会顶格，那不影响判定（判定看比值）。
 *
 * `null` = 这只手这一窗没有数据，显示成"—"而不是空条：空条会被读成"连着但不动"。
 */
function EnergyBar({ label, energy }: { label: string; energy: number | null }) {
  const pct = energy === null ? 0 : Math.min(100, (energy / 2) * 100);
  return (
    <div className="flex items-center gap-1.5 text-[9px] font-mono">
      <span className="text-[#556677] w-4 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-[#1a2030] rounded-full overflow-hidden border border-[#00f0ff]/15">
        <div
          className="h-full rounded-full transition-all duration-200"
          style={{
            width: `${pct}%`,
            backgroundColor: energy !== null && energy >= 1 ? "#00e5a0" : "#556677",
          }}
        />
      </div>
      <span className="text-[#556677] w-8 shrink-0 text-right">
        {energy === null ? "—" : energy.toFixed(2)}
      </span>
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
      <div className="flex items-center gap-2 pb-1 border-b border-[#00f0ff]/15">
        <div className="w-1 h-3 bg-[#00f0ff] rounded-full shadow-[0_0_4px_rgba(0,240,255,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[#00f0ff] font-mono">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}
