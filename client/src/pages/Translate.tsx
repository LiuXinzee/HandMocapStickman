/*
 * Translate — 实时手语翻译页面
 *
 * DESIGN: **浅色**（白卡片 + 浅蓝灰底 + 蓝色强调）。这是 index.css 里 `:root` 的
 * 默认皮肤、全站共用，这一页不需要自己贴类。深色那套原样留在 `.theme-dark` 里。
 * 这一页里凡是颜色都走 `var(--hud-*)`，**别再往回写死十六进制**，写死的那一刻
 * 这一页就又只有一套配色了。
 *
 * 版式：**两个大圆角框**。左＝双手 3D 画面（一张卡，里面并排两个视口），
 * 右＝译文。句子档下右边那张卡整个是聊天窗口（见 SentencePanel）。
 *
 * 功能:
 * 1. 连接手套后实时推理
 * 2. 显示识别结果（大字体 + 置信度）
 * 3. 历史翻译记录（句子拼接）
 *
 * ⚠ 右侧面板（MODE 开关 / 主手能量条 / Top-K 候选 / 阈值滑杆 / MODEL INFO /
 * 逐词历史 / 采集训练导航 / 手套 Hz）**已经撤掉**，这一页现在只剩主区。
 * 后果只有一个要记住的：**MODE 不能在界面上切了**，唯一来源是 URL，默认句子档
 * （见下面 `urlMode`）。置信度阈值和平滑窗口跟着变成固定值（0.7 / 5 帧），
 * 那两个状态还在，只是没有滑杆能调。
 *
 * 三种推理模式（现在靠 `?mode=` 选）：
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
 * 左框是**一手一个视口**（`HandModel` × 2），与第 1 步 /mocap 的自检**同一个组件、
 * 同一份标定、同一套驱动**（见 useHandModelDrive）。默认档也和自检页一致（实心手模），
 * 所以手模看着不对劲时，两页应该长得一模一样 —— 不一样才说明是这一页的问题。
 * 顶栏另有一档骨架：藏掉蒙皮只留关节球，用来看清指节到底停在哪。
 *
 * ⚠ **朝向对不上不是这一页能修的**。这两格只是显示 `applyOrientationCalib` 的结果；
 * 朝向标定（零位 + 轴向矩阵）只在第 1 步 /mocap 做。没标定时 `applyOrientationCalib`
 * 原样返回原始 IMU 四元数，指向完全取决于 IMU 怎么装在手套上，和实手差多少都可能。
 * 所以标题行有那个「朝向未标定」，别把它当成可以忽略的提示。
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
  loadModelFromSaved,
} from "@/lib/signLanguageModel";
import {
  predictSequence,
  isSequenceModelLoaded,
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
import {
  buildDemoTimeline,
  gestureAt,
  gestureWindows,
  DEMO_SCRIPT,
} from "@/lib/translateDemo";
import {
  clipDurations,
  demoClipQualityNote,
  describeDemoClips,
  frameAt,
  loadDemoClips,
  poseAt,
  type DemoClipSet,
} from "@/lib/demoPlayback";
// 演示期的临时东西，连同 demoSubtitles.ts 一起删（那个文件头说了怎么关）
import { demoSubtitleAt } from "@/lib/demoSubtitles";
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
  ChevronRight,
  ChevronDown,
  Play,
  Square,
} from "lucide-react";

interface TranslationEntry {
  word: string;
  label: string;
  confidence: number;
  timestamp: number;
}

/**
 * 已成句的一条。**句子档专用**，和上面那个逐词档的 `TranslationEntry` 是两回事
 * （一个是"一个词"，一个是"一整句"），别合并。
 *
 * 为什么不只存 `text`：历史气泡点开要能看**模型的原始词序**。只留顺句结果的话，
 * 事后翻看一句翻错的话，分不清是模型认错了词、还是顺句规则顺错了 ——
 * 而这两件事一个要补数据重训、一个只要改一行规则表（同 SentencePanel 文件头那段）。
 * `rule` 一并记下来：没记的话，事后重算规则可能命中的已经是另一条了。
 */
export interface SentenceEntry {
  /** 顺句结果，气泡里的大字 */
  text: string;
  /** 模型解出来的原始词序（原始类别 id，含合并类），展开时逐词显示 */
  words: string[];
  /** 命中的顺句规则名；null = 没命中/规则已关 */
  rule: string | null;
  at: number;
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
   * MODE 的**唯一**来源是 URL，默认落在连续句子档。
   *
   * 右侧面板（含三档 MODE 开关）已经撤掉，页面上不再有切档的入口，所以初值不能
   * 再是静态 —— 那样进来就永久停在一个没有开始按钮的档上（逐词档的「开始翻译」
   * 按钮在主区，但这一页的用途已经是连续句子）。
   *
   * `?mode=static` / `?mode=sequence` 仍然认，作为逐词两档的后门：那两条推理链路
   * 和它们的主区 UI 一行没动，只是没有按钮能走过去了。
   *
   * 同时**锁住**下面自动加载 effect 里的自动选档 —— 那段逻辑只认识静态/时序两条
   * （句子模型不在 IndexedDB 里，`getLatestModel` 系列查不到它），留着会让页面
   * 先闪一下句子档再跳去时序。
   */
  const urlMode = useRef<ModelMode | null>(null);
  if (urlMode.current === null) urlMode.current = modeFromUrl();
  const [modelMode, setModelMode] = useState<ModelMode>(
    urlMode.current ?? "sentence"
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
  const gloveError = gloveLeft.error || gloveRight.error;

  /*
   * 底部两个手模视口的驱动。
   *
   * hook 留在页面这一层调（而不是塞进 `HandViewport` 里），是因为上方那一行
   * 连接/标定汇总也要读 `bendCalibrated` / `orientCalibrated`。它自带 rAF、
   * 写 ref 不触发 React 重渲染，100Hz 的手套数据不会打到这个页面组件上。
   */
  /*
   * 离线演示的回放姿态。非 null 时顶掉手套数据（见 `useHandModelDrive` 的第三个参数）。
   *
   * 由下面 `startDemo` 那个 rAF 循环写，不在演示中时置回 null —— 置 null 而不是
   * 留一个"最后的姿态"，是为了让手模回到"没数据"那一档（停在静止姿态），
   * 否则停止演示之后手会永远举在最后一个词的收势位置上。
   */
  const demoPoseLeftRef = useRef<HandDrive | null>(null);
  const demoPoseRightRef = useRef<HandDrive | null>(null);
  const leftHand = useHandModelDrive(gloveLeft, "LH", demoPoseLeftRef);
  const rightHand = useHandModelDrive(gloveRight, "RH", demoPoseRightRef);

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
  /*
   * 已成句的历史 —— 右列聊天窗口里那些气泡。
   *
   * 它同时是「定版」和 `lastLive` 那套语义的账本：连续模式下每句解出来就往这里
   * 追一条，之后的改代词/删词靠下面那个 effect 同步回最后一条，手动「定版」= 停止同步。
   *
   * ⚠ **最后一条可能就是屏幕上那句 live**（`lastLive === true` 时）。渲染前必须切掉，
   * 否则同一句会在窗口里上下出现两次 —— 见下面 `settledHistory`。
   *
   * 只在内存里，刷新即清空：这一档的用法是"现场对话一次"，没有跨会话回看的需求。
   */
  const [sentenceHistory, setSentenceHistory] = useState<SentenceEntry[]>([]);
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
   * ===== 演示译文覆盖的句子计数器 =====
   *
   * `demoSeq` = 屏幕上这一句是第几句（从 0 起；-1 = 还没收过任何一句）。
   * 一次真解码 = 一句，所以在 `decodeUtterance` 里 +1 —— 「一句一次」和「连续」
   * 两档都走那一条路，挂在那儿就不用分别处理两种模式。
   *
   * ref 和 state 两份不是冗余：`decodeUtterance` 是 `[reportMirrored]` 依赖的
   * useCallback，闭包里读到的 state 永远是第一次渲染那个值，只能读 ref；
   * 而渲染和下面那个同步 effect 要的是会触发重渲染的 state。
   *
   * 覆盖是什么、怎么关，见 `@/lib/demoSubtitles`。演示结束后这几行一起删。
   */
  const demoSeqRef = useRef(-1);
  const [demoSeq, setDemoSeq] = useState(-1);
  /** 当前这句要不要换成假字幕（`null` = 不换）。四句之外会自动退回真译文 */
  const subtitleOverride = demoSubtitleAt(demoSeq);

  /*
   * ===== 离线演示 =====
   *
   * 没手套时也能看译文区的效果。`demoOn` 为真时**绕过 `isConnected` 门禁**
   * 渲染真的 `SentencePanel`，词序由 `translateDemo.ts` 的脚本按时序喂进
   * 上面那几个 state；顺句、历史结构、面板全都是真的（理由见那个文件的头注释）。
   *
   * ⚠ 手套真连上时**不给进演示**（下面按钮会 disabled）。两边都在写
   * `sentenceWords`，同时跑的话屏幕上是真词和演示词交替闪 —— 那比看不到效果更糟。
   *
   * timers 存成数组是为了能一把清掉：嵌套 setTimeout 的取消要跟着句子和词两层
   * 索引走，很容易漏掉最里层那个，表现为点了停止又蹦出一个词。
   *
   * ===== 手模动作 =====
   *
   * 演示不只出字，手模也跟着比划：动作是**库里那几个词的真录制**回放
   * （`demoPlayback.ts`，IMU + 弯折逐帧喂进 `demoPose*Ref`），不是合成的曲线。
   *
   * 由此来的两个连带后果，都是刻意的：
   *
   *  - **字的节奏跟着动作走**。一个词停多久 = 那条录制裁剪后的实际长度，不再是
   *    固定的 550ms。整段因此从约 12s 变成 30s 上下 —— 那才是真打手语的速度，
   *    把 2 秒的手势压进 550ms 会播成抽搐，而且那就不是真数据了。
   *  - **库里没有的词手模不动，字照出**。`missing` 会如实报出来，不拿另一个词的
   *    动作凑数：演示里播错的手势比不播更糟。
   *
   * ⚠ 库里 08-13 那批录制与词表描述不符（`signLanguageVocab.ts` 文件头）。脚本里
   * `hello`/`thank_you` 就在这批里，也就是说演示会一本正经地播两个已知/疑似打错的
   * 手势。这件事**走 console.warn，不上屏**（见下面取库那段）；上屏的 `demoClipNote`
   * 只留"看起来像故障其实不是"的那几条。要真修只能重录。
   */
  const [demoOn, setDemoOn] = useState(false);
  const demoTimersRef = useRef<number[]>([]);
  /** 手势回放的 rAF 句柄 + 起点时刻；`clipsRef` 是本轮取到的片段 */
  const demoRafRef = useRef(0);
  const demoClipsRef = useRef<DemoClipSet | null>(null);
  /** 每点一次「演示」自增：异步取库回来时用它判断这一轮是不是已经被取消了 */
  const demoRunRef = useRef(0);
  /** 手模动作那一行提示（缺录制 / 已知打错），与译文区的 `sentenceNote` 分开 */
  const [demoClipNote, setDemoClipNote] = useState<string | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);

  const stopDemo = useCallback(() => {
    demoRunRef.current++;
    for (const id of demoTimersRef.current) clearTimeout(id);
    demoTimersRef.current = [];
    cancelAnimationFrame(demoRafRef.current);
    demoRafRef.current = 0;
    demoClipsRef.current = null;
    // 置 null 让手模回到"没数据"那一档，而不是举在最后一个词的收势位置上
    demoPoseLeftRef.current = null;
    demoPoseRightRef.current = null;
    setDemoOn(false);
    setDemoLoading(false);
    setDemoClipNote(null);
    setSentenceWords(null);
    setSentenceHistory([]);
    setPronOverrides({});
    setLastLive(false);
    setSentenceNote(null);
  }, []);

  const startDemo = useCallback(async () => {
    // 重播 = 先清干净再来一遍，不然第二次会接在上一轮的 5 句后面。
    // stopDemo 自带 `demoRunRef++`，所以连点两下时第一轮的异步回调会自己作废
    stopDemo();
    const run = demoRunRef.current;
    setDemoOn(true);
    setDemoLoading(true);

    /*
     * 先把动作片段取回来，**再**排时间轴 —— 顺序不能反：每个词停多久由它那条
     * 录制的长度决定，时间轴排完了才知道时长就等于排错了。
     *
     * 取库失败不让演示整个失败：`loadDemoClips` 自己吞掉异常、把词记进 missing，
     * 于是最坏情况退回"纯文字演示 + 手模不动"，也就是加手模之前的样子。
     */
    const clips = await loadDemoClips(DEMO_SCRIPT.flatMap((s) => s.words));
    if (demoRunRef.current !== run) return; // 取库期间被停掉了
    demoClipsRef.current = clips;
    setDemoLoading(false);
    setDemoClipNote(describeDemoClips(clips));
    /*
     * 录制质量存疑的那几条只进 console，**不上屏** —— 演示是给人看的，
     * 观众读不懂"与词表描述不符"，只看见产品自己挂了个橙色警告。
     * 信息没丢，受众换了：这条本来就是给做演示的人自己看的。见 demoPlayback 文件头。
     */
    const quality = demoClipQualityNote(clips);
    if (quality) console.warn("[demo] 录制存疑：" + quality);

    // 时间轴只排一次，字和手都从它派生 —— 排两次就等于两张表，必然漂
    const timeline = buildDemoTimeline(DEMO_SCRIPT, clipDurations(clips));
    const windows = gestureWindows(timeline);
    const t0 = performance.now();

    /*
     * 手势回放循环。走 rAF 而不是给每一帧排一个 timer：一条 2 秒的录制在 50Hz
     * 下是 100 帧，14 个词就是 1400 个 timer，而 rAF 天然与屏幕刷新对齐、
     * 掉帧时自己按 `elapsed` 跳到该在的位置（timer 会整体滞后并越积越多）。
     */
    const tick = () => {
      if (demoRunRef.current !== run) return;
      const hit = gestureAt(windows, performance.now() - t0);
      const clip = hit ? demoClipsRef.current?.clips.get(hit.gesture.wordId) : null;
      if (hit && clip) {
        const frame = frameAt(clip, hit.offsetMs);
        for (const side of ["left", "right"] as const) {
          const pose = poseAt(clip, side, frame);
          const ref = side === "left" ? demoPoseLeftRef : demoPoseRightRef;
          /*
           * 单手词的另一只手：`poseAt` 返回 null，这里就**保持上一个词留下的姿态**
           * 不动，而不是置 null。置 null 会让那只手在"有数据/没数据"之间反复切，
           * 表现为一只手每隔一个词就弹一下。它不动才是对的 —— 单手词里那只手
           * 本来就垂着没参与。
           */
          if (pose) ref.current = { ...pose, hasData: true };
        }
      }
      demoRafRef.current = requestAnimationFrame(tick);
    };
    demoRafRef.current = requestAnimationFrame(tick);

    for (const ev of timeline) {
      const id = window.setTimeout(() => {
        if (ev.kind === "word") {
          if (ev.first) {
            /*
             * 新的一句开门。这两下的顺序和真机 `decodeUtterance` 一样，但**必须
             * 显式写在这里**，因为演示是逐词喂的、真机是整句一次到位：
             *
             *  - `setLastLive(false)`：上一句就此定版。不放掉的话，下面那个
             *    "编辑同步回历史最后一条"的 effect 会拿**这一句才打了一个词**的
             *    译文去改写上一句的历史记录 —— 屏幕上"你好！"会变成"你"。
             *    真机上不会撞见这个，是因为那边换词和推历史在同一批 setState 里，
             *    effect 跑的时候两边已经是同一句了。
             *  - `setPronOverrides({})`：它按词下标存（`Record<number, string>`）。
             *    看效果的人多半会点一下上一句的「代词」，不清的话那个 `{0: …}`
             *    会落到这一句的第 0 个词上，而 5 句里代词位置各不相同。
             */
            setLastLive(false);
            setPronOverrides({});
          }
          setSentenceWords(ev.words);
        } else if (ev.kind === "settle") {
          /*
           * 收句：推进历史，`lastLive = true`，**live 保持原样不清**。
           *
           * 三点都是照着真机 `decodeUtterance` 抄的，尤其"不清 live"——
           * 清了的话句子刚译完就自己缩小成历史那一档，而真机是大字继续留着
           * 当前句、等下一句的第一个词进来才退档。面板那边靠 `lastLive` 把历史
           * 最后一条切掉（`settledHistory`），所以不会画两遍。
           *
           * 顺句走真的 `resolveSentence`（和 `commitSentence` 同一个调用），
           * 历史条目里的 `rule` 是真命中的规则名 —— 展开历史气泡看到的
           * 「原始词序 + 规则名」跟真机一模一样。
           *
           * 代词覆盖传 `{}`：演示不预设"用户改过代词"，屏幕上显示的是
           * **模型按位置猜的默认值**，正是要展示的那一层。
           */
          const solved = resolveSentence(ev.words, {}, grammarOnRef.current);
          setSentenceHistory((prev) => [
            ...prev,
            { text: solved.text, words: ev.words, rule: solved.rule, at: Date.now() },
          ]);
          setLastLive(true);
        } else {
          // 播完停住，不循环：自动循环会让人分不清"卡住了"和"又播了一遍"
          setSentenceNote("演示播完了。点「演示」重播，或连上手套开始真的翻译。");
          /*
           * 手模停在最后一个词的收势姿态上（不置 null）。真人打完最后一句也是
           * 手停在那里，而不是弹回原位；而且这时候屏幕上写着"播完了"，
           * 一只静止的手不会被误读成卡住。回放循环留着不停也无所谓 ——
           * `gestureAt` 在末尾之后一直返回最后那一帧。
           */
        }
      }, ev.at);
      demoTimersRef.current.push(id);
    }
  }, [stopDemo]);
  // 卸载时清 timer 和 rAF：不清的话切走页面之后回调还在往已卸载的组件里 setState
  useEffect(() => () => {
    demoRunRef.current++;
    for (const id of demoTimersRef.current) clearTimeout(id);
    cancelAnimationFrame(demoRafRef.current);
  }, []);

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
  /*
   * 置信度阈值与平滑窗口。**右栏的两个滑杆撤掉之后这两个值不可调了**，
   * 就是下面这两个初值。留成 state（而不是写成模块级常量）是为了保住下游那一整套
   * ref 同步：推理循环读的是 `confidenceThresholdRef` / `smoothingWindowRef`，
   * 要重新加回滑杆，把 setter 接到界面上就行，循环那一侧一行都不用改。
   */
  const [confidenceThreshold] = useState(0.7);
  const [smoothingWindow] = useState(5);
  /**
   * 只放**出问题**的提示。它会占一整行，所以只有真出错才写这里。
   *
   * ⚠ 启动时"加载成功"的回执**不要**再塞进来 —— 那条改走 `loadedModels`
   * （挂在 header 的 MODEL 徽标上，收起时零高度）。理由见 `loadedModels`。
   */
  const [message, setMessage] = useState("");
  /**
   * 当前生效的两条模型（静态 / 时序），给 header 那个 MODEL 徽标点开看。
   *
   * 这原来是正文里一条 `✓ 已自动加载：静态单帧 "student_2026..." · 时序滑窗
   * "seq_student(部署)" · 27 词` 的提示带，一行 15px，**开着页面就一直占着**，
   * 而它讲的是启动那一瞬间的事。翻译的时候没人看它，出问题时才想查。
   * 所以挪到徽标上：MODEL 这个字本来就是"模型就绪"的指示灯，"就绪的是哪两个"
   * 正是它该回答的问题，点一下展开成浮层（absolute，不占版面）。
   *
   * 顺带从"这次加载了什么"改成"现在生效的是什么"：返回这一页时模型已经在内存里、
   * 不会重新加载，旧写法下徽标就没得可点了 —— 而"我训的那条到底在不在用"
   * 这个问题跟是不是刚加载完没有关系。
   */
  const [loadedModels, setLoadedModels] = useState<string[]>([]);
  /** MODEL 徽标的浮层开没开 */
  const [modelInfoOpen, setModelInfoOpen] = useState(false);
  /** ⇄ 镜像徽标的浮层开没开 */
  const [mirrorInfoOpen, setMirrorInfoOpen] = useState(false);
  /** 底部手模条开关。默认开；两个 WebGL 画面与 tfjs 推理共用 GPU，卡就关掉 */
  const [showHands, setShowHands] = useState(true);
  /**
   * 左框画骨架还是实心手模。**默认实心手模** —— 和 `/mocap` 第 1 步自检、向导、
   * VirtualMocap 全站一致（那几处都不传 mode，`AnimatedHand` 的默认就是 mesh）。
   *
   * 曾经默认过骨架，被退回来了：骨架读的是同一套骨头世界矩阵、姿态和实心档
   * **完全一致**，但没有蒙皮遮挡，原先被手掌那块肉挡住的指节偏差全暴露出来，
   * 观感上就是"手模变不准了"。要判断是真不准还是画法差异，得能和 `/mocap`
   * 那页比 —— 两页画法一致才比得了，所以基准档必须是 mesh。
   *
   * 骨架那一档留着（下面那排「骨架 / 手模」），它能看清关节位置，是反过来查
   * 遮挡问题的工具；只是不该当默认。
   */
  const [handRender, setHandRender] = useState<"skeleton" | "mesh">("mesh");

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
   * 这里以前只播报静态那一条（"✓ 已自动加载静态模型 xxx"），时序模型是**默默**加载的。
   * 两条都播报（各自写清是哪一条）：不然刚训完的人看不出自己训的那条有没有加载上。
   * 播报的**去处**后来变了：不再写进 `message` 那条占一行的提示带，而是存进
   * `loadedModels`、挂在 header 的 MODEL 徽标上点开看（理由见那个 state）。
   * 出错的 `seqWarn` 仍然走 `message` —— 它必须自己撞到人眼前。
   *
   * 它**不再选 MODE**（那段逻辑已经删掉，见 `urlMode`）：右栏撤掉后默认档是句子，
   * 而这里的比较只认识静态/时序两条，让它插手只会把默认档覆盖成时序。
   *
   * 顺带一句会救人的对照：静态模型叫 `student_...`、时序模型叫 `seq_student_...`
   * （`Train.tsx:136` / `TrainSequence.tsx:211`），名字里没有 seq_ 前缀的一定是静态模型。
   */
  useEffect(() => {
    let cancelled = false;
    /** 部署的词模型读取出错（不是"没部署"）。必须显示出来，见下 */
    let seqWarn: string | null = null;

    // 两条都**先查后判**：已在内存里的模型也要走一遍查询，`staticReady` /
    // `seqReady` 是靠这里的结果置上去的（`modelReady` 又靠它们）
    const staticJob = getLatestModel().then(async (model) => {
      if (!model) return null;
      if (!isModelLoaded()) await loadModelFromSaved(model);
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
        if (!isSequenceModelLoaded()) await loadDeployedWordModel();
        return {
          name: "seq_student(部署)",
          createdAt: deployedAt,
          words: deployed.labels.length,
        };
      }
      if (!saved) return null;
      if (!isSequenceModelLoaded()) await loadSequenceModelFromSaved(saved);
      return {
        name: saved.name,
        createdAt: saved.createdAt,
        words: saved.labels.length,
      };
    })();

    Promise.all([staticJob, seqJob]).then(([staticModel, seqModel]) => {
      if (cancelled) return;
      if (staticModel) setStaticReady(true);
      if (seqModel) setSeqReady(true);
      // 这里**不再自动选档**：MODE 的唯一来源是 URL（见上面 `urlMode`）。
      // 原来那段"谁的 createdAt 更新就切到谁"只认识静态/时序两条，而右栏撤掉后
      // 页面默认就是句子档，让它插手只会把默认档覆盖成时序。
      // 现在生效的是哪两条（不是"这次加载了哪几条"，见 `loadedModels`）
      const active: string[] = [];
      if (staticModel) active.push(`静态单帧 "${staticModel.name}"`);
      if (seqModel)
        active.push(`时序滑窗 "${seqModel.name}" · ${seqModel.words} 词`);
      setLoadedModels(active);
      if (seqWarn) setMessage(`⚠ ${seqWarn}`);
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
        // 演示译文覆盖：一次解码 = 一句。见上面 `demoSeqRef` 那段
        demoSeqRef.current += 1;
        setDemoSeq(demoSeqRef.current);
        const override = demoSubtitleAt(demoSeqRef.current);
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
        /*
         * `|| override` 这一道是演示逼出来的：现场"随便打手语"经常一个词都解不
         * 出来（`post.words` 是空的），不放开的话那一句既不进历史也不朗读 ——
         * 大字倒是有（面板那边只看 `textOverride`），但聊天窗口攒不起来，
         * 四句连不成一段。覆盖关掉之后这个条件自动退回原来的样子。
         */
        if (post.words.length > 0 || override) {
          // rule 也要留下来：历史气泡展开时显示的是"当时命中了哪条规则"，
          // 事后拿 words 重算可能命中的已经是另一条了（规则表会改）
          const solved = resolveSentence(post.words, {}, grammarOnRef.current);
          const text = override ?? solved.text;
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
            setSentenceHistory((prev) => [
              ...prev,
              { text, words: post.words, rule: solved.rule, at: Date.now() },
            ]);
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
      /*
       * 词被删空了 —— 历史里那条也该撤掉，不能留一句已经不存在的话。
       *
       * 但演示覆盖开着的时候**不能撤**：现场解出 0 个词是常事，而那一句上面
       * 已经按假字幕进过历史了，撤掉的效果就是刚出现的那句话自己消失。
       */
      if (subtitleOverride) return;
      setSentenceHistory((prev) => prev.slice(0, -1));
      setLastLive(false);
      return;
    }
    const solved = resolveSentence(sentenceWords, pronOverrides, grammarOn);
    // 覆盖也要走这一遍，否则改代词时这里会拿真译文把历史里那条假字幕冲掉
    const text = subtitleOverride ?? solved.text;
    setSentenceHistory((prev) => {
      if (prev.length === 0 || prev[prev.length - 1].text === text) return prev;
      const last = prev[prev.length - 1];
      // `at` 沿用原值：这条记的是"这句话什么时候说的"，改代词不该让它跳到队尾的时间
      return [
        ...prev.slice(0, -1),
        { text, words: sentenceWords, rule: solved.rule, at: last.at },
      ];
    });
  }, [lastLive, sentenceWords, pronOverrides, grammarOn, subtitleOverride]);

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
   *
   * words / rule 由面板一起传进来（它本来就算好了 `resolved`），这里不重算：
   * 重算要再读一遍 sentenceWords，而这个回调是 `[]` 依赖的，读到的会是旧值。
   */
  const commitSentence = useCallback(
    (text: string, words: string[], rule: string | null) => {
      if (continuousRef.current) {
        setLastLive(false);
        setSentenceWords(null);
        setPronOverrides({});
        setSentenceNote("已定版。接着打下一句就行，不用点按钮。");
        return;
      }
      setSentenceHistory((prev) => [...prev, { text, words, rule, at: Date.now() }]);
      setSentenceWords(null);
      setPronOverrides({});
      setSentenceNote("已成句。再点「开始一句」打下一句。");
    },
    []
  );

  // 清除历史（**逐词档的** history，不是句子档那个）
  const clearHistory = useCallback(() => {
    setHistory([]);
    lastAddedWordRef.current = "";
  }, []);

  /**
   * 清空句子档的聊天记录。
   *
   * 和上面那个 `clearHistory` 是两套账本，不能合并：那个清的是逐词档一个个词的
   * `history`。`lastLive` 必须一起清 —— 不清的话，同步 effect 会以为"历史最后一条
   * 是当前这句"，下一次改代词就会往空数组里同步（`prev.length === 0` 挡住了崩溃，
   * 但状态从此不一致）。
   */
  const clearSentenceHistory = useCallback(() => {
    setSentenceHistory([]);
    setLastLive(false);
    // 演示字幕一并回到第一句。清了记录还接着念第 3 句的话，就没法重演一遍
    demoSeqRef.current = -1;
    setDemoSeq(-1);
  }, []);

  /**
   * 画进聊天窗口的历史 = 全部历史**减掉最后一条 live**。
   *
   * ⚠ 这一步不能省。连续模式下解码那一刻会同时往 `sentenceHistory` 追一条、
   * 又把 `sentenceWords` 设成同一句（`lastLive=true` 就是这个意思）。
   * 不切掉的话，同一句会在窗口里出现两次：一次是历史气泡、一次是下面那个 live 气泡。
   * 历史以前不显示，所以这个重叠一直没露出来过。
   */
  const settledHistory = lastLive ? sentenceHistory.slice(0, -1) : sentenceHistory;

  // 组合翻译文本
  const translatedText = history.map((h) => h.word).join(" ");

  return (
    /* h-screen + overflow-hidden：底部手模条要按"剩下多少高度"占位，
       父级高度必须是确定值；min-h-screen 下长历史会把手模条顶到屏幕外 */
    <div
      /* 底色走 `--hud-page`，不写死。浅色是 index.css 里 `:root` 的默认值、
         全站共用，这一页不再自己贴皮肤类（原来这里有个 `theme-light`，
         那时只有这一页是浅色；现在全站都换过去了，那个类已经不存在）。
         想把某一块退回赛博朋克配色，给它加 `.theme-dark`。 */
      className="h-screen flex flex-col overflow-hidden"
      style={{ backgroundColor: "var(--hud-page)" }}
    >
      {/* 顶部导航 */}
      <header className="h-12 flex items-center justify-between px-4 border-b border-[var(--hud-line)] shrink-0">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[var(--hud-line-strong)]" />
          <span className="text-xs font-bold tracking-widest text-[var(--hud-accent)] font-mono">
            SIGN LANGUAGE TRANSLATOR
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          {/* 显示/隐藏手模挪到这里 —— 它原来钉在左列标题行上，而左列在关掉之后
              整张卡都不渲染了，开关跟着一起消失就再也开不回来 */}
          <button
            onClick={() => setShowHands((v) => !v)}
            className="text-[var(--hud-dim)] hover:text-[var(--hud-accent)] flex items-center gap-1 transition-colors whitespace-nowrap"
            title={
              showHands
                ? "隐藏手模（3D 画面和推理抢同一块 GPU，机器吃力时可以关掉）"
                : "显示手模"
            }
          >
            {showHands ? (
              <>
                <EyeOff className="w-3 h-3" />
                隐藏手模
              </>
            ) : (
              <>
                <Eye className="w-3 h-3" />
                显示手模
              </>
            )}
          </button>
          {/*
            手模 / 骨架切换。`showHands` 关掉时不显示 —— 左框整个没渲染，
            这时候切它没有任何可见效果，只会让人以为按坏了。
            手模在前：它是默认档，也是和 /mocap 一致的那一档（见 `handRender`）。
          */}
          {showHands && (
            <div className="flex items-center rounded-sm border border-[var(--hud-line-strong)] overflow-hidden">
              {(
                [
                  ["mesh", "手模"],
                  ["skeleton", "骨架"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  onClick={() => setHandRender(value)}
                  className={`px-2 py-0.5 transition-colors ${
                    handRender === value
                      ? "bg-[var(--hud-accent)] text-white"
                      : "text-[var(--hud-dim)] hover:text-[var(--hud-accent)]"
                  }`}
                  title={
                    value === "mesh"
                      ? "画实心手模（默认）—— 和第 1 步 /mocap 自检那几格画法一致，两页可以直接对照"
                      : "画关节骨架 —— 藏掉蒙皮只留关节球和骨链，看得清指节到底停在哪；姿态和手模档完全一致"
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          )}
          {/*
            ===== 演示按钮 =====

            连着手套时 disabled：两边都在写 `sentenceWords`，同时跑的话屏幕上是
            真词和演示词交替闪。用 disabled 而不是隐藏 —— 隐藏会让这一行在
            连接/断开时抖一下，而且"连上之后按钮没了"会被当成功能坏了。

            播放中变成「停止演示」：演示是有状态的（一串 timer + 被占用的
            sentenceWords/History），没有出口的话只能靠刷新页面退出。
          */}
          {modelMode === "sentence" && (
            <button
              onClick={demoOn ? stopDemo : startDemo}
              disabled={isConnected}
              className={`cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1 ${
                isConnected ? "opacity-40 cursor-not-allowed" : ""
              }`}
              style={demoOn ? { color: "var(--hud-warn)" } : undefined}
              title={
                isConnected
                  ? "手套已连接，不需要演示 —— 直接打就行（演示和真数据会互相盖）"
                  : `不连手套看一眼译文长什么样：${DEMO_SCRIPT.length} 句，逐词跳出来、停一下自动收句往上滚`
              }
            >
              {demoOn ? (
                <>
                  <Square className="w-3 h-3" />
                  停止演示
                </>
              ) : (
                <>
                  <Play className="w-3 h-3" />
                  演示
                </>
              )}
            </button>
          )}
          {/*
            MODEL 徽标 = "模型就绪"指示灯 + "就绪的是哪两条"的入口。
            点开是浮层（`absolute`），**不占版面** —— 这份清单原来是正文里一条
            常驻的 `✓ 已自动加载：…` 提示带，见 `loadedModels`。
            清单空着（两条都没有）时退回纯 span：没得可点就别装成按钮。
          */}
          {modelReady &&
            (loadedModels.length > 0 ? (
              <div className="relative">
                <button
                  onClick={() => setModelInfoOpen((o) => !o)}
                  className="text-[var(--hud-ok)] flex items-center gap-1 hover:opacity-80 transition-opacity"
                  title="点开看当前生效的模型"
                >
                  <Brain className="w-3 h-3" />
                  MODEL
                  {modelInfoOpen ? (
                    <ChevronDown className="w-2.5 h-2.5" />
                  ) : (
                    <ChevronRight className="w-2.5 h-2.5" />
                  )}
                </button>
                {modelInfoOpen && (
                  <>
                    {/* 点别处关掉。没有这层的话浮层只能靠再点一次徽标关，
                        而人的直觉是点空白处 */}
                    <div
                      className="fixed inset-0 z-40"
                      onClick={() => setModelInfoOpen(false)}
                    />
                    <div className="absolute right-0 top-full mt-1 z-50 min-w-[220px] rounded-sm border border-[var(--hud-line-strong)] bg-[var(--hud-surface)] px-3 py-2 shadow-lg">
                      <div className="text-[9px] uppercase tracking-wider text-[var(--hud-dim)] mb-1">
                        当前生效
                      </div>
                      {loadedModels.map((m) => (
                        <div
                          key={m}
                          className="text-[10px] text-[var(--hud-text)] leading-relaxed whitespace-nowrap"
                        >
                          ✓ {m}
                        </div>
                      ))}
                      {/* 这一句是真会救人的：名字里没有 seq_ 前缀的一定是静态模型
                          （Train.tsx:136 / TrainSequence.tsx:211） */}
                      <div className="mt-1 pt-1 border-t border-[var(--hud-line)] text-[9px] text-[var(--hud-faint)] leading-relaxed">
                        静态模型叫 student_… · 时序模型叫 seq_student_…
                      </div>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <span className="text-[var(--hud-ok)] flex items-center gap-1">
                <Brain className="w-3 h-3" />
                MODEL
              </span>
            ))}
          {/*
            换手归一化徽标。**静默做这件事很危险**：识别结果对不上时，用户没法分辨
            是"手语做错了"还是"软件把左手当右手在算"。所以这条不能删。
            但它在左手用户身上是常亮的 —— 原来那行常驻提示带就是这么一直占着
            正文一行的（见下面正文区那段注释）。这里的形态是：徽标常亮（"这件事
            正在发生"一眼可见），全文点开才给（`absolute`，零高度）。
          */}
          {mirrored && (
            <div className="relative">
              <button
                onClick={() => setMirrorInfoOpen((o) => !o)}
                className="text-[var(--hud-violet)] flex items-center gap-1 hover:opacity-80 transition-opacity whitespace-nowrap"
                title="点开看归一化口径"
              >
                ⇄ 镜像
                {mirrorInfoOpen ? (
                  <ChevronDown className="w-2.5 h-2.5" />
                ) : (
                  <ChevronRight className="w-2.5 h-2.5" />
                )}
              </button>
              {mirrorInfoOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setMirrorInfoOpen(false)}
                  />
                  <div className="absolute right-0 top-full mt-1 z-50 w-[280px] rounded-sm border border-[var(--hud-line-strong)] bg-[var(--hud-surface)] px-3 py-2 shadow-lg">
                    <div className="text-[10px] text-[var(--hud-text)] leading-relaxed">
                      ⇄ 已按镜像归一化到右手口径推理（
                      {dominance ? describeDominance(dominance) : "主手＝左手"}
                      ，换手不改词义）
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
          {isTranslating && (
            <span className="text-[var(--hud-err)] flex items-center gap-1 animate-pulse">
              <Volume2 className="w-3 h-3" />
              LIVE
            </span>
          )}
          {/* 手套状态（左右手分开）+ 下一步 */}
          <StepNav />
        </div>
      </header>

      {/*
       * 主内容区：**上＝译文，下＝手语比划画面**。
       *
       * 这一版是上下分。⚠ 先读完这段再动比例 —— 上一版特意从上下改成了左右，
       * 理由是硬的：手模取景**竖直方向定尺寸**（见 HandModel.tsx 顶部第 2 条），
       * 所以**视口有多高就决定手有多大，把框加宽一点用都没有**。左右分时手模拿到
       * 的是整个正文高度（1440×900 实测 453×740），上下分只能拿到分给它的那一截。
       *
       * 现在这个上下分**不是回退到旧版**：旧版译文在上、手模条只剩 46vh，手很小。
       * 这一版把比例倒过来 —— 手语区拿 70%、译文区拿剩下的 30%
       * （1440×900 下手模视口 687×498）。
       *
       * "手还是小"这件事最后是**在取景里解决的**，不是在这里的宽高比里：
       * 帧内构成原来是「上方空白 8% / 手 53% / 前臂 39%」。相机推近 + 画面中心
       * 上抬之后手占 ~59% 帧高，同时前臂还留着腕下 3.2 个世界单位（那条线索不能丢，
       * 见 HandModel.tsx 第 2 条里三档取景的对照）。
       * **别再想着从这里的百分比找手的大小**：这一区再多给 10%，手也只跟着长 10%；
       * 取景那一刀是成倍的，而且不花译文区的地方。
       *
       * 卡里那行标题（`LIVE HAND · …`）也已经删了，37px 全给了画面 —— 见下面。
       */}
      <div className="flex-1 min-h-0 flex flex-col gap-3 p-3 overflow-hidden">
        {/* ===== 上：翻译输出 =====
         *
         * 句子档下这一列就是**聊天窗口**，由 `SentencePanel` 整个承担：
         * 它自己是那张卡（标题 / 气泡区滚动 / 底部控件三段），所以这里不再包一层
         * 居中容器 —— 聊天记录是从上往下堆、新的在下面，居中会让它在只有一两句时
         * 飘在中间、多几句之后又跳成顶对齐。
         *
         * 其余状态（没模型 / 没连手套 / 逐词档）仍然是"一屏就这么点内容"的样子，
         * 保留居中，但**滚动容器和居中容器必须分成两层**：
         * `justify-center` 和 `overflow-y-auto` 写在同一个元素上是 flexbox 的经典坑，
         * 内容超高时会朝两头溢出，而**朝顶部溢出的那部分滚不到**（scrollTop 到 0 就停）。
         * 上下分之后这一区是**矮而宽**的（1440×900 下约 290px 高），逐词档那摞读数
         * 更容易超高，最先看不见的恰好是排最上面的大字 —— 表现和"输出窗口被删了"
         * 一样。外层只管滚、内层用 `min-h-full` 管居中。
         *
         * 高度是**剩下的那部分**（`flex-1`）：手语区先按 62% 拿走它那一份，
         * 这里吃剩下的。手模关掉时（`showHands` 为假）它整个不渲染，这一区自然吃满。
         */}
        <div className="flex flex-col flex-1 min-h-0 min-w-0 gap-2">
          {/* 句子档就绪 —— 整列换成聊天窗口。
              逐词那套（大字 + 置信度条 + 开始翻译按钮 + 词历史）在这一档全都
              不适用：没有"当前这个词"，也没有逐词置信度，句子的开始/结束
              由面板自己的按钮驱动 */}
          {/* `|| demoOn` 是演示的入口：它**只放开 isConnected 这一道**
              （模型档仍要求 sentence，模型本身仍要求就绪）。演示不需要模型 ——
              它喂的是模型该吐出的词，不跑推理；但档位得对，逐词档下这个面板
              整个不适用。见 `demoOn` 那段。 */}
          {(modelReady && isConnected && modelMode === "sentence") ||
          (demoOn && modelMode === "sentence") ? (
            <SentencePanel
              words={sentenceWords}
              /* ⚠ 传 `settledHistory` 而不是 `sentenceHistory`：最后一条可能就是
                 live 那句，整份传进去会让同一句在窗口里出现两次（见上面那段） */
              history={settledHistory}
              onClearHistory={clearSentenceHistory}
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
              running={isTranslating}
              /* 演示时**故意**保持 disabled：这一项只管「开始 / 结束 / 停止」
                 那三个真正驱动采集的按钮，演示里按它们没有任何意义（没手套、
                 没推理循环）。「代词 / 删词 / 定版 / 朗读」不看这一项，
                 各自按有没有词判断，所以演示里照样能点、能改代词 —— 那正是
                 要展示的东西。 */
              disabled={!modelReady || !isConnected}
              /* 演示译文覆盖。`null` 时这个 prop 等于不存在（见 SentencePanel
                 那边的 `resolved`）。删演示的时候连这一行一起删 */
              textOverride={subtitleOverride}
            />
          ) : (
          <div className="cyber-panel rounded-2xl flex-1 min-h-0 overflow-y-auto">
            <div className="min-h-full flex flex-col items-center justify-center space-y-4 p-4">
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
                    <Brain className="w-16 h-16 mx-auto text-[var(--hud-faint)]" />
                    <p className="text-sm text-[var(--hud-dim)]">
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
                  <Hand className="w-16 h-16 mx-auto text-[var(--hud-dim)]" />
                  <p className="text-sm text-[var(--hud-soft)]">手套未连接</p>
                  {gloveError && (
                    <p className="text-[10px] text-[var(--hud-err)]">{gloveError}</p>
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
                  <p className="text-[10px] text-[var(--hud-dim)]">连接任一只手即可翻译；双手手语请两只都连</p>
                </div>
              )}

              {/* 就绪状态 - 可以翻译（逐词两档）
                  ⚠ 这一档是**横排两栏**，不是竖着一摞。上下分之后这一区是矮而宽的
                  （1440×900 下约 290px 高），而这里要装的东西是：72px 大字 + 置信度条
                  + 三行读数 + 按钮 + 累计输出句 —— 竖着堆下来 400px 都不够，
                  会把最上面的大字顶出可视区（顶部溢出是滚不回去的，见上面那段注释）。
                  横排之后左栏管"当前这一个词"、右栏管"已经攒出来的句子"，
                  刚好也是这一页的两件事。 */}
              {modelReady && isConnected && modelMode !== "sentence" && (
                <div className="w-full flex-1 min-h-0 flex items-stretch gap-5">
                  {/* ===== 左栏：当前识别的这一个词 ===== */}
                  <div className="shrink-0 w-[40%] max-w-[520px] flex flex-col items-center justify-center gap-3 text-center">
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
                                : "var(--hud-dim)",
                            /* 浅色底上不再发光：白底描不出辉光，只会糊成一圈脏边 */
                            textShadow: "none",
                            opacity:
                              currentPrediction.confidence >= 0.5
                                ? 1
                                : 0.4,
                          }}
                        >
                          {currentPrediction.word}
                        </div>
                        {/* 置信度条 + 那几行标定读数。宽度跟着左栏走（原来写死
                            192px，那是右侧窄列时代的值）—— 转角速率那行带阈值说明，
                            192px 下要折成三行，把大字往上顶 */}
                        <div className="w-full max-w-[300px] mx-auto space-y-1">
                          <div className="h-2 bg-[var(--hud-track)] rounded-full overflow-hidden border border-[var(--hud-line-strong)]">
                            <div
                              className="h-full rounded-full transition-all duration-200"
                              style={{
                                width: `${currentPrediction.confidence * 100}%`,
                                backgroundColor:
                                  currentPrediction.confidence >= confidenceThreshold
                                    ? "var(--hud-ok)"
                                    : currentPrediction.confidence >= 0.5
                                    ? "var(--hud-warn)"
                                    : "var(--hud-err)",
                              }}
                            />
                          </div>
                          <div className="text-[10px] font-mono text-[var(--hud-dim)] text-center">
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
                            <div className="text-[10px] font-mono text-center text-[var(--hud-dim)]">
                              转角速率{" "}
                              <span
                                className={
                                  currentPrediction.rotRate >= ROT_RATE_HI
                                    ? "text-[var(--hud-warn)]"
                                    : currentPrediction.rotRate < ROT_RATE_LO
                                      ? "text-[var(--hud-ok)]"
                                      : "text-[var(--hud-mid)]"
                                }
                              >
                                {currentPrediction.rotRate.toFixed(0)}
                              </span>
                              <span className="text-[var(--hud-faint)]">
                                {" "}
                                °/s · 静止&lt;{ROT_RATE_LO} / 画圈≥{ROT_RATE_HI}
                              </span>
                              {/* 中间带要明说，否则读数在 50~70 之间时看不出是谁在做决定 */}
                              {currentPrediction.rotRate >= ROT_RATE_LO &&
                                currentPrediction.rotRate < ROT_RATE_HI && (
                                  <span className="ml-1 text-[var(--hud-mid)]">
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
                            <div className="text-[10px] font-mono text-center text-[var(--hud-dim)]">
                              拇指压力峰值{" "}
                              <span
                                className={
                                  currentPrediction.thumbPeak >= THUMB_PEAK_GATE
                                    ? "text-[var(--hud-ok)]"
                                    : "text-[var(--hud-faint)]"
                                }
                              >
                                {currentPrediction.thumbPeak}
                              </span>
                              <span className="text-[var(--hud-faint)]">
                                {" "}
                                / 闸门 {THUMB_PEAK_GATE}
                              </span>
                              {/* 改判了要说清是**哪个判据**改的，否则调阈值时不知道该调哪一个 */}
                              {currentPrediction.gated && (
                                <span className="ml-1 text-[var(--hud-warn)]">
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
                        <div className="text-4xl text-[var(--hud-faint)] animate-pulse">
                          {gated ? "—" : "..."}
                        </div>
                        <p className="text-[10px] text-[var(--hud-dim)] font-mono">
                          {gated
                            ? "静止中 · 等你起手（手没动时不出词）"
                            : "等待手势输入"}
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <MessageSquare className="w-12 h-12 mx-auto text-[var(--hud-faint)]" />
                        <p className="text-sm text-[var(--hud-dim)]">
                          点击"开始翻译"进入实时识别模式
                        </p>
                      </div>
                    )}

                    {/* 控制按钮跟着左栏走：它控制的是"出不出词"这件事 */}
                    <button
                      onClick={toggleTranslation}
                      className={`cyber-btn px-6 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
                        isTranslating ? "cyber-btn-accent" : ""
                      }`}
                    >
                      {isTranslating ? (
                        <>
                          <div className="w-2 h-2 rounded-full bg-[var(--hud-err)] animate-pulse" />
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

                  <div className="w-px shrink-0 bg-[var(--hud-line)]" />

                  {/* ===== 右栏：已经攒出来的句子 =====
                      这是这一页的**产出**。空着的时候也保留这一栏（只显示占位），
                      不然一出第一个词整个版面会横向跳一次。 */}
                  <div className="flex-1 min-w-0 flex flex-col gap-2 py-1">
                    <div className="shrink-0 flex items-center justify-between">
                      <span className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
                        Translation Output
                      </span>
                      <button
                        onClick={clearHistory}
                        disabled={history.length === 0}
                        className={`transition-colors ${
                          history.length === 0
                            ? "text-[var(--hud-faint)] cursor-not-allowed"
                            : "text-[var(--hud-dim)] hover:text-[var(--hud-err)]"
                        }`}
                        title="清除"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                    {/* 这行是这一页的**产出**，不是调试读数。旁边一圈置信度/
                        转角速率都是 10px，输出句原来 18px，两者拉不开层级。
                        24px 与句子档 live 顺句同一档，两个档看起来是一件事。
                        横排之后这一栏拿到的是整块宽度，长句不用再频繁折行 */}
                    <div className="flex-1 min-h-0 overflow-y-auto">
                      {history.length > 0 ? (
                        <p
                          className="text-2xl leading-relaxed"
                          style={{
                            fontFamily: "'Space Grotesk', sans-serif",
                            color: "var(--hud-text)",
                          }}
                        >
                          {translatedText}
                        </p>
                      ) : (
                        <p className="text-[11px] font-mono text-[var(--hud-faint)]">
                          还没有输出 —— 识别到的词会按顺序攒在这里
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          )}

          {/*
           * 这两条提示放在卡**外面**的一条窄带里，两个档共用。
           * 放进卡里的话，句子档下整张卡是 SentencePanel 的三段结构（中间那段才滚），
           * 它们要么被塞进气泡流里跟着滚走、要么得在面板里再开一个插槽 ——
           * 而它们讲的是"推理口径"，跟聊天内容不是一回事。
           *
           * ⚠ 原来这里有**三**条，第一条是常驻的 `⇄ 已按镜像归一化到右手口径推理（…）`。
           * 它已经挪到 header 上变成一个 `⇄ 镜像` 徽标（点开看全文，零高度）——
           * 那一条在左手用户身上是**一直亮着**的，也就是一直占着一行，而这一页
           * 最缺的就是竖直空间（大字 + 手模都要高度）。
           * 留在这里的两条都是**偶发**的：pending_flip 只在换手那几百毫秒出现，
           * message 只在出错时有内容。偶发的才配占一行。
           */}
          {/* 判定还在滞回确认期间就说明主手可能要换了。不提示的话，这几百毫秒里
              喂给模型的是旧口径，用户只会看到"刚换手那一下识别不准" */}
          {isTranslating && dominance?.reason === "pending_flip" && (
            <div className="shrink-0 text-[10px] font-mono text-[var(--hud-warn)] px-1 leading-relaxed">
              ⚠ 检测到主手可能换了，正在确认（切换要连续几个窗口一致，
              避免一个词做到一半翻转口径）
            </div>
          )}

          {/* 出错提示。**只有出错才有内容** —— 启动时"加载成功"的回执已经挪到
              header 的 MODEL 徽标里了（见 `loadedModels`），所以这一行平时是空的、
              一行都不占。颜色跟着从 ok 换成 warn：现在能出现在这里的都是坏消息 */}
          {message && (
            <div className="shrink-0 text-[10px] font-mono text-[var(--hud-warn)] px-1">
              {message}
            </div>
          )}
        </div>

        {/* ===== 下：手语比划区 =====
         *
         * 两只手合进同一张卡：外面一圈圆角边框，里面并排两个视口。每只手的标题行
         * （LH/RH + FPS + 未标定）和画面内的覆盖提示保持**一手一份** ——
         * 那些是分手别的信息，合并了就说不清是哪只手。
         *
         * 与识别链路完全无关，只反映手套原始数据。
         * 关掉时是**真卸载 Canvas**，不是 hidden —— 隐藏的 WebGL 画面照样在渲染，
         * 而关它的理由正是省 GPU。整张卡不渲染，上面那区接管整个高度。
         *
         * ⚠ 高度写成 `h-[70%] shrink-0`，不是 `flex-1`：
         *  - 用 `flex-1` 的话两区平分，手模会更小；
         *  - `shrink-0` 是必须的 —— 上面那区在逐词档下内容会变多（读数 + 三条提示带），
         *    不锁住的话 flex 会从这里往回抢高度，手在打字最多的时候反而缩得最狠。
         * 百分比是相对父级内容高度算的，父级已经是确定高度（h-screen 那条链），
         * 所以这里能拿到稳定值；改成 vh 会把 header 和 p-3 算重。
         *
         * 62% → 70% 是把 SentencePanel 底部控件条压扁（三行 103px 并成一行 48px）
         * 省出来的高度直接给了这里，译文区当时没有变窄。
         *
         * **70% → 55% 就是真的从这里割给译文区了**，理由是那 30% 只够 ~2.5 行台词，
         * 最上面那句基本被顶边渐隐吃掉 —— 而这一页的主角是台词，手模是旁证
         * （标签行自己写着"不参与识别"）。手模缩掉的是**画面高度**，手会跟着小一圈；
         * 嫌小的话去动 HandModel 的取景（那一刀是成倍的，且不花译文区的地方），
         * 别回来调这个百分比。
         */}
        {showHands && (
          <div className="cyber-panel rounded-2xl h-[55%] shrink-0 min-w-0 flex flex-col overflow-hidden">
            {/*
              ⚠ 这里原来有一行卡标题（`LIVE HAND · BEND + IMU（不参与识别）`），
              **已经删掉**，为的是把那 37px 全给下面的画面。信息没丢：
              `BEND+IMU · 不参与识别` 挪到每只手自己的标签行末尾了（那行本来就在，
              所以这次是真省出高度，不是搬个地方继续占）。
              别把标题行加回来 —— 它说的两件事（这是实时手 / 这不是识别链路）
              标签行都在说，而画面高度直接决定手有多大（见 HandModel.tsx 第 2 条）。
            */}
            {/*
              ===== 演示回放的提示带 =====
              只在演示时出现，所以不占常态版面。挂在**手模这张卡**里而不是译文区：
              说的是手模，而译文区那条 `sentenceNote` 说的是句子，混在一处会让
              "这个词没有录制"被读成"这句译文有问题"。

              ⚠ 这里**只放"看起来像故障、其实不是"的那几条**（某个词库里没有录制 →
              手模不动；没标定 → 握拳不成形）。录制本身准不准已经挪去 console.warn，
              别加回来：那句话观众读不懂，只会看见产品自己挂了个橙色警告。
              于是绝大多数时候这条带子根本不出现 —— 那是对的，不是坏了。

              `demoOn && !isConnected` 里那个 `!isConnected` 是冗余的（连着时进不了
              演示），留着是因为这条提示一旦在实机上出现就是纯误导 —— 实机手模播的
              是手套的真数据，与库里录制无关。
            */}
            {demoOn && !isConnected && (demoLoading || demoClipNote) && (
              <div className="shrink-0 mx-3 mt-3 px-2 py-1 rounded-sm bg-[var(--hud-warn-wash)] border border-[var(--hud-warn-edge)]">
                <span className="text-[9px] font-mono text-[var(--hud-warn)] leading-relaxed">
                  {demoLoading ? "正在从库里取录制…" : demoClipNote}
                </span>
              </div>
            )}
            {/* 左手在左、右手在右 —— 第一人称（照镜子）。和第 1 步自检的四格布局
                同一个顺序，两页对照时不用在脑子里翻一次。 */}
            <div className="flex-1 min-h-0 flex gap-3 p-3">
              <HandViewport
                label="LH · 左手"
                side="left"
                channel={gloveLeft}
                driveRef={leftHand.driveRef}
                bendCalibrated={leftHand.bendCalibrated}
                orientCalibrated={leftHand.orientCalibrated}
                mode={handRender}
                playback={demoOn}
              />
              <HandViewport
                label="RH · 右手"
                side="right"
                channel={gloveRight}
                driveRef={rightHand.driveRef}
                bendCalibrated={rightHand.bendCalibrated}
                orientCalibrated={rightHand.orientCalibrated}
                mode={handRender}
                playback={demoOn}
              />
            </div>
          </div>
        )}
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
      <Brain className="w-16 h-16 mx-auto text-[var(--hud-faint)]" />
      {error ? (
        <>
          <p className="text-sm text-[var(--hud-err)]">句子模型加载失败</p>
          {/* 原文照抄出来：blankIndex 不符、frameDim 不符这类错误是"模型要重训"的
              明确信号，藏起来就只剩"不可用" */}
          <p className="text-[10px] font-mono text-[var(--hud-err)] leading-relaxed break-all">
            {error}
          </p>
        </>
      ) : available === null ? (
        <p className="text-sm text-[var(--hud-dim)]">正在检查有没有句子模型…</p>
      ) : (
        <>
          <p className="text-sm text-[var(--hud-dim)]">还没有句子模型</p>
          <div className="text-[10px] font-mono text-[var(--hud-dim)] leading-relaxed text-left inline-block space-y-1">
            {/* 这条警告要留着：两个模型长得像但不通用，指错一次就是白训一轮 */}
            <p className="text-[var(--hud-warn)]">
              注意：这一档要的<b>不是</b> /train-seq 训的那个模型（那是逐词滑窗、存在
              IndexedDB 里）。句子模型在本机 Python 里训，是一份静态文件。
            </p>
            <p className="text-[var(--hud-faint)]">
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
            style={{
              borderColor: "var(--hud-violet)",
              color: "var(--hud-violet)",
            }}
          >
            前往句子训练 →
          </Link>
          <p className="text-[9px] font-mono text-[var(--hud-faint)] leading-relaxed">
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
 * 驱动和标定与第 1 步 /mocap 的自检**是同一份** —— 同一个 `HandModel` 组件、
 * 同一个相机、同一份标定。**只有画法不同**：这一页默认画骨架，自检页画实心手模。
 * 所以手模看着不对劲时，别直接和自检页对照 —— 先用顶栏的「手模」档切回实心，
 * 那时候两页应该是同一只手，不一样就说明真出问题了。
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
  mode,
  playback = false,
}: {
  label: string;
  side: "left" | "right";
  channel: HandChannel;
  driveRef: RefObject<HandDrive>;
  bendCalibrated: boolean;
  orientCalibrated: boolean;
  mode: "mesh" | "skeleton";
  /**
   * 手模现在放的是**库里的录制回放**（离线演示），不是手套数据。
   *
   * 影响两处显示，都是为了不撒谎也不误导：
   *  - 那层"这只手套没连接"的遮罩要撤掉 —— 手确实在动，压着遮罩就是自相矛盾；
   *  - 状态位从"未连接"改成"回放"，而不是伪装成已连接。手套确实没连。
   */
  playback?: boolean;
}) {
  const connected = channel.isConnected;
  return (
    <div className="flex-1 min-w-0 min-h-0 flex flex-col gap-1">
      <div className="shrink-0 flex items-center gap-1.5 px-0.5 text-[9px] font-mono">
        <span className="tracking-widest text-[var(--hud-warn)]">{label}</span>
        <span
          className={
            connected
              ? "text-[var(--hud-ok)]"
              : playback
              ? "text-[var(--hud-violet)]"
              : "text-[var(--hud-dim)]"
          }
        >
          {connected
            ? `${channel.gloveFps.toFixed(0)} FPS`
            : playback
            ? "回放库里的录制"
            : "未连接"}
        </span>
        {connected && bendCalibrated && !orientCalibrated && (
          <span className="text-[var(--hud-dim)]">朝向未标定</span>
        )}
        {/* 这两句原来在卡的标题行上（整卡一份），标题行删掉后落到这里。
            一手一份是重复的 —— 但它们是**卡级**的事实，而这一行是唯一一条
            不额外占高度的位置。"不参与识别"必须留着：这两个画面在页面上最显眼，
            不说清就会被当成识别链路，"手模动了怎么还不出词"由此而来 */}
        <span className="text-[var(--hud-faint)]">BEND+IMU · 不参与识别</span>
      </div>
      <div
        className="hud-stage relative flex-1 min-h-0 rounded-xl border overflow-hidden"
        style={{
          borderColor:
            connected || playback
              ? "var(--hud-line)"
              : "var(--hud-line-dead)",
        }}
      >
        <HandModel driveRef={driveRef} side={side} mode={mode} />
        {!connected && !playback && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--hud-veil)] pointer-events-none">
            <span className="px-2 py-1 rounded-sm bg-[var(--hud-chip)] border border-[var(--hud-line)] text-[10px] font-mono text-[var(--hud-soft)]">
              这只手套没连接，去第 1 步
            </span>
          </div>
        )}
        {/* 未标定这条必须压在画面里，不能只写在标题行：这个状态下握拳只弯到约
            42%，看起来就是"手模坏了"或"模型不准"，而其实只是没跑标定 */}
        {connected && !bendCalibrated && (
          <div className="absolute bottom-1.5 left-1.5 right-1.5 px-2 py-1 rounded-sm pointer-events-none bg-[var(--hud-warn-wash)] border border-[var(--hud-warn-edge)]">
            <span className="text-[9px] font-mono text-[var(--hud-warn)] leading-relaxed">
              未标定弯折 · 满量程也只弯约 42%，握拳不会成形。去第 1 步跑向导
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
