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
 * 两种推理模式，可在右侧面板切换：
 * - static：旧的单帧 MLP，每 100ms 独立推理一帧（原有逻辑，一行没动）
 * - sequence：TCN 时序模型，手套帧全速写进环形缓冲，每 100ms 取最近 1.5s
 *   窗口推理一次。动态词（再见、来、工作…）只有这条路能识别。
 *
 * 两条路共用下游的置信度阈值 / 平滑窗口 / 2 秒去重逻辑 —— 那套逻辑与模型
 * 无关，只跟"一串预测结果如何变成一句话"有关，不该为时序模型重写一份。
 *
 * 底部有一条**双手 3D 手模**（与 /mocap 同一份标定、同一套驱动，见 useHandModelDrive）。
 * 它不参与识别，是给"为什么不识别"提供第一手判据的：手模不动 = 手套没在出数据；
 * 手模动了但手型不像 = 弯折标定或某一路传感器的问题，跟模型无关。
 * 没有它的时候，这两种硬件问题在这一页表现为"模型不准"，会把人引向重训模型。
 */
import { useGloveFrames, useGloves } from "@/contexts/GloveContext";
import StepNav from "@/components/StepNav";
import HandModel from "@/components/HandModel";
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
import { SequenceWindowBuffer } from "@/lib/sequenceWindow";
import {
  mirrorStaticInputs,
  normalizeHandedness,
  type DominantHand,
} from "@/lib/handMirror";
import type { GloveFrame } from "@/lib/gloveProtocol";
import { getLatestModel, getLatestSequenceModel } from "@/lib/datasetStore";
import {
  getWordById,
  getCategoryColor,
  IDLE_LABEL,
} from "@/lib/signLanguageVocab";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

type ModelMode = "static" | "sequence";

/** 滑窗长度：常见孤立词 1~2s，取 1.5s 兼顾"能装下整个词"和"延迟不刺眼" */
const WINDOW_MS = 1500;

/** 主手选择的 localStorage key */
const DOMINANT_KEY = "dk_translate_dominant_hand";

export default function Translate() {
  const [modelMode, setModelMode] = useState<ModelMode>("static");
  const windowBufRef = useRef<SequenceWindowBuffer | null>(null);
  if (windowBufRef.current === null) {
    windowBufRef.current = new SequenceWindowBuffer({ bufferMs: 3000 });
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
   * 主手（做手语的那只手）。**两只手套都连着时归一化只能靠这个** ——
   * 闲着那只手的槽位有静止数据，从数据上分不出"没戴"和"戴着不动"，
   * 所以 `auto` 在双手都有数据时判不了。惯用手不会天天变，记进 localStorage。
   */
  const [dominantHand, setDominantHand] = useState<DominantHand>(() => {
    const v = localStorage.getItem(DOMINANT_KEY);
    return v === "left" || v === "right" ? v : "auto";
  });
  const dominantRef = useRef(dominantHand);
  useEffect(() => {
    dominantRef.current = dominantHand;
    localStorage.setItem(DOMINANT_KEY, dominantHand);
  }, [dominantHand]);

  // 连接在第 1 步（/mocap）建立、住在 GloveProvider 里，本页只订阅帧、不碰串口
  const { left: gloveLeft, right: gloveRight, anyConnected } = useGloves();
  useGloveFrames(onLeftFrame, onRightFrame);
  const isConnected = anyConnected;
  const bothConnected = gloveLeft.isConnected && gloveRight.isConnected;
  const gloveError = gloveLeft.error || gloveRight.error;

  // 状态
  const [staticReady, setStaticReady] = useState(isModelLoaded());
  const [seqReady, setSeqReady] = useState(isSequenceModelLoaded());
  const modelReady = modelMode === "static" ? staticReady : seqReady;
  const [currentPrediction, setCurrentPrediction] = useState<{
    label: string;
    word: string;
    confidence: number;
    allProbabilities: Array<{ label: string; probability: number }>;
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

    const seqJob = getLatestSequenceModel().then(async (model) => {
      if (!model) return null;
      if (!isSequenceModelLoaded()) {
        await loadSequenceModelFromSaved(model);
        loaded.push(`时序滑窗 "${model.name}"`);
      }
      return model;
    });

    Promise.all([staticJob, seqJob]).then(([staticModel, seqModel]) => {
      if (cancelled) return;
      if (staticModel) setStaticReady(true);
      if (seqModel) setSeqReady(true);
      setLoadedNames({
        static: staticModel?.name ?? null,
        sequence: seqModel?.name ?? null,
      });
      // 谁的 createdAt 更新就切到谁；只有一个就用那一个
      if (seqModel && (!staticModel || seqModel.createdAt >= staticModel.createdAt)) {
        setModelMode("sequence");
      } else if (staticModel) {
        setModelMode("static");
      }
      if (loaded.length) setMessage(`✓ 已自动加载：${loaded.join(" · ")}`);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  // 推理循环 — 使用 ref 读取最新帧，避免闭包陷阱
  useEffect(() => {
    if (!isTranslating || !isConnected || !modelReady) return;

    console.log("[Translate] Starting inference loop...");

    // 归一化状态只在**翻转的那一刻**进 state：推理每 100ms 一次，
    // 无条件 setState 会把整页每秒重渲染十次。ref 记住上一次的值即可
    const reportMirrored = (v: boolean) => {
      if (mirroredRef.current === v) return;
      mirroredRef.current = v;
      setMirrored(v);
    };

    translateIntervalRef.current = setInterval(() => {
      let result:
        | {
            label: string;
            confidence: number;
            allProbabilities: Array<{ label: string; probability: number }>;
          }
        | null = null;

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
        // 判定与序列那条路**同一套**：显式指定左手 → 无条件镜像；
        // 没指定 → 只有在"只连了左手套"时才敢镜像
        const dom = dominantRef.current;
        const doMirror =
          dom === "left" || (dom === "auto" && !!leftInput && !rightInput);
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
        // 归一化到右手口径：整条样本镜像 + 左右槽位互换。模型的两只手是两段独立槽位、
        // 且 137 维指序左右相反，不归一化的话左手做的动作落在训练时只见过静止基线的
        // 那半边，输出会塌到某一个固定的词上（见 handMirror.ts 顶部）
        const norm = normalizeHandedness(raw, dominantRef.current);
        reportMirrored(norm.mirrored);
        result = predictSequence(norm.sample);
      }

      if (!result) {
        console.warn("[Translate] predict() returned null");
        return;
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

      const word = getWordById(pred.label);
      setCurrentPrediction({
        label: pred.label,
        word: word?.label ?? pred.label,
        confidence: pred.confidence,
        allProbabilities: pred.allProbabilities.slice(0, 5),
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
          // 确认识别结果
          const entry: TranslationEntry = {
            word: word?.label ?? pred.label,
            label: pred.label,
            confidence: pred.confidence,
            timestamp: now,
          };
          setHistory((prev) => [...prev, entry]);
          lastAddedWordRef.current = pred.label;
          lastAddedTimeRef.current = now;
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
      // 停下来之后那行提示就不再反映任何正在发生的事，留着是假信息
      reportMirrored(false);
    };
  }, [isTranslating, isConnected, modelReady, modelMode]); // 不再依赖 latestFrame/confidenceThreshold/smoothingWindow

  // 开始/停止翻译
  const toggleTranslation = useCallback(() => {
    if (isTranslating) {
      setIsTranslating(false);
      setCurrentPrediction(null);
      predictionBufferRef.current = [];
    } else {
      // 开始前清掉环形缓冲：里面可能是上次停止翻译前留下的旧动作，
      // 不清会在刚开始的 1.5s 内拿陈旧数据推理
      windowBufRef.current?.clear();
      setIsTranslating(true);
    }
  }, [isTranslating]);

  const switchMode = useCallback((mode: ModelMode) => {
    setModelMode(mode);
    setCurrentPrediction(null);
    predictionBufferRef.current = [];
    windowBufRef.current?.clear();
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
          {/* 翻译输出区 */}
          <div className="flex-1 min-h-0 overflow-y-auto flex flex-col items-center justify-center p-8 space-y-8">
            {/* 模型未加载 */}
            {!modelReady && (
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
            )}

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

            {/* 就绪状态 - 可以翻译 */}
            {modelReady && isConnected && (
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
                                  getWordById(currentPrediction.label)?.category ?? ""
                                )
                              : "#556677",
                          textShadow:
                            currentPrediction.confidence >= confidenceThreshold
                              ? `0 0 30px ${getCategoryColor(
                                  getWordById(currentPrediction.label)?.category ?? ""
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
                      </div>
                    </>
                  ) : isTranslating ? (
                    <div className="space-y-2">
                      <div className="text-4xl text-[#334455] animate-pulse">
                        ...
                      </div>
                      <p className="text-[10px] text-[#556677] font-mono">
                        等待手势输入
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
                {dominantHand === "left" ? "主手＝左手" : "只检测到左手套"}
                ，换手不改词义）
              </div>
            )}

            {/* 双手套都连着又没指定主手时，归一化**判不了**。这一步只能靠用户，
                不提示的话表现就是"左手怎么做都是同一个词"，而页面上什么异常都没有 */}
            {isTranslating && !mirrored && dominantHand === "auto" && bothConnected && (
              <div className="text-[10px] font-mono text-[#f59e0b]">
                ⚠ 两只手套都连着，无法自动判断主手。用左手做手语请在右侧「DOMINANT
                HAND」里选「左手」，否则模型看到的是训练时没出现过的输入。
              </div>
            )}

            {/* 消息 */}
            {message && (
              <div className="text-[10px] font-mono text-[#00e5a0]">
                {message}
              </div>
            )}
          </div>

          {/* 双手 3D 手模条：与识别链路完全无关，只反映手套原始数据 */}
          <div
            className={`shrink-0 border-t border-[#00f0ff]/15 px-3 pt-1.5 pb-2 flex flex-col ${
              showHands ? "h-[300px]" : ""
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
                    ? "隐藏手模（两个 3D 画面和推理抢同一块 GPU，机器吃力时可以关掉）"
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
              <div className="flex-1 min-h-0 flex gap-3 justify-center">
                <HandStripItem channel={gloveLeft} handKey="LH" label="LH · 左手" />
                <HandStripItem channel={gloveRight} handKey="RH" label="RH · 右手" />
              </div>
            )}
          </div>
        </div>

        {/* 右侧面板 */}
        <div className="w-60 border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-4 shrink-0">
          {/* 推理模式切换 */}
          <Section title="MODE">
            <div className="grid grid-cols-2 gap-1">
              <button
                onClick={() => switchMode("static")}
                className={`px-2 py-1.5 rounded-sm text-[10px] font-mono border transition-colors ${
                  modelMode === "static"
                    ? "border-[#00f0ff]/60 text-[#00f0ff] bg-[#00f0ff]/10"
                    : "border-[#00f0ff]/15 text-[#556677]"
                }`}
              >
                静态单帧
              </button>
              <button
                onClick={() => switchMode("sequence")}
                className={`px-2 py-1.5 rounded-sm text-[10px] font-mono border transition-colors ${
                  modelMode === "sequence"
                    ? "border-[#a855f7]/60 text-[#a855f7] bg-[#a855f7]/10"
                    : "border-[#00f0ff]/15 text-[#556677]"
                }`}
              >
                时序滑窗
              </button>
            </div>
            <div className="text-[9px] font-mono text-[#556677] leading-relaxed">
              {modelMode === "static" ? (
                <>每 100ms 推理单帧。看不到运动轨迹，「再见」「来」这类动态词识别不了。</>
              ) : (
                <>
                  每 100ms 取最近 {WINDOW_MS}ms 窗口推理，能识别动态词。
                  {!seqReady && (
                    <span className="text-[#ff2d7b]">
                      {" "}
                      当前没有时序模型，先去 /train-seq 训练。
                    </span>
                  )}
                </>
              )}
            </div>
          </Section>

          {/* 主手 —— 归一化的唯一输入。模型是用右手采的数据训的，
              左手做手语必须整体镜像到右手口径 */}
          <Section title="DOMINANT HAND">
            <div className="grid grid-cols-3 gap-1">
              {(
                [
                  ["auto", "自动"],
                  ["right", "右手"],
                  ["left", "左手"],
                ] as Array<[DominantHand, string]>
              ).map(([v, text]) => (
                <button
                  key={v}
                  onClick={() => setDominantHand(v)}
                  className={`px-1 py-1.5 rounded-sm text-[10px] font-mono border transition-colors ${
                    dominantHand === v
                      ? "border-[#a855f7]/60 text-[#a855f7] bg-[#a855f7]/10"
                      : "border-[#00f0ff]/15 text-[#556677]"
                  }`}
                >
                  {text}
                </button>
              ))}
            </div>
            <div className="text-[9px] font-mono text-[#556677] leading-relaxed">
              {dominantHand === "auto" ? (
                bothConnected ? (
                  <span className="text-[#f59e0b]">
                    两只手套都连着，自动判不了主手（闲着那只手也在出静止数据）。
                    用左手做手语请直接选「左手」。
                  </span>
                ) : (
                  <>按连了哪只手套判：只连左手套时整体镜像到右手口径。</>
                )
              ) : dominantHand === "left" ? (
                <>
                  无条件把两只手镜像并互换槽位，喂给模型的永远是右手口径。
                  单手词换手不改词义，双手词整体镜像后也是同一个词。
                </>
              ) : (
                <>训练口径本身，不做任何变换。</>
              )}
            </div>
          </Section>

          {/* Top-K 候选 */}
          {currentPrediction && isTranslating && (
            <Section title="CANDIDATES">
              <div className="space-y-1">
                {currentPrediction.allProbabilities.map((p, i) => {
                  const word = getWordById(p.label);
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
                          {word?.label ?? p.label}
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
                      .map((l) => getWordById(l)?.label ?? l)
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
 * 底部手模条里的一只手。
 *
 * 标定是**只读**的：连接与标定的唯一入口在第 1 步（/mocap），这里改标定只会让
 * 两页说法不一致。所以未标定时不给按钮，只给一条"去第 1 步标定"的提示 ——
 * 未标定的手模最多弯到 0.42（柔和预览），拿它判断手型会得出错误结论，必须标死。
 */
function HandStripItem({
  channel,
  handKey,
  label,
}: {
  channel: HandChannel;
  handKey: HandKey;
  label: string;
}) {
  const { driveRef, bendCalibrated, orientCalibrated } = useHandModelDrive(
    channel,
    handKey
  );
  const connected = channel.isConnected;

  // aspect-[4/3] w-auto：fiber 只按容器**垂直** FOV 取景，容器越扁手就被上下切得越多
  // （实测 610×208 的扁盒子里手腕直接出画）。所以让高度决定宽度、左右留白，
  // 取景与 /mocap 上那两块保持一致
  return (
    <div className="h-full aspect-[4/3] min-w-0 flex flex-col gap-1">
      <div className="flex items-center justify-between text-[9px] font-mono">
        <span className="tracking-widest text-[#f59e0b]">{label}</span>
        <span className={connected ? "text-[#00e5a0]" : "text-[#556677]"}>
          {connected
            ? `${channel.gloveFps.toFixed(0)} FPS`
            : "未连接"}
          {connected && !bendCalibrated && (
            <span className="ml-1.5 text-[#f59e0b]">未标定弯折</span>
          )}
          {connected && bendCalibrated && !orientCalibrated && (
            <span className="ml-1.5 text-[#556677]">朝向未标定</span>
          )}
        </span>
      </div>
      <div className="relative flex-1 min-h-0 rounded-sm border border-[#00f0ff]/15 overflow-hidden bg-[#070a13]">
        <HandModel driveRef={driveRef} side={handKey === "LH" ? "left" : "right"} />
        {!connected && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#070a13]/70">
            <span className="px-2 py-1 rounded-sm bg-[#0a0e1a]/90 border border-[#00f0ff]/15 text-[10px] font-mono text-[#8899aa]">
              这只手没连接
            </span>
          </div>
        )}
        {connected && !bendCalibrated && (
          <div className="absolute bottom-1 left-1 right-1 text-[9px] font-mono text-[#f59e0b]/90 text-center">
            未标定：手指最多弯到 42%，去第 1 步跑一遍向导
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
