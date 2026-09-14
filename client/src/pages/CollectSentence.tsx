/*
 * CollectSentence —— 真实**句子**采集页（连续手语，整句连着打不停顿）
 *
 * 与 `/collect-seq`（孤立词）的关系：数据进的是同一个 `sequences` store，
 * 区别只在 `segments` 有几个 —— 多于一个就是句子样本（`isSentenceSample`），
 * 孤立词训练那条链路会自动把它挡掉。所以**后端一行都不用改**。
 *
 * ===== 这一页存在的理由 =====
 *
 * 句子模型现在是纯合成数据训的（`numReal: 0`）：把孤立词录制交叉淡化拼起来。
 * 合成数据里**不存在协同发音** —— 真人连着打时上一个词还没收手就开始过渡到下一个
 * 词的手型。那是孤立词录制在结构上给不出的东西，只能真人连着打才有。
 *
 * ===== 唯一容易做错的地方：时间包络必须和推理端同源 =====
 *
 * 推理端（`/translate` 句子档）喂给模型的一段是：
 *   起点 = 第一次判到"手在动"的那一刻（**不是**按按钮那一刻）
 *   终点 = 砍掉尾部 `SETTLE_MS` 的静止
 *
 * 如果这一页沿用孤立词那套空格键起停（`useSequenceRecorder`），录进去的是
 * 「按键 → 起手前静止 → 句子 → 收手后静止 → 按键」，头尾各多一段推理时**永远见不到**
 * 的静止。所有样本最终都被重采样到定长 T=128，多出来的静止会把每个词在归一化时间轴上
 * 的位置整体挪位 —— 和之前修掉的 127× 接缝尖峰是同一类系统性人造特征：
 * 合成 val、真实 val 都看不出来，只有戴上手套才发现打什么都不准。
 *
 * 所以这一页**不自己写取数逻辑**，而是和推理端共用 `SentenceEnvelope`
 * （它把 `captureSpanMs` / `settleDropMs` 成对算出，调用方没有机会拆开）。
 * 包络相同是**构造上**保证的，不是靠"照抄一遍"保证的。
 *
 * ===== 为什么没有「手动结束」按钮 =====
 *
 * `settleDropMs` 只在 `reason === "settled"` 时才砍尾巴（手动收句砍 0）。
 * 手动收句存下来的样本会带着一段收手后的静止 —— 正是上面那个错误。
 * 所以这一页只接受**自动收句**的样本：停手 800ms 自己收。
 * 打到 12s 上限的那种（`maxLength`，说明一直没停手）也**不保存**，直接提示重录。
 *
 * ===== 摄像头：开着，但视觉**不是特征** =====
 *
 * 部署时没有摄像头，句子模型是纯触觉学生 —— 这一条没变，而且不能变。
 * 所以这里的关键点**一个都不会进模型输入**：`buildSequenceFeatures` 的
 * `includeVision` 默认 false，`synth_sentences.py` 合成时把视觉整个丢掉。
 * 摄像头开着是为了两件只在**训练/标注期**成立的事：
 *
 *   1. **裁剪的前两层需要关键点。** `sequenceTrim` 的第一层（可见段）靠关键点判
 *      "哪一帧起看得见手"。没有视觉时它直接判 `no_vision`，整条录制只剩第三层
 *      （触觉静止段）可裁，而第三层要求弯折两点标定做过。
 *      ⚠ 第二层（到位检测）对句子是**关掉**的 —— 它假设一条录制里只有一个手势，
 *      对句子会把第一个词整个切掉。分流在 `sequenceTrim.trimSpanForExport` 里。
 *
 *   2. **量真实句子里的词间过渡时长。** 这是更重要的一条。CTC 是免对齐的，它不会
 *      告诉你第 2 个词从哪一帧开始；触觉也分不出"在打词"和"在换词"（弯折和压力在
 *      过渡期照样在变）。关键点能看出"手稳住了 → 手在移动 → 手又稳住了"，于是能量出
 *      过渡段的真实时长 —— 那就是合成端 `overlap_ms` 该对齐的量。它现在是
 *      `(100, 250)ms`，**是估的**，没有任何测量支撑。判据在 `signTransitions.ts`。
 *
 * 摄像头没开也照样能采（`vision: true` 只是留出通道，视觉缓冲空时 landmarks 仍是
 * null），只是那些条裁得更少、也不参与 `overlap_ms` 的标定 —— 界面上会照实标出来。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";
import { ArrowLeft, Camera, Circle, Trash2 } from "lucide-react";
import { useGloveFrames, useGloves } from "@/contexts/GloveContext";
import { useHandTracking } from "@/hooks/useHandTracking";
import HandCanvas from "@/components/HandCanvas";
import StepNav from "@/components/StepNav";
import { EnergyChart, ImuHealthLine, Metric, useSampleImuVerdict } from "@/components/SampleQc";
import {
  addSequence,
  deleteSequence,
  getSentencesByTemplate,
  getSequenceStats,
  type SequenceSample,
  type SequenceStats,
} from "@/lib/datasetStore";
import { analyzeSequenceImu } from "@/lib/imuHealth";
import { motionEnergy, visionCoverage } from "@/lib/sequenceFeatures";
import { SequenceWindowBuffer } from "@/lib/sequenceWindow";
import { SentenceEnvelope } from "@/lib/sentenceEnvelope";
import { extractHandLandmarks } from "@/lib/visionLandmarks";
import {
  detectSignTransitions,
  summarizeTransitions,
  transitionVerdict,
  type SignTransitions,
} from "@/lib/signTransitions";
import {
  ARM_TIMEOUT_MS,
  MAX_UTTERANCE_MS,
  SETTLE_MS,
  type CaptureAction,
  type CaptureState,
} from "@/lib/sentenceCapture";
import {
  BATCH1_TEMPLATES,
  RECOMMENDED_PER_TEMPLATE,
  SENTENCE_TEMPLATES,
  templateKey,
} from "@/lib/sentenceTemplates";
import { resolveSentence } from "@/lib/sentenceGrammar";
import { getDisplayLabel } from "@/lib/signLanguageVocab";
import type { GloveFrame } from "@/lib/gloveProtocol";
import { loadBendRange, type BendRange } from "@/lib/bendRange";

/** 状态机的推进间隔，与 `/translate` 句子档一致（那里也是 100ms） */
const TICK_MS = 100;

/** 每词时长的两条质检线，见文件末尾 `perWordVerdict` */
const MIN_MS_PER_WORD = 400;
const MAX_MS_PER_WORD = 2500;

/** 界面要显示的捕获状态。比 `CaptureStatus` 多一个 armed 倒计时 */
interface CaptureUi {
  state: CaptureState;
  elapsedMs: number;
  settleRemainMs: number | null;
  /**
   * 当前生效的收句门限。这一页恒等于 `SETTLE_MS`（`adaptiveSettle` 默认 false），
   * 带上它只是为了进度条别去读常量 —— 常量和实际判据分家的那天，
   * 条子会填满后卡在 100% 干等
   */
  settleMs: number;
  /** armed 状态下距离放弃还剩多久；其余状态 null */
  armRemainMs: number | null;
}

const IDLE_UI: CaptureUi = {
  state: "idle",
  elapsedMs: 0,
  settleRemainMs: null,
  settleMs: SETTLE_MS,
  armRemainMs: null,
};

export default function CollectSentence() {
  // 缓冲与推理端同样大小：装不下的部分会被静默丢掉（开头几秒消失），比截断更难发现
  const bufRef = useRef<SequenceWindowBuffer | null>(null);
  if (bufRef.current === null) {
    // `vision: true` 只在这一页开（推理端保持默认 false）。它只是**留出通道** ——
    // 视觉不进特征，用途见文件头。摄像头不开时视觉缓冲是空的，取出来的样本
    // landmarks 仍是 null，与这个开关加进来之前逐位相同
    bufRef.current = new SequenceWindowBuffer({
      bufferMs: MAX_UTTERANCE_MS,
      vision: true,
    });
  }
  // 和 `/translate` **同一个类**。取数逻辑（span 与 dropTail 成对）在它里面，
  // 这一页不要再写一份 tick + snapshotAll
  const envRef = useRef<SentenceEnvelope | null>(null);
  if (envRef.current === null) {
    envRef.current = new SentenceEnvelope(bufRef.current);
  }

  // 连接在第 1 步（/mocap）建立、住在 GloveProvider 里，本页只订阅帧、不碰串口。
  // 必须走全速回调：轮询 latestFrameRef 会按 targetFps 节流丢帧，而协同发音的
  // 过渡细节正好丢在这里
  const onLeftFrame = useCallback((f: GloveFrame) => {
    bufRef.current?.push("left", f);
  }, []);
  const onRightFrame = useCallback((f: GloveFrame) => {
    bufRef.current?.push("right", f);
  }, []);
  const { anyConnected: gloveConnected } = useGloves();
  useGloveFrames(onLeftFrame, onRightFrame);

  const {
    videoRef,
    isRunning: cameraRunning,
    error: cameraError,
    handResults,
    handResultsRef,
    fps: cameraFps,
    startTracking,
    stopTracking,
  } = useHandTracking({ maxHands: 2 });

  /*
   * 视觉采样循环 —— 与 `useSequenceRecorder.sampleVision` 同一套做法：
   * 按**对象引用**去重，MediaPipe 没出新结果就不重复记（rAF 约 60Hz、视觉约 30Hz，
   * 不去重会把每一帧记两遍，重采样时那些重复点会让最近邻偏向后半拍）。
   *
   * 一直跑，不只在录制时跑：缓冲是环形的，`pushVision` 自己按 `bufferMs` 裁，
   * 内存有界。而只在 capturing 时跑会漏掉**起手那一段** —— 状态机是"见到动作才转
   * capturing"，那之前的帧已经在缓冲里了，视觉缺了这一段的话第一层裁剪正好裁不准。
   *
   * 时间戳必须用 `performance.now()`：与手套帧同一个时钟，重采样是按时间最近邻对齐的。
   */
  const lastVisionObjRef = useRef<unknown>(null);
  useEffect(() => {
    if (!cameraRunning) return;
    let raf = 0;
    const loop = () => {
      const r = handResultsRef.current ?? null;
      if (r && r !== lastVisionObjRef.current) {
        lastVisionObjRef.current = r;
        bufRef.current?.pushVision(
          performance.now(),
          extractHandLandmarks(r, "Left"),
          extractHandLandmarks(r, "Right")
        );
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [cameraRunning, handResultsRef]);

  /**
   * 弯折两点标定 —— 动作闸门用它把两只手的量程归一化（实测左右差得很多）。
   * 传给 `env.tick` 的必须是**镜像之前**的量程，这一页从头到尾不做镜像（见下面 save）。
   */
  const bendRangesRef = useRef<{ LH: BendRange | null; RH: BendRange | null } | null>(
    null
  );
  if (bendRangesRef.current === null) {
    bendRangesRef.current = { LH: loadBendRange("LH"), RH: loadBendRange("RH") };
  }

  const [showAll, setShowAll] = useState(false);
  const [template, setTemplate] = useState<readonly string[]>(BATCH1_TEMPLATES[0]);
  /**
   * 起手那一刻锁定的句型。保存时用**这一份**，不用 state ——
   * 录制中途换选句型的话，state 已经变了而手打的还是原来那句，
   * 存下去就是一条标签写错的样本（而且看不出来）
   */
  const activeTemplateRef = useRef<readonly string[]>(template);

  const [ui, setUi] = useState<CaptureUi>(IDLE_UI);
  const armedAtRef = useRef(0);
  const [stats, setStats] = useState<SequenceStats | null>(null);
  const [samples, setSamples] = useState<SequenceSample[]>([]);
  const [lastSample, setLastSample] = useState<SequenceSample | null>(null);
  const [lastSampleId, setLastSampleId] = useState<number | null>(null);
  const [message, setMessage] = useState("");
  const [warn, setWarn] = useState("");

  const templates = showAll ? SENTENCE_TEMPLATES : BATCH1_TEMPLATES;
  const tKey = templateKey(template);

  const refreshStats = useCallback(async () => {
    try {
      setStats(await getSequenceStats());
    } catch (e) {
      console.error("[CollectSent] 统计刷新失败", e);
    }
  }, []);

  const refreshSamples = useCallback(async (labels: readonly string[]) => {
    try {
      const list = await getSentencesByTemplate(labels);
      list.sort((a, b) => b.timestamp - a.timestamp);
      setSamples(list);
    } catch (e) {
      console.error("[CollectSent] 读取该句型样本失败", e);
    }
  }, []);

  useEffect(() => {
    void refreshStats();
  }, [refreshStats]);

  useEffect(() => {
    void refreshSamples(template);
    setLastSample(null);
    setLastSampleId(null);
  }, [template, refreshSamples]);

  // ===== 保存 =====

  const saveUtterance = useCallback(
    async (act: CaptureAction) => {
      const env = envRef.current;
      if (!env || act.kind !== "decode") return;
      const labels = activeTemplateRef.current;

      /*
       * **只收自动收句的样本。** `settleDropMs` 只在 settled 时砍尾部静止；
       * maxLength（一直没停手，12s 到顶）砍 0，存下来就是一条带着尾部静止、
       * 而且大概率中间还打串了的样本 —— 宁可重录
       */
      if (act.reason !== "settled") {
        setMessage("");
        setWarn(
          act.reason === "maxLength"
            ? `到了 ${MAX_UTTERANCE_MS / 1000}s 上限还没停手 —— 这条没保存。打完把手停住 ${SETTLE_MS}ms 会自动收句。`
            : "这一句是手动收的，没保存 —— 手动收句会把收手后的静止一起录进去，和推理时的包络不一致。"
        );
        return;
      }

      const raw = env.take(act, labels[0]);
      if (!raw) {
        setMessage("");
        setWarn("这一段太短，取不出可用的数据段（不足 4 帧），没保存。");
        return;
      }

      /*
       * 这里是这一页唯一真正"造数据"的地方。
       *
       * `snapshotAll` 只会给回一个覆盖整段的 segment。换成句型的完整标签序列，
       * `startFrame/endFrame` 全填整段：**这不是真词边界**，只是占位。
       * CTC 只读 label 的顺序（`load_dataset.py` 的 `label_sequence` 忽略帧下标），
       * 所以填整段是合法的训练目标；但别拿它当对齐结果用，也别写代码依赖它。
       *
       * **不做 `normalizeHandedness`。** 推理端在 `take()` 之后自己镜像（模型是右手
       * 口径训的），采集端不能做 —— 数据集存的必须是原始录制，镜像发生在建特征那一层。
       * 存进库里的镜像不可逆，做了就永久污染。
       */
      const sample: SequenceSample = {
        ...raw,
        segments: labels.map((label) => ({
          label,
          startFrame: 0,
          endFrame: raw.frameCount,
        })),
        primaryLabel: labels[0],
        // 与录制器一致地事后算一遍 IMU 健康度（`snapshotAll` 不算）：
        // 陀螺漂移下帧数/时长/能量曲线全都正常，四元数通道却已经废了
        imuHealth: {
          left: raw.leftImu
            ? analyzeSequenceImu(raw.leftImu, raw.frameCount, raw.timestamps)
            : null,
          right: raw.rightImu
            ? analyzeSequenceImu(raw.rightImu, raw.frameCount, raw.timestamps)
            : null,
        },
      };

      try {
        const id = await addSequence(sample);
        setLastSample(sample);
        setLastSampleId(id);
        setWarn("");
        setMessage(
          `已保存：${sample.frameCount} 帧 / ${Math.round(sample.durationMs)}ms（末尾静止已掐掉）`
        );
        await refreshStats();
        await refreshSamples(labels);
      } catch (e) {
        console.error("[CollectSent] 保存失败", e);
        setWarn(`保存失败：${e}`);
      }
    },
    [refreshStats, refreshSamples]
  );

  // ===== 状态机 =====

  const handleAction = useCallback(
    (act: CaptureAction) => {
      if (act.kind === "abort") {
        setMessage("");
        setWarn(
          act.reason === "armTimeout"
            ? `等了 ${ARM_TIMEOUT_MS / 1000} 秒没见到动作，这一句取消了。再按空格。`
            : "这一段没录到任何动作，没保存。"
        );
        return;
      }
      if (act.kind === "decode") void saveUtterance(act);
    },
    [saveUtterance]
  );

  // 定时器闭包只读这个 ref，所以换 handler 不会重建定时器
  // （重建本身不会丢状态——状态在 envRef 里——但会让这一跳的节奏抖一下）
  const actionRef = useRef(handleAction);
  actionRef.current = handleAction;

  useEffect(() => {
    if (!gloveConnected) return;
    const id = window.setInterval(() => {
      const env = envRef.current;
      if (!env) return;
      const now = performance.now();
      // 量程传**镜像之前**的：镜像后左手数据配的是右手量程，动作能量的分母就错了
      const act = env.tick(now, bendRangesRef.current ?? {});
      const s = env.status();
      setUi((prev) => {
        const next: CaptureUi = {
          state: s.state,
          elapsedMs: s.elapsedMs,
          settleRemainMs: s.settleRemainMs,
          settleMs: s.settleMs,
          armRemainMs:
            s.state === "armed"
              ? Math.max(0, ARM_TIMEOUT_MS - (now - armedAtRef.current))
              : null,
        };
        // 10Hz 无条件 setState 会让整页每秒重渲染 10 次。取 100ms 粒度比较：
        // 数字本来也只显示到 0.1s，看不出差别
        const same =
          prev.state === next.state &&
          Math.round(prev.elapsedMs / 100) === Math.round(next.elapsedMs / 100) &&
          Math.round((prev.settleRemainMs ?? -1) / 100) ===
            Math.round((next.settleRemainMs ?? -1) / 100) &&
          Math.round((prev.armRemainMs ?? -1) / 100) ===
            Math.round((next.armRemainMs ?? -1) / 100);
        return same ? prev : next;
      });
      if (act.kind !== "none") actionRef.current(act);
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [gloveConnected]);

  /*
   * 手套掉线时必须把当前这句丢掉。上面那个定时器的存活条件就是 `gloveConnected`，
   * 掉线后它不再跑 —— 状态机会**停在** capturing/settling 上，界面上一直显示
   * "正在录 3.2s"，而其实一帧都没进来。重连后接着录只会拼出一条断成两半的句子。
   */
  useEffect(() => {
    if (gloveConnected) return;
    envRef.current?.cancel();
    setUi(IDLE_UI);
  }, [gloveConnected]);

  const active = ui.state !== "idle";

  const arm = useCallback(() => {
    if (!gloveConnected) {
      setWarn("手套未连接 —— 触觉是部署时唯一的输入，必须连");
      return;
    }
    // 起手那一刻把句型锁下来，之后换选不影响这一条
    activeTemplateRef.current = template;
    armedAtRef.current = performance.now();
    setMessage("");
    setWarn("");
    setLastSample(null);
    setLastSampleId(null);
    // `arm` 里会清缓冲：不清的话上一句的尾巴会被接到这一句前面一起存库
    envRef.current?.arm(armedAtRef.current);
    setUi({ ...IDLE_UI, state: "armed", armRemainMs: ARM_TIMEOUT_MS });
  }, [gloveConnected, template]);

  const cancel = useCallback(() => {
    envRef.current?.cancel();
    setUi(IDLE_UI);
    setMessage("");
    setWarn("这一句取消了，没保存。");
  }, []);

  // 空格：没在录就起手，在录就取消（**没有手动收句** —— 见文件头）
  useEffect(() => {
    const isTyping = (t: EventTarget | null) =>
      t instanceof HTMLElement &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat || isTyping(e.target)) return;
      e.preventDefault();
      if (active) cancel();
      else arm();
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [active, arm, cancel]);

  const pickTemplate = useCallback(
    (t: readonly string[]) => {
      // 录制中换句型 = 放弃当前这句。留着录会存成一条标签写错的样本
      if (active) {
        envRef.current?.cancel();
        setUi(IDLE_UI);
        setWarn("换了句型，刚才那一句取消了。");
      }
      setTemplate(t);
    },
    [active]
  );

  const handleDelete = useCallback(
    async (id?: number) => {
      if (id === undefined) return;
      await deleteSequence(id);
      if (id === lastSampleId) {
        setLastSample(null);
        setLastSampleId(null);
      }
      await refreshStats();
      await refreshSamples(template);
      setMessage("已删除该条");
    },
    [lastSampleId, refreshStats, refreshSamples, template]
  );

  // ===== 派生 =====

  const energy = useMemo(
    () => (lastSample ? motionEnergy(lastSample) : null),
    [lastSample]
  );
  const imu = useSampleImuVerdict(lastSample);

  const transitions = useMemo(
    () => (lastSample ? detectSignTransitions(lastSample) : null),
    [lastSample]
  );
  const coverage = useMemo(
    () => (lastSample ? visionCoverage(lastSample) : 0),
    [lastSample]
  );

  /**
   * 这个句型已有样本的过渡时长汇总 —— **`overlap_ms` 的实测依据就是这个数**。
   *
   * `summarizeTransitions` 只吃分段与词数对得上的条，所以 `usable < total` 是常态。
   * 界面上必须把两个数都显示出来：只报中位数会让人以为全部样本都量到了。
   */
  const transitionSummary = useMemo(
    () =>
      summarizeTransitions(
        samples.map((s) => ({
          transitions: detectSignTransitions(s),
          wordCount: s.segments.length,
        }))
      ),
    [samples]
  );

  const batchDone = useMemo(() => {
    if (!stats) return 0;
    return BATCH1_TEMPLATES.reduce(
      (n, t) =>
        n + Math.min(RECOMMENDED_PER_TEMPLATE, stats.sentenceCounts[templateKey(t)] ?? 0),
      0
    );
  }, [stats]);
  const batchTarget = BATCH1_TEMPLATES.length * RECOMMENDED_PER_TEMPLATE;

  const done = stats?.sentenceCounts[tKey] ?? 0;
  const preview = resolveSentence([...template]);

  /*
   * `h-screen` 而不是 `min-h-screen`：min-h 下这一层会被内容顶高，`flex-1` 那行跟着
   * 长高，三列的 `overflow-y-auto` 就永远不触发 —— 滚的是整个文档。摄像头挪到中栏
   * 最上面之后这件事更明显：不钉死视口高度，按钮会被整页滚动推到屏幕外。
   */
  return (
    <div className="h-screen overflow-hidden flex flex-col" style={{ backgroundColor: "var(--hud-page)" }}>
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
          <span
            className="text-xs font-bold tracking-widest"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              // 紫色是时序链路的标识色，与 /collect-seq、/train-seq、/translate 的
              // MODE 开关一致。青色留给静态链路
              color: "var(--hud-violet)",
            }}
          >
            SENTENCE COLLECTION
          </span>
          <span className="text-[9px] text-[var(--hud-dim)] font-mono ml-2">
            连续手语 · 整句自动收句
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-4 text-[10px] font-mono">
            {/* 视觉不是特征，所以这里**不能**用红色报警色 —— 摄像头关着照样能采，
                只是裁得少、不参与 overlap_ms 标定。用灰色表示"少了个工具"而不是"坏了" */}
            <span className={cameraRunning ? "text-[var(--hud-ok)]" : "text-[var(--hud-dim)]"}>
              <Camera className="w-3 h-3 inline mr-1" />
              {cameraRunning ? `CAM ${cameraFps}fps` : "CAM OFF"}
            </span>
            <span className="text-[var(--hud-dim)]">
              第一批{" "}
              <span className="text-[var(--hud-violet)]">
                {batchDone}/{batchTarget}
              </span>
            </span>
            <span className="text-[var(--hud-dim)]">
              句子样本总数{" "}
              <span className="text-[var(--hud-accent)]">{stats?.sentenceCount ?? 0}</span>
            </span>
          </div>
          <StepNav />
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 左：句型表 */}
        <div className="w-72 border-r border-[#1677ff]/15 overflow-y-auto shrink-0 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
              Templates ({templates.length})
            </span>
            <button
              onClick={() => setShowAll((v) => !v)}
              className="cyber-btn px-2 py-0.5 rounded-sm text-[9px] font-mono"
            >
              {/* 两个数都从表里取。写死过一次「12 句」，加句型时忘了改，
                  按钮上写 12 而列表里列 15 —— 没人会怀疑按钮的文案 */}
              {showAll
                ? `只看第一批 ${BATCH1_TEMPLATES.length} 句`
                : `看全表 ${SENTENCE_TEMPLATES.length} 句`}
            </button>
          </div>
          {!showAll && (
            <div className="text-[9px] text-[var(--hud-faint)] font-mono leading-relaxed">
              第一批只采这 {BATCH1_TEMPLATES.length} 句，每句{" "}
              {RECOMMENDED_PER_TEMPLATE} 条。剩下的句型仍由合成数据兜着。
            </div>
          )}
          {templates.map((t) => {
            const k = templateKey(t);
            const n = stats?.sentenceCounts[k] ?? 0;
            const selected = k === tKey;
            return (
              <button
                key={k}
                onClick={() => pickTemplate(t)}
                className={`w-full text-left px-2.5 py-2 rounded-sm border transition-all duration-150 ${
                  selected
                    ? "border-[#7c3aed]/60 bg-[#7c3aed]/10"
                    : "border-[#1677ff]/10 hover:border-[#1677ff]/30"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] text-[var(--hud-text)]">
                    {t.map((w) => getDisplayLabel(w)).join(" ")}
                  </span>
                  <span
                    className="text-[9px] font-mono shrink-0"
                    style={{
                      color:
                        n >= RECOMMENDED_PER_TEMPLATE
                          ? "var(--hud-ok)"
                          : n > 0
                            ? "var(--hud-warn)"
                            : "var(--hud-faint)",
                    }}
                  >
                    {n}/{RECOMMENDED_PER_TEMPLATE}
                  </span>
                </div>
                <div className="text-[9px] text-[var(--hud-dim)] mt-0.5">
                  {resolveSentence([...t]).text}
                </div>
              </button>
            );
          })}
        </div>

        {/*
          中：捕获状态。
          ⚠ 面板顺序与 /collect-seq 的中栏对齐 —— **摄像头在最上，采集按钮在下面**。
          两页是同一个动作的两种粒度（一条录制装一个词 / 装一句），采集时人的视线在
          画面和按钮之间来回；两页顺序不一致的话，切页就要重新找按钮在哪。
        */}
        <div className="flex-1 min-h-0 flex flex-col p-4 gap-3 overflow-y-auto">
          {/* 摄像头预览。**视觉不是特征**（见文件头），这块存在的意义是让人确认
              双手在画面里 —— 手出画的那些帧关键点是 NaN，裁剪第一层会把它们当"看不见手" */}
          <div className="cyber-panel p-3 rounded-sm">
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-[10px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
                  Vision — 只做标注，不进特征
                </span>
                <div className="text-[9px] text-[var(--hud-faint)] font-mono mt-0.5">
                  关键点用于裁剪的可见段判据 + 量词间过渡时长（校准合成端 overlap_ms）
                </div>
              </div>
              <button
                className="cyber-btn px-2 py-1 rounded-sm text-[10px] shrink-0"
                onClick={() => (cameraRunning ? stopTracking() : startTracking())}
              >
                {cameraRunning ? "停止摄像头" : "启动摄像头"}
              </button>
            </div>
            {/* 宽高只是摄像头 metadata 到达前的占位值，取 useHandTracking 申请的
                1280×720，避免启动瞬间从 4:3 跳成 16:9 */}
            <HandCanvas
              handResults={handResults}
              videoWidth={1280}
              videoHeight={720}
              videoRef={videoRef}
            />
            {cameraError && (
              <div className="text-[10px] text-[var(--hud-err)] mt-1 font-mono">
                {cameraError}
              </div>
            )}
            {!cameraRunning && !cameraError && (
              <div className="text-[9px] text-[var(--hud-dim)] mt-1 font-mono leading-relaxed">
                摄像头关着也能采 —— 触觉是唯一进模型的通道。代价是这些条只能靠触觉静止段
                裁剪（需要弯折两点标定做过），而且不参与过渡时长的标定。
              </div>
            )}
          </div>

          {/* 四态大字 —— 采集时人在看屏幕做动作，必须一眼看出"系统开始录了没有"。
              放在 Target 上面：/collect-seq 的中栏也是摄像头紧跟着控制面板，
              要打哪句可以边等边看，而"录了没有"是等不起的 */}
          <div className="cyber-panel p-4 rounded-sm">
            <CaptureBanner ui={ui} />
            <div className="flex items-center gap-3 mt-3">
              <button
                onClick={active ? cancel : arm}
                disabled={!gloveConnected}
                className={`flex-1 py-3 rounded-sm border text-[12px] font-mono tracking-wider transition-all ${
                  active
                    ? "border-[var(--hud-err)] bg-[#e11d48]/20 text-[var(--hud-err)]"
                    : "border-[#7c3aed]/50 text-[var(--hud-violet)] hover:bg-[#7c3aed]/10"
                }`}
              >
                {active ? (
                  "取消这一句（空格）"
                ) : (
                  <>
                    <Circle className="w-3 h-3 inline mr-1" />
                    准备（或按空格）
                  </>
                )}
              </button>
            </div>
            {!gloveConnected && (
              <div className="text-[10px] text-[var(--hud-warn)] mt-2 font-mono">
                手套未连接 —— 触觉是部署时唯一的输入，必须连
              </div>
            )}
            {warn && (
              <div className="text-[10px] text-[var(--hud-warn)] mt-2 font-mono">{warn}</div>
            )}
            {message && (
              <div className="text-[10px] text-[var(--hud-ok)] mt-2 font-mono">{message}</div>
            )}
          </div>

          <div className="cyber-panel p-3 rounded-sm">
            <div className="text-[10px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
              Target
            </div>
            <div className="text-[20px] text-[var(--hud-violet)] tracking-wide">
              {template.map((w) => getDisplayLabel(w)).join("  ")}
            </div>
            <div className="text-[12px] text-[var(--hud-text)] mt-0.5">{preview.text}</div>
            <div className="text-[9px] text-[var(--hud-dim)] font-mono mt-1">
              {template.join(" ")} · 已采 {done}/{RECOMMENDED_PER_TEMPLATE}
              {preview.rule === null && "（顺句规则没命中，上面是原样拼接）"}
            </div>
            <div className="text-[9px] text-[var(--hud-dim)] mt-2 leading-relaxed">
              连着打完整句，词之间不要停顿。打完把手停住不动，{SETTLE_MS}ms
              后自动收句入库。
            </div>
          </div>

          {/* 录后质检 */}
          {lastSample && energy && (
            <div className="cyber-panel p-3 rounded-sm">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
                  Last Take — Motion Energy
                </span>
                <button
                  onClick={() => handleDelete(lastSampleId ?? undefined)}
                  className="cyber-btn px-2 py-1 rounded-sm text-[10px] text-[var(--hud-err)]"
                >
                  <Trash2 className="w-3 h-3 inline mr-1" />
                  删除这条
                </button>
              </div>
              <EnergyChart energy={energy} />
              <div className="grid grid-cols-4 gap-2 mt-2 text-[10px] font-mono">
                <Metric label="帧数" value={String(lastSample.frameCount)} />
                <Metric
                  label="时长"
                  value={`${Math.round(lastSample.durationMs)}ms`}
                />
                <Metric
                  label="双手"
                  value={`${lastSample.leftSensor ? "L" : "-"}${
                    lastSample.rightSensor ? "R" : "-"
                  }`}
                />
                <PerWordMetric sample={lastSample} />
              </div>
              <PerWordNote sample={lastSample} />
              {imu && <ImuHealthLine imu={imu} />}
              {transitions && (
                <TransitionLine
                  transitions={transitions}
                  wordCount={lastSample.segments.length}
                  coverage={coverage}
                />
              )}
            </div>
          )}

          <div className="text-[9px] text-[var(--hud-faint)] font-mono leading-relaxed">
            视觉只做标注：关键点一个都不进模型输入（合成数据也一律丢掉视觉），
            部署端没有摄像头。它的两个用途是给裁剪补上可见段判据、以及量出词间过渡的
            真实时长去校准合成端的 overlap_ms（现在那个 100~250ms 是估的）。
          </div>
        </div>

        {/* 右：这个句型已有的样本 */}
        <div className="w-72 border-l border-[#1677ff]/15 overflow-y-auto shrink-0 p-3 space-y-2">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            {template.map((w) => getDisplayLabel(w)).join(" ")} — {samples.length} 条
          </div>
          {samples.length === 0 && (
            <div className="text-[10px] text-[var(--hud-faint)] font-mono">还没有样本</div>
          )}
          {/* 过渡时长汇总。`usable/total` 必须一起显示：只报中位数会被当成全体的结论 */}
          {transitionSummary.total > 0 && (
            <div className="px-2 py-1.5 rounded-sm border border-[#7c3aed]/20 bg-[#7c3aed]/5">
              <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
                词间过渡（overlap_ms 依据）
              </div>
              {transitionSummary.usable === 0 ? (
                <div className="text-[9px] text-[var(--hud-warn)] font-mono mt-0.5 leading-relaxed">
                  {transitionSummary.total} 条里 0 条可用 —— 没有视觉，或分段与词数对不上
                </div>
              ) : (
                <>
                  <div className="text-[11px] font-mono text-[var(--hud-violet)] mt-0.5">
                    中位 {Math.round(transitionSummary.medianMoveMs)}ms
                    <span className="text-[var(--hud-dim)] text-[9px] ml-1">
                      （p10 {Math.round(transitionSummary.p10MoveMs)} / p90{" "}
                      {Math.round(transitionSummary.p90MoveMs)}）
                    </span>
                  </div>
                  <div className="text-[9px] text-[var(--hud-dim)] font-mono mt-0.5">
                    {transitionSummary.usable}/{transitionSummary.total} 条可用 · 合成端现用
                    100~250ms
                  </div>
                </>
              )}
            </div>
          )}
          {samples.map((s) => {
            const v = perWordVerdict(s);
            return (
              <div
                key={s.id}
                className="flex items-center justify-between px-2 py-1.5 rounded-sm border border-[#1677ff]/10"
              >
                <div className="text-[10px] font-mono text-[var(--hud-text)]">
                  #{s.id}
                  <span className="text-[var(--hud-dim)] ml-2">
                    {s.frameCount}f / {Math.round(s.durationMs)}ms
                  </span>
                  <div className="text-[9px]" style={{ color: v.color }}>
                    {Math.round(v.perWordMs)}ms/词
                    {s.origin === "synthesized" && (
                      <span className="text-[var(--hud-warn)] ml-1">合成</span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(s.id)}
                  className="text-[var(--hud-err)] hover:text-[var(--hud-err-hover)]"
                  title="删除这条"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/**
 * 四态大字。**这块是这一页的主界面** —— 采集时人的手在动、眼睛在屏幕上，
 * 需要毫不含糊地知道系统在哪个状态：
 *   armed     等你起手（带 8s 倒计时，超时会放弃）
 *   capturing 正在录，显示已录时长
 *   settling  停手中，显示还差多久收句
 */
function CaptureBanner({ ui }: { ui: CaptureUi }) {
  if (ui.state === "idle") {
    return (
      <div>
        <div className="text-[24px] text-[var(--hud-dim)]">待机</div>
        <div className="text-[10px] text-[var(--hud-dim)] font-mono mt-1">
          按空格 → 见到你起手自动开始录
        </div>
      </div>
    );
  }
  if (ui.state === "armed") {
    return (
      <div>
        <div className="text-[24px] text-[var(--hud-warn)]">等你起手</div>
        <div className="text-[10px] text-[var(--hud-warn)] font-mono mt-1">
          {((ui.armRemainMs ?? 0) / 1000).toFixed(1)}s 内没动作就放弃这一句
        </div>
      </div>
    );
  }
  if (ui.state === "capturing") {
    return (
      <div>
        <div className="text-[24px] text-[var(--hud-err)]">
          正在录 {(ui.elapsedMs / 1000).toFixed(1)}s
        </div>
        <div className="text-[10px] text-[var(--hud-dim)] font-mono mt-1">
          连着打完，词之间不要停。上限 {MAX_UTTERANCE_MS / 1000}s
        </div>
      </div>
    );
  }
  // settling
  const remain = ui.settleRemainMs ?? 0;
  return (
    <div>
      <div className="text-[24px] text-[var(--hud-ok)]">
        停手中… {(remain / 1000).toFixed(1)}s 后收句
      </div>
      {/* 进度条读状态机报的实时门限，不读常量：采集页 `adaptive` 是 false、
          门限恒等于 `SETTLE_MS`，所以这一行行为不变；但常量和实际判据一旦分家，
          条子会填满后卡在 100% 干等 */}
      <div className="h-1.5 mt-2 rounded-sm bg-[#16a34a]/15 overflow-hidden">
        <div
          className="h-full bg-[var(--hud-ok)] transition-[width] duration-100"
          style={{ width: `${(1 - remain / ui.settleMs) * 100}%` }}
        />
      </div>
      <div className="text-[10px] text-[var(--hud-dim)] font-mono mt-1">
        又动一下就回到"正在录"（词间过渡不算句尾）
      </div>
    </div>
  );
}

/**
 * 句子专属质检：**每词时长**。
 *
 * 太短 = 打太快，词都糊在一起，很可能漏词。
 * 太长 = 中间停顿了。真实连续手语不该有停顿，合成数据里也没有
 * （`crossfade_sequences` 是**吃掉**静止而不是插入静止），
 * 停顿会让模型学到"词之间有静止"，反过来把真正连着打的句子判错。
 */
function perWordVerdict(s: SequenceSample): {
  perWordMs: number;
  color: string;
  note: string | null;
} {
  const n = Math.max(1, s.segments.length);
  const perWordMs = s.durationMs / n;
  if (perWordMs < MIN_MS_PER_WORD) {
    return {
      perWordMs,
      color: "var(--hud-err)",
      note: `每词只有 ${Math.round(perWordMs)}ms —— 打太快了，很可能漏词，建议删掉重录`,
    };
  }
  if (perWordMs > MAX_MS_PER_WORD) {
    return {
      perWordMs,
      color: "var(--hud-warn)",
      note: `每词 ${Math.round(perWordMs)}ms —— 词之间大概停顿了。连续手语不该有停顿（合成数据里也没有），模型会学错`,
    };
  }
  return { perWordMs, color: "var(--hud-accent)", note: null };
}

function PerWordMetric({ sample }: { sample: SequenceSample }) {
  const v = perWordVerdict(sample);
  return (
    <div>
      <div className="text-[var(--hud-dim)] uppercase text-[8px]">每词时长</div>
      <div style={{ color: v.color }}>{Math.round(v.perWordMs)}ms</div>
    </div>
  );
}

/**
 * 过渡时长读数 —— **这一页开摄像头的主要产出**。
 *
 * `perWordVerdict` 那一行是 `durationMs / 词数`，是个平均值，看不出词粘在一起还是
 * 中间顿了；这一行是从关键点真量出来的分段。两个都留着：没有视觉时只有前者。
 *
 * 颜色只区分"量到了 / 没量到 / 分段对不上"，**不给过渡时长本身判好坏** ——
 * 现在还没有"多长才对"的基准，那个基准正是要靠这些数字建立起来的。
 */
function TransitionLine({
  transitions,
  wordCount,
  coverage,
}: {
  transitions: SignTransitions;
  wordCount: number;
  coverage: number;
}) {
  const v = transitionVerdict(transitions, wordCount);
  const color =
    v.kind === "match" ? "var(--hud-violet)" : v.kind === "unmeasured" ? "var(--hud-dim)" : "var(--hud-warn)";
  return (
    <div className="text-[9px] font-mono mt-1.5 leading-relaxed" style={{ color }}>
      {v.note}
      {v.kind !== "unmeasured" && (
        <span className="text-[var(--hud-faint)] ml-1">
          · 视觉覆盖 {Math.round(coverage * 100)}%
        </span>
      )}
    </div>
  );
}

function PerWordNote({ sample }: { sample: SequenceSample }) {
  const v = perWordVerdict(sample);
  if (!v.note) return null;
  return (
    <div className="text-[9px] font-mono mt-1.5" style={{ color: v.color }}>
      {v.note}
    </div>
  );
}
