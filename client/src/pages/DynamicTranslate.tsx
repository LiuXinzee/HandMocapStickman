import { useDualGloveSerial } from "@/hooks/useDualGloveSerial";
import { getLatestDynamicModel } from "@/lib/datasetStore";
import {
  DYNAMIC_FEATURE_DIM,
  DYNAMIC_SAMPLE_RATE_HZ,
  DYNAMIC_SEQUENCE_LENGTH,
  DYNAMIC_WINDOW_DURATION_MS,
  getLoadedDynamicGestureLabels,
  isDynamicGestureModelLoaded,
  loadDynamicGestureModelFromSaved,
  predictDynamicGesture,
  setActiveDynamicGestureModel,
  type DynamicGestureFrameInput,
  type DynamicGesturePrediction,
} from "@/lib/dynamicGestureModel";
import type { GloveFrame } from "@/lib/gloveProtocol";
import { getCategoryColor, getWordById } from "@/lib/signLanguageVocab";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  Clock3,
  Hand,
  Pause,
  Play,
  PlugZap,
  Radio,
  Trash2,
  Unplug,
} from "lucide-react";

const IDLE_LABEL = "__idle__";
const INFERENCE_INTERVAL_MS = 100;
const BUFFER_RETENTION_MS = DYNAMIC_WINDOW_DURATION_MS + 750;
const STALE_FRAME_TIMEOUT_MS = 300;

interface TranslationEntry {
  label: string;
  word: string;
  confidence: number;
  timestamp: number;
}

interface PredictionView extends DynamicGesturePrediction {
  word: string;
}

interface StableCandidate {
  label: string;
  count: number;
}

interface LiveSequenceWindow {
  leftFrames: DynamicGestureFrameInput[];
  rightFrames: DynamicGestureFrameInput[];
  progress: number;
  stalledHands: string[];
}

function appendFrame(buffer: GloveFrame[], frame: GloveFrame): void {
  buffer.push(frame);
  const cutoff = frame.timestamp - BUFFER_RETENTION_MS;
  let removeCount = 0;
  while (
    removeCount < buffer.length &&
    buffer[removeCount].timestamp < cutoff
  ) {
    removeCount++;
  }
  if (removeCount > 0) buffer.splice(0, removeCount);
}

function frameToModelInput(
  frame: GloveFrame,
  windowStart: number
): DynamicGestureFrameInput {
  return {
    relativeTimeMs: frame.timestamp - windowStart,
    sensor_data: frame.mapped_data,
    quaternion: frame.quaternion,
  };
}

function makeLiveSequenceWindow(
  leftBuffer: readonly GloveFrame[],
  rightBuffer: readonly GloveFrame[],
  leftConnected: boolean,
  rightConnected: boolean,
  now: number
): LiveSequenceWindow | null {
  const activeBuffers = [
    leftConnected ? { hand: "左手", frames: leftBuffer } : null,
    rightConnected ? { hand: "右手", frames: rightBuffer } : null,
  ].filter((entry): entry is { hand: string; frames: readonly GloveFrame[] } =>
    Boolean(entry)
  );

  if (
    activeBuffers.length === 0 ||
    activeBuffers.some(entry => entry.frames.length === 0)
  ) {
    return null;
  }

  const stalledHands = activeBuffers
    .filter(
      entry =>
        now - entry.frames[entry.frames.length - 1].timestamp >
        STALE_FRAME_TIMEOUT_MS
    )
    .map(entry => entry.hand);
  if (stalledHands.length > 0) {
    return { leftFrames: [], rightFrames: [], progress: 0, stalledHands };
  }

  // Use the latest timestamp shared by every connected hand. This avoids
  // padding one hand with stale values when the two serial streams drift.
  const windowEnd = Math.min(
    ...activeBuffers.map(
      entry => entry.frames[entry.frames.length - 1].timestamp
    )
  );
  const coverageMs = Math.min(
    ...activeBuffers.map(entry => windowEnd - entry.frames[0].timestamp)
  );
  const progress = Math.max(
    0,
    Math.min(1, coverageMs / DYNAMIC_WINDOW_DURATION_MS)
  );
  const windowStart = windowEnd - DYNAMIC_WINDOW_DURATION_MS;

  if (progress < 1) {
    return { leftFrames: [], rightFrames: [], progress, stalledHands: [] };
  }

  const selectFrames = (buffer: readonly GloveFrame[]) =>
    buffer
      .filter(
        frame => frame.timestamp >= windowStart && frame.timestamp <= windowEnd
      )
      .map(frame => frameToModelInput(frame, windowStart));

  return {
    leftFrames: leftConnected ? selectFrames(leftBuffer) : [],
    rightFrames: rightConnected ? selectFrames(rightBuffer) : [],
    progress,
    stalledHands: [],
  };
}

function displayLabel(label: string): string {
  if (label === IDLE_LABEL) return "静止 / 过渡";
  return getWordById(label)?.label ?? label;
}

function predictionColor(label: string, confidence: number): string {
  if (confidence < 0.5 || label === IDLE_LABEL) return "#667788";
  const category = getWordById(label)?.category;
  return category ? getCategoryColor(category) : "#00f0ff";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function DynamicTranslate() {
  const leftFramesRef = useRef<GloveFrame[]>([]);
  const rightFramesRef = useRef<GloveFrame[]>([]);
  const inferenceInFlightRef = useRef(false);
  const stableCandidateRef = useRef<StableCandidate>({ label: "", count: 0 });
  const lastAcceptedLabelRef = useRef("");
  const lastAcceptedAtRef = useRef(0);
  const idleUnlockedRef = useRef(true);

  const handleLeftFrame = useCallback((frame: GloveFrame) => {
    appendFrame(leftFramesRef.current, frame);
  }, []);
  const handleRightFrame = useCallback((frame: GloveFrame) => {
    appendFrame(rightFramesRef.current, frame);
  }, []);

  const {
    left: gloveLeft,
    right: gloveRight,
    isSupported,
    anyConnected,
    bothConnected,
    disconnectAll,
  } = useDualGloveSerial({
    baudRate: 921600,
    targetFps: 30,
    onLeftFrame: handleLeftFrame,
    onRightFrame: handleRightFrame,
  });

  const [modelReady, setModelReady] = useState(isDynamicGestureModelLoaded());
  const [modelName, setModelName] = useState("");
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [isTranslating, setIsTranslating] = useState(false);
  const [currentPrediction, setCurrentPrediction] =
    useState<PredictionView | null>(null);
  const [history, setHistory] = useState<TranslationEntry[]>([]);
  const [warmupProgress, setWarmupProgress] = useState(0);
  const [stalledHands, setStalledHands] = useState<string[]>([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.75);
  const [stableFrames, setStableFrames] = useState(4);
  const [repeatCooldownMs, setRepeatCooldownMs] = useState(2000);
  const [message, setMessage] = useState("");

  const confidenceThresholdRef = useRef(confidenceThreshold);
  const stableFramesRef = useRef(stableFrames);
  const repeatCooldownMsRef = useRef(repeatCooldownMs);
  useEffect(() => {
    confidenceThresholdRef.current = confidenceThreshold;
  }, [confidenceThreshold]);
  useEffect(() => {
    stableFramesRef.current = stableFrames;
  }, [stableFrames]);
  useEffect(() => {
    repeatCooldownMsRef.current = repeatCooldownMs;
  }, [repeatCooldownMs]);

  const clearInferenceState = useCallback(() => {
    stableCandidateRef.current = { label: "", count: 0 };
    setCurrentPrediction(null);
    setWarmupProgress(0);
    setStalledHands([]);
  }, []);

  const clearLiveBuffers = useCallback(() => {
    leftFramesRef.current = [];
    rightFramesRef.current = [];
    inferenceInFlightRef.current = false;
    stableCandidateRef.current = { label: "", count: 0 };
  }, []);

  useEffect(() => {
    if (isDynamicGestureModelLoaded()) {
      setModelReady(true);
      setModelName("当前内存模型");
      setMessage("已使用当前加载的动态模型");
      setIsLoadingModel(false);
      return;
    }

    let cancelled = false;

    const loadLatestModel = async () => {
      let pendingModel: Awaited<
        ReturnType<typeof loadDynamicGestureModelFromSaved>
      > | null = null;
      setIsLoadingModel(true);
      try {
        const saved = await getLatestDynamicModel();
        if (saved) {
          pendingModel = await loadDynamicGestureModelFromSaved(saved, {
            activate: false,
          });
          if (cancelled) {
            pendingModel.dispose();
            pendingModel = null;
            return;
          }
          setActiveDynamicGestureModel(pendingModel, saved.labels, {
            sampleRateHz: saved.sampleRateHz ?? DYNAMIC_SAMPLE_RATE_HZ,
            sequenceLength: saved.sequenceLength ?? DYNAMIC_SEQUENCE_LENGTH,
            featureDim: saved.featureDim ?? DYNAMIC_FEATURE_DIM,
          });
          pendingModel = null;
          setModelReady(true);
          setModelName(saved.name);
          setMessage(`已加载最新动态模型「${saved.name}」`);
        } else if (!cancelled) {
          setModelReady(false);
          setMessage("尚无动态 TCN 模型，请先完成训练");
        }
      } catch (error) {
        pendingModel?.dispose();
        if (!cancelled) {
          setModelReady(false);
          setMessage(`动态模型加载失败：${errorMessage(error)}`);
        }
      } finally {
        if (!cancelled) setIsLoadingModel(false);
      }
    };

    void loadLatestModel();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!gloveLeft.isConnected) leftFramesRef.current = [];
    if (!gloveRight.isConnected) rightFramesRef.current = [];
    if (!anyConnected) clearInferenceState();
  }, [
    anyConnected,
    clearInferenceState,
    gloveLeft.isConnected,
    gloveRight.isConnected,
  ]);

  useEffect(() => {
    if (!isTranslating || !anyConnected || !modelReady) return;

    let cancelled = false;
    const infer = async () => {
      if (cancelled || inferenceInFlightRef.current) return;
      inferenceInFlightRef.current = true;

      try {
        const liveWindow = makeLiveSequenceWindow(
          leftFramesRef.current,
          rightFramesRef.current,
          gloveLeft.isConnected,
          gloveRight.isConnected,
          performance.now()
        );
        const nextStalledHands = liveWindow?.stalledHands ?? [];
        setStalledHands(previous =>
          previous.join("|") === nextStalledHands.join("|")
            ? previous
            : nextStalledHands
        );
        const progress = liveWindow?.progress ?? 0;
        setWarmupProgress(progress);
        if (nextStalledHands.length > 0) {
          stableCandidateRef.current = { label: "", count: 0 };
          setCurrentPrediction(null);
          return;
        }
        if (!liveWindow || progress < 1) return;

        // Yield once so interval ticks never stack around a long TF.js call.
        await Promise.resolve();
        if (cancelled) return;

        const result = predictDynamicGesture({
          leftFrames: liveWindow.leftFrames,
          rightFrames: liveWindow.rightFrames,
        });
        if (!result || cancelled) return;

        setCurrentPrediction({ ...result, word: displayLabel(result.label) });

        if (result.confidence < confidenceThresholdRef.current) {
          stableCandidateRef.current = { label: "", count: 0 };
          return;
        }

        const previous = stableCandidateRef.current;
        const nextCandidate =
          previous.label === result.label
            ? { label: result.label, count: previous.count + 1 }
            : { label: result.label, count: 1 };
        stableCandidateRef.current = nextCandidate;
        if (nextCandidate.count < stableFramesRef.current) return;

        if (result.label === IDLE_LABEL) {
          // A stable idle/transition window separates two intentional signs.
          idleUnlockedRef.current = true;
          return;
        }

        const now = Date.now();
        const repeatsLastLabel = result.label === lastAcceptedLabelRef.current;
        if (repeatsLastLabel && !idleUnlockedRef.current) return;
        const duplicateStillCoolingDown =
          repeatsLastLabel &&
          now - lastAcceptedAtRef.current < repeatCooldownMsRef.current;
        if (duplicateStillCoolingDown) return;

        const entry: TranslationEntry = {
          label: result.label,
          word: displayLabel(result.label),
          confidence: result.confidence,
          timestamp: now,
        };
        setHistory(previousHistory => [...previousHistory.slice(-99), entry]);
        lastAcceptedLabelRef.current = result.label;
        lastAcceptedAtRef.current = now;
        idleUnlockedRef.current = false;
        stableCandidateRef.current = { label: "", count: 0 };
      } catch (error) {
        if (!cancelled) {
          setMessage(`动态推理失败：${errorMessage(error)}`);
          setIsTranslating(false);
        }
      } finally {
        inferenceInFlightRef.current = false;
      }
    };

    void infer();
    const intervalId = window.setInterval(() => {
      void infer();
    }, INFERENCE_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [
    anyConnected,
    gloveLeft.isConnected,
    gloveRight.isConnected,
    isTranslating,
    modelReady,
  ]);

  // Serial hooks also close themselves, but this page owns the live buffers and
  // explicitly releases both ports when navigation unmounts the route.
  useEffect(
    () => () => {
      clearLiveBuffers();
      void disconnectAll();
    },
    [clearLiveBuffers, disconnectAll]
  );

  const handleLeftConnection = useCallback(async () => {
    if (gloveLeft.isConnected) {
      leftFramesRef.current = [];
      await gloveLeft.disconnect();
    } else {
      await gloveLeft.connect();
    }
  }, [gloveLeft.connect, gloveLeft.disconnect, gloveLeft.isConnected]);

  const handleRightConnection = useCallback(async () => {
    if (gloveRight.isConnected) {
      rightFramesRef.current = [];
      await gloveRight.disconnect();
    } else {
      await gloveRight.connect();
    }
  }, [gloveRight.connect, gloveRight.disconnect, gloveRight.isConnected]);

  const handleDisconnectAll = useCallback(async () => {
    setIsTranslating(false);
    clearInferenceState();
    clearLiveBuffers();
    await disconnectAll();
  }, [clearInferenceState, clearLiveBuffers, disconnectAll]);

  const toggleTranslation = useCallback(() => {
    if (isTranslating) {
      setIsTranslating(false);
      clearInferenceState();
      return;
    }
    stableCandidateRef.current = { label: "", count: 0 };
    setIsTranslating(true);
    setMessage(
      bothConnected ? "动态识别已启动" : "动态识别已启动，当前使用单手输入"
    );
  }, [bothConnected, clearInferenceState, isTranslating]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    lastAcceptedLabelRef.current = "";
    lastAcceptedAtRef.current = 0;
    idleUnlockedRef.current = true;
  }, []);

  const loadedLabels = modelReady ? getLoadedDynamicGestureLabels() : [];
  const translatedText = useMemo(
    () => history.map(entry => entry.word).join(" "),
    [history]
  );
  const gloveError = gloveLeft.error ?? gloveRight.error;
  const totalFps = gloveLeft.gloveFps + gloveRight.gloveFps;

  return (
    <div
      className="min-h-screen flex flex-col text-[#ccd6e0]"
      style={{ backgroundColor: "#0a0e1a" }}
    >
      <header className="min-h-12 flex flex-col items-stretch gap-2 px-4 py-2 border-b border-[#00f0ff]/15 shrink-0 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[#00f0ff]/20" />
          <span className="whitespace-nowrap text-[11px] font-bold tracking-widest text-[#00f0ff] font-mono sm:text-xs">
            DYNAMIC SIGN TRANSLATOR
          </span>
        </div>

        <nav
          className="flex w-full items-center border border-[#00f0ff]/20 rounded-sm p-0.5 font-mono text-[10px] sm:w-auto"
          aria-label="识别模式"
        >
          <Link
            href="/translate"
            className="flex-1 px-3 py-1 text-center text-[#667788] hover:text-[#00f0ff] transition-colors sm:flex-none"
          >
            静态识别
          </Link>
          <Link
            href="/translate-dynamic"
            aria-current="page"
            className="flex-1 px-3 py-1 text-center bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30 rounded-sm sm:flex-none"
          >
            动态 TCN
          </Link>
        </nav>

        <div className="flex w-full items-center justify-between gap-4 text-[10px] font-mono sm:w-auto sm:justify-end">
          <StatusBadge
            active={modelReady}
            icon={<BrainCircuit className="w-3 h-3" />}
            label={modelReady ? "TCN READY" : "NO MODEL"}
          />
          <StatusBadge
            active={anyConnected}
            icon={<Hand className="w-3 h-3" />}
            label={anyConnected ? `${totalFps} FPS` : "GLOVE OFFLINE"}
          />
          {isTranslating && (
            <span className="flex items-center gap-1 text-[#ff2d7b] animate-pulse">
              <Radio className="w-3 h-3" /> LIVE
            </span>
          )}
        </div>
      </header>

      <main className="flex-1 grid min-h-0 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0 overflow-y-auto px-4 py-6 sm:px-8">
          <div className="mx-auto max-w-4xl min-h-full flex flex-col items-center justify-center gap-6">
            {!modelReady ? (
              <EmptyState
                icon={<BrainCircuit className="w-14 h-14" />}
                title={
                  isLoadingModel ? "正在加载动态模型" : "没有可用的动态模型"
                }
                detail=""
                action={
                  !isLoadingModel ? (
                    <Link
                      href="/train-dynamic"
                      className="cyber-btn px-4 py-2 rounded-sm text-xs inline-flex items-center gap-2"
                    >
                      <BrainCircuit className="w-4 h-4" />
                      前往动态训练
                    </Link>
                  ) : null
                }
              />
            ) : !anyConnected ? (
              <EmptyState
                icon={<Hand className="w-14 h-14" />}
                title="连接手套"
                detail=""
                action={
                  <div className="flex flex-wrap justify-center gap-2">
                    <ConnectionButton
                      hand="左手"
                      connected={gloveLeft.isConnected}
                      connecting={gloveLeft.isConnecting}
                      supported={isSupported}
                      onClick={handleLeftConnection}
                    />
                    <ConnectionButton
                      hand="右手"
                      connected={gloveRight.isConnected}
                      connecting={gloveRight.isConnecting}
                      supported={isSupported}
                      onClick={handleRightConnection}
                    />
                  </div>
                }
              />
            ) : (
              <>
                <div className="w-full flex flex-wrap justify-center gap-3">
                  <GloveStatus
                    hand="左手"
                    connected={gloveLeft.isConnected}
                    connecting={gloveLeft.isConnecting}
                    fps={gloveLeft.gloveFps}
                    onClick={handleLeftConnection}
                  />
                  <GloveStatus
                    hand="右手"
                    connected={gloveRight.isConnected}
                    connecting={gloveRight.isConnecting}
                    fps={gloveRight.gloveFps}
                    onClick={handleRightConnection}
                  />
                </div>

                <section className="w-full text-center py-4" aria-live="polite">
                  {isTranslating && stalledHands.length > 0 ? (
                    <div className="mx-auto max-w-sm space-y-3">
                      <Unplug className="w-10 h-10 mx-auto text-[#ff2d7b]" />
                      <div className="text-sm text-[#ff2d7b]">
                        {stalledHands.join("、")}数据已停流
                      </div>
                    </div>
                  ) : isTranslating && warmupProgress < 1 ? (
                    <div className="mx-auto max-w-sm space-y-3">
                      <Clock3 className="w-10 h-10 mx-auto text-[#00f0ff] animate-pulse" />
                      <div className="text-sm text-[#8899aa]">
                        正在填充时序窗口
                      </div>
                      <ProgressBar value={warmupProgress} />
                      <div className="text-[10px] font-mono text-[#556677]">
                        {(
                          (warmupProgress * DYNAMIC_WINDOW_DURATION_MS) /
                          1000
                        ).toFixed(1)}{" "}
                        / {(DYNAMIC_WINDOW_DURATION_MS / 1000).toFixed(2)} s
                      </div>
                    </div>
                  ) : currentPrediction ? (
                    <PredictionResult
                      prediction={currentPrediction}
                      threshold={confidenceThreshold}
                    />
                  ) : (
                    <div className="space-y-2">
                      <Activity className="w-12 h-12 mx-auto text-[#334455]" />
                      <p className="text-sm text-[#667788]">
                        {isTranslating ? "等待动态手势" : "动态识别已就绪"}
                      </p>
                    </div>
                  )}
                </section>

                <button
                  type="button"
                  onClick={toggleTranslation}
                  className={`cyber-btn px-6 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
                    isTranslating ? "cyber-btn-accent" : ""
                  }`}
                >
                  {isTranslating ? (
                    <>
                      <Pause className="w-4 h-4" /> 停止识别
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" /> 开始识别
                    </>
                  )}
                </button>

                {history.length > 0 && (
                  <section className="w-full border-t border-[#00f0ff]/15 pt-4">
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <span className="text-[9px] font-mono tracking-widest text-[#556677]">
                        DYNAMIC OUTPUT
                      </span>
                      <button
                        type="button"
                        onClick={clearHistory}
                        className="p-1 text-[#556677] hover:text-[#ff2d7b] transition-colors"
                        title="清空识别记录"
                        aria-label="清空识别记录"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    <p className="text-lg leading-relaxed text-[#ccd6e0] break-words">
                      {translatedText}
                    </p>
                  </section>
                )}
              </>
            )}

            {(message || gloveError) && (
              <div
                className={`max-w-xl text-center text-[10px] font-mono ${
                  gloveError ? "text-[#ff2d7b]" : "text-[#00e5a0]"
                }`}
              >
                {gloveError ?? message}
              </div>
            )}
          </div>
        </div>

        <aside className="border-t lg:border-t-0 lg:border-l border-[#00f0ff]/15 overflow-y-auto p-4 space-y-5">
          {currentPrediction && isTranslating && (
            <Section title="CANDIDATES">
              <div className="space-y-1.5">
                {currentPrediction.allProbabilities
                  .slice(0, 5)
                  .map((candidate, index) => (
                    <CandidateRow
                      key={candidate.label}
                      rank={index + 1}
                      label={candidate.label}
                      probability={candidate.probability}
                      threshold={confidenceThreshold}
                    />
                  ))}
              </div>
            </Section>
          )}

          <Section title="INFERENCE">
            <RangeSetting
              label="置信度阈值"
              value={confidenceThreshold}
              min={0.3}
              max={0.95}
              step={0.05}
              display={`${Math.round(confidenceThreshold * 100)}%`}
              onChange={setConfidenceThreshold}
            />
            <RangeSetting
              label="连续确认"
              value={stableFrames}
              min={2}
              max={10}
              step={1}
              display={`${stableFrames} 次`}
              onChange={setStableFrames}
            />
            <RangeSetting
              label="同词冷却"
              value={repeatCooldownMs}
              min={500}
              max={5000}
              step={500}
              display={`${(repeatCooldownMs / 1000).toFixed(1)} s`}
              onChange={setRepeatCooldownMs}
            />
          </Section>

          <Section title="TCN MODEL">
            <DataRow label="MODEL" value={modelName || "ACTIVE"} />
            <DataRow label="CLASSES" value={String(loadedLabels.length)} />
            <DataRow
              label="INPUT"
              value={`${DYNAMIC_SEQUENCE_LENGTH} x ${DYNAMIC_FEATURE_DIM}D`}
            />
            <DataRow label="RATE" value={`${DYNAMIC_SAMPLE_RATE_HZ} Hz`} />
            <DataRow
              label="WINDOW"
              value={`${(DYNAMIC_WINDOW_DURATION_MS / 1000).toFixed(2)} s`}
            />
          </Section>

          {history.length > 0 && (
            <Section title="HISTORY">
              <div className="max-h-44 overflow-y-auto space-y-1">
                {history
                  .slice(-20)
                  .reverse()
                  .map(entry => (
                    <div
                      key={`${entry.timestamp}-${entry.label}`}
                      className="flex items-center justify-between gap-3 text-[9px] font-mono"
                    >
                      <span className="text-[#8899aa] break-words">
                        {entry.word}
                      </span>
                      <span className="text-[#556677] shrink-0">
                        {(entry.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-2">
            <Link
              href="/collect-dynamic"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center"
            >
              动态数据采集
            </Link>
            <Link
              href="/train-dynamic"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center"
            >
              动态模型训练
            </Link>
            {anyConnected && (
              <button
                type="button"
                onClick={handleDisconnectAll}
                className="w-full px-3 py-1.5 text-[10px] font-mono text-[#667788] hover:text-[#ff2d7b] flex items-center justify-center gap-1.5 transition-colors"
              >
                <Unplug className="w-3 h-3" />
                断开全部串口
              </button>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

function StatusBadge({
  active,
  icon,
  label,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <span
      className={`flex items-center gap-1 ${
        active ? "text-[#00e5a0]" : "text-[#556677]"
      }`}
    >
      {icon}
      {label}
    </span>
  );
}

function EmptyState({
  icon,
  title,
  detail,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  detail: string;
  action: React.ReactNode;
}) {
  return (
    <div className="text-center space-y-4">
      <div className="text-[#334455] flex justify-center">{icon}</div>
      <div>
        <p className="text-sm text-[#8899aa]">{title}</p>
        {detail && <p className="text-[10px] text-[#556677] mt-1">{detail}</p>}
      </div>
      {action}
    </div>
  );
}

function ConnectionButton({
  hand,
  connected,
  connecting,
  supported,
  onClick,
}: {
  hand: string;
  connected: boolean;
  connecting: boolean;
  supported: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!supported || connecting}
      className="cyber-btn px-5 py-2.5 rounded-sm text-xs flex items-center gap-2 disabled:opacity-40"
    >
      {connected ? (
        <Unplug className="w-4 h-4" />
      ) : (
        <PlugZap className="w-4 h-4" />
      )}
      {connecting ? `${hand}连接中` : connected ? `断开${hand}` : `连接${hand}`}
    </button>
  );
}

function GloveStatus({
  hand,
  connected,
  connecting,
  fps,
  onClick,
}: {
  hand: string;
  connected: boolean;
  connecting: boolean;
  fps: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={connecting}
      className="cyber-panel w-40 p-3 rounded-sm flex items-center justify-between gap-3 text-left disabled:opacity-50"
      title={connected ? `断开${hand}串口` : `连接${hand}串口`}
    >
      <span className="flex items-center gap-2 min-w-0">
        <Hand
          className={`w-4 h-4 shrink-0 ${
            connected ? "text-[#00e5a0]" : "text-[#556677]"
          }`}
        />
        <span>
          <span className="block text-[10px] text-[#8899aa]">{hand}</span>
          <span className="block text-[8px] font-mono text-[#556677]">
            {connecting ? "CONNECTING" : connected ? `${fps} FPS` : "OFFLINE"}
          </span>
        </span>
      </span>
      {connected ? (
        <Unplug className="w-3 h-3 text-[#556677]" />
      ) : (
        <PlugZap className="w-3 h-3 text-[#556677]" />
      )}
    </button>
  );
}

function PredictionResult({
  prediction,
  threshold,
}: {
  prediction: PredictionView;
  threshold: number;
}) {
  const color = predictionColor(prediction.label, prediction.confidence);
  const accepted = prediction.confidence >= threshold;

  return (
    <div className="space-y-4">
      <div
        className="text-5xl sm:text-6xl font-bold break-words transition-colors duration-200"
        style={{
          color: accepted ? color : "#556677",
          textShadow: accepted ? `0 0 28px ${color}40` : "none",
        }}
      >
        {prediction.word}
      </div>
      <div className="w-52 mx-auto space-y-1.5">
        <ProgressBar value={prediction.confidence} accepted={accepted} />
        <div className="text-[10px] font-mono text-[#667788]">
          置信度 {(prediction.confidence * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
}

function ProgressBar({
  value,
  accepted = true,
}: {
  value: number;
  accepted?: boolean;
}) {
  return (
    <div className="h-2 bg-[#1a2030] rounded-sm overflow-hidden border border-[#00f0ff]/15">
      <div
        className="h-full transition-[width] duration-150"
        style={{
          width: `${Math.max(0, Math.min(1, value)) * 100}%`,
          backgroundColor: accepted ? "#00e5a0" : "#f59e0b",
        }}
      />
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
    <section className="space-y-2.5">
      <div className="flex items-center gap-2 pb-1 border-b border-[#00f0ff]/15">
        <div className="w-1 h-3 bg-[#00f0ff] rounded-full shadow-[0_0_4px_rgba(0,240,255,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[#00f0ff] font-mono">
          {title}
        </span>
      </div>
      {children}
    </section>
  );
}

function CandidateRow({
  rank,
  label,
  probability,
  threshold,
}: {
  rank: number;
  label: string;
  probability: number;
  threshold: number;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-[10px] font-mono">
      <span className="flex items-center gap-2 min-w-0">
        <span className="w-3 text-[#445566]">{rank}.</span>
        <span
          className="truncate"
          style={{ color: probability >= threshold ? "#ccd6e0" : "#667788" }}
        >
          {displayLabel(label)}
        </span>
      </span>
      <span
        className="shrink-0"
        style={{ color: probability >= threshold ? "#00e5a0" : "#556677" }}
      >
        {(probability * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function RangeSetting({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="flex items-center justify-between gap-3 text-[9px] font-mono text-[#667788]">
        <span>{label}</span>
        <span className="text-[#00f0ff]">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={event => onChange(Number(event.target.value))}
        className="w-full h-1 bg-[#1a2030] rounded-sm appearance-none cursor-pointer"
        style={{ accentColor: "#00f0ff" }}
      />
    </label>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 text-[9px] font-mono">
      <span className="text-[#556677] shrink-0">{label}</span>
      <span className="text-[#8899aa] text-right break-words min-w-0">
        {value}
      </span>
    </div>
  );
}
