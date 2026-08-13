import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Camera,
  Database,
  Play,
  Square,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import HandCanvas from "@/components/HandCanvas";
import { useDualGloveSerial } from "@/hooks/useDualGloveSerial";
import { useDynamicGestureRecorder } from "@/hooks/useDynamicGestureRecorder";
import { useHandTracking } from "@/hooks/useHandTracking";
import {
  addDynamicSequence,
  deleteDynamicSequencesByLabel,
  getDynamicDatasetStats,
  type DynamicDatasetStats,
} from "@/lib/datasetStore";
import {
  SIGN_CATEGORIES,
  SIGN_VOCABULARY,
  getCategoryColor,
  type SignWord,
} from "@/lib/signLanguageVocab";

const DEFAULT_CLIP_DURATION_MS = 2560;
const DEFAULT_TARGET_SEQUENCES = 20;
const DEFAULT_COUNTDOWN_SECONDS = 3;
const MIN_CAPTURE_RATE_HZ = 20;
const MIN_CAPTURE_COVERAGE = 0.8;
const IDLE_WORD: SignWord = {
  id: "__idle__",
  label: "静止/过渡",
  pinyin: "idle",
  category: "transition",
  description: "自然静止与词语之间的过渡动作",
};
const DYNAMIC_VOCABULARY: SignWord[] = [IDLE_WORD, ...SIGN_VOCABULARY];

interface ParticipatingHands {
  left: boolean;
  right: boolean;
}

function validateGloveStream(
  frames: Array<{ relativeTimeMs: number }>,
  durationMs: number,
  handLabel: string
): string | null {
  const minimumFrames = Math.max(
    5,
    Math.floor((durationMs / 1000) * MIN_CAPTURE_RATE_HZ)
  );
  if (frames.length < minimumFrames) {
    return `${handLabel}帧数不足（${frames.length}/${minimumFrames}）`;
  }
  const coverageMs =
    frames[frames.length - 1].relativeTimeMs - frames[0].relativeTimeMs;
  if (coverageMs < durationMs * MIN_CAPTURE_COVERAGE) {
    return `${handLabel}数据仅覆盖 ${(coverageMs / 1000).toFixed(2)}s`;
  }
  return null;
}

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function DynamicCollect() {
  const recorder = useDynamicGestureRecorder();
  const {
    left: gloveLeft,
    right: gloveRight,
    isSupported: gloveSupported,
    anyConnected,
    disconnectAll,
  } = useDualGloveSerial({
    baudRate: 921600,
    onLeftFrame: recorder.recordGloveFrame,
    onRightFrame: recorder.recordGloveFrame,
  });
  const {
    videoRef,
    isLoading: cameraLoading,
    isRunning: cameraRunning,
    error: cameraError,
    handResults,
    handResultsRef,
    fps: cameraFps,
    startTracking,
    stopTracking,
  } = useHandTracking({ maxHands: 2 });

  const [selectedWord, setSelectedWord] = useState<SignWord | null>(null);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [clipDurationMs, setClipDurationMs] = useState(
    DEFAULT_CLIP_DURATION_MS
  );
  const [targetSequences, setTargetSequences] = useState(
    DEFAULT_TARGET_SEQUENCES
  );
  const [countdownSeconds, setCountdownSeconds] = useState(
    DEFAULT_COUNTDOWN_SECONDS
  );
  const [countdown, setCountdown] = useState<number | null>(null);
  const [stats, setStats] = useState<DynamicDatasetStats | null>(null);
  const [message, setMessage] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  const sessionIdRef = useRef(createSessionId());
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoFinishTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previousVisionResultRef = useRef(handResults);
  const participatingHandsRef = useRef<ParticipatingHands | null>(null);

  const refreshStats = useCallback(async () => {
    try {
      setStats(await getDynamicDatasetStats());
    } catch (error) {
      console.error("[DynamicCollect] Failed to load stats:", error);
      setMessage("动态数据集读取失败");
    }
  }, []);

  useEffect(() => {
    void refreshStats();
  }, [refreshStats]);

  useEffect(() => {
    if (
      recorder.isRecording &&
      handResults !== previousVisionResultRef.current
    ) {
      recorder.recordVisionFrame(handResults);
    }
    previousVisionResultRef.current = handResults;
  }, [handResults, recorder.isRecording, recorder.recordVisionFrame]);

  const clearCaptureTimers = useCallback(() => {
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    if (autoFinishTimerRef.current) {
      clearTimeout(autoFinishTimerRef.current);
      autoFinishTimerRef.current = null;
    }
  }, []);

  const finishAndSave = useCallback(async () => {
    clearCaptureTimers();
    setCountdown(null);
    const sequence = recorder.finish();
    if (!sequence) return;
    const participatingHands = participatingHandsRef.current;
    participatingHandsRef.current = null;

    if (sequence.durationMs < 500 || !participatingHands) {
      setMessage("片段过短或没有有效参与手，未保存");
      return;
    }
    const qualityErrors = [
      participatingHands.left
        ? validateGloveStream(sequence.leftFrames, sequence.durationMs, "左手")
        : null,
      participatingHands.right
        ? validateGloveStream(sequence.rightFrames, sequence.durationMs, "右手")
        : null,
    ].filter((error): error is string => Boolean(error));
    if (qualityErrors.length > 0) {
      setMessage(`${qualityErrors.join("；")}，片段未保存`);
      return;
    }

    setIsSaving(true);
    try {
      await addDynamicSequence(sequence);
      await refreshStats();
      const word = DYNAMIC_VOCABULARY.find(item => item.id === sequence.label);
      setMessage(
        `已保存 ${word?.label ?? sequence.label}：${(sequence.durationMs / 1000).toFixed(2)}s，L ${sequence.leftFrames.length} / R ${sequence.rightFrames.length}`
      );
    } catch (error) {
      console.error("[DynamicCollect] Failed to save sequence:", error);
      setMessage("动态片段保存失败");
    } finally {
      setIsSaving(false);
    }
  }, [clearCaptureTimers, recorder.finish, refreshStats]);

  const beginRecording = useCallback(() => {
    if (!selectedWord) return;
    const participatingHands = {
      left: gloveLeft.isConnected,
      right: gloveRight.isConnected,
    };
    if (!participatingHands.left && !participatingHands.right) return;
    const started = recorder.start({
      label: selectedWord.id,
      sessionId: sessionIdRef.current,
      targetDurationMs: clipDurationMs,
    });
    if (!started) return;
    participatingHandsRef.current = participatingHands;

    if (cameraRunning) {
      recorder.recordVisionFrame(handResultsRef.current);
    }
    setCountdown(null);
    setMessage(`正在录制 ${selectedWord.label}`);
    autoFinishTimerRef.current = setTimeout(() => {
      void finishAndSave();
    }, clipDurationMs);
  }, [
    clipDurationMs,
    cameraRunning,
    finishAndSave,
    gloveLeft.isConnected,
    gloveRight.isConnected,
    handResultsRef,
    recorder.recordVisionFrame,
    recorder.start,
    selectedWord,
  ]);

  const startCountdown = useCallback(() => {
    if (!selectedWord || !anyConnected || isSaving || recorder.isRecording) {
      return;
    }

    clearCaptureTimers();
    if (countdownSeconds === 0) {
      beginRecording();
      return;
    }

    let remaining = countdownSeconds;
    setCountdown(remaining);
    setMessage(`准备录制 ${selectedWord.label}`);
    countdownTimerRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        if (countdownTimerRef.current) {
          clearInterval(countdownTimerRef.current);
          countdownTimerRef.current = null;
        }
        beginRecording();
      } else {
        setCountdown(remaining);
      }
    }, 1000);
  }, [
    anyConnected,
    beginRecording,
    clearCaptureTimers,
    countdownSeconds,
    isSaving,
    recorder.isRecording,
    selectedWord,
  ]);

  const cancelCapture = useCallback(() => {
    clearCaptureTimers();
    setCountdown(null);
    recorder.cancel();
    participatingHandsRef.current = null;
    setMessage("已取消当前片段");
  }, [clearCaptureTimers, recorder.cancel]);

  useEffect(() => {
    const participatingHands = participatingHandsRef.current;
    const lostParticipatingHand =
      recorder.isRecording &&
      Boolean(
        (participatingHands?.left && !gloveLeft.isConnected) ||
          (participatingHands?.right && !gloveRight.isConnected)
      );
    if (
      lostParticipatingHand ||
      ((recorder.isRecording || countdown !== null) && !anyConnected)
    ) {
      cancelCapture();
    }
  }, [
    anyConnected,
    cancelCapture,
    countdown,
    gloveLeft.isConnected,
    gloveRight.isConnected,
    recorder.isRecording,
  ]);

  useEffect(() => {
    return () => {
      clearCaptureTimers();
      recorder.cancel();
    };
  }, [clearCaptureTimers, recorder.cancel]);

  const handleDeleteLabel = useCallback(
    async (label: string) => {
      const word = DYNAMIC_VOCABULARY.find(item => item.id === label);
      if (!confirm(`确定删除“${word?.label ?? label}”的所有动态片段？`)) {
        return;
      }
      await deleteDynamicSequencesByLabel(label);
      await refreshStats();
      setMessage(`已删除 ${word?.label ?? label} 的动态片段`);
    },
    [refreshStats]
  );

  const captureActive = recorder.isRecording || countdown !== null;
  const captureLocked = captureActive || isSaving;
  const canStart = Boolean(selectedWord) && anyConnected && !captureLocked;
  const filteredWords =
    selectedCategory === "all"
      ? DYNAMIC_VOCABULARY
      : DYNAMIC_VOCABULARY.filter(word => word.category === selectedCategory);
  const selectedCount = selectedWord
    ? (stats?.labelCounts[selectedWord.id] ?? 0)
    : 0;
  const progress = Math.min(
    100,
    targetSequences > 0 ? (selectedCount / targetSequences) * 100 : 0
  );
  const videoWidth = videoRef.current?.videoWidth || 640;
  const videoHeight = videoRef.current?.videoHeight || 480;

  return (
    <div className="min-h-screen flex flex-col bg-[#0a0e1a] text-[#ccd6e0]">
      <header className="min-h-12 px-4 py-2 border-b border-[#00f0ff]/15 flex flex-col items-stretch gap-2 shrink-0 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[#00f0ff]/20" />
          <span className="whitespace-nowrap text-[11px] font-bold tracking-widest text-[#00f0ff] sm:text-xs">
            DYNAMIC DATA CAPTURE
          </span>
        </div>
        <nav
          className="flex w-full border border-[#00f0ff]/20 rounded-sm overflow-hidden sm:w-auto"
          aria-label="采集模式"
        >
          <Link
            href="/collect"
            className="flex-1 px-3 py-1 text-center text-[9px] font-mono text-[#778899] hover:text-[#00f0ff] sm:flex-none"
          >
            静态
          </Link>
          <span className="flex-1 px-3 py-1 text-center text-[9px] font-mono bg-[#00f0ff]/15 text-[#00f0ff] sm:flex-none">
            动态
          </span>
        </nav>
        <div className="flex w-full items-center justify-between gap-4 text-[9px] font-mono sm:w-auto sm:justify-end">
          <span className={cameraRunning ? "text-[#00e5a0]" : "text-[#556677]"}>
            CAM {cameraRunning ? `${cameraFps}fps` : "OFF"}
          </span>
          <span className={anyConnected ? "text-[#00e5a0]" : "text-[#556677]"}>
            GLOVE {gloveLeft.gloveFps + gloveRight.gloveFps}Hz
          </span>
          <span className="text-[#00f0ff]">
            {stats?.totalSequences ?? 0} SEQUENCES
          </span>
        </div>
      </header>

      <main className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)_300px]">
        <aside className="border-b lg:border-b-0 lg:border-r border-[#00f0ff]/15 p-3 overflow-y-auto max-h-72 lg:max-h-none">
          <div className="text-[9px] font-mono text-[#556677] uppercase mb-2">
            Category
          </div>
          <div className="flex flex-wrap gap-1 mb-4">
            <button
              type="button"
              onClick={() => setSelectedCategory("all")}
              disabled={captureLocked}
              className={`px-2 py-1 border rounded-sm text-[9px] font-mono ${selectedCategory === "all" ? "text-[#00f0ff] border-[#00f0ff]/50 bg-[#00f0ff]/10" : "text-[#667788] border-[#334455]"}`}
            >
              全部
            </button>
            {SIGN_CATEGORIES.map(category => (
              <button
                key={category.id}
                type="button"
                onClick={() => setSelectedCategory(category.id)}
                disabled={captureLocked}
                className={`px-2 py-1 border rounded-sm text-[9px] ${selectedCategory === category.id ? "bg-white/5" : "border-[#334455]"}`}
                style={{
                  color: category.color,
                  borderColor:
                    selectedCategory === category.id
                      ? `${category.color}70`
                      : undefined,
                }}
              >
                {category.label}
              </button>
            ))}
          </div>

          <div className="space-y-1">
            {filteredWords.map(word => {
              const count = stats?.labelCounts[word.id] ?? 0;
              const active = selectedWord?.id === word.id;
              return (
                <button
                  key={word.id}
                  type="button"
                  disabled={captureLocked}
                  onClick={() => setSelectedWord(word)}
                  className={`w-full px-2.5 py-2 border rounded-sm flex items-center justify-between text-left transition-colors ${active ? "bg-[#00f0ff]/10 border-[#00f0ff]/45" : "border-transparent hover:border-[#00f0ff]/15"}`}
                >
                  <span>
                    <span
                      className="text-[11px]"
                      style={{ color: getCategoryColor(word.category) }}
                    >
                      {word.label}
                    </span>
                    <span className="ml-2 text-[8px] text-[#556677]">
                      {word.pinyin}
                    </span>
                  </span>
                  <span className="text-[9px] font-mono text-[#667788]">
                    {count}
                  </span>
                </button>
              );
            })}
          </div>
        </aside>

        <section className="relative min-w-0 min-h-[360px] lg:min-h-0 bg-black/20">
          {cameraRunning ? (
            <HandCanvas
              handResults={handResults}
              videoWidth={videoWidth}
              videoHeight={videoHeight}
              showVideo
              videoRef={videoRef}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <button
                type="button"
                onClick={startTracking}
                disabled={cameraLoading}
                className="cyber-btn px-5 py-2 text-[11px] flex items-center gap-2"
              >
                <Camera className="w-4 h-4" />
                {cameraLoading ? "启动中..." : "启动摄像头"}
              </button>
            </div>
          )}

          {countdown !== null && (
            <div className="absolute inset-0 bg-[#0a0e1a]/60 flex items-center justify-center">
              <div className="text-8xl font-mono font-bold text-[#00f0ff]">
                {countdown}
              </div>
            </div>
          )}

          {recorder.isRecording && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-[#ff2d7b]/90 flex items-center gap-2 rounded-sm">
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-[11px] font-mono font-bold text-white">
                REC {(recorder.durationMs / 1000).toFixed(2)} /{" "}
                {(clipDurationMs / 1000).toFixed(2)}s
              </span>
            </div>
          )}

          <div className="absolute bottom-3 left-3 px-2 py-1 bg-[#0a0e1a]/85 border border-[#00f0ff]/20 text-[9px] font-mono">
            VISION {handResults?.landmarks.length ?? 0} | L{" "}
            {recorder.counts.left} | R {recorder.counts.right}
          </div>
        </section>

        <aside className="border-t lg:border-t-0 lg:border-l border-[#00f0ff]/15 p-4 overflow-y-auto space-y-4">
          <div className="flex items-center gap-2 border-b border-[#00f0ff]/15 pb-2">
            <Zap className="w-4 h-4 text-[#00f0ff]" />
            <span className="text-[10px] font-bold tracking-widest text-[#00f0ff]">
              SEQUENCE
            </span>
          </div>

          {selectedWord ? (
            <div>
              <div
                className="text-3xl font-bold text-center"
                style={{ color: getCategoryColor(selectedWord.category) }}
              >
                {selectedWord.label}
              </div>
              <div className="text-[10px] text-[#778899] text-center mt-1">
                {selectedWord.description}
              </div>
              <div className="mt-3 h-2 bg-[#18202c] overflow-hidden rounded-sm">
                <div
                  className="h-full bg-[#00e5a0] transition-[width]"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[9px] font-mono text-[#667788]">
                <span>{selectedCount} 段</span>
                <span>{targetSequences} 目标</span>
              </div>
            </div>
          ) : (
            <div className="py-6 text-center text-[10px] text-[#667788]">
              未选择词语
            </div>
          )}

          <div className="space-y-2 border-y border-[#00f0ff]/10 py-3">
            <ParameterRow label="片段时长(ms)">
              <input
                type="number"
                min={500}
                max={10000}
                step={20}
                value={clipDurationMs}
                disabled={captureLocked}
                onChange={event =>
                  setClipDurationMs(
                    Math.min(
                      10000,
                      Math.max(500, Number(event.target.value) || 2560)
                    )
                  )
                }
                className="w-20 bg-[#141c28] border border-[#00f0ff]/25 px-2 py-1 text-right text-[#00f0ff]"
              />
            </ParameterRow>
            <ParameterRow label="倒计时(s)">
              <input
                type="number"
                min={0}
                max={10}
                value={countdownSeconds}
                disabled={captureLocked}
                onChange={event =>
                  setCountdownSeconds(
                    Math.min(10, Math.max(0, Number(event.target.value) || 0))
                  )
                }
                className="w-20 bg-[#141c28] border border-[#00f0ff]/25 px-2 py-1 text-right text-[#00f0ff]"
              />
            </ParameterRow>
            <ParameterRow label="目标片段">
              <input
                type="number"
                min={1}
                max={500}
                value={targetSequences}
                disabled={captureLocked}
                onChange={event =>
                  setTargetSequences(
                    Math.min(500, Math.max(1, Number(event.target.value) || 1))
                  )
                }
                className="w-20 bg-[#141c28] border border-[#00f0ff]/25 px-2 py-1 text-right text-[#00f0ff]"
              />
            </ParameterRow>
          </div>

          {!captureActive ? (
            <button
              type="button"
              onClick={startCountdown}
              disabled={!canStart}
              className="w-full cyber-btn px-4 py-2 flex items-center justify-center gap-2 text-[11px] disabled:opacity-35"
            >
              <Play className="w-4 h-4" />
              开始动态采集
            </button>
          ) : recorder.isRecording ? (
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => void finishAndSave()}
                className="cyber-btn px-3 py-2 flex items-center justify-center gap-1 text-[10px]"
              >
                <Square className="w-3.5 h-3.5" />
                停止并保存
              </button>
              <button
                type="button"
                onClick={cancelCapture}
                className="cyber-btn px-3 py-2 flex items-center justify-center gap-1 text-[10px] text-[#ff2d7b]"
              >
                <X className="w-3.5 h-3.5" />
                取消
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={cancelCapture}
              className="w-full cyber-btn px-4 py-2 flex items-center justify-center gap-2 text-[11px] text-[#ff2d7b]"
            >
              <X className="w-4 h-4" />
              取消倒计时
            </button>
          )}

          {message && (
            <div className="text-[9px] font-mono text-center text-[#00e5a0]">
              {message}
            </div>
          )}

          <div className="space-y-2 pt-2 border-t border-[#00f0ff]/10">
            <div className="grid grid-cols-2 gap-2">
              <DeviceButton
                label={
                  gloveLeft.isConnected
                    ? `左手 ${gloveLeft.gloveFps}Hz`
                    : "连接左手"
                }
                active={gloveLeft.isConnected}
                disabled={
                  captureLocked || gloveLeft.isConnecting || !gloveSupported
                }
                onClick={
                  gloveLeft.isConnected
                    ? () => void gloveLeft.disconnect()
                    : () => void gloveLeft.connect()
                }
              />
              <DeviceButton
                label={
                  gloveRight.isConnected
                    ? `右手 ${gloveRight.gloveFps}Hz`
                    : "连接右手"
                }
                active={gloveRight.isConnected}
                disabled={
                  captureLocked || gloveRight.isConnecting || !gloveSupported
                }
                onClick={
                  gloveRight.isConnected
                    ? () => void gloveRight.disconnect()
                    : () => void gloveRight.connect()
                }
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <DeviceButton
                label={cameraRunning ? "关闭摄像头" : "启动摄像头"}
                active={cameraRunning}
                disabled={captureLocked || cameraLoading}
                onClick={cameraRunning ? stopTracking : startTracking}
              />
              <DeviceButton
                label="断开手套"
                active={false}
                disabled={captureLocked || !anyConnected}
                onClick={() => void disconnectAll()}
              />
            </div>
            {(cameraError || gloveLeft.error || gloveRight.error) && (
              <div className="text-[8px] text-[#ff2d7b]">
                {cameraError || gloveLeft.error || gloveRight.error}
              </div>
            )}
          </div>

          <div className="pt-2 border-t border-[#00f0ff]/10">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-3.5 h-3.5 text-[#00f0ff]" />
              <span className="text-[9px] font-mono text-[#00f0ff]">
                DATASET
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center mb-3">
              <Metric label="片段" value={stats?.totalSequences ?? 0} />
              <Metric label="类别" value={stats?.labels.length ?? 0} />
              <Metric label="帧" value={stats?.totalFrames ?? 0} />
            </div>
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {stats?.labels.map(label => {
                const word = DYNAMIC_VOCABULARY.find(item => item.id === label);
                return (
                  <div
                    key={label}
                    className="flex items-center justify-between px-2 py-1 bg-white/[0.02]"
                  >
                    <span className="text-[9px] text-[#8899aa]">
                      {word?.label ?? label}
                    </span>
                    <span className="flex items-center gap-2">
                      <span className="text-[9px] font-mono text-[#556677]">
                        {stats.labelCounts[label]}
                      </span>
                      <button
                        type="button"
                        disabled={captureLocked}
                        onClick={() => void handleDeleteLabel(label)}
                        title="删除该词语动态片段"
                        className="text-[#556677] hover:text-[#ff2d7b]"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <Link
            href="/train-dynamic"
            className="w-full cyber-btn px-3 py-2 flex items-center justify-center text-[10px]"
          >
            前往 TCN 训练
          </Link>
        </aside>
      </main>
    </div>
  );
}

function ParameterRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center justify-between text-[9px] font-mono text-[#778899]">
      <span>{label}</span>
      {children}
    </label>
  );
}

function DeviceButton({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="border px-2 py-1.5 text-[9px] font-mono disabled:opacity-35"
      style={{
        color: active ? "#00e5a0" : "#667788",
        borderColor: active ? "#00e5a050" : "#334455",
      }}
    >
      {label}
    </button>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="border border-[#00f0ff]/10 px-1 py-2">
      <div className="text-[7px] text-[#556677] font-mono">{label}</div>
      <div className="text-[11px] text-[#00f0ff] font-mono mt-1">{value}</div>
    </div>
  );
}
