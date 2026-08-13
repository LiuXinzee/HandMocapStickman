import {
  deleteDynamicModel,
  getAllDynamicModels,
  getAllDynamicSequences,
  getDynamicDatasetStats,
  saveDynamicModel,
  type DynamicDatasetStats,
  type SavedDynamicModelRecord,
} from "@/lib/datasetStore";
import {
  DYNAMIC_FEATURE_DIM,
  DYNAMIC_SAMPLE_RATE_HZ,
  DYNAMIC_SEQUENCE_LENGTH,
  isDynamicGestureModelLoaded,
  loadDynamicGestureModelFromSaved,
  serializeDynamicGestureModel,
  setActiveDynamicGestureModel,
  trainDynamicGestureModel,
  type DynamicTrainingProgress,
} from "@/lib/dynamicGestureModel";
import { getWordById } from "@/lib/signLanguageVocab";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  Play,
  Square,
  Trash2,
  Upload,
} from "lucide-react";

const MIN_SEQUENCE_COUNT = 4;
const IDLE_LABEL = "__idle__";

function displayDynamicLabel(label: string): string {
  if (label === IDLE_LABEL) return "静止/过渡";
  return getWordById(label)?.label ?? label;
}

export default function DynamicTrain() {
  const mountedRef = useRef(true);
  const trainingAbortRef = useRef<AbortController | null>(null);
  const modelLoadRequestRef = useRef(0);
  const [stats, setStats] = useState<DynamicDatasetStats | null>(null);
  const [models, setModels] = useState<SavedDynamicModelRecord[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<DynamicTrainingProgress | null>(
    null
  );
  const [history, setHistory] = useState<DynamicTrainingProgress[]>([]);
  const [message, setMessage] = useState("");
  const [activeModelLoaded, setActiveModelLoaded] = useState(
    isDynamicGestureModelLoaded()
  );

  const [epochs, setEpochs] = useState(60);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(0.0005);
  const [dropoutRate, setDropoutRate] = useState(0.15);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      modelLoadRequestRef.current += 1;
      trainingAbortRef.current?.abort();
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [nextStats, nextModels] = await Promise.all([
        getDynamicDatasetStats(),
        getAllDynamicModels(),
      ]);
      if (!mountedRef.current) return;
      setStats(nextStats);
      setModels(nextModels.sort((a, b) => b.createdAt - a.createdAt));
      setActiveModelLoaded(isDynamicGestureModelLoaded());
    } catch (error) {
      if (mountedRef.current) {
        setMessage(`✗ 无法读取动态数据: ${getErrorMessage(error)}`);
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const canTrain =
    (stats?.totalSequences ?? 0) >= MIN_SEQUENCE_COUNT &&
    (stats?.labels.length ?? 0) >= 2 &&
    Boolean(stats?.labels.includes(IDLE_LABEL));

  const handleTrain = useCallback(async () => {
    if (trainingAbortRef.current) return;
    const controller = new AbortController();
    trainingAbortRef.current = controller;
    modelLoadRequestRef.current += 1;
    let pendingModel: { dispose: () => unknown } | null = null;
    let savedModelId: number | null = null;
    setIsTraining(true);
    setProgress(null);
    setHistory([]);
    setMessage("正在加载动态手势片段...");

    try {
      const sequences = await getAllDynamicSequences();
      if (sequences.length < MIN_SEQUENCE_COUNT) {
        throw new Error(`至少需要 ${MIN_SEQUENCE_COUNT} 个动态片段`);
      }

      const labels = new Set(sequences.map(sequence => sequence.label));
      if (labels.size < 2) {
        throw new Error("至少需要 2 个不同的动态词语类别");
      }
      if (!labels.has(IDLE_LABEL)) {
        throw new Error("请先采集“静止/过渡”类别，实时识别需要它区分词语边界");
      }

      const labelCounts = new Map<string, number>();
      for (const sequence of sequences) {
        labelCounts.set(
          sequence.label,
          (labelCounts.get(sequence.label) ?? 0) + 1
        );
      }
      const sparseLabels = Array.from(labelCounts.entries())
        .filter(([, count]) => count < 2)
        .map(([label]) => displayDynamicLabel(label));
      if (sparseLabels.length > 0) {
        throw new Error(`以下类别少于 2 个片段: ${sparseLabels.join("、")}`);
      }

      setMessage(
        `训练中... (${sequences.length} 个片段, ${labels.size} 类, ${DYNAMIC_SEQUENCE_LENGTH} 帧窗口)`
      );

      const result = await trainDynamicGestureModel(
        sequences,
        {
          epochs,
          batchSize,
          learningRate,
          dropoutRate,
        },
        nextProgress => {
          if (!controller.signal.aborted && mountedRef.current) {
            setProgress(nextProgress);
            setHistory(previous => [...previous, nextProgress]);
          }
        },
        controller.signal
      );
      pendingModel = result.model;
      if (controller.signal.aborted || !mountedRef.current) {
        pendingModel.dispose();
        pendingModel = null;
        return;
      }

      const timestamp = new Date()
        .toISOString()
        .slice(0, 16)
        .replace(/[:-]/g, "");
      const saved = await serializeDynamicGestureModel(
        result.model,
        result.labels,
        result.accuracy,
        `tcn_${timestamp}`,
        result.validationLoss
      );
      if (controller.signal.aborted || !mountedRef.current) {
        pendingModel.dispose();
        pendingModel = null;
        return;
      }
      savedModelId = await saveDynamicModel(saved);
      if (controller.signal.aborted || !mountedRef.current) {
        await deleteDynamicModel(savedModelId);
        savedModelId = null;
        pendingModel.dispose();
        pendingModel = null;
        return;
      }

      setActiveDynamicGestureModel(result.model, result.labels, result.config);
      pendingModel = null;
      savedModelId = null;
      setActiveModelLoaded(true);

      const finalProgress = result.history.at(-1);
      const validationText =
        finalProgress && result.validationSequenceCount > 0
          ? `，验证准确率 ${(finalProgress.valAccuracy * 100).toFixed(1)}%`
          : "，当前仅一个采集会话，未生成独立验证集";
      setMessage(
        `✓ 动态 TCN 训练完成：准确率 ${(result.accuracy * 100).toFixed(1)}%${validationText}`
      );
      await refresh();
    } catch (error) {
      if (savedModelId !== null) {
        try {
          await deleteDynamicModel(savedModelId);
        } catch (rollbackError) {
          console.error(
            "[DynamicTrain] Failed to roll back an uncommitted model:",
            rollbackError
          );
        }
      }
      pendingModel?.dispose();
      if (mountedRef.current) {
        if (controller.signal.aborted || getErrorName(error) === "AbortError") {
          setMessage("已取消动态训练");
        } else {
          setMessage(`✗ 训练失败: ${getErrorMessage(error)}`);
        }
      }
    } finally {
      if (trainingAbortRef.current === controller) {
        trainingAbortRef.current = null;
      }
      if (mountedRef.current) setIsTraining(false);
    }
  }, [batchSize, dropoutRate, epochs, learningRate, refresh]);

  const cancelTraining = useCallback(() => {
    trainingAbortRef.current?.abort();
    setMessage("正在停止动态训练...");
  }, []);

  const handleLoadModel = useCallback(
    async (model: SavedDynamicModelRecord) => {
      const requestId = ++modelLoadRequestRef.current;
      try {
        const loaded = await loadDynamicGestureModelFromSaved(model, {
          activate: false,
        });
        if (!mountedRef.current || requestId !== modelLoadRequestRef.current) {
          loaded.dispose();
          return;
        }
        setActiveDynamicGestureModel(loaded, model.labels, {
          sampleRateHz: model.sampleRateHz ?? DYNAMIC_SAMPLE_RATE_HZ,
          sequenceLength: model.sequenceLength ?? DYNAMIC_SEQUENCE_LENGTH,
          featureDim: model.featureDim ?? DYNAMIC_FEATURE_DIM,
        });
        setActiveModelLoaded(true);
        setMessage(`✓ 已加载动态模型 "${model.name}"`);
      } catch (error) {
        if (mountedRef.current && requestId === modelLoadRequestRef.current) {
          setMessage(`✗ 加载失败: ${getErrorMessage(error)}`);
        }
      }
    },
    []
  );

  const handleDeleteModel = useCallback(
    async (id: number) => {
      if (!confirm("确定删除这个动态 TCN 模型？")) return;
      try {
        await deleteDynamicModel(id);
        await refresh();
      } catch (error) {
        setMessage(`✗ 删除失败: ${getErrorMessage(error)}`);
      }
    },
    [refresh]
  );

  const totalDuration = useMemo(
    () => formatDuration(stats?.totalDurationMs ?? 0),
    [stats?.totalDurationMs]
  );

  return (
    <div
      className="min-h-screen flex flex-col"
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
            DYNAMIC TCN TRAINING
          </span>
        </div>

        <nav
          className="flex w-full items-center border border-[#00f0ff]/20 rounded-sm p-0.5 font-mono text-[10px] sm:w-auto"
          aria-label="训练模式"
        >
          <Link
            href="/train"
            className="flex-1 px-3 py-1 text-center text-[#667788] hover:text-[#00f0ff] transition-colors sm:flex-none"
          >
            静态训练
          </Link>
          <Link
            href="/train-dynamic"
            aria-current="page"
            className="flex-1 px-3 py-1 text-center bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30 rounded-sm sm:flex-none"
          >
            动态 TCN
          </Link>
        </nav>

        <div className="flex w-full justify-end text-[10px] font-mono sm:w-auto sm:min-w-36">
          {activeModelLoaded ? (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <BrainCircuit className="w-3 h-3" />
              TCN MODEL ACTIVE
            </span>
          ) : (
            <span className="text-[#556677]">NO ACTIVE TCN</span>
          )}
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        <aside className="w-full lg:w-72 border-b lg:border-b-0 lg:border-r border-[#00f0ff]/15 overflow-y-auto p-4 space-y-4 shrink-0">
          <Section title="DYNAMIC DATASET">
            <DataRow
              label="CLIPS"
              value={String(stats?.totalSequences ?? 0)}
              color="#00f0ff"
            />
            <DataRow
              label="CLASSES"
              value={String(stats?.labels.length ?? 0)}
              color="#00e5a0"
            />
            <DataRow
              label="RAW FRAMES"
              value={String(stats?.totalFrames ?? 0)}
              color="#8899aa"
            />
            <DataRow label="DURATION" value={totalDuration} color="#f59e0b" />
            <DataRow
              label="MODEL INPUT"
              value={`${DYNAMIC_SEQUENCE_LENGTH} x ${DYNAMIC_FEATURE_DIM}`}
              color="#da77f2"
            />
            {stats && stats.labels.length > 0 && (
              <div className="mt-2 space-y-0.5 max-h-40 overflow-y-auto">
                {stats.labels.map(label => (
                  <div
                    key={label}
                    className="flex justify-between gap-2 text-[9px] font-mono"
                  >
                    <span className="text-[#8899aa] truncate">
                      {displayDynamicLabel(label)}
                    </span>
                    <span className="text-[#556677] shrink-0">
                      {stats.labelCounts[label]} clips
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Section>

          <Section title="PARAMETERS">
            <ParamInput
              label="Epochs"
              value={epochs}
              onChange={setEpochs}
              min={10}
              max={300}
            />
            <ParamInput
              label="Batch Size"
              value={batchSize}
              onChange={setBatchSize}
              min={2}
              max={64}
            />
            <ParamInput
              label="Learning Rate"
              value={learningRate}
              onChange={setLearningRate}
              min={0.00001}
              max={0.01}
              step={0.0001}
              isFloat
            />
            <ParamInput
              label="Dropout"
              value={dropoutRate}
              onChange={setDropoutRate}
              min={0}
              max={0.6}
              step={0.05}
              isFloat
            />
            <DataRow
              label="Sample Rate"
              value={`${DYNAMIC_SAMPLE_RATE_HZ} Hz`}
              color="#556677"
            />
            <DataRow
              label="Window"
              value={`${(DYNAMIC_SEQUENCE_LENGTH / DYNAMIC_SAMPLE_RATE_HZ).toFixed(2)} s`}
              color="#556677"
            />
          </Section>

          <button
            onClick={isTraining ? cancelTraining : () => void handleTrain()}
            disabled={!isTraining && !canTrain}
            className="w-full cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {isTraining ? (
              <>
                <Square className="w-3.5 h-3.5" />
                停止训练
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                开始 TCN 训练
              </>
            )}
          </button>

          {!canTrain && (
            <p className="text-[9px] text-[#556677] text-center leading-relaxed">
              至少需要 {MIN_SEQUENCE_COUNT} 个动态片段、2
              个类别，并包含“静止/过渡”
            </p>
          )}

          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-1.5">
            <Link
              href="/collect-dynamic"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            >
              ← 采集动态片段
            </Link>
            <Link
              href="/translate-dynamic"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{ borderColor: "rgba(0, 229, 160, 0.3)" }}
            >
              动态识别 →
            </Link>
          </div>
        </aside>

        <main className="flex-1 min-h-[420px] flex flex-col items-center justify-center overflow-y-auto p-6 space-y-6">
          {isTraining && progress && (
            <div className="w-full max-w-lg space-y-4">
              <div className="text-center space-y-1">
                <span className="text-sm font-mono text-[#00f0ff]">
                  Epoch {progress.epoch} / {progress.totalEpochs}
                </span>
                <p className="text-[10px] font-mono text-[#00e5a0]">
                  DILATED TEMPORAL CONVOLUTION
                </p>
              </div>
              <div className="h-2 bg-[#1a2030] rounded-full overflow-hidden border border-[#00f0ff]/20">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min((progress.epoch / progress.totalEpochs) * 100, 100)}%`,
                    background: "linear-gradient(90deg, #00f0ff, #00e5a0)",
                    boxShadow: "0 0 10px rgba(0,240,255,0.5)",
                  }}
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard
                  label="Loss"
                  value={formatMetric(progress.loss)}
                  color="#ff2d7b"
                />
                <MetricCard
                  label="Accuracy"
                  value={formatPercent(progress.accuracy)}
                  color="#00e5a0"
                />
                <MetricCard
                  label="Val Loss"
                  value={formatMetric(progress.valLoss)}
                  color="#f59e0b"
                />
                <MetricCard
                  label="Val Accuracy"
                  value={formatPercent(progress.valAccuracy)}
                  color="#00f0ff"
                />
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="w-full max-w-lg">
              <TrainingChart history={history} />
            </div>
          )}

          {!isTraining && history.length === 0 && (
            <div className="text-center space-y-3 max-w-md">
              <Activity className="w-16 h-16 mx-auto text-[#334455]" />
              <p className="text-sm text-[#556677]">
                配置参数后开始训练动态词语模型
              </p>
              <p className="text-[10px] text-[#334455] leading-relaxed">
                每个片段会重采样为固定长度序列。TCN
                沿时间轴学习压力变化、双手协同与 IMU
                旋转轨迹，推理阶段只需要手套数据。
              </p>
            </div>
          )}

          {message && <StatusMessage message={message} />}
        </main>

        <aside className="w-full lg:w-60 border-t lg:border-t-0 lg:border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-3 shrink-0">
          <Section title="SAVED TCN MODELS">
            {models.length === 0 ? (
              <p className="text-[9px] text-[#334455] italic">暂无动态模型</p>
            ) : (
              <div className="space-y-2">
                {models.map(model => (
                  <div
                    key={model.id ?? `${model.name}-${model.createdAt}`}
                    className="cyber-panel p-2 rounded-sm space-y-1"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className="text-[9px] font-mono text-[#8899aa] truncate"
                        title={model.name}
                      >
                        {model.name}
                      </span>
                      {model.id != null && (
                        <button
                          onClick={() => void handleDeleteModel(model.id!)}
                          disabled={isTraining}
                          className="text-[#556677] hover:text-[#ff2d7b] transition-colors shrink-0 disabled:opacity-30"
                          title="删除模型"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                    <div className="flex justify-between text-[8px] font-mono text-[#556677]">
                      <span>Acc {formatPercent(model.accuracy)}</span>
                      <span>{model.labels.length} classes</span>
                    </div>
                    {model.validationLoss != null && (
                      <div className="text-[8px] font-mono text-[#556677]">
                        Val Loss {formatMetric(model.validationLoss)}
                      </div>
                    )}
                    <div className="flex justify-between gap-2 text-[8px] font-mono">
                      <span className="px-1 rounded-sm bg-[#00f0ff]/10 text-[#00f0ff]">
                        TCN
                      </span>
                      <span className="text-[#334455]">
                        {new Date(model.createdAt).toLocaleDateString("zh-CN")}
                      </span>
                    </div>
                    <button
                      onClick={() => void handleLoadModel(model)}
                      disabled={isTraining}
                      className="w-full cyber-btn px-2 py-1 rounded-sm text-[9px] flex items-center justify-center gap-1 mt-1 disabled:opacity-30"
                    >
                      <Upload className="w-2.5 h-2.5" />
                      加载模型
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Section>

          <Section title="ARCHITECTURE">
            <div className="text-[8px] font-mono text-[#556677] space-y-1">
              <p className="text-[#00f0ff]">Temporal input</p>
              <p>
                {DYNAMIC_SEQUENCE_LENGTH} frames x {DYNAMIC_FEATURE_DIM}D
              </p>
              <p>{DYNAMIC_SAMPLE_RATE_HZ}Hz normalized stream</p>
              <p className="text-[#00e5a0] pt-1">Residual TCN</p>
              <p>Shifted dilated temporal blocks</p>
              <p>Temporal pooling</p>
              <p>Dense(N) → Softmax</p>
              <p className="pt-1 text-[#334455]">Optimizer: Adam</p>
              <p className="text-[#334455]">Loss: categorical CE</p>
            </div>
          </Section>
        </aside>
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
    <section className="space-y-2">
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

function DataRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex justify-between gap-3 text-[10px] font-mono">
      <span className="text-[#556677]">{label}</span>
      <span className="text-right break-words" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

function ParamInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  isFloat = false,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  isFloat?: boolean;
}) {
  return (
    <label className="flex items-center justify-between gap-3 text-[10px] font-mono">
      <span className="text-[#556677]">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={event => {
          const nextValue = isFloat
            ? Number.parseFloat(event.target.value)
            : Number.parseInt(event.target.value, 10);
          if (
            Number.isFinite(nextValue) &&
            nextValue >= min &&
            nextValue <= max
          ) {
            onChange(nextValue);
          }
        }}
        className="w-20 bg-[#1a2030] border border-[#00f0ff]/20 rounded-sm px-1.5 py-0.5 text-[#00f0ff] text-center text-[10px]"
      />
    </label>
  );
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="cyber-panel p-2 rounded-sm text-center">
      <div className="text-[8px] font-mono text-[#556677] uppercase">
        {label}
      </div>
      <div
        className="text-sm font-bold font-mono mt-0.5"
        style={{ color, textShadow: `0 0 8px ${color}40` }}
      >
        {value}
      </div>
    </div>
  );
}

function StatusMessage({ message }: { message: string }) {
  const isSuccess = message.startsWith("✓");
  const isError = message.startsWith("✗");
  const isWarning = message.startsWith("⚠");
  const color = isSuccess
    ? "#00e5a0"
    : isError
      ? "#ff2d7b"
      : isWarning
        ? "#f59e0b"
        : "#8899aa";

  return (
    <div
      className="max-w-lg text-[11px] font-mono px-4 py-2 rounded-sm border text-center"
      style={{ color, borderColor: `${color}4d` }}
    >
      {message}
    </div>
  );
}

function TrainingChart({ history }: { history: DynamicTrainingProgress[] }) {
  const width = 500;
  const height = 160;
  const padding = 28;
  const maxLoss = Math.max(
    ...history.flatMap(item => [
      finiteMetric(item.loss),
      finiteMetric(item.valLoss),
    ]),
    0.1
  );
  const scaleX = (index: number) =>
    padding + (index / Math.max(history.length - 1, 1)) * (width - padding * 2);
  const scaleLoss = (value: number) =>
    height - padding - (finiteMetric(value) / maxLoss) * (height - padding * 2);
  const scaleAccuracy = (value: number) =>
    height -
    padding -
    Math.max(0, Math.min(1, finiteMetric(value))) * (height - padding * 2);

  const pathFor = (
    select: (item: DynamicTrainingProgress) => number,
    scale: (value: number) => number
  ) =>
    history
      .map(
        (item, index) =>
          `${index === 0 ? "M" : "L"} ${scaleX(index)} ${scale(select(item))}`
      )
      .join(" ");

  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider mb-2">
        TCN Training Curves
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        aria-label="TCN 训练损失和准确率曲线"
      >
        {[0.25, 0.5, 0.75].map(value => (
          <line
            key={value}
            x1={padding}
            y1={scaleAccuracy(value)}
            x2={width - padding}
            y2={scaleAccuracy(value)}
            stroke="rgba(0,240,255,0.1)"
            strokeDasharray="4 4"
          />
        ))}
        <path
          d={pathFor(item => item.loss, scaleLoss)}
          fill="none"
          stroke="#ff2d7b"
          strokeWidth="1.5"
        />
        <path
          d={pathFor(item => item.valLoss, scaleLoss)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth="1.5"
          strokeDasharray="4 2"
        />
        <path
          d={pathFor(item => item.accuracy, scaleAccuracy)}
          fill="none"
          stroke="#00e5a0"
          strokeWidth="1.5"
        />
        <path
          d={pathFor(item => item.valAccuracy, scaleAccuracy)}
          fill="none"
          stroke="#00f0ff"
          strokeWidth="1.5"
          strokeDasharray="4 2"
        />
      </svg>
      <div className="flex flex-wrap items-center justify-center gap-4 mt-2 text-[8px] font-mono">
        <Legend color="#ff2d7b" label="Loss" />
        <Legend color="#f59e0b" label="Val Loss" dashed />
        <Legend color="#00e5a0" label="Accuracy" />
        <Legend color="#00f0ff" label="Val Acc" dashed />
      </div>
    </div>
  );
}

function Legend({
  color,
  label,
  dashed = false,
}: {
  color: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <div className="flex items-center gap-1">
      <div
        className="w-3 h-0.5"
        style={{
          backgroundColor: dashed ? "transparent" : color,
          borderTop: dashed ? `1px dashed ${color}` : undefined,
        }}
      />
      <span className="text-[#556677]">{label}</span>
    </div>
  );
}

function finiteMetric(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function formatMetric(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : "--";
}

function formatPercent(value: number): string {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";
}

function formatDuration(durationMs: number): string {
  if (!Number.isFinite(durationMs) || durationMs <= 0) return "0.0 s";
  const totalSeconds = durationMs / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)} s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}m ${seconds}s`;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function getErrorName(error: unknown): string | undefined {
  return error instanceof Error ? error.name : undefined;
}
