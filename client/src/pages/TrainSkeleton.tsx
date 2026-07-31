/*
 * TrainSkeleton — 触觉→骨架回归模型训练页面
 * DESIGN: Cyberpunk HUD 风格
 *
 * 功能:
 * 1. 显示数据集统计（含视觉数据的样本数）
 * 2. 配置训练参数
 * 3. 浏览器端 TensorFlow.js 回归训练（触觉141D → 骨架63D）
 * 4. 保存/加载骨架回归模型
 */
import {
  getAllSamples,
  getDatasetStats,
  saveSkeletonModel,
  getAllSkeletonModels,
  deleteSkeletonModel,
  pickPrimaryHand,
  type DatasetStats,
  type SavedSkeletonModelRecord,
} from "@/lib/datasetStore";
import {
  trainSkeletonModel,
  serializeSkeletonModel,
  loadSkeletonModelFromSaved,
  isSkeletonModelLoaded,
  type SkeletonTrainingProgress,
} from "@/lib/skeletonModel";
import { useCallback, useEffect, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Bone,
  Play,
  Trash2,
  Upload,
} from "lucide-react";

export default function TrainSkeleton() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [validSampleCount, setValidSampleCount] = useState(0);
  const [models, setModels] = useState<SavedSkeletonModelRecord[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<SkeletonTrainingProgress | null>(null);
  const [history, setHistory] = useState<SkeletonTrainingProgress[]>([]);
  const [message, setMessage] = useState("");
  const [modelLoaded, setModelLoaded] = useState(isSkeletonModelLoaded());

  // 训练参数
  const [epochs, setEpochs] = useState(80);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [augmentNoise, setAugmentNoise] = useState(0.015);

  const refresh = useCallback(async () => {
    const s = await getDatasetStats();
    setStats(s);
    const samples = await getAllSamples();
    const valid = samples.filter(
      (s) => { const h = pickPrimaryHand(s); return !!h && h.landmarks.length === 21; }
    ).length;
    setValidSampleCount(valid);
    const m = await getAllSkeletonModels();
    setModels(m.sort((a, b) => b.createdAt - a.createdAt));
    setModelLoaded(isSkeletonModelLoaded());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleTrain = useCallback(async () => {
    setIsTraining(true);
    setHistory([]);
    setProgress(null);
    setMessage("正在加载数据集...");

    try {
      const samples = await getAllSamples();
      const validSamples = samples.filter(
        (s) => { const h = pickPrimaryHand(s); return !!h && h.landmarks.length === 21; }
      );

      if (validSamples.length < 10) {
        setMessage(
          `✗ 有效样本不足（需要至少10个含视觉骨架数据的样本，当前仅${validSamples.length}个）`
        );
        setIsTraining(false);
        return;
      }

      setMessage(
        `训练中... (${validSamples.length} 有效样本, 触觉141D → 骨架63D)`
      );

      const result = await trainSkeletonModel(
        validSamples,
        { epochs, batchSize, learningRate, augmentNoise },
        (p) => {
          setProgress(p);
          setHistory((prev) => [...prev, p]);
        }
      );

      // 保存模型
      const timestamp = new Date()
        .toISOString()
        .slice(0, 16)
        .replace(/[:-]/g, "");
      const saved = await serializeSkeletonModel(
        result.model,
        result.finalLoss,
        result.finalMae,
        `skeleton_${timestamp}`
      );
      await saveSkeletonModel(saved);

      setMessage(
        `✓ 训练完成！Val Loss: ${result.finalLoss.toFixed(6)} | Val MAE: ${result.finalMae.toFixed(6)}`
      );
      refresh();
    } catch (e: any) {
      setMessage(`✗ 训练失败: ${e.message}`);
    } finally {
      setIsTraining(false);
    }
  }, [epochs, batchSize, learningRate, augmentNoise, refresh]);

  const handleLoadModel = useCallback(
    async (model: SavedSkeletonModelRecord) => {
      try {
        await loadSkeletonModelFromSaved(model);
        setModelLoaded(true);
        setMessage(`✓ 已加载骨架模型 "${model.name}"`);
      } catch (e: any) {
        setMessage(`✗ 加载失败: ${e.message}`);
      }
    },
    []
  );

  const handleDeleteModel = useCallback(
    async (id: number) => {
      if (!confirm("确定删除该模型？")) return;
      await deleteSkeletonModel(id);
      refresh();
    },
    [refresh]
  );

  return (
    <div
      className="min-h-screen flex flex-col"
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
          <span className="text-xs font-bold tracking-widest text-[#f59e0b] font-mono">
            SKELETON REGRESSION
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            TACTILE → SKELETON
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono">
          {modelLoaded && (
            <span className="text-[#f59e0b] flex items-center gap-1">
              <Bone className="w-3 h-3" />
              SKELETON MODEL ACTIVE
            </span>
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 左侧：数据集和参数 */}
        <div className="w-72 border-r border-[#00f0ff]/15 overflow-y-auto p-4 space-y-4 shrink-0">
          <Section title="DATASET">
            <DataRow
              label="TOTAL SAMPLES"
              value={String(stats?.totalSamples ?? 0)}
              color="#00f0ff"
            />
            <DataRow
              label="WITH LANDMARKS"
              value={String(validSampleCount)}
              color="#00e5a0"
            />
            <DataRow
              label="INPUT DIM"
              value="141 (tactile)"
              color="#f59e0b"
            />
            <DataRow
              label="OUTPUT DIM"
              value="63 (skeleton)"
              color="#da77f2"
            />
            {validSampleCount < (stats?.totalSamples ?? 0) && (
              <p className="text-[8px] text-[#f59e0b] mt-1">
                ⚠ {(stats?.totalSamples ?? 0) - validSampleCount} 个样本缺少视觉数据，将被跳过
              </p>
            )}
          </Section>

          <Section title="PARAMETERS">
            <ParamInput
              label="Epochs"
              value={epochs}
              onChange={setEpochs}
              min={20}
              max={300}
            />
            <ParamInput
              label="Batch Size"
              value={batchSize}
              onChange={setBatchSize}
              min={8}
              max={128}
            />
            <ParamInput
              label="Learning Rate"
              value={learningRate}
              onChange={setLearningRate}
              min={0.0001}
              max={0.01}
              step={0.0001}
              isFloat
            />
            <ParamInput
              label="Noise Aug."
              value={augmentNoise}
              onChange={setAugmentNoise}
              min={0}
              max={0.1}
              step={0.005}
              isFloat
            />
          </Section>

          <button
            onClick={handleTrain}
            disabled={isTraining || validSampleCount < 10}
            className="w-full cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center justify-center gap-2 disabled:opacity-50"
            style={{ borderColor: "rgba(245, 158, 11, 0.4)" }}
          >
            {isTraining ? (
              <>
                <div className="w-3 h-3 border border-[#f59e0b] border-t-transparent rounded-full animate-spin" />
                训练中...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                训练骨架模型
              </>
            )}
          </button>

          {validSampleCount < 10 && (
            <p className="text-[9px] text-[#556677] text-center">
              需要至少 10 个含视觉骨架数据的样本
            </p>
          )}

          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-1.5">
            <Link
              href="/collect"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            >
              ← 采集数据
            </Link>
            <Link
              href="/mocap"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{ borderColor: "rgba(245, 158, 11, 0.3)" }}
            >
              虚拟动捕 →
            </Link>
          </div>
        </div>

        {/* 中间：训练可视化 */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-6">
          {isTraining && progress && (
            <div className="w-full max-w-lg space-y-4">
              <div className="text-center space-y-1">
                <span className="text-sm font-mono text-[#f59e0b]">
                  Epoch {progress.epoch} / {progress.totalEpochs}
                </span>
                <div className="text-[10px] font-mono text-[#8899aa]">
                  触觉(141D) → 骨架(63D) 回归训练
                </div>
              </div>
              <div className="h-2 bg-[#1a2030] rounded-full overflow-hidden border border-[#f59e0b]/20">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${(progress.epoch / progress.totalEpochs) * 100}%`,
                    background: "linear-gradient(90deg, #f59e0b, #00e5a0)",
                    boxShadow: "0 0 10px rgba(245,158,11,0.5)",
                  }}
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard
                  label="MSE Loss"
                  value={progress.loss.toFixed(6)}
                  color="#ff2d7b"
                />
                <MetricCard
                  label="Val Loss"
                  value={progress.valLoss.toFixed(6)}
                  color="#f59e0b"
                />
                <MetricCard
                  label="MAE"
                  value={progress.mae.toFixed(6)}
                  color="#00e5a0"
                />
                <MetricCard
                  label="Val MAE"
                  value={progress.valMae.toFixed(6)}
                  color="#00f0ff"
                />
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="w-full max-w-lg">
              <RegressionChart history={history} />
            </div>
          )}

          {!isTraining && history.length === 0 && (
            <div className="text-center space-y-3">
              <Bone className="w-16 h-16 mx-auto text-[#334455]" />
              <p className="text-sm text-[#556677]">
                触觉 → 骨架回归训练
              </p>
              <p className="text-[10px] text-[#334455] max-w-sm mx-auto">
                用同步采集的视觉骨架关键点作为 ground truth，训练模型从触觉传感器数据预测手部骨架姿态。训练完成后，仅需手套即可驱动火柴人动画。
              </p>
            </div>
          )}

          {message && (
            <div
              className="text-[11px] font-mono px-4 py-2 rounded-sm border"
              style={{
                color: message.startsWith("✓")
                  ? "#00e5a0"
                  : message.startsWith("✗")
                  ? "#ff2d7b"
                  : "#8899aa",
                borderColor: message.startsWith("✓")
                  ? "rgba(0,229,160,0.3)"
                  : message.startsWith("✗")
                  ? "rgba(255,45,123,0.3)"
                  : "rgba(245,158,11,0.2)",
              }}
            >
              {message}
            </div>
          )}
        </div>

        {/* 右侧：已保存模型 */}
        <div className="w-60 border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-3 shrink-0">
          <Section title="SAVED MODELS">
            {models.length === 0 ? (
              <p className="text-[9px] text-[#334455] italic">暂无骨架模型</p>
            ) : (
              <div className="space-y-2">
                {models.map((m) => (
                  <div
                    key={m.id}
                    className="cyber-panel p-2 rounded-sm space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] font-mono text-[#8899aa] truncate max-w-[120px]">
                        {m.name}
                      </span>
                      <button
                        onClick={() => handleDeleteModel(m.id!)}
                        className="text-[#556677] hover:text-[#ff2d7b] transition-colors"
                      >
                        <Trash2 className="w-2.5 h-2.5" />
                      </button>
                    </div>
                    <div className="flex justify-between text-[8px] font-mono text-[#556677]">
                      <span>MAE: {m.valMae.toFixed(6)}</span>
                      <span>Loss: {m.valLoss.toFixed(6)}</span>
                    </div>
                    <div className="flex justify-between text-[8px] font-mono">
                      <span
                        className="px-1 rounded-sm"
                        style={{
                          backgroundColor: "rgba(245,158,11,0.15)",
                          color: "#f59e0b",
                        }}
                      >
                        骨架回归
                      </span>
                      <span className="text-[#334455]">
                        {new Date(m.createdAt).toLocaleDateString("zh-CN")}
                      </span>
                    </div>
                    <button
                      onClick={() => handleLoadModel(m)}
                      className="w-full cyber-btn px-2 py-1 rounded-sm text-[9px] flex items-center justify-center gap-1 mt-1"
                      style={{ borderColor: "rgba(245,158,11,0.3)" }}
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
            <div className="text-[8px] font-mono text-[#556677] space-y-0.5">
              <p className="text-[#f59e0b]">— 回归网络 —</p>
              <p>Input: 141D (137 sensors + 4 quat)</p>
              <p>Dense(256) → BN → Drop(0.2)</p>
              <p>Dense(128) → BN → Drop(0.15)</p>
              <p>Dense(64) → Drop(0.1)</p>
              <p>Dense(63) → Sigmoid</p>
              <p className="pt-1 text-[#334455]">Loss: MSE</p>
              <p className="text-[#334455]">Metric: MAE</p>
              <p className="text-[#334455]">Optimizer: Adam</p>
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}

// ===== 辅助组件 =====

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 pb-1 border-b border-[#f59e0b]/15">
        <div className="w-1 h-3 bg-[#f59e0b] rounded-full shadow-[0_0_4px_rgba(245,158,11,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[#f59e0b] font-mono">
          {title}
        </span>
      </div>
      {children}
    </div>
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
    <div className="flex justify-between text-[10px] font-mono">
      <span className="text-[#556677]">{label}</span>
      <span style={{ color }}>{value}</span>
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
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  isFloat?: boolean;
}) {
  return (
    <div className="flex items-center justify-between text-[10px] font-mono">
      <span className="text-[#556677]">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => {
          const v = isFloat
            ? parseFloat(e.target.value)
            : parseInt(e.target.value);
          if (!isNaN(v) && v >= min && v <= max) onChange(v);
        }}
        min={min}
        max={max}
        step={step}
        className="w-16 bg-[#1a2030] border border-[#f59e0b]/20 rounded-sm px-1.5 py-0.5 text-[#f59e0b] text-center text-[10px]"
      />
    </div>
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

function RegressionChart({
  history,
}: {
  history: SkeletonTrainingProgress[];
}) {
  const width = 500;
  const height = 150;
  const padding = 30;

  const maxEpoch = history.length;
  const maxLoss = Math.max(...history.map((h) => Math.max(h.loss, h.valLoss)), 0.001);
  const maxMae = Math.max(...history.map((h) => Math.max(h.mae, h.valMae)), 0.001);

  const scaleX = (idx: number) =>
    padding + (idx / Math.max(maxEpoch - 1, 1)) * (width - 2 * padding);
  const scaleYLoss = (loss: number) =>
    height - padding - (loss / maxLoss) * (height - 2 * padding);
  const scaleYMae = (mae: number) =>
    height - padding - (mae / maxMae) * (height - 2 * padding);

  const lossPath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYLoss(h.loss)}`)
    .join(" ");
  const valLossPath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYLoss(h.valLoss)}`)
    .join(" ");
  const maePath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYMae(h.mae)}`)
    .join(" ");
  const valMaePath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYMae(h.valMae)}`)
    .join(" ");

  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider mb-2">
        Regression Training Curves
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height: "auto" }}
      >
        {[0.25, 0.5, 0.75].map((v) => (
          <line
            key={v}
            x1={padding}
            y1={padding + v * (height - 2 * padding)}
            x2={width - padding}
            y2={padding + v * (height - 2 * padding)}
            stroke="rgba(245,158,11,0.1)"
            strokeDasharray="4 4"
          />
        ))}
        <path d={lossPath} fill="none" stroke="#ff2d7b" strokeWidth="1.5" opacity="0.8" />
        <path d={valLossPath} fill="none" stroke="#f59e0b" strokeWidth="1.5" opacity="0.6" strokeDasharray="4 2" />
        <path d={maePath} fill="none" stroke="#00e5a0" strokeWidth="1.5" opacity="0.8" />
        <path d={valMaePath} fill="none" stroke="#00f0ff" strokeWidth="1.5" opacity="0.6" strokeDasharray="4 2" />
      </svg>
      <div className="flex items-center justify-center gap-4 mt-2 text-[8px] font-mono">
        <Legend color="#ff2d7b" label="MSE Loss" />
        <Legend color="#f59e0b" label="Val Loss" dashed />
        <Legend color="#00e5a0" label="MAE" />
        <Legend color="#00f0ff" label="Val MAE" dashed />
      </div>
    </div>
  );
}

function Legend({
  color,
  label,
  dashed,
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
      <span style={{ color: "#556677" }}>{label}</span>
    </div>
  );
}
