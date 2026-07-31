/*
 * Train — 模型训练页面（双分支架构：教师融合 + 学生触觉）
 * DESIGN: Cyberpunk HUD 风格
 *
 * 功能:
 * 1. 显示数据集统计
 * 2. 配置训练参数（含蒸馏温度和权重）
 * 3. 浏览器端 TensorFlow.js 双阶段训练
 *    - Phase 1: 教师模型（视觉63D + 触觉141D = 204D）
 *    - Phase 2: 学生模型（仅触觉141D，知识蒸馏）
 * 4. 保存/加载模型（保存学生模型用于推理）
 */
import {
  getAllSamples,
  getDatasetStats,
  saveModel,
  getAllModels,
  deleteModel,
  sampleHasVision,
  type DatasetStats,
  type SavedModel,
} from "@/lib/datasetStore";
import {
  trainModel,
  serializeModel,
  loadModelFromSaved,
  setActiveModel,
  isModelLoaded,
  type TrainingProgress,
} from "@/lib/signLanguageModel";
import { getWordById } from "@/lib/signLanguageVocab";
import { useCallback, useEffect, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Brain,
  Play,
  Trash2,
  Upload,
} from "lucide-react";

export default function Train() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [models, setModels] = useState<SavedModel[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<TrainingProgress[]>([]);
  const [trainMessage, setTrainMessage] = useState("");
  const [activeModelLoaded, setActiveModelLoaded] = useState(isModelLoaded());

  // 训练参数
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [augmentNoise, setAugmentNoise] = useState(0.02);
  const [distillationTemp, setDistillationTemp] = useState(3.0);
  const [distillationAlpha, setDistillationAlpha] = useState(0.5);

  // 加载数据
  const refresh = useCallback(async () => {
    const s = await getDatasetStats();
    setStats(s);
    const m = await getAllModels();
    setModels(m.sort((a, b) => b.createdAt - a.createdAt));
    setActiveModelLoaded(isModelLoaded());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // 开始训练
  const handleTrain = useCallback(async () => {
    setIsTraining(true);
    setHistory([]);
    setProgress(null);
    setTrainMessage("正在加载数据集...");

    try {
      const samples = await getAllSamples();
      if (samples.length < 10) {
        setTrainMessage("✗ 样本数量不足，至少需要 10 个样本");
        setIsTraining(false);
        return;
      }

      const uniqueLabels = new Set(samples.map((s) => s.label));
      if (uniqueLabels.size < 2) {
        setTrainMessage("✗ 至少需要 2 个不同的手语词汇才能训练");
        setIsTraining(false);
        return;
      }

      // 检查是否有视觉数据
      const withLandmarks = samples.filter(sampleHasVision);
      if (withLandmarks.length < samples.length * 0.5) {
        setTrainMessage(
          `⚠ 仅 ${withLandmarks.length}/${samples.length} 个样本包含视觉数据，建议重新采集`
        );
      }

      setTrainMessage(
        `训练中... (${samples.length} 样本, ${uniqueLabels.size} 类, 双分支架构)`
      );

      const result = await trainModel(
        samples,
        {
          epochs,
          batchSize,
          learningRate,
          augmentNoise,
          distillationTemp,
          distillationAlpha,
        },
        (p) => {
          setProgress(p);
          setHistory((prev) => [...prev, p]);
        }
      );

      // 设学生模型为活跃推理模型
      setActiveModel(result.studentModel, result.labels);
      setActiveModelLoaded(true);

      // 保存学生模型（用于推理）
      const timestamp = new Date()
        .toISOString()
        .slice(0, 16)
        .replace(/[:-]/g, "");
      const studentSaved = await serializeModel(
        result.studentModel,
        result.labels,
        result.studentAccuracy,
        `student_${timestamp}`,
        "tactile"
      );
      await saveModel(studentSaved);

      // 也保存教师模型（备份）
      const teacherSaved = await serializeModel(
        result.teacherModel,
        result.labels,
        result.teacherAccuracy,
        `teacher_${timestamp}`,
        "fused"
      );
      await saveModel(teacherSaved);

      setTrainMessage(
        `✓ 训练完成！教师准确率: ${(result.teacherAccuracy * 100).toFixed(1)}% | 学生准确率: ${(result.studentAccuracy * 100).toFixed(1)}%`
      );
      refresh();
    } catch (e: any) {
      setTrainMessage(`✗ 训练失败: ${e.message}`);
    } finally {
      setIsTraining(false);
    }
  }, [
    epochs,
    batchSize,
    learningRate,
    augmentNoise,
    distillationTemp,
    distillationAlpha,
    refresh,
  ]);

  // 加载已保存的模型
  const handleLoadModel = useCallback(async (model: SavedModel) => {
    try {
      await loadModelFromSaved(model);
      setActiveModelLoaded(true);
      setTrainMessage(
        `✓ 已加载模型 "${model.name}" (${model.modelType || "tactile"})`
      );
    } catch (e: any) {
      setTrainMessage(`✗ 加载失败: ${e.message}`);
    }
  }, []);

  // 删除模型
  const handleDeleteModel = useCallback(
    async (id: number) => {
      if (!confirm("确定删除该模型？")) return;
      await deleteModel(id);
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
          <span className="text-xs font-bold tracking-widest text-[#00f0ff] font-mono">
            MODEL TRAINING
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            DUAL-BRANCH DISTILLATION
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono">
          {activeModelLoaded && (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <Brain className="w-3 h-3" />
              STUDENT MODEL ACTIVE
            </span>
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 左侧：数据集和参数 */}
        <div className="w-72 border-r border-[#00f0ff]/15 overflow-y-auto p-4 space-y-4 shrink-0">
          {/* 数据集统计 */}
          <Section title="DATASET">
            <DataRow
              label="SAMPLES"
              value={String(stats?.totalSamples ?? 0)}
              color="#00f0ff"
            />
            <DataRow
              label="CLASSES"
              value={String(stats?.labels.length ?? 0)}
              color="#00e5a0"
            />
            <DataRow
              label="FEATURES"
              value="204D (63V+141T)"
              color="#da77f2"
            />
            {stats && stats.labels.length > 0 && (
              <div className="mt-2 space-y-0.5 max-h-40 overflow-y-auto">
                {stats.labels.map((label) => {
                  const word = getWordById(label);
                  return (
                    <div
                      key={label}
                      className="flex justify-between text-[9px] font-mono"
                    >
                      <span className="text-[#8899aa]">
                        {word?.label ?? label}
                      </span>
                      <span className="text-[#556677]">
                        {stats.labelCounts[label]}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </Section>

          {/* 训练参数 */}
          <Section title="PARAMETERS">
            <ParamInput
              label="Epochs"
              value={epochs}
              onChange={setEpochs}
              min={10}
              max={200}
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

          {/* 蒸馏参数 */}
          <Section title="DISTILLATION">
            <ParamInput
              label="Temperature"
              value={distillationTemp}
              onChange={setDistillationTemp}
              min={1}
              max={10}
              step={0.5}
              isFloat
            />
            <ParamInput
              label="Alpha (soft)"
              value={distillationAlpha}
              onChange={setDistillationAlpha}
              min={0}
              max={1}
              step={0.1}
              isFloat
            />
            <p className="text-[8px] text-[#445566] mt-1">
              Alpha 越大，学生越依赖教师的 soft labels
            </p>
          </Section>

          {/* 训练按钮 */}
          <button
            onClick={handleTrain}
            disabled={isTraining || (stats?.totalSamples ?? 0) < 10}
            className="w-full cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {isTraining ? (
              <>
                <div className="w-3 h-3 border border-[#00f0ff] border-t-transparent rounded-full animate-spin" />
                训练中...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                开始训练
              </>
            )}
          </button>

          {(stats?.totalSamples ?? 0) < 10 && (
            <p className="text-[9px] text-[#556677] text-center">
              需要至少 10 个样本和 2 个词汇类别
            </p>
          )}

          {/* 导航 */}
          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-1.5">
            <Link
              href="/collect"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            >
              ← 采集数据
            </Link>
            <Link
              href="/translate"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{ borderColor: "rgba(0, 229, 160, 0.3)" }}
            >
              前往翻译 →
            </Link>
          </div>
        </div>

        {/* 中间：训练可视化 */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-6">
          {/* 训练进度 */}
          {isTraining && progress && (
            <div className="w-full max-w-lg space-y-4">
              <div className="text-center space-y-1">
                <span className="text-sm font-mono text-[#00f0ff]">
                  Epoch {progress.epoch} / {progress.totalEpochs}
                </span>
                <div className="text-[10px] font-mono" style={{
                  color: progress.phase === "teacher" ? "#da77f2" : "#00e5a0"
                }}>
                  {progress.phase === "teacher"
                    ? "Phase 1: 教师模型 (视觉+触觉融合)"
                    : "Phase 2: 学生模型 (触觉蒸馏)"}
                </div>
              </div>
              {/* 进度条 */}
              <div className="h-2 bg-[#1a2030] rounded-full overflow-hidden border border-[#00f0ff]/20">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${(progress.epoch / progress.totalEpochs) * 100}%`,
                    background:
                      progress.phase === "teacher"
                        ? "linear-gradient(90deg, #da77f2, #00f0ff)"
                        : "linear-gradient(90deg, #00f0ff, #00e5a0)",
                    boxShadow: "0 0 10px rgba(0,240,255,0.5)",
                  }}
                />
              </div>
              {/* 指标 */}
              <div className="grid grid-cols-2 gap-3">
                <MetricCard
                  label="Loss"
                  value={progress.loss.toFixed(4)}
                  color="#ff2d7b"
                />
                <MetricCard
                  label="Accuracy"
                  value={`${(progress.accuracy * 100).toFixed(1)}%`}
                  color="#00e5a0"
                />
                <MetricCard
                  label="Val Loss"
                  value={progress.valLoss.toFixed(4)}
                  color="#f59e0b"
                />
                <MetricCard
                  label="Val Accuracy"
                  value={`${(progress.valAccuracy * 100).toFixed(1)}%`}
                  color="#00f0ff"
                />
              </div>
            </div>
          )}

          {/* 训练曲线 */}
          {history.length > 0 && (
            <div className="w-full max-w-lg">
              <TrainingChart history={history} />
            </div>
          )}

          {/* 空状态 */}
          {!isTraining && history.length === 0 && (
            <div className="text-center space-y-3">
              <Brain className="w-16 h-16 mx-auto text-[#334455]" />
              <p className="text-sm text-[#556677]">
                配置参数后点击"开始训练"
              </p>
              <p className="text-[10px] text-[#334455] max-w-sm mx-auto">
                双分支架构：先用视觉+触觉融合数据训练教师模型，再通过知识蒸馏训练仅触觉的学生模型。最终推理只需手套数据。
              </p>
            </div>
          )}

          {/* 消息 */}
          {trainMessage && (
            <div
              className="text-[11px] font-mono px-4 py-2 rounded-sm border"
              style={{
                color: trainMessage.startsWith("✓")
                  ? "#00e5a0"
                  : trainMessage.startsWith("✗")
                  ? "#ff2d7b"
                  : trainMessage.startsWith("⚠")
                  ? "#f59e0b"
                  : "#8899aa",
                borderColor: trainMessage.startsWith("✓")
                  ? "rgba(0,229,160,0.3)"
                  : trainMessage.startsWith("✗")
                  ? "rgba(255,45,123,0.3)"
                  : "rgba(0,240,255,0.2)",
              }}
            >
              {trainMessage}
            </div>
          )}
        </div>

        {/* 右侧：已保存模型 */}
        <div className="w-60 border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-3 shrink-0">
          <Section title="SAVED MODELS">
            {models.length === 0 ? (
              <p className="text-[9px] text-[#334455] italic">暂无已保存模型</p>
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
                      <span>Acc: {(m.accuracy * 100).toFixed(1)}%</span>
                      <span>{m.labels.length} classes</span>
                    </div>
                    <div className="flex justify-between text-[8px] font-mono">
                      <span
                        className="px-1 rounded-sm"
                        style={{
                          backgroundColor:
                            m.modelType === "tactile"
                              ? "rgba(0,229,160,0.15)"
                              : "rgba(218,119,242,0.15)",
                          color:
                            m.modelType === "tactile" ? "#00e5a0" : "#da77f2",
                        }}
                      >
                        {m.modelType === "tactile" ? "学生/触觉" : "教师/融合"}
                      </span>
                      <span className="text-[#334455]">
                        {new Date(m.createdAt).toLocaleDateString("zh-CN")}
                      </span>
                    </div>
                    <button
                      onClick={() => handleLoadModel(m)}
                      className="w-full cyber-btn px-2 py-1 rounded-sm text-[9px] flex items-center justify-center gap-1 mt-1"
                    >
                      <Upload className="w-2.5 h-2.5" />
                      加载模型
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* 模型架构说明 */}
          <Section title="ARCHITECTURE">
            <div className="text-[8px] font-mono text-[#556677] space-y-0.5">
              <p className="text-[#da77f2]">— 教师 (融合) —</p>
              <p>Input: 204D (63V + 141T)</p>
              <p>Dense(256) → BN → Drop(0.3)</p>
              <p>Dense(128) → BN → Drop(0.2)</p>
              <p>Dense(64) → Drop(0.1)</p>
              <p>Dense(N) → Softmax</p>
              <p className="text-[#00e5a0] mt-1.5">— 学生 (触觉) —</p>
              <p>Input: 141D (137 sensors + 4 quat)</p>
              <p>Dense(128) → BN → Drop(0.3)</p>
              <p>Dense(64) → BN → Drop(0.2)</p>
              <p>Dense(32) → Drop(0.1)</p>
              <p>Dense(N) → Softmax</p>
              <p className="pt-1 text-[#334455]">Distillation: KL + CE</p>
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
        className="w-16 bg-[#1a2030] border border-[#00f0ff]/20 rounded-sm px-1.5 py-0.5 text-[#00f0ff] text-center text-[10px]"
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

function TrainingChart({ history }: { history: TrainingProgress[] }) {
  const width = 500;
  const height = 150;
  const padding = 30;

  const maxEpoch = history.length;
  const maxLoss = Math.max(
    ...history.map((h) => Math.max(h.loss, h.valLoss)),
    0.1
  );

  const scaleX = (idx: number) =>
    padding + (idx / Math.max(maxEpoch - 1, 1)) * (width - 2 * padding);
  const scaleYLoss = (loss: number) =>
    height - padding - (loss / maxLoss) * (height - 2 * padding);
  const scaleYAcc = (acc: number) =>
    height - padding - acc * (height - 2 * padding);

  const lossPath = history
    .map(
      (h, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYLoss(h.loss)}`
    )
    .join(" ");
  const valLossPath = history
    .map(
      (h, i) =>
        `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYLoss(h.valLoss)}`
    )
    .join(" ");
  const accPath = history
    .map(
      (h, i) =>
        `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYAcc(h.accuracy)}`
    )
    .join(" ");
  const valAccPath = history
    .map(
      (h, i) =>
        `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleYAcc(h.valAccuracy)}`
    )
    .join(" ");

  // 找到教师/学生分界线
  const phaseChangeIdx = history.findIndex((h) => h.phase === "student");

  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider mb-2">
        Training Curves (Teacher → Student)
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height: "auto" }}
      >
        {/* 网格 */}
        {[0.25, 0.5, 0.75].map((v) => (
          <line
            key={v}
            x1={padding}
            y1={scaleYAcc(v)}
            x2={width - padding}
            y2={scaleYAcc(v)}
            stroke="rgba(0,240,255,0.1)"
            strokeDasharray="4 4"
          />
        ))}
        {/* 阶段分界线 */}
        {phaseChangeIdx > 0 && (
          <line
            x1={scaleX(phaseChangeIdx)}
            y1={padding}
            x2={scaleX(phaseChangeIdx)}
            y2={height - padding}
            stroke="rgba(218,119,242,0.4)"
            strokeDasharray="4 4"
          />
        )}
        {/* Loss 曲线 */}
        <path
          d={lossPath}
          fill="none"
          stroke="#ff2d7b"
          strokeWidth="1.5"
          opacity="0.8"
        />
        <path
          d={valLossPath}
          fill="none"
          stroke="#f59e0b"
          strokeWidth="1.5"
          opacity="0.6"
          strokeDasharray="4 2"
        />
        {/* Accuracy 曲线 */}
        <path
          d={accPath}
          fill="none"
          stroke="#00e5a0"
          strokeWidth="1.5"
          opacity="0.8"
        />
        <path
          d={valAccPath}
          fill="none"
          stroke="#00f0ff"
          strokeWidth="1.5"
          opacity="0.6"
          strokeDasharray="4 2"
        />
      </svg>
      {/* 图例 */}
      <div className="flex items-center justify-center gap-4 mt-2 text-[8px] font-mono">
        <Legend color="#ff2d7b" label="Loss" />
        <Legend color="#f59e0b" label="Val Loss" dashed />
        <Legend color="#00e5a0" label="Accuracy" />
        <Legend color="#00f0ff" label="Val Acc" dashed />
        <Legend color="#da77f2" label="Phase" dashed />
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
