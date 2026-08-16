/*
 * TrainSequence — 时序模型训练页（TCN 骨干 + 两阶段知识蒸馏）
 * DESIGN: Cyberpunk HUD 风格（沿用 Train.tsx 的视觉语言）
 *
 * 与 /train（单帧静态 MLP）并存，互不干扰：那条链路仍被骨架回归依赖。
 *
 * 三个这里独有的东西：
 * 1) 真实/合成分开计数 —— 合成序列的帧间统计与真实录制不完全一致，
 *    混在一起统计会让人误判数据够了
 * 2)「从静态样本合成序列」—— 把旧的单帧样本迁移过来，立刻能跑通整条链路
 * 3)「导出给 Python」—— tfjs 4.22 没有 CTC，句子级训练只能走 python_train/
 */
import { useCallback, useEffect, useState } from "react";
import { Link } from "wouter";
import { ArrowLeft, Brain, Download, Play, Trash2, Upload, Wand2 } from "lucide-react";
import {
  getAllSamples,
  getAllSequences,
  addSequences,
  getSequenceStats,
  deleteSynthesizedSequences,
  exportSequencesBinary,
  saveModel,
  deleteModel,
  getAllSequenceModels,
  type SequenceStats,
  type SavedModel,
  type SequenceSample,
} from "@/lib/datasetStore";
import { analyzeSequenceImu, isImuSuspect } from "@/lib/imuHealth";
import StepNav from "@/components/StepNav";
import TfBackendBadge from "@/components/TfBackendBadge";
import type { TrimStats } from "@/lib/sequenceTrim";
import {
  trainSequenceModel,
  serializeSequenceModel,
  loadSequenceModelFromSaved,
  isSequenceModelLoaded,
  sequenceVisionRatio,
  DEFAULT_SEQ_CONFIG,
  type SeqBackbone,
  type SeqTrainingProgress,
} from "@/lib/sequenceModel";
import {
  synthesizeFromStatic,
  DEFAULT_SYNTHESIZE,
  DEFAULT_AUGMENT,
} from "@/lib/sequenceFeatures";
import { getDisplayLabel, IDLE_LABEL } from "@/lib/signLanguageVocab";

/**
 * 陀螺漂移可疑的样本条数。优先读录制时写进样本的 `imuHealth`；没有该字段的
 * **存量样本就地从 leftImu/rightImu 重算** —— 所以这个指标对以前录的数据同样有效，
 * 不需要为了它重录任何东西。
 *
 * 合成样本跳过：它们的 IMU 是从静态单帧复制出来的，没有真加速度，判不出漂移，
 * 算进去只会得到一堆 unknown 噪声。
 */
function countImuSuspect(seqs: SequenceSample[]): number {
  let n = 0;
  for (const s of seqs) {
    if (s.origin !== "recorded") continue;
    const h =
      s.imuHealth ??
      {
        left: s.leftImu
          ? analyzeSequenceImu(s.leftImu, s.frameCount, s.timestamps)
          : null,
        right: s.rightImu
          ? analyzeSequenceImu(s.rightImu, s.frameCount, s.timestamps)
          : null,
      };
    if (isImuSuspect(h.left) || isImuSuspect(h.right)) n++;
  }
  return n;
}

/**
 * 起手段裁剪的一句话汇报。
 *
 * 裁剪是**没有 UI 的自动预处理**（见 sequenceTrim.ts），所以这一行是它唯一的出口：
 * 裁了几条、平均留下多少、哪几条没裁成必须能读到。全部没裁成时更要说 —— 那通常意味着
 * 这批数据是没开摄像头录的（no_vision），起手段还在里面。
 */
function describeTrim(t: TrimStats): string {
  if (t.total === 0) return "";
  if (t.applied === 0) {
    const why = t.skipped.no_vision > 0 ? "没有视觉，判不出入画时刻" : "判据没成立";
    return ` — 起手段裁剪：0/${t.total} 条（${why}，起手段仍在数据里）`;
  }
  const kept = (t.meanKeptRatio * 100).toFixed(0);
  const skipped = (
    [
      ["no_vision", "无视觉"],
      ["no_run", "全程没看见手"],
      ["too_short", "裁完太短"],
      ["full_span", "本来就全程在画面里"],
    ] as const
  )
    .filter(([k]) => t.skipped[k] > 0)
    .map(([k, label]) => `${label} ${t.skipped[k]}`)
    .join(" / ");
  // 到位检测（第二层）单列。它切的是**画面内**那段抬手，第一层看不见，
  // 所以"裁了几条"这一个数字分不清两层各自出了多少力。
  const arrival =
    t.arrivalApplied > 0
      ? `，其中 ${t.arrivalApplied} 条另切掉画面内抬手平均 ${t.meanArrivalDroppedMs.toFixed(0)}ms` +
        (t.arrivalBendClamped > 0
          ? `（${t.arrivalBendClamped} 条因手型提前成形而少切）`
          : "")
      : "";
  return (
    ` — 起手段裁剪：${t.applied}/${t.total} 条，平均留下 ${kept}%${arrival}` +
    (skipped ? `（未裁：${skipped}）` : "")
  );
}

export default function TrainSequence() {
  const [stats, setStats] = useState<SequenceStats | null>(null);
  const [models, setModels] = useState<SavedModel[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [progress, setProgress] = useState<SeqTrainingProgress | null>(null);
  const [history, setHistory] = useState<SeqTrainingProgress[]>([]);
  const [message, setMessage] = useState("");
  const [visionRatio, setVisionRatio] = useState(0);
  const [imuSuspect, setImuSuspect] = useState(0);
  const [recordedTotal, setRecordedTotal] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(isSequenceModelLoaded());

  // 训练参数
  const [seqLen, setSeqLen] = useState(DEFAULT_SEQ_CONFIG.seqLen);
  const [epochs, setEpochs] = useState(DEFAULT_SEQ_CONFIG.epochs);
  const [batchSize, setBatchSize] = useState(DEFAULT_SEQ_CONFIG.batchSize);
  const [learningRate, setLearningRate] = useState(
    DEFAULT_SEQ_CONFIG.learningRate
  );
  const [distillationTemp, setDistillationTemp] = useState(
    DEFAULT_SEQ_CONFIG.distillationTemp
  );
  const [distillationAlpha, setDistillationAlpha] = useState(
    DEFAULT_SEQ_CONFIG.distillationAlpha
  );
  const [backbone, setBackbone] = useState<SeqBackbone>("tcn");
  const [augmentCopies, setAugmentCopies] = useState(
    DEFAULT_SEQ_CONFIG.augmentCopies
  );
  const [timeWarp, setTimeWarp] = useState(DEFAULT_AUGMENT.timeWarp);

  const refresh = useCallback(async () => {
    const s = await getSequenceStats();
    setStats(s);
    setModels(await getAllSequenceModels());
    setModelLoaded(isSequenceModelLoaded());
    const seqs = await getAllSequences();
    setVisionRatio(sequenceVisionRatio(seqs));
    setImuSuspect(countImuSuspect(seqs));
    setRecordedTotal(seqs.filter((s) => s.origin === "recorded").length);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // ===== 从静态样本合成序列 =====

  const handleSynthesize = useCallback(async () => {
    setIsBusy(true);
    setMessage("正在读取旧的静态样本...");
    try {
      const statics = await getAllSamples();
      if (statics.length === 0) {
        setMessage("没有静态样本可迁移（/collect 里还没采过）");
        return;
      }
      const seqs = statics.map((s) => synthesizeFromStatic(s, DEFAULT_SYNTHESIZE));
      await addSequences(seqs);
      setMessage(
        `已从 ${statics.length} 条静态样本合成序列。注意：合成序列的帧间统计与真实录制不完全一致，正式训练前建议重录真实序列做校准。`
      );
      await refresh();
    } catch (e) {
      setMessage(`合成失败：${e}`);
    } finally {
      setIsBusy(false);
    }
  }, [refresh]);

  const handleDeleteSynth = useCallback(async () => {
    setIsBusy(true);
    try {
      const n = await deleteSynthesizedSequences();
      setMessage(`已删除 ${n} 条合成序列`);
      await refresh();
    } finally {
      setIsBusy(false);
    }
  }, [refresh]);

  // ===== 训练 =====

  const handleTrain = useCallback(async () => {
    setIsTraining(true);
    setHistory([]);
    setProgress(null);
    setMessage("正在加载序列数据集...");
    try {
      const seqs = await getAllSequences();
      if (seqs.length < 10) {
        setMessage(`序列样本只有 ${seqs.length} 条，太少了，先去 /collect-seq 采集`);
        return;
      }

      const result = await trainSequenceModel(
        seqs,
        {
          seqLen,
          epochs,
          batchSize,
          learningRate,
          distillationTemp,
          distillationAlpha,
          backbone,
          augmentCopies,
          augment: { ...DEFAULT_AUGMENT, timeWarp },
        },
        (p) => {
          setProgress(p);
          setHistory((h) => [...h, p]);
        }
      );

      setMessage("训练完成，正在保存模型...");
      const ts = new Date().toISOString().slice(0, 16).replace("T", " ");

      if (result.teacherModel) {
        await saveModel(
          await serializeSequenceModel(
            result.teacherModel,
            result.labels,
            result.teacherAccuracy,
            `seq_teacher_${ts}`,
            "seq_fused",
            seqLen,
            backbone
          )
        );
      }
      const student = await serializeSequenceModel(
        result.studentModel,
        result.labels,
        result.studentAccuracy,
        `seq_student_${ts}`,
        "seq_tactile",
        seqLen,
        backbone
      );
      await saveModel(student);
      await loadSequenceModelFromSaved(student);

      // 验证集是**按整条录制**切的、且不含增强副本，所以这个准确率可以当真
      // （详见 sequenceModel.ts 的 planTrainValSplit）。valSamples=0 时必须说清不可信
      setMessage(
        `完成。教师 ${(result.teacherAccuracy * 100).toFixed(1)}% / 学生 ${(result.studentAccuracy * 100).toFixed(1)}%（学生已设为当前推理模型）` +
          (result.split.valSamples > 0
            ? ` — 验证集 ${result.split.valSamples} 条 / 训练 ${result.split.trainSamples} 条（按整条录制划分，验证集不做增强）`
            : " — ⚠ 验证集为空，准确率不可信，每类至少要 2 条") +
          (result.teacherModel
            ? ""
            : " — 视觉覆盖不足 80%，本次跳过了教师与蒸馏") +
          describeTrim(result.trim)
      );
      await refresh();
    } catch (e) {
      console.error("[TrainSeq] 训练失败", e);
      setMessage(`训练失败：${e}`);
    } finally {
      setIsTraining(false);
    }
  }, [
    seqLen,
    epochs,
    batchSize,
    learningRate,
    distillationTemp,
    distillationAlpha,
    backbone,
    augmentCopies,
    timeWarp,
    refresh,
  ]);

  // ===== 导出给 Python =====

  const handleExport = useCallback(async () => {
    setIsBusy(true);
    setMessage("正在打包序列数据集...");
    try {
      const { bin, manifest } = await exportSequencesBinary();
      downloadBlob(new Blob([bin]), "dataset.bin");
      downloadBlob(
        new Blob([JSON.stringify(manifest)], { type: "application/json" }),
        "dataset.json"
      );
      setMessage(
        `已导出 ${manifest.totalSequences} 条（${(bin.byteLength / 1024 / 1024).toFixed(1)} MB）。把两个文件放进 python_train/data/ 再跑 train_seq.py`
      );
    } catch (e) {
      setMessage(`导出失败：${e}`);
    } finally {
      setIsBusy(false);
    }
  }, []);

  const handleLoadModel = useCallback(
    async (m: SavedModel) => {
      await loadSequenceModelFromSaved(m);
      setModelLoaded(true);
      setMessage(`已载入 ${m.name} 作为当前推理模型`);
    },
    []
  );

  const handleDeleteModel = useCallback(
    async (id?: number) => {
      if (id === undefined) return;
      await deleteModel(id);
      await refresh();
    },
    [refresh]
  );

  const labelRows = stats
    ? Object.entries(stats.labelCounts).sort((a, b) =>
        a[0].localeCompare(b[0])
      )
    : [];
  const hasIdle = (stats?.labelCounts[IDLE_LABEL]?.recorded ?? 0) > 0;
  const hc = stats?.handCounts ?? {
    leftOnly: 0,
    rightOnly: 0,
    both: 0,
    neither: 0,
  };
  // 只在**确实有单手样本、且全偏在一侧**时告警。两边都有、或全是双手样本都不算偏
  const handSkewed =
    hc.leftOnly + hc.rightOnly > 0 && (hc.leftOnly === 0 || hc.rightOnly === 0);

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: "#0a0e1a" }}
    >
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
          <span
            className="text-xs font-bold tracking-widest"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              // 紫＝时序链路（与 /collect-seq、/translate 的 MODE 开关同色），
              // 青＝静态链路（/train）。两个训练页长得像，颜色是第一道区分
              color: "#a855f7",
            }}
          >
            SEQUENCE TRAINING
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            时序滑窗 · TCN + DISTILLATION
          </span>
          <TfBackendBadge />
        </div>
        <div className="flex items-center gap-4">
          <div className="text-[10px] font-mono text-[#556677]">
            MODEL:{" "}
            <span className={modelLoaded ? "text-[#00e5a0]" : "text-[#556677]"}>
              {modelLoaded ? "LOADED" : "NONE"}
            </span>
          </div>
          <StepNav />
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 左：数据集 + 参数 */}
        <div className="w-80 border-r border-[#00f0ff]/15 overflow-y-auto shrink-0 p-3 space-y-4">
          <Section title="DATASET">
            <div className="space-y-1">
              <DataRow
                label="序列总数"
                value={String(stats?.totalSequences ?? 0)}
                color="#00f0ff"
              />
              <DataRow
                label="真实录制"
                value={String(stats?.recordedCount ?? 0)}
                color="#00e5a0"
              />
              <DataRow
                label="静态合成"
                value={String(stats?.synthesizedCount ?? 0)}
                color="#f59e0b"
              />
              <DataRow
                label="平均时长"
                value={`${Math.round(stats?.avgDurationMs ?? 0)}ms`}
                color="#ccd6e0"
              />
              <DataRow
                label="估算占用"
                value={`${((stats?.estimatedBytes ?? 0) / 1024 / 1024).toFixed(1)} MB`}
                color="#ccd6e0"
              />
              <DataRow
                label="视觉覆盖"
                value={`${(visionRatio * 100).toFixed(0)}%`}
                color={visionRatio >= 0.8 ? "#00e5a0" : "#f59e0b"}
              />
              <DataRow
                label="IMU 可疑"
                value={`${imuSuspect} / ${recordedTotal} 条`}
                color={imuSuspect === 0 ? "#00e5a0" : "#f59e0b"}
              />
              <DataRow
                label="手别分布"
                value={`左 ${hc.leftOnly} · 右 ${hc.rightOnly} · 双 ${hc.both}`}
                color={handSkewed ? "#f59e0b" : "#00e5a0"}
              />
            </div>
            {handSkewed && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                单手样本全在{hc.leftOnly === 0 ? "右" : "左"}手 —— 模型在另一只手的槽位上
                从没见过信号，直接戴另一只手套做同一个词会塌到某个固定的词上。
                单手词已由 /translate 的镜像归一化兜住（只戴一只手套时生效）；
                双手词兜不住，得真去补采。
              </div>
            )}
            {imuSuspect > 0 && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                有 {imuSuspect} 条样本录制时陀螺姿态在漂 —— 它们的四元数通道不可信，
                帧数和运动能量都看不出问题。先去 /mocap 做「静置自检」确认手套状态，
                再决定是重录还是接受。
              </div>
            )}
            {visionRatio < 0.8 && (stats?.totalSequences ?? 0) > 0 && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                视觉覆盖低于 80%，训练会跳过教师与蒸馏，直接用 hard label
                训学生 —— 准确率通常会明显下降。
              </div>
            )}
            {!hasIdle && (stats?.totalSequences ?? 0) > 0 && (
              <div className="text-[9px] text-[#ff2d7b] font-mono leading-relaxed">
                还没有 `_idle` 样本。滑窗推理下模型对任意窗口都会强行输出一个词，
                没有空闲类，翻译页会在手放松时持续乱吐词。去 /collect-seq 补录。
              </div>
            )}
          </Section>

          <Section title="PER LABEL">
            <div className="space-y-0.5 max-h-48 overflow-y-auto">
              {labelRows.map(([label, c]) => (
                <div
                  key={label}
                  className="flex justify-between text-[10px] font-mono"
                >
                  <span className="text-[#556677]">
                    {getDisplayLabel(label)}
                  </span>
                  <span>
                    <span className="text-[#00e5a0]">{c.recorded}</span>
                    <span className="text-[#334455]">/</span>
                    <span className="text-[#f59e0b]">{c.synthesized}</span>
                  </span>
                </div>
              ))}
              {labelRows.length === 0 && (
                <div className="text-[10px] text-[#334455] font-mono">
                  还没有序列样本
                </div>
              )}
            </div>
          </Section>

          <Section title="MIGRATION">
            <button
              onClick={handleSynthesize}
              disabled={isBusy || isTraining}
              className="cyber-btn w-full px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1"
            >
              <Wand2 className="w-3 h-3" />
              从静态样本合成序列
            </button>
            <button
              onClick={handleDeleteSynth}
              disabled={isBusy || isTraining}
              className="cyber-btn w-full px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1 text-[#ff2d7b]"
            >
              <Trash2 className="w-3 h-3" />
              删除全部合成序列
            </button>
          </Section>

          <Section title="HYPERPARAMS">
            <ParamInput label="序列长度 T" value={seqLen} onChange={setSeqLen} min={8} max={128} />
            <ParamInput label="Epochs" value={epochs} onChange={setEpochs} min={5} max={500} />
            <ParamInput label="Batch" value={batchSize} onChange={setBatchSize} min={1} max={128} />
            <ParamInput label="学习率" value={learningRate} onChange={setLearningRate} min={0.00001} max={0.1} step={0.0001} isFloat />
            <ParamInput label="蒸馏温度" value={distillationTemp} onChange={setDistillationTemp} min={1} max={10} step={0.5} isFloat />
            <ParamInput label="蒸馏 α" value={distillationAlpha} onChange={setDistillationAlpha} min={0} max={1} step={0.05} isFloat />
            <ParamInput label="增强副本数" value={augmentCopies} onChange={setAugmentCopies} min={0} max={5} />
            <ParamInput label="时间扭曲" value={timeWarp} onChange={setTimeWarp} min={0} max={0.5} step={0.05} isFloat />
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[#556677]">骨干网络</span>
              <select
                value={backbone}
                onChange={(e) => setBackbone(e.target.value as SeqBackbone)}
                className="bg-[#1a2030] border border-[#00f0ff]/20 rounded-sm px-1.5 py-0.5 text-[#00f0ff] text-[10px]"
              >
                <option value="tcn">tcn</option>
                <option value="tcn_bigru">tcn_bigru</option>
              </select>
            </div>
            {backbone === "tcn_bigru" && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                BiGRU 在浏览器里训练明显更慢（RNN 无法在时间维并行）。
                孤立词阶段建议留在 tcn，这一项主要留给句子阶段的 Python 训练。
              </div>
            )}
          </Section>

          <Section title="ACTIONS">
            <button
              onClick={handleTrain}
              disabled={isTraining || isBusy}
              className="cyber-btn w-full px-2 py-2 rounded-sm text-[11px] flex items-center justify-center gap-1"
            >
              <Play className="w-3 h-3" />
              {isTraining ? "训练中..." : "开始训练"}
            </button>
            <button
              onClick={handleExport}
              disabled={isTraining || isBusy}
              className="cyber-btn w-full px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1"
            >
              <Download className="w-3 h-3" />
              导出数据集给 Python
            </button>
          </Section>
        </div>

        {/* 右：进度 + 模型列表 */}
        <div className="flex-1 p-4 space-y-4 overflow-y-auto">
          {message && (
            <div className="cyber-panel p-2 rounded-sm text-[10px] font-mono text-[#00e5a0] leading-relaxed">
              {message}
            </div>
          )}

          <div className="grid grid-cols-4 gap-2">
            <MetricCard
              label="Phase"
              value={progress?.phase ?? "—"}
              color="#a855f7"
            />
            <MetricCard
              label="Epoch"
              value={
                progress ? `${progress.epoch}/${progress.totalEpochs}` : "—"
              }
              color="#00f0ff"
            />
            <MetricCard
              label="Train Acc"
              value={progress ? `${(progress.accuracy * 100).toFixed(1)}%` : "—"}
              color="#00e5a0"
            />
            <MetricCard
              label="Val Acc"
              value={
                progress ? `${(progress.valAccuracy * 100).toFixed(1)}%` : "—"
              }
              color="#f59e0b"
            />
          </div>

          <div className="cyber-panel p-3 rounded-sm">
            <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-2">
              Training Curve
            </div>
            <SeqTrainingChart history={history} />
          </div>

          <div className="cyber-panel p-3 rounded-sm">
            <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-2">
              Saved Sequence Models
            </div>
            <div className="space-y-1">
              {models.length === 0 && (
                <div className="text-[10px] text-[#334455] font-mono">
                  还没有时序模型
                </div>
              )}
              {models.map((m) => (
                <div
                  key={m.id}
                  className="flex items-center justify-between px-2 py-1.5 rounded-sm border border-[#00f0ff]/10 text-[10px] font-mono"
                >
                  <div>
                    <span className="text-[#ccd6e0]">{m.name}</span>
                    <span className="text-[#556677] ml-2">
                      {m.modelType} · T={m.seqLen} · {m.backbone} ·{" "}
                      {m.labels.length}类 · {(m.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => void handleLoadModel(m)}
                      className="cyber-btn px-1.5 py-0.5 rounded-sm"
                      title="设为当前推理模型"
                    >
                      <Upload className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => void handleDeleteModel(m.id)}
                      className="text-[#ff2d7b]"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="cyber-panel p-3 rounded-sm text-[10px] font-mono text-[#556677] leading-relaxed">
            <div className="flex items-center gap-1 text-[#00f0ff] mb-1">
              <Brain className="w-3 h-3" />
              句子级（连续手语）
            </div>
            tfjs 4.22 没有 CTC loss 和 CTC 解码器，句子级训练无法在浏览器里做。
            数据 schema 已经按句子级设计好（每条序列存 segments 词边界列表，
            孤立词只是长度为 1 的特例），骨干与 head 也分离了，
            所以到句子阶段直接用「导出数据集给 Python」→ python_train/train_seq.py --ctc，
            训完用 export_to_tfjs.py 转回来即可，不需要重新采集数据。
          </div>
        </div>
      </div>
    </div>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
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

/** 训练曲线。教师段与学生段用竖线分开，否则两段 loss 尺度不同看起来像发散 */
function SeqTrainingChart({ history }: { history: SeqTrainingProgress[] }) {
  const W = 600;
  const H = 160;
  const pad = 28;
  if (history.length < 2) {
    return (
      <div className="text-[10px] text-[#334455] font-mono h-[160px] flex items-center justify-center">
        等待训练数据...
      </div>
    );
  }

  const n = history.length;
  const x = (i: number) => pad + (i / (n - 1)) * (W - pad * 2);
  const y = (v: number) => H - pad - v * (H - pad * 2);
  const line = (pick: (p: SeqTrainingProgress) => number) =>
    history.map((p, i) => `${x(i).toFixed(1)},${y(pick(p)).toFixed(1)}`).join(" ");

  const switchIdx = history.findIndex((p) => p.phase === "student");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={pad} y1={y(0)} x2={W - pad} y2={y(0)} stroke="#00f0ff20" />
        <line x1={pad} y1={y(1)} x2={W - pad} y2={y(1)} stroke="#00f0ff20" />
        {switchIdx > 0 && (
          <line
            x1={x(switchIdx)}
            y1={pad}
            x2={x(switchIdx)}
            y2={H - pad}
            stroke="#a855f7"
            strokeDasharray="3 3"
          />
        )}
        <polyline points={line((p) => p.accuracy)} fill="none" stroke="#00e5a0" strokeWidth="1.5" />
        <polyline points={line((p) => p.valAccuracy)} fill="none" stroke="#f59e0b" strokeWidth="1.5" />
      </svg>
      <div className="flex gap-3 text-[9px] font-mono mt-1">
        <span className="text-[#00e5a0]">train acc</span>
        <span className="text-[#f59e0b]">val acc</span>
        {switchIdx > 0 && <span className="text-[#a855f7]">教师→学生切换</span>}
      </div>
    </div>
  );
}
