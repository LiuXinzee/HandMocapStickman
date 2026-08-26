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
import {
  ArrowLeft,
  Brain,
  ClipboardCopy,
  Download,
  Play,
  Stethoscope,
  Trash2,
  Upload,
  Wand2,
  X,
} from "lucide-react";
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
import { loadBendRange } from "@/lib/bendRange";
import {
  summarizeHandedness,
  describeHandedness,
  type BendRanges,
  type HandednessStats,
} from "@/lib/dominantHand";
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
import { auditDataset, formatAuditReport } from "@/lib/datasetAudit";
import { probeYawReference, formatYawProbe } from "@/lib/yawDrift";
import { classesRemovedBy } from "@/lib/labelMerge";
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

/**
 * 两只手的弯折两点标定。主手判定要按各自量程归一化（实测左右量程差 40~176），
 * 而标定只存在 `localStorage` 里 —— 库里读不到，只能页面读了往下传。
 * 没标定不会让判定失效，两只手一起走兜底量程，会在归一化汇总里报出来。
 */
function currentBendRanges(): BendRanges {
  return { LH: loadBendRange("LH"), RH: loadBendRange("RH") };
}

/**
 * 训练时排除哪些词。**过滤，不删数据** —— 库里一条不动，取消排除就原样训回来。
 *
 * 存进 localStorage 而不是每次重置：排除是用来做对照实验的（"去掉这几个词，
 * 剩下的会不会准"），一轮实验要跑训练 → 去 /translate 试 → 回来看，中间刷新页面
 * 很正常。重置会让人在不知情的情况下训出一个全类别模型、却以为是排除过的，
 * 那就把整个实验读反了。
 *
 * 代价是它会一直生效，所以要有两道明显的出口：非空时界面上有红色横幅 +
 * 一键取消，且模型名带 `_ex{n}` 后缀（保存列表里那一行的"N 类"也会跟着变）。
 */
const EXCLUDED_KEY = "seq_train_excluded";

function loadExcluded(): Set<string> {
  try {
    const raw = localStorage.getItem(EXCLUDED_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return new Set(
      Array.isArray(arr) ? arr.filter((x): x is string => typeof x === "string") : []
    );
  } catch {
    return new Set();
  }
}

function saveExcluded(s: Set<string>): void {
  try {
    localStorage.setItem(EXCLUDED_KEY, JSON.stringify(Array.from(s)));
  } catch {
    // 隐私模式/配额满时写不进去。实验照样能跑完，只是刷新后要重新勾
  }
}

/**
 * 会话开头总是先蹦出来的那三个词。
 *
 * 它们不是随机的：滑窗要攒满一整个窗口（2000ms）才出第一个预测，所以会话里
 * 第一个窗口装的是「准备时的静止 + 第一个词刚起手的一小截」—— 这个形状恰好
 * 等于一条静止头段很长的训练样本。这三个词就是头段最长的那几个，模型输出
 * 它们其实是照它学到的东西正确作答。
 *
 * 做成一键预设而不是让人点三行，是因为漏点一行的后果不对称：训出来的是个
 * 全类别模型，却会被当成排除过的那版去读 —— 那是把实验结论读反，比没做更糟。
 */
const PHANTOM_TRIO = ["hello", "name", "happy"];

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
  const [handedness, setHandedness] = useState<HandednessStats | null>(null);
  const [auditText, setAuditText] = useState("");
  const [modelLoaded, setModelLoaded] = useState(isSequenceModelLoaded());
  const [excluded, setExcluded] = useState<Set<string>>(loadExcluded);

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
    setHandedness(summarizeHandedness(seqs, currentBendRanges()));
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggleExcluded = useCallback((label: string) => {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      saveExcluded(next);
      return next;
    });
  }, []);

  const clearExcluded = useCallback(() => {
    setExcluded(new Set());
    saveExcluded(new Set());
  }, []);

  /**
   * 一键排除/恢复幽灵三词。
   *
   * `words` 只传库里真有的那几个 —— 传了不存在的标签，过滤时一条也匹配不到，
   * 但 `excluded.size` 会虚高、模型名上的 `_ex{n}` 跟着虚高，下次看保存列表
   * 就对不上号了
   */
  const toggleTrio = useCallback((words: string[]) => {
    setExcluded((prev) => {
      const next = new Set(prev);
      const allOff = words.every((w) => next.has(w));
      for (const w of words) {
        if (allOff) next.delete(w);
        else next.add(w);
      }
      saveExcluded(next);
      return next;
    });
  }, []);

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
      const all = await getAllSequences();
      // 排除是**过滤不是删除**：库里一条不动。类别表是从传进去的样本里现推的
      // （sequenceModel.ts 的 `labels`），所以过滤完类别数、softmax 宽度、
      // 验证集划分会自动跟着缩，这里不需要再动模型侧任何东西
      const seqs = excluded.size
        ? all.filter((s) => !excluded.has(s.primaryLabel))
        : all;
      const droppedRows = all.length - seqs.length;
      const remainingLabels = new Set(seqs.map((s) => s.primaryLabel)).size;

      if (seqs.length < 10) {
        setMessage(
          `可用序列只有 ${seqs.length} 条，太少了` +
            (droppedRows > 0
              ? `（已被排除 ${droppedRows} 条 —— 去左栏 PER LABEL 点一下取消）`
              : "，先去 /collect-seq 采集")
        );
        return;
      }
      if (remainingLabels < 2) {
        setMessage(
          `排除之后只剩 ${remainingLabels} 个词，至少要 2 个才能训。去左栏 PER LABEL 取消几个排除。`
        );
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
          bendRanges: currentBendRanges(),
        },
        (p) => {
          setProgress(p);
          setHistory((h) => [...h, p]);
        }
      );

      setMessage("训练完成，正在保存模型...");
      const ts = new Date().toISOString().slice(0, 16).replace("T", " ");
      // 名字里带上排除数：保存列表里两个模型只差几个类别时，光看时间戳分不出
      // 哪个是对照实验那一版，而选错了会把结论读反
      const tag = excluded.size ? `_ex${excluded.size}` : "";

      if (result.teacherModel) {
        await saveModel(
          await serializeSequenceModel(
            result.teacherModel,
            result.labels,
            result.teacherAccuracy,
            `seq_teacher_${ts}${tag}`,
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
        `seq_student_${ts}${tag}`,
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
          describeTrim(result.trim) +
          (result.handedness
            ? ` — 主手归一化：${describeHandedness(result.handedness)}`
            : "") +
          // 合并同样会减类别数，所以和排除一样必须紧贴着准确率出现。
          // 这里说的是**为什么**合并，不是"合并了"—— 光说合并了，下次看到的人
          // 只会想着把它关掉
          (result.merge.merged > 0
            ? ` — 已合并 ${result.merge.groups
                .map((g) => g.display)
                .join("、")}（共 ${result.merge.merged} 条）：这几个词只差指向，` +
              `六轴 IMU 测不到绝对 yaw，实测漂移已盖过类间距，分不开是硬件限制而非训练不足。`
            : "") +
          // 句子样本被挡掉了这件事必须说 —— 采集页里它们是实实在在录进去的条数，
          // 这里不说的话会被当成"数据丢了"
          (result.sentencesExcluded > 0
            ? ` — 另有 ${result.sentencesExcluded} 条句子级样本未参与（多个词连着打的连续手语，` +
              `孤立词模型吃不了；句子模型走 python_train/train_seq.py --ctc）。`
            : "") +
          // 类别数变了，准确率就不可比 —— 24 类天生比 27 类容易。这句话必须紧贴着
          // 那个百分数出现，否则很容易把"数字变高了"读成"排除起作用了"
          (excluded.size
            ? ` — ⚠ 本次排除了 ${excluded.size} 个词（${Array.from(excluded)
                .map(getDisplayLabel)
                .join("、")}）共 ${droppedRows} 条，模型只有 ${remainingLabels} 类。` +
              `类别越少准确率天生越高，这个百分数不能和全类别那次直接比 —— 要比就去 /translate 看实际输出。`
            : "")
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
    excluded,
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

  /**
   * 数据体检。**只读** —— 不删、不改、不写库，跑完随时可以关掉。
   *
   * 之所以是一份文本报告而不是几个界面数字：左栏那些统计是全局聚合的，
   * 而这批数据是分两次采的、两次的采集方式不一样，全局平均会把差异整个抹平。
   * 要定"删哪些、阈值多少"必须先看到**按批次**和**按词**的分布。
   */
  const handleAudit = useCallback(async () => {
    setIsBusy(true);
    setMessage("正在体检...");
    try {
      const seqs = await getAllSequences();
      if (seqs.length === 0) {
        setMessage("没有序列样本");
        return;
      }
      const report = auditDataset(seqs, currentBendRanges());
      // YAW 参考系探针拼在同一份报告里：它也是全库只读扫描，单独给一个按钮
      // 只会多一次点击、多一次复制，而这两份结论本来就要放在一起看
      const yaw = probeYawReference(seqs, currentBendRanges());
      setAuditText(
        formatAuditReport(report) + "\n\n" + formatYawProbe(yaw)
      );
      setMessage(
        `体检完成：${report.total} 条、${report.byDay.length} 个采集批次、${report.flags.length} 项待处理。` +
          `YAW 参考系：${yaw.verdict === "viable" ? "可行" : yaw.verdict === "marginal" ? "勉强" : yaw.verdict === "dead" ? "不可行" : "数据不足"}。` +
          `报告在右侧，复制给我。`
      );
    } catch (e) {
      setMessage(`体检失败：${e}`);
    } finally {
      setIsBusy(false);
    }
  }, []);

  const handleCopyAudit = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(auditText);
      setMessage("报告已复制到剪贴板");
    } catch {
      // 剪贴板 API 在非 https / 无权限时会拒绝；报告本身就在 <pre> 里，手选也能复制
      setMessage("复制失败，手动选中右侧文本复制即可");
    }
  }, [auditText]);

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

  /**
   * 「这次点下去到底会训什么」—— 训练**之前**就摊开。
   *
   * 之前只有训完才能从模型名的 `_ex{n}` 后缀反推排除有没有生效，而那时候
   * 结果已经出来了，人自然会拿它当排除过的那版去读
   */
  // 合并会减类别数，所以这里要按合并**之后**的口径报，否则训练前说 24 类、
  // 训完变 20 类，那个读数就没用了
  const mergedAway = classesRemovedBy(
    labelRows.filter(([l]) => !excluded.has(l)).map(([l]) => l)
  );
  const plan = labelRows.reduce(
    (a, [label, c]) => {
      const n = c.recorded + c.synthesized;
      if (excluded.has(label)) {
        a.droppedLabels++;
        a.droppedRows += n;
      } else {
        a.labels++;
        a.rows += n;
      }
      return a;
    },
    { labels: 0, rows: 0, droppedLabels: 0, droppedRows: 0 }
  );
  plan.labels -= mergedAway;

  // 只对库里真有的那几个词提供一键排除
  const trioPresent = PHANTOM_TRIO.filter((w) => w in (stats?.labelCounts ?? {}));
  const trioOff =
    trioPresent.length > 0 && trioPresent.every((w) => excluded.has(w));
  const hc = stats?.handCounts ?? {
    leftOnly: 0,
    rightOnly: 0,
    both: 0,
    neither: 0,
  };
  // 只在**确实有单手样本、且全偏在一侧**时告警。两边都有、或全是双手样本都不算偏
  const handSkewed =
    hc.leftOnly + hc.rightOnly > 0 && (hc.leftOnly === 0 || hc.rightOnly === 0);
  // 主手混采：左右都有一批。这不是问题，训练会自动镜像归一化 —— 这一行只是让
  // 「我这批数据是混采的」这件事看得见（`handCounts` 那一行看不见，见 currentBendRanges 上面的注释）
  const handMixed = !!handedness && handedness.left > 0 && handedness.right > 0;
  const tieHeavy =
    !!handedness &&
    handedness.left + handedness.right > 0 &&
    handedness.nearTie > (handedness.left + handedness.right) * 0.3;

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
              {/* 只在真有句子样本时出现。它们不进下面的每词条数（primaryLabel 是第一个词，
                  算进去会给那个词虚增），所以必须在这里单独有一行，否则"总数比每词之和多"
                  这件事在页面上无处可查 */}
              {(stats?.sentenceCount ?? 0) > 0 && (
                <DataRow
                  label="句子样本"
                  value={`${stats?.sentenceCount} · 不参与孤立词`}
                  color="#a855f7"
                />
              )}
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
                label="手套连接"
                value={`左 ${hc.leftOnly} · 右 ${hc.rightOnly} · 双 ${hc.both}`}
                color={handSkewed ? "#f59e0b" : "#00e5a0"}
              />
              <DataRow
                label="主手分布"
                value={
                  handedness
                    ? `左 ${handedness.left} · 右 ${handedness.right} · 静止 ${handedness.idle}`
                    : "—"
                }
                color={handMixed ? "#a855f7" : "#00e5a0"}
              />
            </div>
            {handMixed && (
              <div className="text-[9px] text-[#a855f7] font-mono leading-relaxed">
                左右手混采（不是问题）：训练会把 {handedness!.left} 条左手样本整条镜像到
                右手口径再喂模型，与 /translate 的推理口径一致。不做这一步的话，同一个词
                的两半会落进两段零重叠的槽位，网络只能退回类先验 —— 表现就是"打什么都输出
                同一个词"。
              </div>
            )}
            {tieHeavy && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                有 {handedness!.nearTie} 条样本左右手活动量接近，主手判定基本是掷硬币。
                多为双手词（镜像与否影响本来就小）；如果里面有单手词，说明闲着那只手
                动得太多，采集时让它自然垂下。
              </div>
            )}
            {handedness && handedness.calibrated < handedness.total && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                有 {handedness.total - handedness.calibrated} 条样本判定时没有两点标定可用，
                走了兜底量程。左右手弯折量程实测差 40~176，不标定会系统性偏向量程大的那只手。
                去第 1 步给两只手都做一次张开/握拳标定。
              </div>
            )}
            {handSkewed && (
              <div className="text-[9px] text-[#f59e0b] font-mono leading-relaxed">
                只连了一只手套的样本全在{hc.leftOnly === 0 ? "右" : "左"}手 ——
                这些样本的另一段槽位是**全 0**，而两只手套都戴着录的样本，闲着那只手
                出的是静止基线、不是 0。两种输入长得不一样，混在一起训会多出一个
                与词义无关的因子。要么统一戴两只，要么统一戴一只。
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
            <div className="text-[9px] text-[#334455] font-mono">
              点一个词把它从训练里排除（不删数据，再点一下恢复）
            </div>
            {trioPresent.length > 0 && (
              <button
                onClick={() => toggleTrio(trioPresent)}
                disabled={isTraining}
                title={
                  trioOff
                    ? "把这三个词放回训练"
                    : "会话开头总是先输出这三个词 —— 一键把它们从训练里排除，验证它们是不是靠静止头段赢的"
                }
                className={
                  "w-full px-2 py-1 rounded-sm text-[10px] font-mono border " +
                  (trioOff
                    ? "border-[#ff2d7b] text-[#ff2d7b]"
                    : "border-[#334455] text-[#556677] hover:border-[#00f0ff] hover:text-[#00f0ff]")
                }
              >
                {trioOff ? "✓ 已排除" : "排除"}开头那三个词（
                {trioPresent.map(getDisplayLabel).join(" / ")}）
              </button>
            )}
            {/* 训练前的口径。数字对不上就别点训练 —— 上一轮就是这么读反的 */}
            <div className="text-[9px] font-mono">
              <span className="text-[#334455]">本次将训练 </span>
              <span className="text-[#00e5a0]">{plan.labels}</span>
              <span className="text-[#334455]"> 类 / </span>
              <span className="text-[#00e5a0]">{plan.rows}</span>
              <span className="text-[#334455]"> 条</span>
              {plan.droppedLabels > 0 && (
                <span className="text-[#ff2d7b]">
                  ，排除 {plan.droppedLabels} 类 / {plan.droppedRows} 条
                </span>
              )}
              {mergedAway > 0 && (
                <span className="text-[#a855f7]">（含合并省掉 {mergedAway} 类）</span>
              )}
            </div>
            {mergedAway > 0 && (
              <div className="text-[9px] text-[#a855f7] font-mono leading-relaxed">
                我/你/他 与 我们/你们/他们 各自合并为一类：这几个词只差指向(yaw)，
                六轴 IMU 无磁力计测不到绝对 yaw，实测漂移 P90 15.9°/s、10 秒累计 159°，
                而类间距仅 45°。**库里的原始标签没动** —— 换九轴 IMU 后重测漂移即可退回。
              </div>
            )}
            <div className="space-y-0.5 max-h-48 overflow-y-auto">
              {labelRows.map(([label, c]) => {
                const off = excluded.has(label);
                return (
                  <button
                    key={label}
                    onClick={() => toggleExcluded(label)}
                    disabled={isTraining}
                    title={
                      off
                        ? "已排除：本次训练不含这个词，模型也不会输出它。点一下恢复"
                        : "点一下把这个词从训练里排除（只是过滤，数据不删）"
                    }
                    className="w-full flex justify-between items-center text-[10px] font-mono px-1 py-0.5 rounded-sm hover:bg-[#00f0ff]/5"
                  >
                    <span
                      className={
                        off ? "text-[#ff2d7b] line-through" : "text-[#556677]"
                      }
                    >
                      {getDisplayLabel(label)}
                    </span>
                    <span className={off ? "opacity-25" : ""}>
                      <span className="text-[#00e5a0]">{c.recorded}</span>
                      <span className="text-[#334455]">/</span>
                      <span className="text-[#f59e0b]">{c.synthesized}</span>
                    </span>
                  </button>
                );
              })}
              {labelRows.length === 0 && (
                <div className="text-[10px] text-[#334455] font-mono">
                  还没有序列样本
                </div>
              )}
            </div>
            {excluded.size > 0 && (
              <>
                <div className="text-[9px] text-[#ff2d7b] font-mono leading-relaxed">
                  已排除 {excluded.size} 个词。这是**过滤不是删除** ——
                  库里一条没动，但训出来的模型里没有这些类，/translate 永远不会输出它们。
                  排除状态会一直保留（刷新页面也在），别忘了做完实验取消。
                  另外类别数变了，准确率不能和之前那次比，类别越少本身就越容易。
                </div>
                {excluded.has(IDLE_LABEL) && (
                  <div className="text-[9px] text-[#ff2d7b] font-mono leading-relaxed">
                    你把 `_idle` 也排除了。没有空闲类，滑窗推理时手放松的窗口会被强行
                    判成某个词 —— 这正是要查的那个症状，排除它会让实验结论没法读。
                  </div>
                )}
                <button
                  onClick={clearExcluded}
                  disabled={isTraining}
                  className="cyber-btn w-full px-2 py-1 rounded-sm text-[10px] flex items-center justify-center gap-1"
                >
                  <X className="w-3 h-3" />
                  取消全部排除（{excluded.size}）
                </button>
              </>
            )}
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
              {isTraining
                ? "训练中..."
                : excluded.size
                  ? `开始训练（排除 ${excluded.size} 词）`
                  : "开始训练"}
            </button>
            <button
              onClick={handleAudit}
              disabled={isTraining || isBusy}
              className="cyber-btn w-full px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1"
            >
              <Stethoscope className="w-3 h-3" />
              数据体检（只读）
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

          {auditText && (
            <div className="cyber-panel p-3 rounded-sm">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider">
                  Dataset Audit · 只读
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={handleCopyAudit}
                    className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
                  >
                    <ClipboardCopy className="w-3 h-3" />
                    复制
                  </button>
                  <button
                    onClick={() => setAuditText("")}
                    className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
                  >
                    <X className="w-3 h-3" />
                    关闭
                  </button>
                </div>
              </div>
              <pre className="text-[10px] font-mono text-[#ccd6e0] leading-relaxed whitespace-pre-wrap max-h-[60vh] overflow-y-auto select-text">
                {auditText}
              </pre>
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
