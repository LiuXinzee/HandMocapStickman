/*
 * TrainSentence — 句子级（连续手语 / CTC）训练页。
 *
 * 与 /train-seq 的分工：那页管**词**模型和**数据导出**，这页管**句子**模型。
 *
 * ## 训练不在浏览器里跑，这不是偷懒
 *
 * tfjs 4.22 既没有 CTC loss 也没有 CTC 解码器。句子级训练只能在本机
 * `python_train/` 里跑。本页做的是**桥**（`lib/trainBridge.ts` ↔
 * `vite-plugin-train-bridge.ts`）：送数据、发起、看进度、看结果。
 * 桥只在 `npm run dev` 存在 —— 生产构建里整个控制区禁用并写明原因。
 *
 * ## 为什么有「训练输入」那一大块
 *
 * 因为输入出过一次事故，而且事故的形状是**看不出来的**：第一层裁剪取最长连续
 * 可见段，句子中途 MediaPipe 掉一次手就把录制劈成两段，较短那半整段被丢掉 ——
 * 包括第一个词。45 条真实句里 13 条中招，最坏一条 301 个可见帧只留下 96 个。
 * 数字读数看不见这件事：`keptRatio` 0.23 既可能是"录制两头等太久"，也可能是
 * "前 4 秒的动作被扔了"。画成条就一秒可见。
 *
 * 所以这页把每条真实句都画出来。硬指标是 `uncoveredRuns`（整段落在保留区间外的
 * 可见段数）—— 它必须是 0。
 *
 * ⚠ 不要改回"被裁掉的词数"。那个读数恒为 0：句子录制的词边界是占位值
 * （`CollectSentence.tsx` 把每个词都填成 `[0, frameCount)`），每个 segment 都
 * 横跨整条、永远与保留区间相交。一个恒为 0 的指标读起来像"已验证干净"，
 * 比没有指标更糟。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Brain,
  ClipboardCopy,
  Play,
  RefreshCw,
  Square,
  Upload,
} from "lucide-react";
import {
  getAllSequences,
  isSentenceSample,
  sampleTemplateKey,
  type SequenceSample,
} from "@/lib/datasetStore";
import { trimSpanForExport, visibleRuns } from "@/lib/sequenceTrim";
import { loadBendRange } from "@/lib/bendRange";
import type { BendRanges } from "@/lib/dominantHand";
import {
  backboneVerdict,
  bridgeAvailable,
  BRIDGE_ABSENT_REASON,
  exportWeights,
  fetchEvents,
  getBackbone,
  getDeployedModel,
  getStatus,
  ping,
  startTraining,
  stopTraining,
  subscribeLog,
  type BackboneInfo,
  type DeployedModel,
  type EpochEvent,
  type ParsedEvents,
} from "@/lib/trainBridge";
import {
  stripGeometry,
  summarizeStrips,
  sortStripRows,
  uncoveredRunsOf,
  type StripRow,
} from "@/lib/trimStrip";
import { Section, DataRow, ParamInput, MetricCard, ToggleRow } from "@/components/CyberPanels";
import StepNav from "@/components/StepNav";
import {
  BATCH1_TEMPLATES,
  RECOMMENDED_PER_TEMPLATE,
  templateKey,
  UNTRAINED_WORDS,
} from "@/lib/sentenceTemplates";

/** 与 /train-seq 同一套取法：标定只在 localStorage 里，库里读不到 */
function currentBendRanges(): BendRanges {
  return { LH: loadBendRange("LH"), RH: loadBendRange("RH") };
}

const STRIP_W = 560;

export default function TrainSentence() {
  const hasBridge = bridgeAvailable();

  const [rows, setRows] = useState<StripRow[]>([]);
  const [loadingRows, setLoadingRows] = useState(true);
  const [templateCounts, setTemplateCounts] = useState<Map<string, number>>(new Map());

  const [epochs, setEpochs] = useState(80);
  const [synthPerTemplate, setSynthPerTemplate] = useState(24);
  const [noSynth, setNoSynth] = useState(false);
  const [noTrim, setNoTrim] = useState(false);

  const [running, setRunning] = useState(false);
  const [argv, setArgv] = useState<string[] | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [events, setEvents] = useState<ParsedEvents | null>(null);
  const [pyError, setPyError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [deployed, setDeployed] = useState<DeployedModel | null>(null);
  const [backbone, setBackbone] = useState<BackboneInfo | null>(null);

  const logRef = useRef<HTMLPreElement>(null);

  // ===== 训练输入：从 IndexedDB 现读，现算裁剪 =====

  const loadRows = useCallback(async () => {
    setLoadingRows(true);
    try {
      const all = await getAllSequences();
      // 只要真实录制的句子。合成句是孤立词首尾相接出来的，没有"掉手"这回事，
      // 混进来只会把 13/45 稀释成 13/几百
      const sentences = all.filter(
        (s: SequenceSample) => isSentenceSample(s) && s.origin === "recorded"
      );
      const ranges = currentBendRanges();
      const built: StripRow[] = sentences.map((s) => {
        const vis = visibleRuns(s);
        return {
          id: s.id ?? 0,
          text: s.segments.map((g) => g.label).join(" "),
          totalFrames: s.frameCount,
          durationMs: s.durationMs,
          visibleRuns: vis.runs,
          visibleFrames: vis.visibleFrames,
          // **与导出走同一个函数。** 各算各的话这页会显示一套区间、Python 收到
          // 另一套，而两者不一致时页面看着完全正常
          trimSpan: trimSpanForExport(s, ranges),
          segments: s.segments,
          multiRun: vis.runs.length > 1,
        };
      });
      setRows(sortStripRows(built));

      const counts = new Map<string, number>();
      for (const s of sentences) {
        // 与采集页同一个 key 函数。各写各的话「我 爱 你」在两页会归到不同格子
        const k = sampleTemplateKey(s);
        counts.set(k, (counts.get(k) ?? 0) + 1);
      }
      setTemplateCounts(counts);
    } catch (e) {
      setMessage(`读取录制失败：${e}`);
    } finally {
      setLoadingRows(false);
    }
  }, []);

  useEffect(() => {
    void loadRows();
  }, [loadRows]);

  // ===== 桥：接回运行中的训练 + 订阅日志 =====

  useEffect(() => {
    if (!hasBridge) return;
    let alive = true;

    // 页面刷新后接回来。SSE 自己不带历史，只订阅的话已经跑了 40 个 epoch
    // 的训练在页面上会是一片空白
    void (async () => {
      try {
        const st = await getStatus();
        if (!alive) return;
        setRunning(st.running);
        setArgv(st.argv);
        setLines(st.lines);
      } catch {
        /* 桥在但状态取不到：下面的 ping 会报出真正的原因 */
      }
      try {
        const p = await ping();
        if (alive) setPyError(p.pythonError);
      } catch (e) {
        if (alive) setPyError(String(e));
      }
      if (alive) setEvents(await fetchEvents());
      if (alive) setDeployed(await getDeployedModel());
      if (alive) setBackbone(await getBackbone());
    })();

    const unsub = subscribeLog(
      (line) => setLines((prev) => [...prev, line]),
      (info) => {
        setRunning(false);
        // 措辞中性：这条订阅同时服务训练和导出（两者共用一个进程槽位），
        // 写死"训练结束"会在导出完成时说错话
        setMessage(
          info.error
            ? `进程没能起来：${info.error}`
            : info.exitCode === 0
              ? "跑完了（exit 0）"
              : `进程退出，exit code ${info.exitCode}`
        );
        void fetchEvents().then((e) => alive && setEvents(e));
        // 训练结束 out/ 里有了新 keras（deployedAt 就落后了）；导出结束
        // 部署目录换了新的。两种情况都要重读，所以不分支
        void getDeployedModel().then((d) => alive && setDeployed(d));
        // 骨干训练也走这条订阅（三个 target 共用一个进程槽位）。不重读的话
        // 骨干训完那段还写着「没见过：漂亮」，看起来像刚才那次没生效
        void getBackbone().then((b) => alive && setBackbone(b));
      }
    );
    return () => {
      alive = false;
      unsub();
    };
  }, [hasBridge]);

  // 跑起来的时候轮询 events 文件。SSE 推的是 stdout 原文，结构化的那份在文件里 ——
  // 每 3 秒够了，一个 epoch 本来就要几秒
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => void fetchEvents().then(setEvents), 3000);
    return () => clearInterval(t);
  }, [running]);

  // 日志自动滚底
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const handleStart = useCallback(async () => {
    setMessage(null);
    setLines([]);
    setEvents(null);
    try {
      const r = await startTraining({
        target: "ctc",
        epochs,
        noSynth,
        noTrim,
        synthPerTemplate,
      });
      setRunning(true);
      setArgv(r.argv);
    } catch (e) {
      setMessage(`起训练失败：${e}`);
    }
  }, [epochs, noSynth, noTrim, synthPerTemplate]);

  /*
   * 词骨干（CTC 的前置步骤）。
   *
   * 跑的是 `train_seq.py` **不带** `--ctc`，产物是 `out/student.keras` ——
   * 也就是 `transfer_backbone` 要读的那个文件。桥那边的 target 叫 `distill`
   * （它顺带加 `--distill`），但界面上不写"蒸馏"：`--distill` 只在视觉覆盖
   * ≥80% 时才真的训教师，覆盖不够 Python 自己会跳过并打印原因。这个按钮
   * 真正的产物是骨干，不是蒸馏。
   *
   * **不传 epochs。** 左边那个 epochs 输入是给 CTC 调的（默认 80，训到 WER
   * 收敛）；孤立词训练的 Python 默认也是 80，但两者没有理由联动 ——
   * 把 CTC 那个值透传过来，改一次 CTC 的 epochs 会静默改掉骨干的训练量。
   */
  const handleTrainBackbone = useCallback(async () => {
    setMessage(null);
    setLines([]);
    try {
      const r = await startTraining({ target: "distill" });
      setRunning(true);
      setArgv(r.argv);
    } catch (e) {
      setMessage(`起骨干训练失败：${e}`);
    }
  }, []);

  const handleExport = useCallback(async () => {
    setMessage(null);
    try {
      const r = await exportWeights();
      setRunning(true);
      setArgv(r.argv);
    } catch (e) {
      setMessage(`导出失败：${e}`);
    }
  }, []);

  const handleStop = useCallback(async () => {
    try {
      await stopTraining();
    } catch (e) {
      setMessage(`停止失败：${e}`);
    }
  }, []);

  const summary = summarizeStrips(rows);
  const bb = backboneVerdict(backbone, UNTRAINED_WORDS);
  // 骨干不行的时候**不禁用**「开始训练」，只套一圈黄边。理由：`--no-trim` 那类
  // 对照实验有时就是要"不迁移骨干"，硬挡会让它做不了；而 transfer_backbone
  // 本身是优雅降级的（打一行 ⚠️ 然后随机初始化起跑）。禁用等于把一个合法用法
  // 从界面上删掉，而这里真正要防的只是"不知不觉地"那样跑
  const bbWarn = bb.level === "missing" || bb.level === "stale";
  const epochsSeen = events?.epochs ?? [];
  const last = epochsSeen[epochsSeen.length - 1] ?? null;
  const bestWer = epochsSeen.reduce<number | null>(
    (b, e) => (e.valWer != null && (b == null || e.valWer < b) ? e.valWer : b),
    null
  );

  const pct = (v: number | null | undefined) =>
    v == null ? "—" : `${(v * 100).toFixed(1)}%`;

  return (
    <div className="min-h-screen bg-[#0a0e17] text-[#c8d4e0]">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-[#00f0ff]/15">
        <Link href="/" className="cyber-btn px-2 py-1 rounded-sm">
          <ArrowLeft className="w-3 h-3" />
        </Link>
        {/* 紫＝连续句子那条链路（与 /collect-sentence、首页步骤卡同色）。
            这页和 /train-seq 长得几乎一样，颜色是第一道区分 */}
        <Brain className="w-4 h-4 text-[#a855f7]" />
        <span className="text-xs font-mono tracking-widest text-[#a855f7]">
          SENTENCE MODEL · CTC
        </span>
        <Link
          href="/train-seq"
          className="text-[10px] font-mono text-[#556677] hover:text-[#00f0ff]"
        >
          ← 词模型 / 数据导出
        </Link>
        {/* 步号 + 手套状态 + 下一步。六个流程页共用，查表在 StepNav.tsx。
            漏挂的表现是这页头上比别的页空一块，而不是报错 */}
        <div className="ml-auto">
          <StepNav />
        </div>
      </div>

      <div className="flex">
        {/* 左：控制 */}
        <div className="w-64 shrink-0 p-4 space-y-4 border-r border-[#00f0ff]/15">
          <Section title="TRAINING">
            <ParamInput label="epochs" value={epochs} onChange={setEpochs} min={1} max={2000} />
            <ParamInput
              label="合成/句型"
              value={synthPerTemplate}
              onChange={setSynthPerTemplate}
              min={1}
              max={500}
            />
            <ToggleRow
              label="--no-synth"
              hint="只用真实句。真实句现在 45 条，多半训不动"
              checked={noSynth}
              onChange={setNoSynth}
            />
            <ToggleRow
              label="--no-trim"
              hint="不裁头尾静止。对照实验用，正常别开"
              checked={noTrim}
              onChange={setNoTrim}
            />
          </Section>

          <Section title="ACTIONS">
            {/* 黄边指向右栏的 ⓪ 段。按钮**能点** —— 见上面 bbWarn 处的注释 */}
            <button
              onClick={handleStart}
              disabled={!hasBridge || running || !!pyError}
              title={bbWarn ? `⚠ ${bb.detail}` : undefined}
              style={bbWarn ? { borderColor: "rgba(245,158,11,0.6)" } : undefined}
              className="cyber-btn w-full px-2 py-2 rounded-sm text-[11px] flex items-center justify-center gap-1"
            >
              <Play className="w-3 h-3" />
              {running ? "训练中..." : "开始训练（CTC）"}
            </button>
            <button
              onClick={handleExport}
              disabled={!hasBridge || running || !!pyError}
              title="跑 export_weights.py，把权重导到 /translate 读的那个目录"
              className="cyber-btn w-full px-2 py-2 rounded-sm text-[11px] flex items-center justify-center gap-1"
            >
              <Upload className="w-3 h-3" />
              导出到浏览器
            </button>
            <button
              onClick={handleStop}
              disabled={!hasBridge || !running}
              className="cyber-btn w-full px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1"
            >
              <Square className="w-3 h-3" />
              停止
            </button>
            <button
              onClick={() => void loadRows()}
              disabled={loadingRows}
              className="cyber-btn w-full px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              重算裁剪区间
            </button>

            {/* 禁用要说清为什么。死按钮比没有按钮更糟 */}
            {!hasBridge && (
              <div className="text-[9px] font-mono text-[#f59e0b] leading-relaxed border border-[#f59e0b]/30 rounded-sm p-2">
                {BRIDGE_ABSENT_REASON}
              </div>
            )}
            {hasBridge && pyError && (
              <div className="text-[9px] font-mono text-[#ff3b6b] leading-relaxed border border-[#ff3b6b]/30 rounded-sm p-2">
                {pyError}
              </div>
            )}
          </Section>

          <Section title="INPUT">
            <DataRow label="真实句" value={`${summary.total} 条`} color="#00e5a0" />
            <DataRow
              label="多段可见"
              value={`${summary.multiRun} 条`}
              color={summary.multiRun ? "#f59e0b" : "#556677"}
            />
            {/*
              这里原本报的是「被裁掉的词」。那个读数恒为 0 —— 句子录制的词边界是
              占位值（全填整段），永远与保留区间相交，测不出任何东西。
              换成不依赖词边界的判据：整段落在保留区间外的可见段数
            */}
            <DataRow
              label="整段被扔的可见区"
              value={`${summary.uncoveredRuns} 段`}
              color={summary.uncoveredRuns ? "#ff3b6b" : "#00e5a0"}
            />
            <DataRow
              label="区间外可见帧"
              value={`${summary.uncoveredFrames} 帧`}
              // 只是参考量：头尾静止帧被切掉是裁剪本来就该做的，正常也不为 0
              color="#556677"
            />
            <DataRow
              label="平均保留比"
              value={summary.meanKeptRatio == null ? "未算" : pct(summary.meanKeptRatio)}
              color="#00f0ff"
            />
            {summary.notComputed > 0 && (
              <DataRow
                label="没算过裁剪"
                value={`${summary.notComputed} 条`}
                color="#f59e0b"
              />
            )}
          </Section>
        </div>

        {/*
          右：结果。

          这一栏有十块面板。平铺的时候每块都得自己猜"这是给我看什么的"，
          所以分成四段、每段的小标题里直接写它回答哪个问题。分段顺序 =
          看这页的顺序：训得怎么样 → /translate 用的是哪份 → 喂进去的是什么 →
          原始输出。
        */}
        <div className="flex-1 min-h-0 p-4 space-y-5 overflow-y-auto">
          {message && (
            <div className="cyber-panel p-2 rounded-sm text-[10px] font-mono text-[#00e5a0]">
              {message}
            </div>
          )}

          {/*
            排在 ① 前面，因为它是**时间上**的第一步：CTC 一开头就
            `transfer_backbone(out/student.keras)`，那一步不成，下面所有指标
            都是在一个跑歪的模型上读出来的。
            桥不在的时候整段不显示 —— 判据要读本机文件，生产构建里没法查
          */}
          {hasBridge && (
            <Band n="⓪" title="前置" question="词骨干准备好了吗？">
              <BackbonePanel
                v={bb}
                info={backbone}
                disabled={running || !!pyError}
                onTrain={handleTrainBackbone}
              />
            </Band>
          )}

          <Band n="①" title="结果" question="这次训得怎么样？">
          <div className="grid grid-cols-4 gap-2">
            <MetricCard
              label="Epoch"
              value={last ? `${last.epoch}/${last.total}` : "—"}
              color="#00f0ff"
            />
            <MetricCard
              label="Loss"
              value={last ? last.loss.toFixed(4) : "—"}
              color="#00e5a0"
            />
            <MetricCard label="Val WER" value={pct(last?.valWer)} color="#f59e0b" />
            <MetricCard label="Best WER" value={pct(bestWer)} color="#a855f7" />
          </div>

          <CtcTrainingChart epochs={epochsSeen} />

          {events?.errors && <ErrorList errors={events.errors} />}
          </Band>

          {/* 训完不等于生效：导出是另一个动作，而两边看起来一模一样。
              单独一段就是为了让"我训到 WER 2% 了翻译还是老样子"能在这里找到答案 */}
          {hasBridge && (
            <Band n="②" title="部署" question="/translate 现在用的是哪一份？">
              <DeployPanel d={deployed} />
            </Band>
          )}

          <Band n="③" title="输入" question="这次到底喂进去了什么？">
          {events?.data && (
            <div className="cyber-panel p-3 rounded-sm space-y-1">
              <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-1">
                本轮训练读到的数据
              </div>
              <DataRow
                label="真实句 训练/验证"
                value={`${events.data.numRealTrain} / ${events.data.numRealVal}`}
                color="#00e5a0"
              />
              <DataRow
                label="合成句 训练/验证"
                value={`${events.data.numSynthTrain} / ${events.data.numSynthVal}`}
                color="#556677"
              />
              <DataRow
                label="类别数"
                value={`${events.data.classes.length} + blank(${events.data.blankIndex})`}
                color="#00f0ff"
              />
              <DataRow
                label="特征"
                value={`${events.data.featShape.join(" × ")}，最长 ${events.data.maxLabelLen} 词`}
                color="#00f0ff"
              />
              <DataRow
                label="裁剪"
                value={events.data.trimmed ? "已按 trimSpan 裁" : "⚠ 未裁（与合成句口径不一致）"}
                color={events.data.trimmed ? "#00e5a0" : "#f59e0b"}
              />
              {events.data.orphans.length > 0 && (
                <div className="text-[9px] font-mono text-[#f59e0b] leading-relaxed pt-1">
                  ⚠ {events.data.orphans.join("、")} 有类别但没有任何句型用到 ——
                  这个输出单元只会学到「永远别输出」。要么编一句用到它的手语句子，
                  要么加进 UNTRAINED_WORDS。
                </div>
              )}
              {/* 验证集里真实句占比极低时，总 WER 主要由合成句决定 */}
              {events.data.numVal > 0 &&
                events.data.numRealVal / events.data.numVal < 0.3 && (
                  <div className="text-[9px] font-mono text-[#f59e0b] leading-relaxed pt-1">
                    ⚠ 验证集 {events.data.numVal} 条里真实句只有 {events.data.numRealVal} 条
                    （{pct(events.data.numRealVal / events.data.numVal)}）。上面那个 WER
                    主要由合成句决定，**不能**当真实连续手语的表现看 ——
                    合成数据里没有协同发音。
                  </div>
                )}
            </div>
          )}

          <TemplateCoverage counts={templateCounts} />

          <TrimStrips rows={rows} loading={loadingRows} />
          </Band>

          <Band n="④" title="原始输出" question="上面哪个数字对不上，就来这里核。">
          {argv && (
            <div className="cyber-panel p-2 rounded-sm text-[9px] font-mono text-[#556677] break-all">
              {/* 真跑的那条命令行，桥回显的。页面自己拼一条显示出来毫无意义 */}
              $ {argv.join(" ")}
            </div>
          )}

          <div className="cyber-panel p-3 rounded-sm">
            <div className="flex items-center justify-between mb-2">
              <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider">
                stdout · 唯一的真相来源
              </div>
              <button
                onClick={() => void navigator.clipboard.writeText(lines.join("\n"))}
                className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
              >
                <ClipboardCopy className="w-3 h-3" />
                复制
              </button>
            </div>
            <pre
              ref={logRef}
              className="text-[10px] font-mono text-[#8fa3b8] leading-relaxed max-h-72 overflow-y-auto whitespace-pre-wrap"
            >
              {lines.length ? lines.join("\n") : "（还没有输出）"}
            </pre>
            {/* 上面那些卡片是从 events JSONL 解析来的旁路。两边对不上时以这里为准 */}
            <div className="text-[9px] font-mono text-[#3d4a5c] mt-1">
              上面的图表和卡片解析自 events 旁路；与这里对不上时，以这里为准。
              {events && events.skipped > 0 && `（本次有 ${events.skipped} 行没认出来）`}
            </div>
          </div>
          </Band>
        </div>
      </div>
    </div>
  );
}

/**
 * 结果栏的分段小标题。
 *
 * 这一栏有十块面板，全平铺的时候每块都得自己猜"这是给我看什么的"。
 * 所以 `question` 不是装饰：它写的是这一段**回答哪个问题**，比给面板起一个更精确
 * 的名词有用 —— 名词只说了这是什么，问题说了什么时候该往这儿看。
 */
function Band({
  n,
  title,
  question,
  children,
}: {
  n: string;
  title: string;
  question: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-baseline gap-2">
        <span className="text-[11px] font-mono font-bold tracking-widest text-[#a855f7] shrink-0">
          {n} {title}
        </span>
        <span className="text-[9px] font-mono text-[#556677] shrink-0">
          {question}
        </span>
        <div className="flex-1 h-px bg-[#a855f7]/20" />
      </div>
      {children}
    </div>
  );
}

// ===== 错例 =====

function ErrorList({
  errors,
}: {
  errors: NonNullable<ParsedEvents["errors"]>;
}) {
  const headPct = errors.nBad ? (errors.headBad / errors.nBad) * 100 : 0;
  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-2">
        验证集错例 · {errors.nBad}/{errors.nTotal} 条整句不完全一致
      </div>
      {errors.nBad > 0 && (
        // 句首错和句中错的成因完全不同：句首错是时间包络对不上，句中错是切词能力。
        // 总 WER 会把这件事稀释掉，所以单独报
        <div className="text-[10px] font-mono text-[#f59e0b] mb-2 leading-relaxed">
          其中 {errors.headBad}/{errors.nBad} 条错在第 1 个词（{headPct.toFixed(0)}%）
          —— 句首错是时间包络问题，句中错是切词问题，两者要分开看。
        </div>
      )}
      <div className="space-y-1.5 max-h-80 overflow-y-auto">
        {errors.examples.map((ex, i) => {
          const headWrong = !ex.hyp.length || ex.hyp[0] !== ex.ref[0];
          return (
            <div key={i} className="text-[10px] font-mono leading-relaxed">
              <div className="text-[#556677]">{ex.ref.join(" ")}</div>
              <div className={headWrong ? "text-[#ff3b6b]" : "text-[#f59e0b]"}>
                → {ex.hyp.length ? ex.hyp.join(" ") : "(空)"}
                {headWrong && <span className="ml-1 text-[9px]">句首</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ===== 部署状态 =====

/**
 * 「/translate 现在用的是哪一份模型」。
 *
 * 这一块存在的唯一理由是**它和上面那些指标不是同一份东西**。上面的 epoch/WER
 * 来自 `python_train/out/`（刚训完的），`/translate` 读的是
 * `client/public/models/seq_sentence/`（上次导出的）。两者可以差很远，
 * 而页面上看起来一模一样 —— 于是"我明明训到 WER 2% 了，翻译还是老样子"。
 */
/**
 * 「词骨干准备好了吗」。
 *
 * CTC 训练的第一件事是 `transfer_backbone(model, out/student.keras)`，而那一步
 * **决定收不收敛**（train_seq.py 里的原话）。它的两种失败都不报错：
 *
 *   文件不存在 → 打一行 ⚠️ 然后随机初始化起跑。表现是 loss 半天不降
 *   文件旧了   → 照样打印「骨干迁移 N 层」，因为 conv/bn 的形状**不随类别数变**。
 *                 骨干没见过新词也照搬成功，表现只是那个词的 WER 特别差
 *
 * 后一种是这块面板存在的理由 —— 它是唯一能把"骨干没见过漂亮"说出口的地方。
 *
 * 判据是**标签差集**不是时间戳，理由见 `backboneVerdict()`。
 */
function BackbonePanel({
  v,
  info,
  disabled,
  onTrain,
}: {
  v: { level: string; missing: string[]; detail: string };
  info: BackboneInfo | null;
  disabled: boolean;
  onTrain: () => void;
}) {
  const color =
    v.level === "ok"
      ? "#00e5a0"
      : v.level === "stale"
        ? "#f59e0b"
        : v.level === "missing"
          ? "#ff3b6b"
          : "#556677";
  const mark = v.level === "ok" ? "✅" : v.level === "missing" ? "⛔" : "⚠";
  const when = (ms?: number | null) =>
    ms == null ? "—" : new Date(ms).toLocaleString("zh-CN", { hour12: false });

  return (
    <div
      className="cyber-panel p-3 rounded-sm space-y-2"
      style={v.level === "ok" ? undefined : { borderColor: `${color}55` }}
    >
      <div className="text-[10px] font-mono leading-relaxed" style={{ color }}>
        {mark} {v.detail}
      </div>

      {v.level === "ok" && info && (
        <div className="text-[9px] font-mono text-[#556677]">
          out/student.keras · 训于 {when(info.trainedAt)} · T={info.seqLen ?? "—"}
        </div>
      )}

      {v.level !== "ok" && (
        <div className="flex items-center gap-2">
          <button
            onClick={onTrain}
            disabled={disabled}
            title="跑 train_seq.py（不带 --ctc），产物是 out/student.keras"
            className="cyber-btn px-2 py-1.5 rounded-sm text-[10px] flex items-center gap-1"
            style={{ borderColor: "rgba(0,240,255,0.4)", color: "#00f0ff" }}
          >
            <Play className="w-3 h-3" />
            训练词骨干
          </button>
          <span className="text-[9px] font-mono text-[#556677]">
            日志走下面 ④ 那一段（和 CTC 共用一个进程槽位）
          </span>
        </div>
      )}

      {/*
        差集为空但 unknown：多半是还没直送过数据集。这时候按钮照样给 ——
        骨干本身可能确实没训过，而"先直送再回来看"是两次点击的绕路
      */}
      {v.level === "stale" && (
        <div className="text-[9px] font-mono text-[#556677] leading-relaxed">
          训完骨干再点左边「开始训练（CTC）」。骨干只影响收敛，不改类别表 ——
          类别表是 CTC 自己从数据集标签推的。
        </div>
      )}
    </div>
  );
}

function DeployPanel({ d }: { d: DeployedModel | null }) {
  if (!d) return null;
  const when = (ms?: number | null) =>
    ms == null ? "—" : new Date(ms).toLocaleString("zh-CN", { hour12: false });

  if (!d.exists) {
    return (
      <div className="cyber-panel p-3 rounded-sm text-[10px] font-mono text-[#f59e0b] leading-relaxed">
        <span className="text-[#556677] uppercase tracking-wider">部署状态 · </span>
        还没导出过。<Link href="/translate?mode=sentence" className="text-[#00f0ff]">/translate</Link>{" "}
        的「连续句子」那一档现在不可用 —— 点左边「导出到浏览器」。
      </div>
    );
  }

  const stale = d.trainedAt != null && d.deployedAt != null && d.trainedAt > d.deployedAt;
  return (
    <div className="cyber-panel p-3 rounded-sm space-y-1">
      <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-1">
        部署状态 · /translate 用的这一份
      </div>
      <DataRow label="导出于" value={when(d.deployedAt)} color={stale ? "#f59e0b" : "#00e5a0"} />
      <DataRow label="out/ 里的模型" value={when(d.trainedAt)} color="#00f0ff" />
      <DataRow
        label="权重"
        value={d.bytes == null ? "—" : `${(d.bytes / 1024).toFixed(0)} KB`}
        color="#556677"
      />
      {d.meta?.labels && (
        <DataRow label="类别" value={`${d.meta.labels.length} 类`} color="#556677" />
      )}
      {d.meta?.valWer != null && (
        <DataRow
          label="导出时的 val WER"
          value={`${(d.meta.valWer * 100).toFixed(1)}%`}
          color="#a855f7"
        />
      )}
      {stale ? (
        <div className="text-[9px] font-mono text-[#f59e0b] leading-relaxed pt-1">
          ⚠ out/ 里的模型比部署的这份**新** —— 训完还没导出。上面的指标属于新的那份，
          而 /translate 用的还是旧的。点「导出到浏览器」。
        </div>
      ) : (
        <div className="text-[9px] font-mono text-[#3d4a5c] leading-relaxed pt-1">
          与 out/ 同步。去{" "}
          <Link href="/translate?mode=sentence" className="text-[#00f0ff]">
            /translate
          </Link>{" "}
          试「连续句子」。导出会触发 Vite 整页刷新（publicDir 在 watch 范围里），不是出错。
        </div>
      )}
    </div>
  );
}

// ===== 句型覆盖度 =====

/**
 * `BATCH1_TEMPLATES` 里每句各采了几条。
 *
 * 单看"真实句 45 条"是不够的：45 条全挤在 3 个句型上和均匀铺在十几个句型上，
 * 对 CTC 是完全不同的两件事 —— 前者学到的是那 3 条的整体模板，不是切词。
 * 目标 20 条是为了能划出训练/验证两半而验证集不至于只剩 1~2 条。
 */
function TemplateCoverage({ counts }: { counts: Map<string, number> }) {
  const rows = BATCH1_TEMPLATES.map((t) => {
    const key = templateKey(t);
    return { key, n: counts.get(key) ?? 0 };
  });
  const done = rows.filter((r) => r.n >= RECOMMENDED_PER_TEMPLATE).length;
  const total = rows.reduce((a, r) => a + r.n, 0);
  // 采了但不在第一批清单里的句型。别让它们凭空消失 —— 它们也进训练集
  // forEach 而不是展开 keys()：根 tsconfig 没设 target，迭代 Map/Set 会触发 TS2802。
  // 为一处写法去改全项目的 target 不值得
  const extra: string[] = [];
  counts.forEach((_, k) => {
    if (!rows.some((r) => r.key === k)) extra.push(k);
  });
  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-2">
        第一批句型覆盖 · {done}/{rows.length} 句采满（共 {total} 条 / 目标{" "}
        {rows.length * RECOMMENDED_PER_TEMPLATE}）
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {rows.map((r) => {
          const ratio = Math.min(1, r.n / RECOMMENDED_PER_TEMPLATE);
          return (
            <div key={r.key} className="text-[10px] font-mono">
              <div className="flex justify-between">
                <span className="text-[#8fa3b8]">{r.key}</span>
                <span className={r.n >= RECOMMENDED_PER_TEMPLATE ? "text-[#00e5a0]" : "text-[#556677]"}>
                  {r.n}/{RECOMMENDED_PER_TEMPLATE}
                </span>
              </div>
              <div className="h-0.5 bg-[#1e2836] mt-0.5">
                <div
                  className="h-full"
                  style={{
                    width: `${ratio * 100}%`,
                    background: r.n >= RECOMMENDED_PER_TEMPLATE ? "#00e5a0" : "#00f0ff",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {extra.length > 0 && (
        <div className="text-[9px] font-mono text-[#f59e0b] mt-2 leading-relaxed">
          另有不在第一批清单里的句型 {extra.length} 个：{extra.join(" / ")}
          —— 它们照样会进训练集。
        </div>
      )}
    </div>
  );
}

// ===== 裁剪条 =====

function TrimStrips({ rows, loading }: { rows: StripRow[]; loading: boolean }) {
  if (loading) {
    return (
      <div className="cyber-panel p-3 rounded-sm text-[10px] font-mono text-[#556677]">
        正在读取录制并重算裁剪区间...
      </div>
    );
  }
  if (!rows.length) {
    return (
      <div className="cyber-panel p-3 rounded-sm text-[10px] font-mono text-[#556677]">
        还没有真实句子录制。去{" "}
        <Link href="/collect-sentence" className="text-[#00f0ff]">
          /collect-sentence
        </Link>{" "}
        录几条。
      </div>
    );
  }
  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider mb-1">
        训练输入 · 逐条裁剪区间（{rows.length} 条真实句）
      </div>
      <div className="text-[9px] font-mono text-[#3d4a5c] mb-3 leading-relaxed">
        <span className="text-[#00e5a0]">■</span> 可见段（MediaPipe 看得见手）
        <span className="ml-3 text-[#ff3b6b]">■</span> 整段落在保留区间外 = 这段动作没进训练集
        <span className="ml-3 text-[#00f0ff]">▭</span> 保留区间（真正喂进模型的）。
        可见段被劈成多段 = 录制中途掉了手；此时保留区间必须<b>跨过空洞</b>，
        没跨过就会有整段动作被扔掉（红条）。
        <div className="mt-1 text-[#556677]">
          不画词边界：句子录制的 startFrame/endFrame 是占位值（全填整段），
          画出来会全部堆在最左边。真边界要等「用词模型给句子做强制对齐」才有。
        </div>
      </div>
      <div className="space-y-2.5 max-h-[32rem] overflow-y-auto">
        {rows.map((r) => (
          <Strip key={r.id} row={r} />
        ))}
      </div>
    </div>
  );
}

function Strip({ row }: { row: StripRow }) {
  const g = stripGeometry(row, STRIP_W);
  const H = 22;
  const uncovered = uncoveredRunsOf(row);
  return (
    <div>
      <div className="flex items-baseline gap-2 text-[9px] font-mono mb-0.5">
        <span className="text-[#c8d4e0]">{row.text}</span>
        <span className="text-[#3d4a5c]">
          {row.totalFrames} 帧 / {(row.durationMs / 1000).toFixed(1)}s
        </span>
        {row.multiRun && (
          <span className="text-[#f59e0b]">可见 {row.visibleRuns.length} 段</span>
        )}
        {row.trimSpan && (
          <span className="text-[#556677]">
            保留 {row.trimSpan.startFrame}~{row.trimSpan.endFrame}（
            {(row.trimSpan.keptRatio * 100).toFixed(0)}%，{row.trimSpan.reason}）
          </span>
        )}
        {!row.trimSpan && <span className="text-[#f59e0b]">裁剪未算</span>}
        {uncovered.length > 0 && (
          <span className="text-[#ff3b6b]">
            ⚠ {uncovered.length} 段可见区被整个扔掉：
            {uncovered.map((r) => `${r.start}~${r.end}`).join("、")}
          </span>
        )}
      </div>
      <svg width={STRIP_W} height={H} className="block">
        {/* 底：总帧长 */}
        <rect x={0} y={0} width={STRIP_W} height={H} fill="#141a26" />
        {/* 可见段。整段落在保留区间外的标红 —— 那一段动作根本没进训练集 */}
        {g.visible.map((b, i) => (
          <rect
            key={i}
            x={b.x}
            y={0}
            width={b.w}
            height={H}
            fill={b.covered ? "#00e5a0" : "#ff3b6b"}
            opacity={b.covered ? 0.22 : 0.4}
          />
        ))}
        {/* 保留区间 */}
        {g.trim && (
          <rect
            x={g.trim.x}
            y={0}
            width={g.trim.w}
            height={H}
            fill="none"
            stroke="#00f0ff"
            strokeWidth={1.5}
          />
        )}
        {/*
          词边界不画。句子录制的 startFrame/endFrame 是占位值（全填 [0, frameCount)），
          画出来每条竖线都在 x=0 —— 看着像"所有词都从头开始"。
          真边界要等强制对齐（用词模型给句子标注）做完才有。
          `stripGeometry` 检测到占位时直接给空数组，所以这里也不会误画孤立词的边界
        */}
        {g.segments.map((s, i) => (
          <g key={i}>
            <line x1={s.x} y1={0} x2={s.x} y2={H} stroke="#a855f7" strokeWidth={1} />
            <text x={s.x + 2} y={H - 6} fontSize={8} fontFamily="monospace" fill="#8fa3b8">
              {s.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

// ===== 曲线 =====

/**
 * CTC 训练曲线。两条线两个轴。
 *
 * **loss 走对数轴。** 典型值 0.02，而 31/65 epoch 那两次尖峰到 0.84 —— 线性轴下
 * 整条曲线会压成一条贴着底边的直线，唯一看得见的就是那两个尖峰，而收敛过程
 * （从 84 降到 0.02）完全糊成一团。对数轴下两者都能看。
 *
 * **WER 轴不写死 0~100%。** WER 的分母是参考词数，插词能让它超过 100%
 * （实测未收敛时到过 248%）。写死上限的话前几个 epoch 会齐平顶在天花板上，
 * 看着像"一开始全错然后突然好了"。
 */
function CtcTrainingChart({ epochs }: { epochs: EpochEvent[] }) {
  const W = 600;
  const H = 180;
  const pad = 32;

  if (epochs.length < 2) {
    return (
      <div className="cyber-panel p-3 rounded-sm h-[180px] flex items-center justify-center text-[10px] font-mono text-[#556677]">
        等待训练数据...（至少 2 个 epoch 才能画线）
      </div>
    );
  }

  const n = epochs.length;
  const x = (i: number) => pad + (i / (n - 1)) * (W - pad * 2);

  // log10。loss 理论上不会是 0，但真到了 0 取 log 会是 -Infinity 把整条线弄没，
  // 所以夹一个下限
  const lg = (v: number) => Math.log10(Math.max(v, 1e-6));
  const losses = epochs.map((e) => lg(e.loss));
  const loMin = Math.min(...losses);
  const loMax = Math.max(...losses);
  const loSpan = loMax - loMin || 1;
  const yLoss = (v: number) => H - pad - ((lg(v) - loMin) / loSpan) * (H - pad * 2);

  const wers = epochs.map((e) => e.valWer).filter((v): v is number => v != null);
  const werMax = wers.length ? Math.max(...wers) : 1;
  const yWer = (v: number) => H - pad - (v / werMax) * (H - pad * 2);

  const lossPts = epochs.map((e, i) => `${x(i)},${yLoss(e.loss)}`).join(" ");
  const werPts = epochs
    .map((e, i) => (e.valWer == null ? null : `${x(i)},${yWer(e.valWer)}`))
    .filter(Boolean)
    .join(" ");

  return (
    <div className="cyber-panel p-3 rounded-sm">
      <div className="flex items-center gap-4 text-[9px] font-mono mb-1">
        <span className="text-[#00e5a0]">— loss（左轴，对数）</span>
        <span className="text-[#f59e0b]">— val WER（右轴，0~{(werMax * 100).toFixed(0)}%）</span>
      </div>
      <svg width={W} height={H} className="block">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#1e2836" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#1e2836" />
        {/* 保存点：WER 创新低的那些 epoch。盘上的 checkpoint 来自这里的最后一个 */}
        {epochs.map((e, i) =>
          e.saved ? (
            <circle key={i} cx={x(i)} cy={yWer(e.valWer ?? 0)} r={2} fill="#a855f7" />
          ) : null
        )}
        <polyline points={lossPts} fill="none" stroke="#00e5a0" strokeWidth={1.5} />
        {werPts && (
          <polyline points={werPts} fill="none" stroke="#f59e0b" strokeWidth={1.5} />
        )}
        <text x={2} y={pad + 4} fontSize={8} fontFamily="monospace" fill="#556677">
          {Math.pow(10, loMax).toFixed(3)}
        </text>
        <text x={2} y={H - pad} fontSize={8} fontFamily="monospace" fill="#556677">
          {Math.pow(10, loMin).toFixed(3)}
        </text>
        <text x={W - pad + 4} y={pad + 4} fontSize={8} fontFamily="monospace" fill="#556677">
          {(werMax * 100).toFixed(0)}%
        </text>
        <text
          x={W / 2}
          y={H - 4}
          fontSize={8}
          fontFamily="monospace"
          fill="#556677"
          textAnchor="middle"
        >
          epoch 1 → {epochs[n - 1].epoch}
        </text>
      </svg>
    </div>
  );
}
