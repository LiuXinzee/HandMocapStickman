/*
 * SequenceCollect — 动态手语词的序列采集页
 * DESIGN: Cyberpunk HUD 风格（沿用 DataCollect 的视觉语言）
 *
 * 与 /collect（静态单帧采集）的关系：那个页面仍被骨架回归训练依赖，保持原样不动。
 * 这里录的是整段动作序列，存进 IndexedDB 的 sequences store。
 *
 * 两个必须有的东西，否则数据质量无从判断：
 * 1) 录完立刻画**运动能量曲线** —— 一条误录的静止样本会是贴地直线，肉眼秒判
 * 2) 单条删除 —— 序列样本比静态样本贵得多，录废了要能单独剔而不是整类重录
 *
 * 还有一个采集纪律问题：滑窗推理下模型对任意窗口都会强行输出某个词，
 * 所以必须专门录 `_idle`（手放松/动作过渡/手往起始位置移动），
 * 且样本数建议是单词类的 2~3 倍。词表左侧第一项就是它。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Camera,
  Circle,
  Square,
  Trash2,
} from "lucide-react";
import { useGloveFrames, useGloves } from "@/contexts/GloveContext";
import { useHandTracking } from "@/hooks/useHandTracking";
import { useSequenceRecorder } from "@/hooks/useSequenceRecorder";
import HandCanvas from "@/components/HandCanvas";
import StepNav from "@/components/StepNav";
import {
  SIGN_VOCABULARY,
  SIGN_CATEGORIES,
  IDLE_LABEL,
  getCategoryColor,
  getDisplayLabel,
  type SignWord,
} from "@/lib/signLanguageVocab";
import {
  addSequence,
  deleteSequence,
  deleteWordSequencesByLabel,
  getSequenceStats,
  getSequencesByLabel,
  type SequenceSample,
  type SequenceStats,
} from "@/lib/datasetStore";
import { motionEnergy, visionCoverage } from "@/lib/sequenceFeatures";
// 质检显示件与句子采集页共用（见 SampleQc.tsx 顶部：抄一份的话门限会走样）
import {
  EnergyChart,
  ImuHealthLine,
  Metric,
  useSampleImuVerdict,
} from "@/components/SampleQc";

/** idle 伪类在词表里的展示条目 */
const IDLE_WORD: SignWord = {
  id: IDLE_LABEL,
  label: "空闲 / 过渡",
  pinyin: "idle",
  category: "idle",
  description: "手自然放松、动作之间的过渡、手移动到起始位置的过程",
};

const RECOMMENDED_PER_WORD = 30;
const RECOMMENDED_IDLE = 90; // 约 3 倍

export default function SequenceCollect() {
  const recorderRef = useRef<ReturnType<typeof useSequenceRecorder> | null>(
    null
  );

  // 连接由 /mocap（第 1 步）建立并住在 GloveProvider 里，本页只订阅帧、不碰串口
  const { anyConnected: gloveConnected } = useGloves();
  // 全速回调直通录制器；轮询 latestFrameRef 会因节流丢帧
  useGloveFrames(
    (f) => recorderRef.current?.pushLeftFrame(f),
    (f) => recorderRef.current?.pushRightFrame(f)
  );

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

  const recorder = useSequenceRecorder({ handResultsRef });
  recorderRef.current = recorder;

  const [selectedWord, setSelectedWord] = useState<SignWord>(IDLE_WORD);
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [stats, setStats] = useState<SequenceStats | null>(null);
  const [message, setMessage] = useState("");
  const [lastSample, setLastSample] = useState<SequenceSample | null>(null);
  const [lastSampleId, setLastSampleId] = useState<number | null>(null);
  const [labelSamples, setLabelSamples] = useState<SequenceSample[]>([]);
  const [countdown, setCountdown] = useState(0);

  const bothReady = gloveConnected && cameraRunning;

  const refreshStats = useCallback(async () => {
    try {
      setStats(await getSequenceStats());
    } catch (e) {
      console.error("[SeqCollect] 统计刷新失败", e);
    }
  }, []);

  const refreshLabelSamples = useCallback(async (label: string) => {
    try {
      const list = await getSequencesByLabel(label);
      list.sort((a, b) => b.timestamp - a.timestamp);
      setLabelSamples(list);
    } catch (e) {
      console.error("[SeqCollect] 读取该词样本失败", e);
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  useEffect(() => {
    refreshLabelSamples(selectedWord.id);
    setLastSample(null);
    setLastSampleId(null);
  }, [selectedWord, refreshLabelSamples]);

  // ===== 录制 =====

  const beginRecord = useCallback(() => {
    if (!gloveConnected) {
      setMessage("手套未连接，无法录制");
      return;
    }
    setMessage("");
    setLastSample(null);
    setLastSampleId(null);
    recorder.start();
  }, [gloveConnected, recorder]);

  const endRecord = useCallback(async () => {
    const sample = recorder.stop(selectedWord.id);
    if (!sample) {
      setMessage("这条录废了（时长太短或没有手套数据），请重录");
      return;
    }
    try {
      const id = await addSequence(sample);
      setLastSample(sample);
      setLastSampleId(id);
      setMessage(
        `已保存：${sample.frameCount} 帧 / ${Math.round(sample.durationMs)}ms`
      );
      await refreshStats();
      await refreshLabelSamples(selectedWord.id);
    } catch (e) {
      console.error("[SeqCollect] 保存失败", e);
      setMessage(`保存失败：${e}`);
    }
  }, [recorder, selectedWord.id, refreshStats, refreshLabelSamples]);

  /**
   * 开始 / 结束。**不设固定时长** —— 「你好」这类简单词一秒不到就做完了，
   * 硬录 1.5s 只会在尾巴上堆一串多余的静止帧。
   * 长度不用担心：录制器只在 T<4 帧（约 133ms）时判废，
   * 特征层 resampleSequence 会把每条统一重采样到 SEQ_LEN，短样本照常能训。
   */
  const toggleRecord = useCallback(() => {
    if (recorder.isRecording) {
      void endRecord();
    } else {
      beginRecord();
    }
  }, [recorder.isRecording, beginRecord, endRecord]);

  /** 倒计时录制：3-2-1 后开始，**不自动停**，仍由用户按结束 */
  const startCountdownRecord = useCallback(() => {
    if (recorder.isRecording || countdown > 0) return;
    let n = 3;
    setCountdown(n);
    const tick = window.setInterval(() => {
      n -= 1;
      setCountdown(n);
      if (n <= 0) {
        clearInterval(tick);
        beginRecord();
      }
    }, 1000);
  }, [recorder.isRecording, countdown, beginRecord]);

  // 空格键切换开始/结束（e.repeat 挡住长按的连发）
  useEffect(() => {
    const isTyping = (t: EventTarget | null) =>
      t instanceof HTMLElement &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.isContentEditable);

    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat || isTyping(e.target)) return;
      e.preventDefault();
      toggleRecord();
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [toggleRecord]);

  const handleDelete = useCallback(
    async (id?: number) => {
      if (id === undefined) return;
      await deleteSequence(id);
      if (id === lastSampleId) {
        setLastSample(null);
        setLastSampleId(null);
      }
      await refreshStats();
      await refreshLabelSamples(selectedWord.id);
      setMessage("已删除该条");
    },
    [lastSampleId, refreshStats, refreshLabelSamples, selectedWord.id]
  );

  /**
   * 删掉当前词的全部孤立词样本。
   *
   * 为什么要有这个（而不是点 N 次单条删除）：**换打法重录**时旧样本必须先清空。
   * 同一个词的两种打法混进一个类 = 人为造一个双峰类，模型分不开，而且会得出
   * "这个词本来就不可分"的错误结论。典型场景是「我」—— 它有两种打法，选带身体
   * 接触的那种才能和「你/他」分开（见 labelMerge.ts）。
   *
   * 二次确认里把条数**和合成条数分开报**：合成件是从录制派生的，一起删是对的，
   * 但用户得先知道那些也会没。句子样本不在删除范围内（见 deleteWordSequencesByLabel）。
   */
  const handleDeleteAllOfWord = useCallback(async () => {
    const c = stats?.labelCounts[selectedWord.id];
    const recorded = c?.recorded ?? 0;
    const synth = c?.synthesized ?? 0;
    if (recorded + synth === 0) return;
    const detail = synth > 0 ? `${recorded} 条录制 + ${synth} 条合成` : `${recorded} 条录制`;
    if (
      !confirm(
        `删除「${getDisplayLabel(selectedWord.id)}」的全部 ${detail}？\n\n` +
          `以这个词开头的句子样本不会被删。\n此操作不可撤销。`
      )
    )
      return;
    const n = await deleteWordSequencesByLabel(selectedWord.id);
    setLastSample(null);
    setLastSampleId(null);
    await refreshStats();
    await refreshLabelSamples(selectedWord.id);
    setMessage(`已删除「${getDisplayLabel(selectedWord.id)}」的 ${n} 条样本`);
  }, [stats, selectedWord.id, refreshStats, refreshLabelSamples]);

  // ===== 词表 =====

  const words = useMemo(() => {
    const list =
      selectedCategory === "all"
        ? SIGN_VOCABULARY
        : SIGN_VOCABULARY.filter((w) => w.category === selectedCategory);
    // 动态词排前面：它们才是这个页面存在的理由
    const sorted = [...list].sort(
      (a, b) => Number(!!b.dynamic) - Number(!!a.dynamic)
    );
    return selectedCategory === "all" ? [IDLE_WORD, ...sorted] : sorted;
  }, [selectedCategory]);

  const energy = useMemo(
    () => (lastSample ? motionEnergy(lastSample) : null),
    [lastSample]
  );
  const coverage = useMemo(
    () => (lastSample ? visionCoverage(lastSample) : 0),
    [lastSample]
  );
  const imu = useSampleImuVerdict(lastSample);

  /*
   * `h-screen` 而不是 `min-h-screen`（/collect-sentence 同款，两页要一致）：min-h 下
   * 这一层会被内容顶高，`flex-1` 那行跟着长高，三列的 `overflow-y-auto` 永远不触发 ——
   * 滚的是整个文档，翻词表就得连摄像头一起推走。
   */
  return (
    <div
      className="h-screen overflow-hidden flex flex-col"
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
              // 紫色是时序链路的标识色，与 /translate 的 MODE 开关、/train-seq 一致；
              // 青色留给静态链路。进错页面时颜色比文字先被注意到
              color: "#a855f7",
            }}
          >
            SEQUENCE COLLECTION
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            时序滑窗 · 动态词
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            DYNAMIC WORD / TEMPORAL
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-4 text-[10px] font-mono">
            <span
              className={
                cameraRunning ? "text-[#00e5a0]" : "text-[#556677]"
              }
            >
              <Camera className="w-3 h-3 inline mr-1" />
              {cameraRunning ? `CAM ${cameraFps}fps` : "CAM OFF"}
            </span>
            <span className="text-[#556677]">
              SEQ:{" "}
              <span className="text-[#00f0ff]">
                {stats?.totalSequences ?? 0}
              </span>
              <span className="text-[#334455]">
                {" "}
                ({stats?.recordedCount ?? 0}真/{stats?.synthesizedCount ?? 0}合)
              </span>
            </span>
          </div>
          {/* 手套状态（左右手分开）+ 下一步，统一由 StepNav 给 */}
          <StepNav />
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 左：词汇 */}
        <div className="w-64 border-r border-[#00f0ff]/15 overflow-y-auto shrink-0 p-3 space-y-3">
          <div className="space-y-1.5">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              Category Filter
            </div>
            <div className="flex flex-wrap gap-1">
              <CategoryChip
                label="全部"
                color="#00f0ff"
                active={selectedCategory === "all"}
                onClick={() => setSelectedCategory("all")}
              />
              {SIGN_CATEGORIES.map((c) => (
                <CategoryChip
                  key={c.id}
                  label={c.label}
                  color={c.color}
                  active={selectedCategory === c.id}
                  onClick={() => setSelectedCategory(c.id)}
                />
              ))}
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              Vocabulary ({words.length})
            </div>
            {words.map((w) => {
              const c = stats?.labelCounts[w.id];
              const recorded = c?.recorded ?? 0;
              const synth = c?.synthesized ?? 0;
              const target =
                w.id === IDLE_LABEL ? RECOMMENDED_IDLE : RECOMMENDED_PER_WORD;
              const selected = selectedWord.id === w.id;
              return (
                <button
                  key={w.id}
                  onClick={() => setSelectedWord(w)}
                  className={`w-full text-left px-2.5 py-2 rounded-sm border transition-all duration-150 ${
                    selected
                      ? "border-[#00f0ff]/60 bg-[#00f0ff]/10"
                      : "border-[#00f0ff]/10 hover:border-[#00f0ff]/30"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-1.5 h-1.5 rounded-full"
                        style={{
                          backgroundColor:
                            w.id === IDLE_LABEL
                              ? "#556677"
                              : getCategoryColor(w.category),
                        }}
                      />
                      <span className="text-[11px] text-[#ccd6e0]">
                        {w.label}
                      </span>
                      {w.dynamic && (
                        <span className="text-[8px] font-mono text-[#a855f7] border border-[#a855f7]/40 px-1 rounded-sm">
                          DYN
                        </span>
                      )}
                    </div>
                    <span
                      className="text-[9px] font-mono"
                      style={{
                        color:
                          recorded >= target
                            ? "#00e5a0"
                            : recorded > 0
                              ? "#f59e0b"
                              : "#334455",
                      }}
                    >
                      {recorded}
                      {synth > 0 && (
                        <span className="text-[#334455]">+{synth}</span>
                      )}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* 中：预览 + 录制 */}
        <div className="flex-1 min-h-0 flex flex-col p-4 gap-3 overflow-y-auto">
          <div className="cyber-panel p-3 rounded-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono text-[#556677] uppercase tracking-wider">
                Live Preview
              </span>
              <div className="flex gap-2">
                <button
                  className="cyber-btn px-2 py-1 rounded-sm text-[10px]"
                  onClick={() =>
                    cameraRunning ? stopTracking() : startTracking()
                  }
                >
                  {cameraRunning ? "停止摄像头" : "启动摄像头"}
                </button>
              </div>
            </div>
            {/* 宽高只是摄像头 metadata 到达前的占位值，取 useHandTracking 申请的
                1280×720，避免启动瞬间从 4:3 跳成 16:9；之后 HandCanvas 自己跟随真实尺寸 */}
            <HandCanvas
              handResults={handResults}
              videoWidth={1280}
              videoHeight={720}
              videoRef={videoRef}
            />
            {cameraError && (
              <div className="text-[10px] text-[#ff2d7b] mt-1 font-mono">
                {cameraError}
              </div>
            )}
          </div>

          {/* 录制控制 */}
          <div className="cyber-panel p-3 rounded-sm">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-[10px] font-mono text-[#556677] uppercase tracking-wider">
                  Target
                </div>
                <div className="text-[15px] text-[#00f0ff]">
                  {selectedWord.label}
                </div>
                <div className="text-[10px] text-[#556677] mt-0.5">
                  {selectedWord.description}
                </div>
              </div>
              <div className="text-right font-mono text-[10px] text-[#556677]">
                <div>
                  L {recorder.stats.leftFrames} / R{" "}
                  {recorder.stats.rightFrames}
                </div>
                <div>VIS {recorder.stats.visionFrames}</div>
                <div className="text-[#00f0ff]">
                  {Math.round(recorder.stats.elapsedMs)}ms
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={toggleRecord}
                disabled={!gloveConnected || countdown > 0}
                className={`flex-1 py-3 rounded-sm border text-[12px] font-mono tracking-wider transition-all ${
                  recorder.isRecording
                    ? "border-[#ff2d7b] bg-[#ff2d7b]/20 text-[#ff2d7b]"
                    : "border-[#00f0ff]/40 text-[#00f0ff] hover:bg-[#00f0ff]/10"
                }`}
              >
                {recorder.isRecording ? (
                  <>
                    <Square className="w-3 h-3 inline mr-1" />
                    结束采集 · {Math.round(recorder.stats.elapsedMs)}ms
                  </>
                ) : (
                  <>
                    <Circle className="w-3 h-3 inline mr-1" />
                    开始采集（或按空格）
                  </>
                )}
              </button>
              <button
                onClick={startCountdownRecord}
                disabled={!gloveConnected || recorder.isRecording}
                className="cyber-btn px-3 py-3 rounded-sm text-[11px] font-mono"
              >
                {countdown > 0 ? `${countdown}...` : "倒计时开始"}
              </button>
            </div>

            {!bothReady && (
              <div className="text-[10px] text-[#f59e0b] mt-2 font-mono">
                {!gloveConnected
                  ? "手套未连接 —— 触觉是部署时唯一的输入，必须连"
                  : "摄像头未启动 —— 不开就没有视觉教师，只能训出无蒸馏的学生"}
              </div>
            )}
            {message && (
              <div className="text-[10px] text-[#00e5a0] mt-2 font-mono">
                {message}
              </div>
            )}
          </div>

          {/* 录后质检 */}
          {lastSample && energy && (
            <div className="cyber-panel p-3 rounded-sm">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-[#556677] uppercase tracking-wider">
                  Last Take — Motion Energy
                </span>
                <button
                  onClick={() => handleDelete(lastSampleId ?? undefined)}
                  className="cyber-btn px-2 py-1 rounded-sm text-[10px] text-[#ff2d7b]"
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
                  value={`${lastSample.leftSensor ? "L" : "-"}${lastSample.rightSensor ? "R" : "-"}`}
                />
                <Metric
                  label="视觉覆盖"
                  value={`${(coverage * 100).toFixed(0)}%`}
                  warn={coverage < 0.8}
                />
              </div>
              {imu && <ImuHealthLine imu={imu} />}
            </div>
          )}
        </div>

        {/* 右：该词已有样本 */}
        <div className="w-72 border-l border-[#00f0ff]/15 overflow-y-auto shrink-0 p-3 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              {getDisplayLabel(selectedWord.id)} — {labelSamples.length} 条
            </div>
            {/* 只在真有样本时出现：空列表上摆一个"全删"只会让人误点 */}
            {labelSamples.length > 0 && (
              <button
                onClick={handleDeleteAllOfWord}
                className="text-[9px] font-mono text-[#556677] hover:text-[#ff2d7b] transition-colors shrink-0"
                title="删除这个词的全部样本（换打法重录时用）"
              >
                全部删除
              </button>
            )}
          </div>
          {labelSamples.length === 0 && (
            <div className="text-[10px] text-[#334455] font-mono">
              还没有样本
            </div>
          )}
          {labelSamples.map((s) => (
            <div
              key={s.id}
              className="flex items-center justify-between px-2 py-1.5 rounded-sm border border-[#00f0ff]/10"
            >
              <div className="text-[10px] font-mono text-[#ccd6e0]">
                #{s.id}
                <span className="text-[#556677] ml-2">
                  {s.frameCount}f / {Math.round(s.durationMs)}ms
                </span>
                {s.origin === "synthesized" && (
                  <span className="text-[8px] text-[#f59e0b] ml-1">合成</span>
                )}
              </div>
              <button
                onClick={() => handleDelete(s.id)}
                className="text-[#ff2d7b] hover:text-[#ff6b9d]"
                title="删除这条"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function CategoryChip({
  label,
  color,
  active,
  onClick,
}: {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="px-2 py-0.5 rounded-sm text-[9px] font-mono border transition-all"
      style={{
        borderColor: active ? color : "#00f0ff20",
        color: active ? color : "#556677",
        backgroundColor: active ? `${color}15` : "transparent",
      }}
    >
      {label}
    </button>
  );
}
