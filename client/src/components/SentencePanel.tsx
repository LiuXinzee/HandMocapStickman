/*
 * SentencePanel — 连续句子模式的输出区。
 *
 * 抽成组件而不是继续堆在 Translate.tsx 里（那个文件已经 1250 行）：这里的交互
 * （词条点选、代词三选、删词、成句、朗读）和推理链路完全解耦，只吃一个词序列。
 *
 * ===== 两行都要在，不能只显示顺句结果 =====
 *
 * 上面一行是**原始词序**（灰、小字、合并类原样显示成「我/你/他」），
 * 下面一行是顺句结果（大字）。顺句结果是规则表加工出来的猜测，原始词序才是模型的输出。
 * 只显示顺句结果的话，用户分不清"模型认错了"和"规则顺错了" —— 而这两件事一个要
 * 补数据重训、一个只要改一行规则表，混在一起会把人引向重训模型。
 */
import {
  PRON_SG_OPTIONS,
  displayWord,
  pronounChoices,
  resolveSentence,
} from "@/lib/sentenceGrammar";
import type { CaptureStatus } from "@/lib/sentenceCapture";
import { SETTLE_MS } from "@/lib/sentenceCapture";
import { speakChinese, speechSupported } from "@/lib/speech";
import { useMemo, useState } from "react";
import { Check, Delete, Mic, Square, Volume2 } from "lucide-react";

const SENT_COLOR = "#f59e0b";

export interface SentencePanelProps {
  /** 模型解出的词序列（原始类别 id，含合并类）；null = 还没解过 */
  words: string[] | null;
  /** 词序列下标 → 用户手动指定的代词。由父组件持有，删词时要跟着重排 */
  overrides: Record<number, string>;
  onOverride: (index: number, member: string) => void;
  onDeleteWord: (index: number) => void;
  grammarOn: boolean;
  onToggleGrammar: (on: boolean) => void;
  /** 捕获状态机的实时状态；null = 没在捕获 */
  status: CaptureStatus | null;
  /** 上一次捕获/解码的提示（起手超时、没录到动作、解码失败…） */
  note: string | null;
  onArm: () => void;
  onFinish: () => void;
  onCommit: (text: string) => void;
  /** 已成句的历史 */
  history: string[];
  onClearHistory: () => void;
  /** 手套/模型没就绪时按钮要灰掉 */
  disabled: boolean;
}

export default function SentencePanel({
  words,
  overrides,
  onOverride,
  onDeleteWord,
  grammarOn,
  onToggleGrammar,
  status,
  note,
  onArm,
  onFinish,
  onCommit,
  history,
  onClearHistory,
  disabled,
}: SentencePanelProps) {
  const [picking, setPicking] = useState<number | null>(null);
  const [speakNote, setSpeakNote] = useState<string | null>(null);

  const resolved = useMemo(
    () => (words ? resolveSentence(words, overrides, grammarOn) : null),
    [words, overrides, grammarOn]
  );

  const capturing = status?.state === "capturing" || status?.state === "settling";
  const armed = status?.state === "armed";

  const doSpeak = () => {
    if (!resolved?.text) return;
    const r = speakChinese(resolved.text);
    // 失败原因一定要显示：静默失败会让人以为是音箱坏了
    setSpeakNote(r.ok ? r.reason : r.reason ?? "朗读失败");
  };

  return (
    <div className="w-full max-w-3xl space-y-4">
      {/* ===== 捕获状态 ===== */}
      <div className="cyber-panel p-3 rounded-sm space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
            Utterance Capture
          </span>
          <span
            className="text-[10px] font-mono"
            style={{ color: capturing || armed ? SENT_COLOR : "#556677" }}
          >
            {status?.state === "armed"
              ? "已就绪 · 等你起手"
              : status?.state === "capturing"
              ? `录制中 ${(status.elapsedMs / 1000).toFixed(1)}s`
              : status?.state === "settling"
              ? `静止中 ${(status.stillMs / 1000).toFixed(1)}s / ${(
                  SETTLE_MS / 1000
                ).toFixed(1)}s`
              : "未开始"}
          </span>
        </div>
        {/* settling 进度条：正在被判"这一句结束了"。看得见才知道为什么被收句 */}
        {status?.state === "settling" && status.settleRemainMs !== null && (
          <div className="h-1 bg-[#1a2030] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-100"
              style={{
                width: `${Math.min(100, (status.stillMs / SETTLE_MS) * 100)}%`,
                backgroundColor: SENT_COLOR,
              }}
            />
          </div>
        )}
        <div className="text-[9px] font-mono text-[#556677] leading-relaxed">
          点「开始一句」→ 整句连着打完，中间不用停 → 停手约 {SETTLE_MS / 1000}s
          自动收句（或点「结束」）。起手前的静止不算句尾。
        </div>
        {note && (
          <div className="text-[10px] font-mono text-[#f59e0b]">{note}</div>
        )}
      </div>

      {/* ===== 原始词序（永远显示） ===== */}
      <div className="cyber-panel p-3 rounded-sm space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
            原始词序 · 模型输出
          </span>
          <label className="flex items-center gap-1.5 text-[9px] font-mono text-[#556677] cursor-pointer">
            <input
              type="checkbox"
              checked={grammarOn}
              onChange={(e) => onToggleGrammar(e.target.checked)}
              className="accent-[#f59e0b]"
            />
            套用顺句规则
          </label>
        </div>

        {!words ? (
          <p className="text-[11px] font-mono text-[#334455]">还没有解出句子</p>
        ) : words.length === 0 ? (
          /* 空词序列是有信息的：模型看了这段但一个词都没解出来（全 blank）。
             显示成空白会被读成"程序卡了" */
          <p className="text-[11px] font-mono text-[#ff2d7b]">
            这一段没解出任何词（输出全是 blank）—— 动作幅度太小、或这句话超出了词表
          </p>
        ) : (
          <div className="flex flex-wrap items-center gap-1.5">
            {words.map((w, i) => {
              const choices = pronounChoices(w);
              const shown = overrides[i] ?? resolved?.resolved[i] ?? w;
              return (
                <div key={`${i}-${w}`} className="relative">
                  <div
                    className="flex items-center rounded-sm border text-[11px] font-mono"
                    style={{
                      borderColor: choices ? `${SENT_COLOR}66` : "#00f0ff26",
                    }}
                  >
                    {/* 合并类可点：三个候选模型分不出来，只能人来定 */}
                    <button
                      onClick={() => choices && setPicking(picking === i ? null : i)}
                      disabled={!choices}
                      className={`px-2 py-1 ${
                        choices ? "text-[#f59e0b] hover:bg-[#f59e0b]/10" : "text-[#8899aa]"
                      }`}
                      title={
                        choices
                          ? `模型分不出这三个（六轴 IMU 观测不到朝向），现在按位置默认取「${displayWord(
                              shown
                            )}」，点一下可改`
                          : undefined
                      }
                    >
                      {displayWord(w)}
                      {choices && (
                        <span className="ml-1 text-[#ccd6e0]">
                          →{displayWord(shown)}
                        </span>
                      )}
                    </button>
                    <button
                      onClick={() => {
                        setPicking(null);
                        onDeleteWord(i);
                      }}
                      className="px-1.5 py-1 text-[#556677] hover:text-[#ff2d7b] border-l border-[#00f0ff]/15"
                      title="删掉这个词（模型多解出来一个时用）"
                    >
                      ×
                    </button>
                  </div>
                  {picking === i && choices && (
                    <div className="absolute z-10 top-full left-0 mt-1 flex gap-1 p-1 rounded-sm bg-[#0a0e1a] border border-[#f59e0b]/40">
                      {choices.map((m) => (
                        <button
                          key={m}
                          onClick={() => {
                            onOverride(i, m);
                            setPicking(null);
                          }}
                          className={`px-2 py-1 text-[11px] font-mono rounded-sm ${
                            shown === m
                              ? "bg-[#f59e0b]/20 text-[#f59e0b]"
                              : "text-[#8899aa] hover:text-[#f59e0b]"
                          }`}
                        >
                          {displayWord(m)}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ===== 顺句结果 ===== */}
      {resolved && words && words.length > 0 && (
        <div className="cyber-panel p-4 rounded-sm space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              顺句结果
            </span>
            <span className="text-[9px] font-mono text-[#556677]">
              {/* 命中了哪条规则要说出来 —— 顺错时才知道该改哪条 */}
              {resolved.rule
                ? `规则：${resolved.rule}`
                : grammarOn
                ? "没命中规则 · 原样拼接"
                : "规则已关"}
            </span>
          </div>
          <p
            className="text-3xl leading-snug"
            style={{ fontFamily: "'Space Grotesk', sans-serif", color: "#ccd6e0" }}
          >
            {resolved.text}
          </p>
          {speakNote && (
            <p className="text-[9px] font-mono text-[#f59e0b]">{speakNote}</p>
          )}
        </div>
      )}

      {/* ===== 按钮 ===== */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <button
          onClick={onArm}
          disabled={disabled}
          className={`cyber-btn px-5 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
            armed || capturing ? "cyber-btn-accent" : ""
          } ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
        >
          <Mic className="w-4 h-4" />
          {armed || capturing ? "重新开始一句" : "开始一句"}
        </button>
        <button
          onClick={onFinish}
          disabled={disabled || !(armed || capturing)}
          className={`cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
            disabled || !(armed || capturing) ? "opacity-40 cursor-not-allowed" : ""
          }`}
        >
          <Square className="w-3.5 h-3.5" />
          结束
        </button>
        <button
          onClick={() => words?.length && onDeleteWord(words.length - 1)}
          disabled={!words?.length}
          className={`cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
            !words?.length ? "opacity-40 cursor-not-allowed" : ""
          }`}
          title="删掉最后一个词"
        >
          <Delete className="w-3.5 h-3.5" />
          删词
        </button>
        <button
          onClick={() => resolved?.text && onCommit(resolved.text)}
          disabled={!resolved?.text}
          className={`cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
            !resolved?.text ? "opacity-40 cursor-not-allowed" : ""
          }`}
        >
          <Check className="w-3.5 h-3.5" />
          成句
        </button>
        <button
          onClick={doSpeak}
          disabled={!resolved?.text || !speechSupported()}
          className={`cyber-btn px-4 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
            !resolved?.text || !speechSupported()
              ? "opacity-40 cursor-not-allowed"
              : ""
          }`}
          title={speechSupported() ? "朗读顺句结果" : "这个浏览器不支持语音合成"}
        >
          <Volume2 className="w-3.5 h-3.5" />
          朗读
        </button>
      </div>

      {/* ===== 句子历史 ===== */}
      {history.length > 0 && (
        <div className="cyber-panel p-3 rounded-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              Sentences ({history.length})
            </span>
            <button
              onClick={onClearHistory}
              className="text-[9px] font-mono text-[#556677] hover:text-[#ff2d7b]"
            >
              清除
            </button>
          </div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {history
              .slice()
              .reverse()
              .map((s, i) => (
                <p key={i} className="text-[12px] text-[#8899aa] leading-relaxed">
                  {s}
                </p>
              ))}
          </div>
        </div>
      )}

      {/* 合并类的说明放在最下面：第一次看到「我/你/他」的人需要一句解释，
          但它不该占住视线中心 */}
      <p className="text-[9px] font-mono text-[#556677] text-center leading-relaxed">
        「{displayWord("merged_pron_sg")}」是**一个类**：
        {PRON_SG_OPTIONS.map((m) => displayWord(m)).join(" / ")}
        在手语里只差指向，而六轴 IMU 观测不到绝对朝向，模型永远分不出 ——
        界面上给的是按位置猜的默认值，点词条可改。
      </p>
    </div>
  );
}
