/*
 * SentencePanel — 连续句子模式的**整条右列**：一个聊天窗口。
 *
 * 抽成组件而不是继续堆在 Translate.tsx 里（那个文件已经 1900 行）：这里的交互
 * （词条点选、代词三选、删词、成句、朗读、翻看历史）和推理链路完全解耦，
 * 只吃一个词序列 + 一份历史。
 *
 * ===== 版式：三段，中间那段才滚 =====
 *
 *   标题行 TRANSLATION + 清空        shrink-0
 *   ─────────────────────────
 *   气泡区                           flex-1 overflow-y-auto
 *     历史气泡（已定版的，默认收起，越旧越小越灰）
 *     live 气泡（当前这句，只有大字）
 *   ─────────────────────────
 *   控件行（按钮 + 开关 + 状态）      shrink-0，**一行**
 *
 * 按钮和状态**钉在底部**、不跟着滚：它们是随时要能按到的控件，
 * 混在气泡流里的话，打了几句之后就被顶出屏幕了。
 *
 * 底部原来是**三行堆叠**（开关行 / 提示行 / 按钮行），实测 103px。那是这个面板还在
 * 窄右列（373px）时代的版式 —— 5 个按钮横着排不下才拆开的。翻译页改成上下分之后
 * 它拿到整块宽度（1440 屏上 1416px），三行就只剩浪费了：并成一行 48px，省下的
 * 55px 归了下面的手模视口。窄屏靠 `flex-wrap` 自己折回多行，不会挤爆。
 * 两个不常驻的东西也顺势挪掉了行高：settling 进度条变成贴上边框的 2px 细条
 * （absolute，不占位），提示文字和停顿读数收进右侧状态区、跟状态读数同一行。
 *
 * ===== live 那一层只有大字 =====
 *
 * live 那一层**只有大字** —— 没有箭头、没有可展开的词条。翻译时对面的人正隔着
 * 桌子读这块屏幕，句子底下挂一排 11px 的调试词条只是噪声。
 *
 * ===== 字幕栏只放两行：当前句 + 上一句 =====
 *
 * 排在同一条流里、上一句在上，就是歌词栏那个样子。**只留两行**是明确要求的：
 * 这块栏的用途是"对面的人隔着桌子读"，不是回看记录 —— 行数越多，每行就越小，
 * 而真正要读的只有最下面那行。所以省下来的高度全给了字号（见 `SENT_FONT_LIVE`，
 * 10vh → 14vh）。
 *
 * 更早的历史**仍然在 DOM 里、往上滚就能看见**（连同它们的展开层），只是默认在
 * 折线以上。别为了"只有两行"把 `history` 截断了传进来：那会把原始词序的回查入口
 * 一起弄没，而那个入口是分辨"模型认错"还是"规则顺错"的唯一办法（见下一段）。
 *
 * ⚠ 这里曾经给 live 加过 `min-h-full`，把它撑到至少一屏高、贴底时顶边压住可视区
 * 顶边，于是翻译中屏幕上只剩当前这一句。**那条需求后来被取消了**，撑高也一起
 * 撤掉了。别再加回来：它和"两行字幕"是互斥的两种东西 —— 撑高之后上一句永远在
 * 折线以上，只有主动往上滚才看得见，而这里要的就是"上一句还挂在那儿"。
 *
 * ===== 原始词序仍然查得到，别再把它塞回句子区 =====
 *
 * 顺句结果是规则表加工出来的猜测，**原始词序**（模型输出）才是模型真正说的话。
 * 两者都要留痕：只看顺句结果的话，用户分不清"模型认错了"和"规则顺错了" ——
 * 一个要补数据重训、一个只要改一行规则表，混在一起会把人引向重训模型。
 *
 * 所以它搬到了**历史气泡的展开层**：点上一句就能看原始词序 + 命中的规则名。
 * 那里不碍事（默认收起，而且翻译中根本不在视野里），要查的时候一定找得到。
 *
 * 「改代词」搬到了**底部控件行的「代词」按钮**。这一步不能省：我/你/他 在手语里
 * 只差指向，六轴 IMU 观测不到绝对朝向、模型永远分不出，现在给的是按位置猜的
 * 默认值 —— 没有修正入口，含代词的句子就永久错着，用户还看不出错在哪。
 *
 * 两样东西留在 live 那一层，因为它们不是细节而是内容本身：
 *  - 空词序列的报错（"这一段没解出任何词"）—— 不说的话只剩一片空白，会被读成程序卡了；
 *  - `speakNote`（朗读失败）—— 是用户刚点的那个动作的回执。
 */
import {
  PRON_SG_OPTIONS,
  displayWord,
  pronounChoices,
  resolveSentence,
} from "@/lib/sentenceGrammar";
import {
  CONTINUOUS_SETTLE_MS,
  SETTLE_MS,
  type CaptureStatus,
} from "@/lib/sentenceCapture";
import type { SentenceEntry } from "@/pages/Translate";
import { speakChinese, speechSupported } from "@/lib/speech";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Delete,
  Mic,
  Square,
  StopCircle,
  Trash2,
  Volume2,
} from "lucide-react";

/*
 * 句子档主色。指向 `--hud-warn`（深色档=琥珀 #f59e0b，浅色档=#d97706），
 * 换肤时不用回来改这一行。
 *
 * 因为它现在是 `var(...)` 而不是字面十六进制，**不能再拼后缀做透明度**
 * （原来那句 `${SENT_COLOR}66` 会拼成非法色值）—— 要半透明的边框请用
 * `--hud-warn-edge` / `--hud-warn-wash` 这类专门的变量。
 */
const SENT_COLOR = "var(--hud-warn)";

/*
 * 句子大字的字号。**跟视口高度走，不是写死的 px。**
 *
 * 这个值原来是 Tailwind 的 `text-2xl`(24)，那是"正文级别"的字。
 * 这一页的用法不是读文档：一个人在打手语、另一个人在**隔着桌子看屏幕**，
 * 24px 在 2560 宽的屏幕上小得没法用。所以字号跟着 `vh` 走 ——
 *   1440×900  → 126px（14vh）
 *   2560×1440 → 176px（14vh 本来是 202px，**撞上 11rem 的上限被削到 176**）
 * `clamp` 的下限保证矮窗口下不至于挤没，上限防止超宽屏上一个字就占满一屏。
 *
 * ⚠ 用 `vh` 而不是 `%` 或容器查询：气泡区的高度是 flex 算出来的，
 * 拿它当字号基准会形成"字大→区高→字更大"的回环。
 *
 * 加粗是**去掉橙色底框之后补上的**：底框原来负责"这句是当前句"，
 * 现在这件事全靠排版说（黑 + 粗 + 最大）。见 `SENT_FONT_PREV`。
 *
 * 从 `clamp(2rem,10vh,8rem)` 抬到现在这条：字幕栏现在**只放两行**（见
 * `SENT_FONT_PREV`），省下来的高度全给了字号。上限也跟着 8rem→11rem 抬了一档，
 * 否则 1440 高的屏上 14vh 会整段被 8rem(128px) 削平，抬 vh 等于白抬。
 */
const SENT_FONT_LIVE = "clamp(2.5rem, 14vh, 11rem)";

/**
 * 上一句的字号 = 当前句的 55%。
 *
 * **写成 `calc(当前句 × 系数)` 而不是另一条 clamp**，这是这里唯一要紧的一件事：
 * 两条独立的 clamp 在窗口高度扫过各自的拐点时比例会漂（原来那版就是这样 ——
 * live 是 `clamp(2rem,10vh,8rem)`、上一句是 `clamp(1.25rem,5vh,4rem)`，
 * 矮窗口下两者都撞下限，比例从 0.5 变成 0.63，"上一句明显小一号"这个层级就糊了）。
 * 乘出来的话，改 `SENT_FONT_LIVE` 一个数，上一句自动跟着按比例走，任何窗口高度下
 * 都严格是 55%。
 */
const SENT_PREV_RATIO = 0.55;
const SENT_FONT_PREV = `calc(${SENT_FONT_LIVE} * ${SENT_PREV_RATIO})`;

/** 两行字都用这个行高。改它要连着改 `FLOW_MAX_H`，那条是按它算的 */
const SENT_LINE_H = 1.2;

/**
 * 字幕区的**可视高度上限：正好两行**（当前句 1 行 + 上一句 1 行 + 上下内边距 22px）。
 *
 * "只放两行"是靠这条实现的，**不是**靠把 `history` 截断 —— 更早的句子照样在流里，
 * 往上滚就能看见、也照样能展开看原始词序。截断会把那个回查入口一起弄没，
 * 见文件头那段。
 *
 * 高度跟着字号算而不是写死 px：`SENT_FONT_LIVE` 是 `vh` 的，写死的话在别的窗口
 * 高度下就不是两行了 —— 要么切掉半行，要么露出第三行。
 */
const FLOW_MAX_H = `calc(${SENT_FONT_LIVE} * ${SENT_LINE_H * (1 + SENT_PREV_RATIO)} + 22px)`;

/**
 * 顶边那道渐隐的高度（`topCut` 才开，见它那段）。
 * 也跟着字号走：原来是写死的 44px，那是按老的历史字号（~4rem）配的，
 * 字号抬上去之后 44px 只够化掉字顶一线、硬切口照样看得见。
 * 0.35 × live 在 1440×900 下正好还是 44px。
 */
const TOP_FADE = `calc(${SENT_FONT_LIVE} * 0.35)`;

/**
 * 这一列的字重。**只有这两个值可用** —— MiSans 那个包按小米自己的字重刻度
 * 声明 `@font-face`，Regular 是 330、Semibold 是 520，不是 400/600。
 *
 * ⚠ 写 700（这里原来就是 700，那时用的是 Space Grotesk 的 300~700）会**没有
 * 任何一档匹配得上**，Chrome 会拿 520 那档合成加粗 —— 当前句最大能到 176px
 * （`SENT_FONT_LIVE` 的 11rem 上限），这么大的字上笔画会糊出一圈毛边，一眼能看
 * 出来。想更粗只能再 @import 一档 Bold，见 index.css 那段。
 *
 * 当前句 520 / 历史 330 这个对比也是必需的：只引 Semibold 一档的话整列同粗，
 * "当前句加粗"就没了，梯度只剩字号和颜色两维。
 */
const SENT_WEIGHT_LIVE = 520;
const SENT_WEIGHT_HISTORY = 330;

/** 上一句的颜色。比当前句（`--hud-text`）退一档，和字号一起说明"这句已经过去了" */
const SENT_COLOR_PREV = "var(--hud-soft)";

/**
 * 退档时的过渡。**这就是"滚动"的那一下** —— 收句的瞬间当前句退成上一句，
 * 不给过渡的话是一帧跳变，看着像重新渲染了一遍而不是同一句往上走。
 *
 * 只动 `font-size` 和 `color`，不动 `transform`：气泡的**位置**变化是列表重排，
 * 那个要 FLIP 才能补，代价远大于收益。
 */
const TIER_TRANSITION = "font-size 320ms ease, color 320ms ease";


/** 贴底判定的容差（px）。见 `useLayoutEffect` 那段：超过这个距离就认为用户在往回翻 */
const STICK_TO_BOTTOM_PX = 80;

export interface SentencePanelProps {
  /** 模型解出的词序列（原始类别 id，含合并类）；null = 还没解过 */
  words: string[] | null;
  /**
   * 已定版的历史，从旧到新。
   *
   * ⚠ 调用方**必须已经把 live 那条切掉**（`Translate.tsx` 的 `settledHistory`）。
   * 连续模式下当前这句同时存在于历史数组和 `words` 里，直接把整个数组传进来
   * 会让同一句在窗口里出现两次。
   */
  history: SentenceEntry[];
  onClearHistory: () => void;
  /** 词序列下标 → 用户手动指定的代词。由父组件持有，删词时要跟着重排 */
  overrides: Record<number, string>;
  onOverride: (index: number, member: string) => void;
  onDeleteWord: (index: number) => void;
  grammarOn: boolean;
  onToggleGrammar: (on: boolean) => void;
  /**
   * 收句后自动朗读顺句结果。**念这一步不在本组件里做** —— 触发点是"刚解出来
   * 那一刻"，只有父组件的 `decodeUtterance` 知道。这里放在本组件里的话只能靠
   * watch `resolved.text`，那会在用户改代词/删词时每改一下念一遍。
   */
  autoSpeak: boolean;
  onToggleAutoSpeak: (on: boolean) => void;
  /**
   * 连续模式：一句解完自动接着等下一句，起手就自动开始。
   *
   * 开着的时候「成句」的含义变成**定版** —— 句子在解出来那一刻就已经进历史了
   * （不然下一句会把它覆盖掉），所以父组件在这一档不会再追加一遍。
   */
  continuous: boolean;
  onToggleContinuous: (on: boolean) => void;
  /** 捕获状态机的实时状态；null = 没在捕获 */
  status: CaptureStatus | null;
  /** 上一次捕获/解码的提示（起手超时、没录到动作、解码失败…） */
  note: string | null;
  onArm: () => void;
  /** 连续模式的「停止」：丢掉手头这半句 + 关掉推理循环 */
  onStop: () => void;
  onFinish: () => void;
  /** 成句/定版。words 与 rule 一并回传，父组件不再重算（它拿不到最新的） */
  onCommit: (text: string, words: string[], rule: string | null) => void;
  /**
   * 推理循环开着吗（`isTranslating`）。**连续模式下主按钮靠它决定是开始还是停止**，
   * 不能用 `status.state` —— 一句解完到下一次起手之间状态是 `armed`，
   * 而"干等着"和"没开始"在状态机里长得一样，用状态判会让按钮在每句之间抖动。
   */
  running: boolean;
  /** 手套/模型没就绪时按钮要灰掉 */
  disabled: boolean;
  /**
   * 演示用：把顺句结果的 `text` 换成这个字符串（`null` = 不换，走真的顺句）。
   *
   * **只换 `text` 一个字段**，`words` / `resolved` / `rule` 全保持真值 —— 所以
   * 底部的代词按钮、词序列显示还是模型真的输出，不会因为对不上而错位或崩。
   *
   * 由来和"忘了关"的代价见 `@/lib/demoSubtitles`。这是演示期的临时东西，
   * 那个文件删掉的时候这个 prop 一起删。
   */
  textOverride?: string | null;
}

export default function SentencePanel({
  words,
  history,
  onClearHistory,
  overrides,
  onOverride,
  onDeleteWord,
  grammarOn,
  onToggleGrammar,
  autoSpeak,
  onToggleAutoSpeak,
  continuous,
  onToggleContinuous,
  status,
  note,
  onArm,
  onStop,
  onFinish,
  onCommit,
  running,
  disabled,
  textOverride = null,
}: SentencePanelProps) {
  const [speakNote, setSpeakNote] = useState<string | null>(null);
  /** 展开了原始词序的历史气泡（按 `at` 记，下标会因为撤回而错位） */
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  /** 底部「代词」浮层开没开。见 `代词` 按钮那段 */
  const [pronOpen, setPronOpen] = useState(false);

  /*
   * 这一份是**全部译文的唯一来源**：大字（下面那个 `resolved?.text`）、
   * 朗读、「成句」回传给父组件的 text、贴底判定用的 `liveText`，全从它派生。
   * 所以 `textOverride` 只需要挂在这一处，不用逐个去改那四个地方。
   */
  const resolved = useMemo(() => {
    if (!words) return null;
    const real = resolveSentence(words, overrides, grammarOn);
    return textOverride ? { ...real, text: textOverride } : real;
  }, [words, overrides, grammarOn, textOverride]);

  const capturing = status?.state === "capturing" || status?.state === "settling";
  const armed = status?.state === "armed";

  /*
   * ===== 自动滚到底 =====
   *
   * `useLayoutEffect` 而不是 `useEffect`：要在浏览器画之前把 scrollTop 设好，
   * 否则新气泡会先闪一下在视野外的位置。
   *
   * **只在用户本来就贴着底部时才滚**。无条件滚到底的话，用户往上翻看前面某一句时，
   * 下一句一解出来就把他拽回底部 —— 等于历史根本没法回看，而回看正是加这个窗口的理由。
   *
   * "滚到底"就是字面意思：当前这句露在最下面，上面接着上一句、上上句。
   * （曾经有一版给 live 加 `min-h-full`，让"滚到底"变成"当前句顶到可视区顶边"、
   * 历史全压到折线以上；那条需求取消了，见文件头那段。）
   */
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const liveText = resolved?.text ?? "";
  /**
   * 顶边是不是真的切着东西（`scrollTop > 0`）。只有这时候才给顶边那道渐隐。
   *
   * 别写成常开：只有一句、上面什么都没切的时候，渐隐会白白把这唯一一行压暗一截，
   * 看着像它自己也"退了一档"。现在两行历史同字号同颜色（`SENT_FONT_PREV` /
   * `SENT_COLOR_PREV`），一压暗就更扎眼 —— 同一档的字画出来却一深一浅。
   */
  const [topCut, setTopCut] = useState(false);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (stickRef.current) el.scrollTop = el.scrollHeight;
    setTopCut(el.scrollTop > 0);
  }, [history.length, liveText, words?.length]);

  // 用户手动滚动时重新判定"还贴着底吗"、以及顶边还切不切
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      stickRef.current =
        el.scrollHeight - el.scrollTop - el.clientHeight < STICK_TO_BOTTOM_PX;
      // 同值 setState 会被 React 直接短路，不怕滚动事件的频率
      setTopCut(el.scrollTop > 0);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  /**
   * 画进度条／写读数用的门限。**优先取状态机报的实测值** —— 自适应之后它会变，
   * 拿常量画的条会填满后卡在 100% 干等，看着像卡死了。
   * 状态为 null（还没开始过）时没有实测值，用对应档的基线，只为了帮助文字有个数。
   */
  const settleMs =
    status?.settleMs ?? (continuous ? CONTINUOUS_SETTLE_MS : SETTLE_MS);

  const doSpeak = () => {
    if (!resolved?.text) return;
    const r = speakChinese(resolved.text);
    // 失败原因一定要显示：静默失败会让人以为是音箱坏了
    setSpeakNote(r.ok ? r.reason : r.reason ?? "朗读失败");
  };

  const toggleExpanded = (at: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (!next.delete(at)) next.add(at);
      return next;
    });

  const hasLive = !!words && words.length > 0;
  const empty = history.length === 0 && !hasLive;
  /**
   * 这一句里有没有合并类。有才显示那段解释 —— 原来它固定钉在面板最底下，
   * 在这条窄列里会一直占三行；而它只在**看见「我/你/他」这种词条时**才有用。
   */
  const hasMerged = !!words?.some((w) => pronounChoices(w));

  return (
    <div className="cyber-panel rounded-2xl flex flex-col min-h-0 h-full overflow-hidden">
      {/* ===== 标题行 ===== */}
      <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-[var(--hud-line)]">
        <span className="text-[10px] font-mono text-[var(--hud-dim)] uppercase tracking-widest">
          Translation
        </span>
        <div className="flex items-center gap-3">
          <span className="text-[9px] font-mono text-[var(--hud-faint)]">
            {history.length > 0 && `${history.length} 句`}
          </span>
          <button
            onClick={onClearHistory}
            disabled={history.length === 0}
            className={`transition-colors ${
              history.length === 0
                ? "text-[var(--hud-faint)] cursor-not-allowed"
                : "text-[var(--hud-dim)] hover:text-[var(--hud-err)]"
            }`}
            title="清空聊天记录"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ===== 气泡区（唯一会滚的一段） =====
          纵向内边距写在两个子元素身上（容器只有 `px-4`）：历史那一坨自己
          `pt-3 pb-2.5`，live 自己 `pb-3`。留在这儿没坏处，就不搬回来了。

          外面这层 `justify-end` 让两行字**贴着底**（也就是贴着下面的控件行）：
          区高被 `FLOW_MAX_H` 卡成两行之后，剩下的空当必须有个去处，落在上面才对
          —— 字幕本来就是从底下往上冒的，而且贴着底时它离下面那两块手模视口最近。

          顶边那道渐隐（`topCut` 才开，见 `TOP_FADE`）：往上还有更早的句子时，
          最上面那条会被卡在半截，硬切口看着像渲染坏了。 */}
      <div className="flex-1 min-h-0 flex flex-col justify-end">
      <div
        ref={scrollRef}
        className="overflow-y-auto px-4"
        style={{
          maxHeight: FLOW_MAX_H,
          ...(topCut
            ? {
                maskImage: `linear-gradient(to bottom, transparent 0, #000 ${TOP_FADE})`,
                WebkitMaskImage: `linear-gradient(to bottom, transparent 0, #000 ${TOP_FADE})`,
              }
            : {}),
        }}
      >
        {empty && (
          <div className="h-full flex flex-col items-center justify-center text-center gap-2 px-2">
            <p className="text-[11px] font-mono text-[var(--hud-faint)]">
              还没有句子
            </p>
            {/* 帮助文字只在空的时候给：打过一句之后它就是纯占地方了，
                而这一列本来就窄 */}
            <p className="text-[10px] font-mono text-[var(--hud-dim)] leading-relaxed">
              {continuous ? (
                <>
                  点「开始」一次 → 打一句 → 停手约 {(settleMs / 1000).toFixed(1)}s
                  自动收句 → 直接接着打下一句，不用再点。
                  句内犹豫过一次之后，门限会自动抬到比那次犹豫更长。
                </>
              ) : (
                <>
                  点「开始一句」→ 整句连着打完，中间不用停 → 停手约{" "}
                  {(settleMs / 1000).toFixed(1)}s 自动收句（或点「结束」）。
                  起手前的静止不算句尾。
                </>
              )}
              {autoSpeak && speechSupported() && "收句后自动念出来。"}
            </p>
          </div>
        )}

        {/* 已定版的历史。调用方已经切掉 live 那条了（见 props.history 的说明）。
            **整份都渲染**，只有最底下那条（刚定版的上一句）落在可视区里 ——
            两行的限制由 `FLOW_MAX_H` 卡可视高度实现，不是在这里 slice。
            原来这里按 `depth`（从下往上数）分档递淡，现在两行之内分不出档，
            已经并成一个样子，见 `HistoryBubble`。 */}
        {history.length > 0 && (
          <div className="pt-3 pb-2.5 space-y-2.5">
            {history.map((entry) => (
              <HistoryBubble
                key={entry.at}
                entry={entry}
                open={expanded.has(entry.at)}
                onToggle={() => toggleExpanded(entry.at)}
              />
            ))}
          </div>
        )}

        {/* ===== live 气泡：当前这一句 =====
            **没有底框**（原来是 `--hud-warn-wash` 的橙色圆角块）。这么大的字
            外面再套一层有色块，块本身比字还抢眼；而且它和历史句一齐排在滚动流里时，
            读起来是"一张卡片下面挂着几行字"，不是歌词栏那种同一条流。
            "这是当前句"改由排版承担：最大 + 加粗 + 唯一的纯黑 —— 上一句是
            `SENT_FONT_PREV`（= live × 0.55）+ `SENT_COLOR_PREV`，永远小一档、淡一档。
            左右内边距跟历史气泡的 `px-1` 对齐，否则去掉框之后两边的字会错开一截。

            ⚠ 这里**不要加 `min-h-full`**。加过一版（把它撑到至少一屏高，贴底时
            顶边压住可视区顶边，历史全挤到折线以上），需求取消后撤掉了；
            理由见文件头那段。 */}
        {words && (
          <div className="px-1 pb-3">
            {words.length === 0 ? (
              /* 空词序列是有信息的：模型看了这段但一个词都没解出来（全 blank）。
                 显示成空白会被读成"程序卡了" */
              <p className="pt-3 text-[11px] font-mono text-[var(--hud-err)] leading-relaxed">
                这一段没解出任何词（输出全是 blank）—— 动作幅度太小、或这句话超出了词表
              </p>
            ) : (
              <>
                {/*
                  当前句**只有大字，没有箭头、没有可展开的词条层**。

                  原来这里点一下会展开「原始词序」：一排可点的词条（点合并类改代词、
                  × 删词）+ 命中的规则名。整层已经删掉 —— 翻译时对面的人在读这块屏幕，
                  底下挂一排 11px 的调试词条只是噪声。

                  信息没丢，两件事各自搬了家：
                   - **看**原始词序 / 命中规则：历史气泡的展开层还在，点上一句就能查
                     （"模型认错"还是"规则顺错"仍然分得清，见文件头那段）；
                   - **改**代词：挪到底部控件行的「代词」按钮（见那处注释）。
                     它必须有地方去 —— 我/你/他 在手语里只差指向，六轴 IMU 观测不到
                     朝向、模型永远分不出，没有修正入口这类句子就永久错着。
                */}
                <div
                  className="min-w-0"
                  style={{
                    fontFamily: "var(--font-sentence)",
                    color: "var(--hud-text)",
                    fontSize: SENT_FONT_LIVE,
                    fontWeight: SENT_WEIGHT_LIVE,
                    // 与上一句共用同一个行高：`FLOW_MAX_H` 是按
                    // `SENT_LINE_H × (1 + 0.55)` 算出"正好两行"的，这里写死就对不上了
                    lineHeight: SENT_LINE_H,
                  }}
                >
                  {resolved?.text}
                </div>
              </>
            )}
            {speakNote && (
              <p className="text-[9px] font-mono text-[var(--hud-warn)]">{speakNote}</p>
            )}
          </div>
        )}
      </div>
      </div>

      {/* ===== 底部：控件行（钉住，不跟着滚） ===== */}
      <div className="shrink-0 relative border-t border-[var(--hud-line)] px-4 py-2">
        {/* settling 进度条：正在被判"这一句结束了"。看得见才知道为什么被收句。
            画法是贴在上边框上的 2px 细条（absolute，**不占行高**）—— 它只在收句前
            那一两秒出现，原来自己占一行，为它常驻留一行高度不值得。 */}
        {status?.state === "settling" && status.settleRemainMs !== null && (
          <div
            className="absolute left-0 top-0 h-[2px] transition-all duration-100"
            style={{
              width: `${Math.min(100, (status.stillMs / settleMs) * 100)}%`,
              backgroundColor: SENT_COLOR,
            }}
          />
        )}

        {/*
          一行装完：按钮组 · 开关组 · （右对齐）提示 + 读数 + 状态。
          `flex-wrap` 是宽度不够时的退路（窄窗口会折成两行），宽屏下用不到 ——
          1440 屏上这一行内容约 600px，容器 1416px。
        */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <div className="flex items-center gap-1.5">
          {/*
            连续模式下主按钮是**开始 / 停止**（点一次管一整段对话），
            一句一次模式下是「开始一句 / 重新开始一句」。
            两档共用一个按钮位：并排放两个会让人不知道该点哪个。
          */}
          {continuous && running ? (
            <button
              onClick={onStop}
              disabled={disabled}
              className={`cyber-btn cyber-btn-accent px-3 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
                disabled ? "opacity-40 cursor-not-allowed" : ""
              }`}
              title="不再自动接下一句。手头这半句会被丢掉（不解码）"
            >
              <StopCircle className="w-3.5 h-3.5" />
              停止
            </button>
          ) : (
            <button
              onClick={onArm}
              disabled={disabled}
              className={`cyber-btn px-3 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
                armed || capturing ? "cyber-btn-accent" : ""
              } ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              <Mic className="w-3.5 h-3.5" />
              {continuous ? "开始" : armed || capturing ? "重开" : "开始一句"}
            </button>
          )}
          <button
            onClick={onFinish}
            disabled={disabled || !(armed || capturing)}
            className={`cyber-btn px-2.5 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
              disabled || !(armed || capturing) ? "opacity-40 cursor-not-allowed" : ""
            }`}
            title="现在就把这一句切掉（不等自动收句）"
          >
            <Square className="w-3 h-3" />
            结束
          </button>
          <button
            onClick={() => words?.length && onDeleteWord(words.length - 1)}
            disabled={!words?.length}
            className={`cyber-btn px-2.5 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
              !words?.length ? "opacity-40 cursor-not-allowed" : ""
            }`}
            title="删掉最后一个词"
          >
            <Delete className="w-3 h-3" />
            删词
          </button>
          {/*
            ===== 代词：原来那排词条唯一**必须保留**的能力 =====

            句子区里那层可点词条已经删掉了（翻译时对面在读屏幕，不该挂调试信息）。
            但「我/你/他」不是调试信息：这三个在手语里只差指向，六轴 IMU 观测不到
            绝对朝向、模型**永远**分不出，现在给的是按位置猜的默认值。没有修正入口，
            含代词的句子就永久错着，而且用户看不出错在哪 —— 所以它下来了、但没被删。

            放在这一行而不是句子区：这里本来就是"控件"的地方（删词/定版/朗读），
            而且只有真出现合并类时才亮成主色，平时是一颗灰按钮，不抢那句大字。
            句子里没有合并类时**禁用**而不是隐藏：按钮位置固定，不会让整行抖一下。
          */}
          <div className="relative">
            <button
              onClick={() => setPronOpen((o) => !o)}
              disabled={!hasMerged}
              className={`cyber-btn px-2.5 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
                !hasMerged ? "opacity-40 cursor-not-allowed" : ""
              }`}
              style={hasMerged ? { color: SENT_COLOR } : undefined}
              title={
                hasMerged
                  ? "这句里有模型分不出的代词（我/你/他），点一下改"
                  : "这句里没有需要人来定的代词"
              }
            >
              代词
              {pronOpen ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
            {pronOpen && hasMerged && (
              <>
                {/* 点外面收起来。零高度浮层那套的老写法，见 Translate.tsx 的徽标 */}
                <div className="fixed inset-0 z-40" onClick={() => setPronOpen(false)} />
                <div className="absolute left-0 bottom-full mb-1 z-50 w-[260px] rounded-sm border border-[var(--hud-line-strong)] bg-[var(--hud-surface)] px-3 py-2 shadow-lg space-y-2">
                  {/* 一句里可能有**多个**合并类（"我问你"），所以按出现顺序逐个列，
                      每个给三个候选。只列合并类：普通词不需要人来定 */}
                  {words?.map((w, i) => {
                    const choices = pronounChoices(w);
                    if (!choices) return null;
                    const shown = overrides[i] ?? resolved?.resolved[i] ?? w;
                    return (
                      <div key={`${i}-${w}`} className="flex items-center gap-1.5">
                        <span className="text-[10px] font-mono text-[var(--hud-dim)] w-10 shrink-0">
                          第 {i + 1} 个
                        </span>
                        {choices.map((m) => (
                          <button
                            key={m}
                            onClick={() => onOverride(i, m)}
                            className={`px-2 py-1 text-[11px] font-mono rounded-sm border ${
                              shown === m
                                ? "bg-[var(--hud-warn-wash)] text-[var(--hud-warn)] border-[var(--hud-warn-edge)]"
                                : "text-[var(--hud-soft)] border-[var(--hud-line)] hover:text-[var(--hud-warn)]"
                            }`}
                          >
                            {displayWord(m)}
                          </button>
                        ))}
                      </div>
                    );
                  })}
                  {/* 为什么要人来定，说一次。不说的话这个浮层看着像多余的选择题 */}
                  <p className="text-[9px] font-mono text-[var(--hud-dim)] leading-relaxed border-t border-[var(--hud-line)] pt-1.5">
                    「{displayWord("merged_pron_sg")}」是一个类：
                    {PRON_SG_OPTIONS.map((m) => displayWord(m)).join(" / ")}
                    在手语里只差指向，六轴 IMU 观测不到绝对朝向、模型永远分不出 ——
                    上面高亮的是按位置猜的默认值。
                  </p>
                </div>
              </>
            )}
          </div>
          <button
            onClick={() =>
              resolved?.text && words && onCommit(resolved.text, words, resolved.rule)
            }
            disabled={!resolved?.text}
            className={`cyber-btn px-2.5 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
              !resolved?.text ? "opacity-40 cursor-not-allowed" : ""
            }`}
            title={
              continuous
                ? "定版：这一句已经在历史里了，点一下不再让后续编辑同步过去"
                : "把这一句加进历史"
            }
          >
            <Check className="w-3 h-3" />
            {continuous ? "定版" : "成句"}
          </button>
          <button
            onClick={doSpeak}
            disabled={!resolved?.text || !speechSupported()}
            className={`cyber-btn px-2.5 py-1.5 rounded-sm text-[11px] flex items-center gap-1.5 ${
              !resolved?.text || !speechSupported()
                ? "opacity-40 cursor-not-allowed"
                : ""
            }`}
            title={speechSupported() ? "朗读顺句结果" : "这个浏览器不支持语音合成"}
          >
            <Volume2 className="w-3 h-3" />
            朗读
          </button>
          </div>

          <div className="w-px self-stretch bg-[var(--hud-line)]" />

          {/* 三个开关。跟按钮同一行，但用竖线隔开 —— 它们改的是"怎么录/怎么念"
              这类设定，跟左边那组"现在做什么"不是一类操作，混在一起会误点 */}
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-[9px] font-mono text-[var(--hud-dim)] cursor-pointer">
              <input
                type="checkbox"
                checked={continuous}
                onChange={(e) => onToggleContinuous(e.target.checked)}
                className="accent-[var(--hud-accent)]"
              />
              连续
            </label>
            <label className="flex items-center gap-1.5 text-[9px] font-mono text-[var(--hud-dim)] cursor-pointer">
              <input
                type="checkbox"
                checked={grammarOn}
                onChange={(e) => onToggleGrammar(e.target.checked)}
                className="accent-[var(--hud-accent)]"
              />
              顺句
            </label>
            <label
              className={`flex items-center gap-1.5 text-[9px] font-mono ${
                speechSupported()
                  ? "text-[var(--hud-dim)] cursor-pointer"
                  : "text-[var(--hud-faint)] cursor-not-allowed"
              }`}
              title={
                speechSupported()
                  ? "收句后自动把顺句结果念出来。改完代词要重念请点「朗读」"
                  : "这个浏览器不支持语音合成（Web Speech API）"
              }
            >
              <input
                type="checkbox"
                checked={autoSpeak && speechSupported()}
                disabled={!speechSupported()}
                onChange={(e) => onToggleAutoSpeak(e.target.checked)}
                className="accent-[var(--hud-accent)]"
              />
              朗读
            </label>
          </div>

          {/* 把状态区顶到最右。用空的 flex-1 而不是 justify-between：
              wrap 之后 justify-between 会把折下来的那一行也两端拉开 */}
          <div className="flex-1 min-w-0" />

          <div className="flex items-center gap-2 min-w-0">
            {/* 提示文字。原来自己占一行；`truncate` 是必须的 —— 它的长度不可控，
                不截断的话长提示会把这一行撑成两行，等于白压 */}
            {note && (
              <span className="text-[10px] font-mono text-[var(--hud-warn)] truncate">
                {note}
              </span>
            )}

            {/* 实测的句内最长停顿：门限为什么被抬高，只有这个数说得清。
                现场读数也靠它 —— `CONTINUOUS_SETTLE_MS` 那个基线是估的 */}
            {continuous && !!status && status.maxIntraStillMs > 0 && (
              <span className="text-[9px] font-mono text-[var(--hud-dim)] whitespace-nowrap">
                最长停顿 {(status.maxIntraStillMs / 1000).toFixed(2)}s · 门限{" "}
                {(settleMs / 1000).toFixed(2)}s
              </span>
            )}

            <span
              className="text-[10px] font-mono whitespace-nowrap"
              style={{ color: capturing || armed ? SENT_COLOR : "var(--hud-dim)" }}
            >
              {status?.state === "armed"
                ? "等你起手"
                : status?.state === "capturing"
                  ? `录制中 ${(status.elapsedMs / 1000).toFixed(1)}s`
                  : status?.state === "settling"
                    ? `静止 ${(status.stillMs / 1000).toFixed(1)}s / ${(
                        settleMs / 1000
                      ).toFixed(1)}s`
                    : "未开始"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * 一条已定版的历史气泡。
 *
 * 默认只显示顺句结果（大字）；点一下展开，露出**模型的原始词序**和当时命中的规则名。
 * 默认收起是因为这一列很窄，每条都摊开两行的话，屏幕上放不下几句；
 * 但原始词序必须能点出来 —— 见文件头那段（分不清"模型认错"还是"规则顺错"）。
 *
 * **所有历史条目一个样子**（`SENT_FONT_PREV` / `SENT_COLOR_PREV`），不再按 `depth`
 * 逐档缩。原来那套三档梯度是为"一屏挂四五条歌词"配的；现在可视区被
 * `FLOW_MAX_H` 卡成两行，能看见的历史只有最下面那一条，逐档缩下去的那几档
 * 全在折线以上 —— 留着只是让往上滚的人越看越小。
 *
 * 展开区里的小字**不跟着缩**：那是要读的数据，淡下去就白展开了。
 */
function HistoryBubble({
  entry,
  open,
  onToggle,
}: {
  entry: SentenceEntry;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="px-1">
      <button
        onClick={onToggle}
        className="w-full flex items-start gap-1.5 text-left group"
        title={open ? "收起原始词序" : "展开看模型解出的原始词序"}
      >
        {/* 箭头跟着字号长，理由同 live 气泡那处 */}
        <span
          className="shrink-0 flex items-center"
          style={{
            fontSize: SENT_FONT_PREV,
            height: `${SENT_LINE_H}em`,
            transition: TIER_TRANSITION,
          }}
        >
          {open ? (
            <ChevronDown className="w-[0.4em] h-[0.4em] text-[var(--hud-faint)]" />
          ) : (
            <ChevronRight className="w-[0.4em] h-[0.4em] text-[var(--hud-faint)] group-hover:text-[var(--hud-dim)]" />
          )}
        </span>
        {/* 比当前句小一号、淡一档 —— 见 `SENT_FONT_PREV` */}
        <span
          className="min-w-0"
          style={{
            fontFamily: "var(--font-sentence)",
            color: SENT_COLOR_PREV,
            fontSize: SENT_FONT_PREV,
            fontWeight: SENT_WEIGHT_HISTORY,
            lineHeight: SENT_LINE_H,
            transition: TIER_TRANSITION,
          }}
        >
          {entry.text}
        </span>
      </button>
      {open && (
        <div className="mt-2 pl-4.5 space-y-1.5 border-t border-[var(--hud-line)] pt-2">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            原始词序 · 模型输出
          </div>
          <div className="flex flex-wrap items-center gap-1">
            {entry.words.map((w, i) => (
              <span
                key={`${i}-${w}`}
                className="px-1.5 py-0.5 rounded-sm border border-[var(--hud-line)] text-[10px] font-mono text-[var(--hud-soft)]"
              >
                {displayWord(w)}
              </span>
            ))}
          </div>
          <div className="text-[9px] font-mono text-[var(--hud-faint)]">
            {entry.rule ? `规则：${entry.rule}` : "没命中规则 · 原样拼接"}
            {" · "}
            {new Date(entry.at).toLocaleTimeString("zh-CN", { hour12: false })}
          </div>
        </div>
      )}
    </div>
  );
}
