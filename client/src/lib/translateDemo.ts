/*
 * translateDemo — 翻译页的**离线演示脚本**：不连手套也能看到译文区长什么样。
 *
 * 存在的理由：`SentencePanel` 的渲染门禁是 `modelReady && isConnected &&
 * modelMode === "sentence"`（Translate.tsx），所以没手套时那一列永远是"未连接"
 * 占位卡 —— 想看一眼排版就得插手套、连蓝牙、标定，只为看效果。
 *
 * ===== 只伪造"输入"，不伪造"输出" =====
 *
 * 这里给出的只有 `words`（模型本该吐出的词 id 序列）和节奏。**顺句仍然走真的
 * `resolveSentence`**，历史条目仍然由真的 `commitSentence` 那套结构承载，
 * 面板仍然是真的 `SentencePanel`。
 *
 * 这条界线是刻意的：如果把译文也写死成字符串，演示就变成一张**截图**——
 * 规则表改了它不会跟着变，于是它会慢慢和真实行为分叉，而分叉的方向恰好是
 * "演示里好看、真机上不对"。现在反过来：规则表改坏了，演示会立刻跟着坏，
 * `sentenceGrammar.test.ts` 也会红。
 *
 * ⚠ 所以**词 id 必须是词表里真有的**。下面每条都在 `sentenceGrammar.ts` 的
 * RULES 里命中了一条具体规则（注释里写了规则名），不是随手编的。加句子之前
 * 先跑一遍 `resolveSentence` 确认 `rule !== null` —— rule 为 null 时它会退化成
 * 原样拼接，演示里看着"能出字"，实际展示的是**规则没命中**那条退路。
 *
 * ===== 每句都带 merged_pron_sg =====
 *
 * 我/你/他 在手语里只差指向，六轴 IMU 观测不到绝对朝向、模型永远分不出，
 * 所以句子里的代词一律是合并类，界面靠「代词」按钮让人改（见 SentencePanel）。
 * 演示句刻意全都含它 —— 那个按钮是亮的、可点的，看效果的人能看到这条
 * "模型分不出、需要人定"的链路，而不是一个假装全自动的 demo。
 */

/** 一句演示：模型该吐出的词序 + 这句打完后停多久收句 */
export interface DemoSentence {
  /** 词 id 序列，必须是 sentenceGrammar 词表里的真词 */
  words: string[];
  /** 命中的规则名（仅注释用途，代码不读它 —— 读了就等于把规则表抄了两份） */
  rule: string;
  /** 顺句后的样子（同上，只为读代码的人，运行时不用） */
  preview: string;
}

/**
 * 5 句演示脚本。挑选标准：
 *  - 覆盖不同**长度**（2 / 3 / 4 词）—— 长度决定命中哪条规则，也决定大字会不会折行；
 *  - 覆盖不同**句式**（陈述 / 疑问 / 赞美 / 致谢），句尾标点各不相同，
 *    因为顺句规则里句号和问号是分别写死的，一律陈述句就看不出这一层；
 *  - 全部含 `merged_pron_sg`（理由见文件头）。
 */
export const DEMO_SCRIPT: DemoSentence[] = [
  { words: ["hello", "merged_pron_sg"], rule: "问候", preview: "你好！" },
  {
    words: ["merged_pron_sg", "name", "what"],
    rule: "问名字",
    preview: "你叫什么名字？",
  },
  {
    words: ["merged_pron_sg", "smile", "resemble", "sun"],
    rule: "笑起来像",
    preview: "你笑起来像太阳。",
  },
  { words: ["thank_you", "merged_pron_sg"], rule: "致谢", preview: "谢谢你！" },
  {
    words: ["merged_pron_sg", "beautiful"],
    rule: "赞美",
    preview: "你真好看！",
  },
];

/**
 * 节奏。
 *
 * `WORD_MS` 是"逐词跳出来"的间隔 —— 真机上词是一个一个解出来的，一次性整句
 * 冒出来会看不到那个过程。550ms 是照着实机手感定的：真机一个词大约 0.6~1.2s
 * （取决于手势幅度），取偏快的一端，因为演示要连放 5 句。
 *
 * `SETTLE_MS` 是句末停顿 → 收句。注意收句这一下**屏幕上几乎没有变化**（只有右上角
 * 句数 +1）：真机收句后大字仍然是这一句，退档要等下一句的第一个词进来才发生 ——
 * 见 `kind: "settle"` 那条的说明。所以这段停顿加上后面 `WORD_MS` 才是人感觉到的
 * "说完一句歇一下"，而退档过渡（320ms 的 `TIER_TRANSITION`）落在那之后，动画能走完。
 *
 * `HOLD_MS` 是最后一句留在屏幕上的时间，之后停住不动（不自动重播）。
 * 不循环是刻意的：自动循环的 demo 会让人分不清"卡住了"和"又播了一遍"。
 *
 * ⚠ `WORD_MS` 现在只是**兜底**。手模动作接上之后，一个词该停多久由那个词在库里
 * 那条录制的实际时长决定（`buildDemoTimeline` 的 `durations`）—— 把 2 秒的手势
 * 压进 550ms 会播成抽搐，而且那就不是"真录制"了，是把真数据变速成假的。
 * 只有库里查不到录制的词才用这个数。
 */
export const DEMO_WORD_MS = 550;
export const DEMO_SETTLE_MS = 900;
export const DEMO_HOLD_MS = 2200;

/**
 * 把脚本摊平成一串**定时事件**，交给调用方按 `at` 依次执行。
 *
 * 摊平而不是在组件里写嵌套 `setTimeout`：嵌套写法的取消逻辑要跟着句子索引和
 * 词索引两层走，中途点「停止」很容易漏掉最里层那个 timer，表现为停下之后
 * 又蹦出一个词。摊平之后取消只需要清一个数组。
 */
/**
 * 一个词的手势占据的时间窗。
 *
 * `endMs` 与那个词的 `word` 事件的 `at` **是同一个数**：词是打完才出来的
 * （真机也是这样 —— 滑窗攒够、闸门放行，都在动作结束之后）。所以手模在演示里
 * 总是"先比划、字后出"，而不是字先出来手再动。
 */
export interface DemoGesture {
  wordId: string;
  startMs: number;
  endMs: number;
}

export type DemoEvent =
  /**
   * 当前这句多了一个词（`words` 是**累积**到此刻的完整词序，不是增量）。
   *
   * `first` 标出"这是新一句的头一个词"—— 上一句就在这一刻退档变小。调用方还得
   * 靠它把 `lastLive` 放掉，理由见 Translate.tsx 里 `startDemo` 的那段。
   *
   * `gesture` 是这个词的手势窗。挂在事件上、而不是另开一张时间表，是因为
   * 两张表必然会漂：手势窗的长度**就是**这个词的停留时长，同一个数派生出
   * "字什么时候出"和"手什么时候动"两件事，只能有一个来源。
   */
  | { at: number; kind: "word"; words: string[]; first: boolean; gesture: DemoGesture }
  /**
   * 这句收句：进历史，但**大字仍然留着这一句**。
   *
   * 这里曾经顺手把 live 清空了，于是句子刚译完、还没开始下一句就自己缩小成
   * 历史那一档 —— 而真机不是这样：`decodeUtterance` 收句时是
   * `setSentenceWords(post.words)` + 推历史 + `lastLive = true`，大字继续是这一句，
   * 面板那边 `settledHistory` 会把历史最后那条切掉以免画两遍。**别再清 live。**
   */
  | { at: number; kind: "settle"; words: string[] }
  /** 整段播完 */
  | { at: number; kind: "done" };

/**
 * @param durations 每个词 id 的手势时长（ms）—— 库里那条录制裁剪后的长度。
 *   查不到的词用 `DEMO_WORD_MS` 兜底。不传就是全部兜底（纯文字演示，
 *   加手模动作之前的行为逐位不变）。
 */
export function buildDemoTimeline(
  script: DemoSentence[] = DEMO_SCRIPT,
  durations?: ReadonlyMap<string, number>
): DemoEvent[] {
  const events: DemoEvent[] = [];
  let t = 0;
  for (const sentence of script) {
    for (let i = 0; i < sentence.words.length; i++) {
      const wordId = sentence.words[i];
      const startMs = t;
      const dwell = durations?.get(wordId);
      // `> 0` 而不是 `?? 兜底`：0 或负数（空录、时间戳坏了）会让整段挤在一起，
      // 而那种样本进不进得来是库那边的事，这里只保证时间轴单调
      t += dwell && dwell > 0 ? dwell : DEMO_WORD_MS;
      events.push({
        at: t,
        kind: "word",
        words: sentence.words.slice(0, i + 1),
        first: i === 0,
        gesture: { wordId, startMs, endMs: t },
      });
    }
    t += DEMO_SETTLE_MS;
    events.push({ at: t, kind: "settle", words: sentence.words });
  }
  t += DEMO_HOLD_MS;
  events.push({ at: t, kind: "done" });
  return events;
}

/** 时间轴里所有手势窗，按先后排好。手模播放器只需要这一份 */
export function gestureWindows(events: DemoEvent[]): DemoGesture[] {
  return events.flatMap((e) => (e.kind === "word" ? [e.gesture] : []));
}

/**
 * `elapsedMs` 这一刻手模该摆成什么 —— 哪个手势、放到它自己的第几毫秒。
 *
 * **间隙里返回上一个手势的最后一刻**（`offsetMs` 钳到 `endMs - startMs`），
 * 不是返回 null。理由：句末那 `SETTLE_MS` 的停顿里如果把手放回静止，屏幕上
 * 就是"打完一个词手就弹回原位"，而真人是保持在收势位置上等一下。而且
 * 弹回原位会让人以为演示卡住了。
 *
 * 第一个手势之前返回 null —— 那时候还什么都没发生，手模就该是静止姿态。
 */
export function gestureAt(
  windows: readonly DemoGesture[],
  elapsedMs: number
): { gesture: DemoGesture; offsetMs: number } | null {
  let current: DemoGesture | null = null;
  for (const g of windows) {
    if (g.startMs > elapsedMs) break;
    current = g;
  }
  if (!current) return null;
  const span = current.endMs - current.startMs;
  const offsetMs = Math.min(Math.max(0, elapsedMs - current.startMs), span);
  return { gesture: current, offsetMs };
}
