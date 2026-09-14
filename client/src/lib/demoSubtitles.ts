/*
 * demoSubtitles —— 演示专用的**译文覆盖**。用完即删。
 *
 * ⚠⚠ 开着的时候，屏幕上的译文与模型实际解出来的东西**无关**。
 *
 * 做什么：真机上每收一句（不管手上打的是什么、模型解出了什么、甚至一个词都没
 * 解出来），译文区依次显示 `DEMO_SUBTITLES` 里的下一条。第 N 句就是第 N 条。
 *
 * 为什么这么干：演示现场只要那四句按顺序出来，而模型对这四句的识别率还没到能
 * 上台的程度。
 *
 * 为什么**不做成界面开关**：试过那个方案，否决了 —— 屏幕上多一个亮着的
 * 「演示字幕」按钮，等于当众声明译文是假的。
 *
 * ===== 关它只有一条路：把下面 `DEMO_SUBTITLE_OVERRIDE` 改成 false =====
 *
 * 这是这个设计唯一的代价，摆明说：忘了关，**界面上没有任何痕迹**能提醒你。
 * 唯一的提醒是 console 里那行 warn（见 `demoSubtitleAt`）—— 观众看不见屏幕背后的
 * devtools，做演示的人一开 F12 就看得见。这套"观众看屏幕、操作者看 console"的
 * 分法，和 `demoPlayback.ts` 里那条录制质量提示是同一个理由。
 *
 * ⚠ 开着的时候采的数据不能用：`sentenceHistory` 里存的 `text` 是这里的假字符串，
 * 不是模型输出。要训练/评测/算指标，先把开关关掉。
 */
export const DEMO_SUBTITLE_OVERRIDE = true;

/**
 * 按顺序上屏的四句。第 i 句 = 第 i 条，与实际手势无关。
 *
 * 这里是**直接写字符串**，不过词表也不过顺句规则 —— 所以"漂亮"这种词表里根本
 * 没有的词也能写（`beautiful` 在词表里的标签是「好看」，走规则只能顺出
 * 「你真好看！」）。反过来也成立：改这里一个字都不会让模型多认一个词。
 */
export const DEMO_SUBTITLES: readonly string[] = [
  "你好！",
  "你笑起来真好看。",
  "你真漂亮。",
  "我爱你。",
];

/** warn 只打一次。每句都打会把 console 刷满，真正的报错就淹了 */
let warned = false;

/**
 * 第 `seq` 句该显示什么。返回 `null` = 不覆盖，照常走真的顺句结果。
 *
 * **超出四句之后退回真译文**，不循环也不卡在最后一句：演示只准备了四句，
 * 第五句继续喂假字幕的话，之后整场（包括演示结束后忘了关的那些天）全是假的。
 * 退回真译文至少是诚实的 —— 顺出来难看，正好也是个提醒。
 *
 * `seq < 0` = 还没收过任何一句。离线演示（`Translate.tsx` 的 `demoOn`）那条路
 * 不经过真解码、计数器一直停在 -1，所以它**不受这里影响**，仍然显示真的顺句
 * 结果 —— 那条路本来就是拿来验规则表的，覆盖了就白验了。
 */
export function demoSubtitleAt(seq: number): string | null {
  if (!DEMO_SUBTITLE_OVERRIDE) return null;
  if (!warned) {
    warned = true;
    console.warn(
      "[demo] 译文覆盖已开启：屏幕上的句子来自 demoSubtitles.ts，与模型输出无关。" +
        "要训练/评测，先把 DEMO_SUBTITLE_OVERRIDE 改成 false。"
    );
  }
  if (seq < 0 || seq >= DEMO_SUBTITLES.length) return null;
  return DEMO_SUBTITLES[seq];
}
