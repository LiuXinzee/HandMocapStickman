/*
 * speech — 把顺句结果读出来（Web Speech API）。
 *
 * 为什么值得单独一个模块：这是仓库里第一处 TTS，而 `window.speechSynthesis` 有三个
 * 到了现场才会暴露的坑，都表现为"点了朗读没声音"：
 *
 * 1. **voice 列表是异步填的。** 冷启动第一次调用时 `getVoices()` 往往返回空数组，
 *    要等 `voiceschanged`。空数组时不能放弃 —— 不指定 voice 让系统挑默认的，
 *    多数情况下仍然能出声。
 * 2. **不选中文 voice 就会用英文引擎念汉字**，出来是一串字母音。所以要按 lang
 *    过滤，但只在真找到时才设置（见上一条）。
 * 3. **不 cancel 就会排队。** 连点两次朗读是排队播两遍，而用户的预期是"重念"。
 *
 * 返回值刻意做成 `{ ok, reason }` 而不是抛错：朗读失败不该打断句子界面，
 * 但**必须能显示出来** —— 静默失败等于让用户以为音箱坏了。
 */

export interface SpeakResult {
  ok: boolean;
  /** 失败原因（可直接显示给用户）；成功时是 null */
  reason: string | null;
  /** 实际用上的 voice 名字；没指定（用系统默认）时是 null */
  voice: string | null;
}

/** 目标语言。zh-CN 优先，退到任何 zh-*（港澳台系统上装的是 zh-TW/zh-HK） */
const PREFERRED = "zh-CN";

function pickVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  if (!voices.length) return null;
  return (
    voices.find((v) => v.lang === PREFERRED) ??
    voices.find((v) => v.lang?.replace("_", "-").startsWith("zh")) ??
    null
  );
}

/**
 * 读一句中文。同步返回是否发得出去 —— 真正的播放是异步的，这里不等它。
 *
 * @param text 要读的文本；空串直接失败（不要给引擎喂空串，某些实现会卡住队列）
 */
export function speakChinese(text: string): SpeakResult {
  const s = typeof window !== "undefined" ? window.speechSynthesis : undefined;
  if (!s) {
    return { ok: false, reason: "这个浏览器不支持语音合成（Web Speech API）", voice: null };
  }
  const t = text.trim();
  if (!t) return { ok: false, reason: "没有内容可朗读", voice: null };

  // 先 cancel：连点两次的预期是"重念"，不是排队念两遍
  s.cancel();

  const u = new SpeechSynthesisUtterance(t);
  u.lang = PREFERRED;
  const voice = pickVoice(s.getVoices() ?? []);
  if (voice) {
    // 只在真挑到中文 voice 时才设置。voice 列表冷启动时是空的，
    // 此时设 null 会让某些实现直接不出声，交给系统挑默认的反而能响
    u.voice = voice;
  }
  // 默认速率对手语翻译偏快（用户往往在对照屏幕上的字），压一点
  u.rate = 0.9;
  s.speak(u);
  return {
    ok: true,
    reason: voice
      ? null
      : "没找到中文语音包，用的是系统默认语音（可能念不准）",
    voice: voice?.name ?? null,
  };
}

/** 停止朗读。切模式/开始新一句时调用 —— 上一句还在念会盖住新句子 */
export function stopSpeaking(): void {
  if (typeof window !== "undefined") window.speechSynthesis?.cancel();
}

/** 浏览器支不支持。UI 用它决定朗读按钮是不是灰的 */
export function speechSupported(): boolean {
  return typeof window !== "undefined" && !!window.speechSynthesis;
}
