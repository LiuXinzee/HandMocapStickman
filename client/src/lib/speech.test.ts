/*
 * speech.test —— 朗读的三个"点了没声音"坑
 *
 * 全部用假的 speechSynthesis 驱动。真引擎在 vitest（jsdom/node）里不存在，
 * 而这里要断言的本来就是**调用序列**，不是有没有出声。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { speakChinese, speechSupported, stopSpeaking } from "./speech";

interface FakeUtterance {
  text: string;
  lang: string;
  rate: number;
  voice: unknown;
}

let spoken: FakeUtterance[];
let cancels: number;
let voices: Array<{ name: string; lang: string }>;

beforeEach(() => {
  spoken = [];
  cancels = 0;
  voices = [];
  // SpeechSynthesisUtterance 在 node 里不存在，自己造一个
  (globalThis as any).SpeechSynthesisUtterance = class {
    text: string;
    lang = "";
    rate = 1;
    voice: unknown = null;
    constructor(t: string) {
      this.text = t;
    }
  };
  (globalThis as any).window = {
    speechSynthesis: {
      getVoices: () => voices,
      speak: (u: FakeUtterance) => spoken.push(u),
      cancel: () => {
        cancels++;
      },
    },
  };
});

afterEach(() => {
  delete (globalThis as any).window;
  delete (globalThis as any).SpeechSynthesisUtterance;
  vi.restoreAllMocks();
});

describe("speakChinese", () => {
  it("挑 zh-CN 的语音", () => {
    voices = [
      { name: "Microsoft David", lang: "en-US" },
      { name: "Microsoft Huihui", lang: "zh-CN" },
    ];
    const r = speakChinese("你好");
    expect(r.ok).toBe(true);
    expect(r.voice).toBe("Microsoft Huihui");
    expect((spoken[0].voice as any).name).toBe("Microsoft Huihui");
  });

  it("没有 zh-CN 时退到任意 zh-*（港澳台系统装的是 zh-TW）", () => {
    voices = [{ name: "Hanhan", lang: "zh-TW" }];
    expect(speakChinese("你好").voice).toBe("Hanhan");
  });

  it("voice 列表为空时不设 voice，但仍然发得出去", () => {
    // 冷启动时 getVoices() 常常是空的（列表异步填）。这时候放弃 = 第一次点朗读永远没声音；
    // 设 voice=null 又会让某些实现直接不响。所以：不设，交给系统挑默认的
    voices = [];
    const r = speakChinese("你好");
    expect(r.ok).toBe(true);
    expect(r.voice).toBeNull();
    expect(spoken).toHaveLength(1);
    expect(spoken[0].voice).toBeNull();
    // 但要**说出来**用的是默认语音 —— 念不准时用户得知道为什么
    expect(r.reason).toContain("系统默认语音");
  });

  it("只有英文语音时也不硬塞，走系统默认", () => {
    voices = [{ name: "David", lang: "en-US" }];
    const r = speakChinese("你好");
    expect(r.voice).toBeNull();
    expect(spoken[0].voice).toBeNull();
  });

  it("每次朗读前先 cancel（连点是重念，不是排队念两遍）", () => {
    speakChinese("第一句");
    speakChinese("第二句");
    expect(cancels).toBe(2);
    expect(spoken.map((u) => u.text)).toEqual(["第一句", "第二句"]);
  });

  it("lang 固定 zh-CN，语速压到 0.9", () => {
    speakChinese("我爱你。");
    expect(spoken[0].lang).toBe("zh-CN");
    expect(spoken[0].rate).toBe(0.9);
  });

  it("空串/纯空白不发给引擎", () => {
    // 某些实现喂空串会把队列卡住，之后所有朗读都不响
    expect(speakChinese("").ok).toBe(false);
    expect(speakChinese("   ").ok).toBe(false);
    expect(spoken).toHaveLength(0);
  });

  it("不支持时给能显示的原因，而不是抛错", () => {
    (globalThis as any).window = {};
    const r = speakChinese("你好");
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("不支持");
    expect(speechSupported()).toBe(false);
  });

  it("文本首尾空白被去掉", () => {
    speakChinese("  我高兴。  ");
    expect(spoken[0].text).toBe("我高兴。");
  });
});

describe("stopSpeaking", () => {
  it("调 cancel", () => {
    stopSpeaking();
    expect(cancels).toBe(1);
  });

  it("不支持时静默返回，不抛", () => {
    (globalThis as any).window = {};
    expect(() => stopSpeaking()).not.toThrow();
  });
});
