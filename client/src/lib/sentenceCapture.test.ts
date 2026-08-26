/*
 * sentenceCapture.test —— 整句捕获的时序判据
 *
 * 这里每一条判错的代价都是"整句没了"，而且都不会报错：
 *
 *   - 没有 armed 状态 → 按下按钮时手是静止的，起手前就被收句，永远录不到东西
 *   - settling 中途见到动作不回退 → 词间过渡的短暂停顿被当成句尾，一句被切成两半
 *   - 上限只在 capturing 里查 → 一直静止时无限等下去
 *   - 收句时不减掉那 800ms 静止 → 整段重采样到 128 帧，尾部静止挤掉真正有词的帧
 *
 * 全部用显式时间戳驱动，不依赖真实时钟。
 */
import { describe, expect, it } from "vitest";
import {
  ARM_TIMEOUT_MS,
  MAX_UTTERANCE_MS,
  SETTLE_MS,
  SentenceCapture,
} from "./sentenceCapture";

/** 按 100ms 一跳喂进去，返回第一个非 none 的动作和它发生的时刻 */
function run(
  cap: SentenceCapture,
  from: number,
  steps: Array<{ ms: number; moving: boolean }>
) {
  let t = from;
  for (const s of steps) {
    for (let elapsed = 0; elapsed < s.ms; elapsed += 100) {
      t += 100;
      const a = cap.tick(t, s.moving);
      if (a.kind !== "none") return { action: a, at: t };
    }
  }
  return { action: { kind: "none" } as const, at: t };
}

describe("SentenceCapture", () => {
  it("idle 状态下 tick 不做任何事", () => {
    const cap = new SentenceCapture();
    expect(cap.tick(1000, true).kind).toBe("none");
    expect(cap.tick(2000, false).kind).toBe("none");
    expect(cap.status().state).toBe("idle");
  });

  it("armed 状态下的静止不收句（否则起手前就被收掉）", () => {
    // 这是最关键的一条：按下按钮的那一刻手必然是静止的。
    // 少了 armed 状态，"静止 800ms 收句"会在人刚要抬手时触发，一个字都录不到
    const cap = new SentenceCapture();
    cap.arm(0);
    const r = run(cap, 0, [{ ms: SETTLE_MS * 3, moving: false }]);
    expect(r.action.kind).toBe("none");
    expect(cap.status().state).toBe("armed");
  });

  it("见到动作才开始计时，起点是动作那一刻不是按钮那一刻", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    run(cap, 0, [{ ms: 2000, moving: false }]); // 按了之后干等 2 秒
    cap.tick(2100, true); // 这一刻才起手
    expect(cap.status().state).toBe("capturing");
    cap.tick(3100, true);
    // 已录时长从 2100 算，不是从 0 算 —— 从 0 算会把 2 秒静止塞进句子
    expect(cap.status().elapsedMs).toBe(1000);
  });

  it("静止满 800ms 收句", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true); // 起手
    const r = run(cap, 100, [
      { ms: 2000, moving: true },
      { ms: SETTLE_MS + 200, moving: false },
    ]);
    expect(r.action).toEqual({ kind: "decode", reason: "settled" });
    expect(cap.status().state).toBe("idle");
  });

  it("静止不满 800ms 又动了 → 不收句，计时作废", () => {
    // 词与词之间的过渡有短暂停顿。一停就收的话「我 爱 你」会被切成三句
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true);
    const r = run(cap, 100, [
      { ms: 1000, moving: true },
      { ms: SETTLE_MS - 200, moving: false }, // 停一下，没到门限
      { ms: 1000, moving: true }, // 又动了
      { ms: SETTLE_MS - 200, moving: false }, // 再停一下，还是没到
      { ms: 500, moving: true },
    ]);
    expect(r.action.kind).toBe("none");
    expect(cap.status().state).toBe("capturing");
  });

  it("退回 capturing 后静止计时重新从 0 开始，不累计", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true);
    cap.tick(200, false); // 开始静止
    cap.tick(700, false); // 静止 500ms（未到 800）
    cap.tick(800, true); // 动了 → 作废
    cap.tick(900, false); // 重新开始静止
    // 如果计时是累计的，这里 500+100 已经超过 800 的一半，再过 300 就会误收
    expect(cap.tick(1200, false).kind).toBe("none");
    expect(cap.status().stillMs).toBe(300);
  });

  it("到 12s 上限强制收句", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true);
    const r = run(cap, 100, [{ ms: MAX_UTTERANCE_MS + 1000, moving: true }]);
    expect(r.action).toEqual({ kind: "decode", reason: "maxLength" });
    expect(r.at - 100).toBeGreaterThanOrEqual(MAX_UTTERANCE_MS);
  });

  it("settling 里也查上限（一直静止不能无限等）", () => {
    // 只在 capturing 里查上限的话：起手后立刻停住，会卡在 settling 直到静止满
    // 800ms 才收 —— 这条其实自然会收。真正的漏洞是 settle 门限被调很大时，
    // settling 可以停留任意久。这里锁住上限在两个状态里都有效
    const cap = new SentenceCapture({ settleMs: 1e9 });
    cap.arm(0);
    cap.tick(100, true);
    const r = run(cap, 100, [
      { ms: 500, moving: true },
      { ms: MAX_UTTERANCE_MS, moving: false },
    ]);
    expect(r.action).toEqual({ kind: "decode", reason: "maxLength" });
  });

  it("起手超时后放弃，不留在 armed 里空转", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    const r = run(cap, 0, [{ ms: ARM_TIMEOUT_MS + 500, moving: false }]);
    expect(r.action).toEqual({ kind: "abort", reason: "armTimeout" });
    expect(cap.status().state).toBe("idle");
  });

  it("手动结束：录到动作了就解码", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true);
    cap.tick(1000, true);
    expect(cap.finish()).toEqual({ kind: "decode", reason: "manual" });
    expect(cap.status().state).toBe("idle");
  });

  it("手动结束：一个动作都没录到就 abort，不送空段去解码", () => {
    // 空段送进 CTC 会解出空句或者一串乱词（模型对任意输入都会输出某个东西），
    // 而人看到的是"打了一句但出来的完全不对"，会去怀疑模型
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, false);
    expect(cap.finish()).toEqual({ kind: "abort", reason: "noMotion" });
  });

  it("idle 状态下 finish 也是 abort", () => {
    expect(new SentenceCapture().finish()).toEqual({ kind: "abort", reason: "noMotion" });
  });

  it("重复 arm 等于放弃前一句", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true);
    cap.tick(2000, true);
    cap.arm(3000);
    expect(cap.status().state).toBe("armed");
    expect(cap.status().elapsedMs).toBe(0);
  });

  it("cancel 回到 idle 且不产生动作", () => {
    const cap = new SentenceCapture();
    cap.arm(0);
    cap.tick(100, true);
    cap.cancel();
    expect(cap.status().state).toBe("idle");
    expect(cap.tick(200, false).kind).toBe("none");
  });

  describe("captureSpanMs", () => {
    it("settled 收句要减掉收句用掉的那段静止", () => {
      // 整段会被重采样到定长 128 帧。尾部多 800ms 静止 = 每个词分到的帧变少，
      // 而那 800ms 里没有任何信息
      const cap = new SentenceCapture();
      cap.arm(0);
      cap.tick(100, true);
      const r = run(cap, 100, [
        { ms: 3000, moving: true },
        { ms: SETTLE_MS + 100, moving: false },
      ]);
      expect(r.action.kind).toBe("decode");
      const span = cap.captureSpanMs(r.action);
      const raw = r.at - 100;
      expect(span).toBe(raw - SETTLE_MS);
      expect(span).toBeGreaterThan(2500); // 真正有词的那段还在
    });

    it("手动收句不减 —— 那段静止不是判据造成的", () => {
      const cap = new SentenceCapture();
      cap.arm(0);
      cap.tick(100, true);
      cap.tick(3100, true);
      const a = cap.finish();
      expect(cap.captureSpanMs(a)).toBe(3000);
    });

    it("maxLength 收句不减，且不超过上限", () => {
      const cap = new SentenceCapture();
      cap.arm(0);
      cap.tick(100, true);
      const r = run(cap, 100, [{ ms: MAX_UTTERANCE_MS + 500, moving: true }]);
      const span = cap.captureSpanMs(r.action);
      expect(span).toBeLessThanOrEqual(MAX_UTTERANCE_MS);
      expect(span).toBeGreaterThan(MAX_UTTERANCE_MS - 200);
    });

    it("减完不会变成负数", () => {
      // 起手后立刻停住：raw 可能小于 settleMs
      const cap = new SentenceCapture();
      cap.arm(0);
      cap.tick(100, true);
      const r = run(cap, 100, [{ ms: SETTLE_MS + 200, moving: false }]);
      expect(r.action.kind).toBe("decode");
      expect(cap.captureSpanMs(r.action)).toBeGreaterThanOrEqual(0);
    });
  });
});
