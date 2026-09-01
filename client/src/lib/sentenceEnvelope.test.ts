/*
 * sentenceEnvelope.test —— 时间包络。
 *
 * 这个文件锁的是**采集端和推理端会不会产生不同长度的样本**。这类 bug 编译得过、
 * 跑得通、结果看着正常，只有戴上手套打真句子才发现打什么都不准，所以必须在这里挡住。
 *
 * 用假帧喂缓冲：`judgeWindowMotion` 靠弯折能量判动作，这里给不带标定量程的帧
 * （`ranges = {}`），走的是未标定分支 —— 判据仍然成立（能量按原始 ADC 量程算），
 * 只是门限没那么准。测的是包络算术，不是门限精度。
 */
import { describe, expect, it } from "vitest";
import { SentenceEnvelope } from "./sentenceEnvelope";
import {
  SETTLE_MS,
  SentenceCapture,
  type CaptureAction,
} from "./sentenceCapture";
import { SequenceWindowBuffer } from "./sequenceWindow";
import { SEQ_SENSOR_N, type SequenceSample } from "./datasetStore";
import type { GloveFrame } from "./gloveProtocol";

/**
 * 一帧假手套数据。传感器全给同一个值 —— 值变化 = 有动作，值不变 = 静止。
 *
 * 字段名必须和 `SequenceWindowBuffer.push` 读的那几个对上
 * （`mapped_data` / `quaternion` / `acceleration` / `attitude`），
 * 拼错的话缓冲里存的是 undefined，重采样时才炸。
 */
function frame(t: number, v: number): GloveFrame {
  return {
    timestamp: t,
    mapped_data: new Array(SEQ_SENSOR_N).fill(v),
    quaternion: [1, 0, 0, 0],
    acceleration: [0, 0, 0],
    attitude: [0, 0, 0],
  } as unknown as GloveFrame;
}

const FPS = 50;
const DT = 1000 / FPS;
/** 推理端的定时器周期（`Translate.tsx` 的 100ms 循环） */
const TICK_MS = 100;

/**
 * 一段数据的性质：
 *   - `still` 钉在 128（静止基线）
 *   - `move`  在 60~139 之间摆动
 *   - `open`  在 200~249 之间摆动 —— 也是动作，但**值域和 move 不重叠**，
 *             于是"某个样本里还有没有这一段"可以直接从数据里查出来
 */
type Phase = { ms: number; kind: "still" | "move" | "open" };

function value(kind: Phase["kind"], i: number): number {
  if (kind === "still") return 128;
  return kind === "open" ? 200 + ((i * 13) % 50) : 60 + ((i * 37) % 80);
}

/** 样本里有没有 `open` 那一段（`move` 和 `still` 都在 190 以下） */
function hasOpen(s: SequenceSample): boolean {
  const a = s.leftSensor!;
  for (let i = 0; i < a.length; i++) if (a[i] > 190) return true;
  return false;
}

/** 样本尾部连续静止了多少帧（重采样网格是 50fps，1 帧 = 20ms） */
function trailStillFrames(s: SequenceSample): number {
  const a = s.leftSensor!;
  let n = 0;
  for (let t = s.frameCount - 1; t >= 0; t--) {
    const o = t * SEQ_SENSOR_N;
    let moved = false;
    for (let c = 0; c < SEQ_SENSOR_N; c++) {
      if (a[o + c] !== 128) {
        moved = true;
        break;
      }
    }
    if (moved) break;
    n++;
  }
  return n;
}

/**
 * 按**真实时序**喂数据：帧 50fps 进缓冲，每 100ms 跳一次 tick，
 * 见到第一个非 `none` 的 action 就停 —— 推理端就是在那一跳里立刻 `take()` 的。
 *
 * 这个交错是必须的，不能"先喂完所有帧再补跳 tick"：那样缓冲会比状态机的 `lastNow`
 * 多出一截，而 `snapshotAll` 的区间贴着**缓冲尾部**对齐、`captureSpanMs` 却是按
 * `lastNow` 算的，多出来的那一截会让整个区间平移，测出来的东西就不是包络本身了。
 */
class Player {
  private t: number;
  private i = 0;
  private nextTick: number;

  constructor(
    private readonly buf: SequenceWindowBuffer,
    private readonly env: SentenceEnvelope,
    t0 = 0
  ) {
    this.t = t0;
    this.nextTick = t0;
  }

  play(...phases: Phase[]): CaptureAction {
    for (const ph of phases) {
      const end = this.t + ph.ms;
      for (; this.t < end; this.t += DT, this.i++) {
        const v = value(ph.kind, this.i);
        this.buf.push("left", frame(this.t, v));
        this.buf.push("right", frame(this.t, v));
        while (this.nextTick <= this.t) {
          const act = this.env.tick(this.nextTick, {});
          this.nextTick += TICK_MS;
          if (act.kind !== "none") return act;
        }
      }
    }
    return { kind: "none" };
  }
}

/** 只填缓冲、不走状态机（测 arm 清缓冲用） */
function fill(buf: SequenceWindowBuffer, t0: number, ms: number): number {
  let t = t0;
  for (let i = 0; t < t0 + ms; t += DT, i++) {
    buf.push("left", frame(t, value("move", i)));
    buf.push("right", frame(t, value("move", i)));
  }
  return t - DT;
}

describe("SentenceEnvelope", () => {
  describe("arm", () => {
    it("清缓冲 —— 上一句的尾巴不能接到这一句前面", () => {
      // 这是 arm 必须住在 envelope 里、不能交给调用方的原因：
      // snapshotAll 是"有多少取多少"，漏清一次就把两句拼成一句
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      fill(buf, 0, 3000);
      const env = new SentenceEnvelope(buf);
      env.arm(3000);
      // 缓冲空了 → 探测窗口取不到东西 → 判"没动"，状态停在 armed
      expect(env.tick(3100, {}).kind).toBe("none");
      expect(env.status().state).toBe("armed");
    });
  });

  describe("起点是见到动作的那一刻，不是 arm 的那一刻", () => {
    it("arm 之后先静止 2 秒再起手，样本里不含那 2 秒", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const env = new SentenceEnvelope(buf);
      env.arm(0);
      const p = new Player(buf, env);
      // 2s 静止（模拟"按了按钮才慢慢把手举起来"）
      expect(p.play({ ms: 2000, kind: "still" }).kind).toBe("none");
      expect(env.status().state).toBe("armed");

      // 2s 动作
      expect(p.play({ ms: 2000, kind: "move" }).kind).toBe("none");
      expect(env.status().state).toBe("capturing");

      const sample = env.take({ kind: "decode", reason: "manual" }, "_x");
      expect(sample).not.toBeNull();
      // 只该拿到动作那 2s 左右。含上那 2s 静止就会是 ~4s ——
      // 那多出来的一倍会让每个词在归一化时间轴上缩到一半
      expect(sample!.durationMs).toBeLessThan(3000);
      expect(sample!.durationMs).toBeGreaterThan(1000);
    });
  });

  describe("take：span 与 dropTail 必须成对", () => {
    it("停手收句时掐掉的是**尾部**静止，句子开头留着", () => {
      /*
       * 这一条是整个文件的重点。
       *
       * 缓冲区间贴着尾部对齐，所以：
       *   - 只调小 span（不砍尾巴）→ 区间整体往后平移 `settleMs`，
       *     砍掉的是句子**开头**，尾部静止照样在
       *   - 两个一起传               → 掐掉尾巴、保住开头
       * 两种取法的帧数**一模一样**，光看 durationMs / frameCount 分不出对错，
       * 所以这里比的是数据内容：句子开头那一段还在不在。
       *
       * 开头用 `open`（值域 200~249，和后面的 move 不重叠）做记号。
       * 记号必须落在 [起手时刻, 起手时刻 + SETTLE_MS) 里 —— 正确取法的区间从起手
       * 时刻开始，错误取法的区间从起手时刻 + SETTLE_MS 开始，记号只会出现在前者里。
       *
       * 起手时刻的下界是 600ms（= 动作窗口长度，探测窗口攒满之前判不出 moving），
       * 所以记号放在 700~1300ms。
       */
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const env = new SentenceEnvelope(buf);
      env.arm(0);
      const act = new Player(buf, env).play(
        { ms: 700, kind: "move" },
        { ms: 600, kind: "open" }, // ← 句子开头的记号
        { ms: 800, kind: "move" },
        // 静止要给够：判到静止本身就滞后一个动作窗口(600ms)，再数 SETTLE_MS 才收句。
        // 只给 SETTLE_MS 长度的静止是永远收不了句的
        { ms: 3000, kind: "still" }
      );
      expect(act).toEqual({ kind: "decode", reason: "settled" });

      const good = env.take(act, "_x")!;
      expect(good).not.toBeNull();

      // 对照：错法（只缩 span、不砍尾巴），直接手搓
      const cap = env.capture;
      const bad = buf.snapshotAll(cap.captureSpanMs(act), "_x", 0)!;

      // 两者帧数相同 —— 这就是为什么时长比不出对错
      expect(bad.frameCount).toBe(good.frameCount);

      // 正确取法保住了句子开头
      expect(hasOpen(good), "正确取法把句子开头砍掉了").toBe(true);
      // 错误取法把开头那 800ms 平移掉了，记号跟着没了
      expect(hasOpen(bad), "错法居然还留着开头 —— 这条测试就白写了").toBe(false);

      // 另一头：错法尾部多留了 SETTLE_MS 的静止（40 帧 @ 50fps 网格）。
      // 注意正确取法的尾部**也还有**一段静止（约一个动作窗口）——
      // 那是收句判据本身的滞后，采集端和推理端走的是同一条路、留的一样多，
      // 所以不是包络错位。这里比的是两种取法的差，不是绝对值
      const dropped = trailStillFrames(bad) - trailStillFrames(good);
      expect(dropped).toBeGreaterThanOrEqual(SETTLE_MS / DT - 2);
      expect(dropped).toBeLessThanOrEqual(SETTLE_MS / DT + 2);
    });

    it("手动收句不掐尾巴（那段静止不是收句判据用掉的）", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const env = new SentenceEnvelope(buf);
      env.arm(0);
      new Player(buf, env).play({ ms: 2000, kind: "move" });
      const act = env.finish();
      expect(act).toEqual({ kind: "decode", reason: "manual" });
      expect(env.capture.settleDropMs(act)).toBe(0);
    });

    it("非 decode 的 action 取不到东西（abort 不该产生样本）", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const env = new SentenceEnvelope(buf);
      fill(buf, 0, 2000);
      expect(env.take({ kind: "none" }, "_x")).toBeNull();
      expect(env.take({ kind: "abort", reason: "noMotion" }, "_x")).toBeNull();
    });
  });

  describe("采集端与推理端产生同一种包络", () => {
    it("同一段数据、同一条路 → 帧数与时长逐一相等", () => {
      /*
       * 两个 envelope 各喂一份相同的帧，走相同的 tick 序列。
       * 这是"两边共用 take()"这件事的回归测试：哪天有人在其中一端另写一份
       * snapshotAll 调用，这里就会红。
       */
      const mk = () => {
        const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
        const env = new SentenceEnvelope(buf);
        env.arm(0);
        const last = new Player(buf, env).play(
          { ms: 300, kind: "still" },
          { ms: 2400, kind: "move" },
          { ms: 2500, kind: "still" }
        );
        return { buf, env, last };
      };
      const a = mk();
      const b = mk();
      expect(a.last.kind).toBe("decode");
      expect(b.last).toEqual(a.last);
      const sa = a.env.take(a.last, "_sentence")!;
      const sb = b.env.take(b.last, "_x")!;
      expect(sb.frameCount).toBe(sa.frameCount);
      expect(sb.durationMs).toBeCloseTo(sa.durationMs, 5);
    });
  });

  describe("take 返回的样本形状", () => {
    it("不带视觉（句子模型是纯触觉学生）", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const env = new SentenceEnvelope(buf);
      env.arm(0);
      new Player(buf, env).play({ ms: 2000, kind: "move" });
      const s = env.take(env.finish(), "_x")!;
      expect(s.leftLandmarks).toBeNull();
      expect(s.rightLandmarks).toBeNull();
    });

    it("segments 只有一个占位段 —— 采集端要自己换成句型的标签序列", () => {
      const buf = new SequenceWindowBuffer({ bufferMs: 12000 });
      const env = new SentenceEnvelope(buf);
      env.arm(0);
      new Player(buf, env).play({ ms: 2000, kind: "move" });
      const s = env.take(env.finish(), "_x")!;
      expect(s.segments).toHaveLength(1);
      expect(s.segments[0].label).toBe("_x");
      expect(s.primaryLabel).toBe("_x");
    });
  });

  describe("可注入的 SentenceCapture", () => {
    it("传进来的状态机就是对外暴露的那一个（参数不会被偷偷换掉）", () => {
      const cap = new SentenceCapture({ settleMs: 200 });
      const env = new SentenceEnvelope(new SequenceWindowBuffer(), cap);
      expect(env.capture).toBe(cap);
    });
  });
});
