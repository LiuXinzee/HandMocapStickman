import { describe, it, expect } from "vitest";
import {
  parseEvents,
  bridgeAvailable,
  bridgeConfig,
  backboneVerdict,
} from "./trainBridge";

/**
 * 这里只测纯函数。桥的路由本身测不了（要一个真的 dev server + 真的 python），
 * 那部分靠方案里的端到端清单人工过。
 *
 * 而 `parseEvents` 恰恰是**必须**自动化的那一块：它的输入是一个边写边读的文件，
 * 三种畸形输入都会真的发生，而每一种抛出去的后果都是整张图表白屏。
 */

const DATA_LINE = JSON.stringify({
  kind: "data",
  classes: ["eat", "what"],
  blankIndex: 2,
  orphans: ["is"],
  seqLen: 128,
  outputFrames: 32,
  numReal: 45,
  numRealTrain: 36,
  numRealVal: 9,
  numSynthTrain: 106,
  numSynthVal: 53,
  numTrain: 142,
  numVal: 62,
  featShape: [142, 128, 294],
  maxLabelLen: 4,
  trimmed: true,
  epochs: 80,
});

const epochLine = (n: number, loss: number, wer: number | null, saved = false) =>
  JSON.stringify({
    kind: "epoch",
    epoch: n,
    total: 80,
    loss,
    valWer: wer,
    saved,
    bestWer: wer,
  });

describe("parseEvents", () => {
  it("空输入给出一份空结果而不是抛", () => {
    // 训练刚起来、文件刚 truncate 完，这是最常见的一次读取
    const r = parseEvents("");
    expect(r.data).toBeNull();
    expect(r.epochs).toEqual([]);
    expect(r.errors).toBeNull();
    expect(r.done).toBeNull();
    expect(r.skipped).toBe(0);
  });

  it("按 kind 分流，epoch 按出现顺序累积", () => {
    const r = parseEvents(
      [DATA_LINE, epochLine(1, 84.9, 2.48, true), epochLine(2, 55.6, 2.19)].join("\n")
    );
    expect(r.data?.classes).toEqual(["eat", "what"]);
    expect(r.data?.orphans).toEqual(["is"]);
    expect(r.data?.numRealVal).toBe(9);
    expect(r.epochs.map((e) => e.epoch)).toEqual([1, 2]);
    expect(r.epochs[0].saved).toBe(true);
    expect(r.skipped).toBe(0);
  });

  it("valWer 大于 1 原样透出（未收敛时插词能让 WER 超过 100%）", () => {
    // 实测 2 epoch 时到过 248%。图表的 WER 轴不能写死 0~1，
    // 而解析这一层更不能把它夹到 1 —— 夹了就再也看不出"插词插疯了"
    const r = parseEvents(epochLine(1, 84.9, 2.48));
    expect(r.epochs[0].valWer).toBe(2.48);
  });

  it("最后一行写了一半：前面的行照常拿到，只多记一次 skipped", () => {
    // 这是真实场景，不是造的：训练在追加写，页面在轮询读
    const half = '{"kind":"epoch","epoch":3,"total":80,"loss":0.12,"valW';
    const r = parseEvents([DATA_LINE, epochLine(1, 84.9, 2.48), half].join("\n"));
    expect(r.data).not.toBeNull();
    expect(r.epochs).toHaveLength(1);
    expect(r.skipped).toBe(1);
  });

  it("认不出的 kind 不抛、不污染已知字段", () => {
    // Python 那边加一个新事件，不该让旧页面崩
    const r = parseEvents(
      [DATA_LINE, JSON.stringify({ kind: "lr_schedule", lr: 1e-4 })].join("\n")
    );
    expect(r.data).not.toBeNull();
    expect(r.skipped).toBe(1);
  });

  it("合法 JSON 但不是对象（null / 数字 / 字符串）也不抛", () => {
    const r = parseEvents(["null", "42", '"hello"', "[1,2]"].join("\n"));
    expect(r.skipped).toBe(4);
    expect(r.data).toBeNull();
  });

  it("空行和 CRLF 都不算 skipped", () => {
    // 桥在 Windows 上跑，行尾是 \r\n；文件末尾还有一个空行
    const r = parseEvents(`${DATA_LINE}\r\n${epochLine(1, 1, 0.5)}\r\n\r\n`);
    expect(r.data).not.toBeNull();
    expect(r.epochs).toHaveLength(1);
    expect(r.skipped).toBe(0);
  });

  it("重复的 data / done 取最后一条", () => {
    // 同一个文件里出现两次意味着 truncate 没生效（换 out 目录之类）。
    // 取最后一条至少是本轮的，取第一条会长期显示上一轮的数字
    const second = JSON.parse(DATA_LINE);
    second.numReal = 60;
    const r = parseEvents([DATA_LINE, JSON.stringify(second)].join("\n"));
    expect(r.data?.numReal).toBe(60);
  });

  it("errors 事件带词表形式的 ref/hyp 和句首错计数", () => {
    const r = parseEvents(
      JSON.stringify({
        kind: "errors",
        nBad: 62,
        nTotal: 62,
        headBad: 48,
        examples: [{ ref: ["eat", "what"], hyp: ["eat"] }],
      })
    );
    expect(r.errors?.headBad).toBe(48);
    // 词表而不是拼好的字符串：界面要单独标红句首那个词
    expect(r.errors?.examples[0].hyp).toEqual(["eat"]);
  });

  it("done 里的 meta 原样保留", () => {
    const r = parseEvents(
      JSON.stringify({ kind: "done", meta: { valWer: 0.021, blankIndex: 21 } })
    );
    expect(r.done?.meta.valWer).toBe(0.021);
  });
});

describe("bridgeConfig", () => {
  it("没有注入时是 null，而不是一个半成品对象", () => {
    // vitest 环境里 window.__TRAIN_BRIDGE__ 不存在 —— 与生产构建同一支。
    // 这条锁的是"桥不在时调用方拿到明确的 null"，界面才能禁用并写明原因
    expect(bridgeConfig()).toBeNull();
    expect(bridgeAvailable()).toBe(false);
  });

  it("字段类型不对也算没有桥", () => {
    // 测试环境是 node（没有 window），所以自己搭一个 —— 上一条锁的正是
    // `typeof window === "undefined"` 那道守卫，这条锁的是字段校验
    const g = globalThis as unknown as { window?: { __TRAIN_BRIDGE__?: unknown } };
    g.window = { __TRAIN_BRIDGE__: { token: 123, base: "/__train__" } };
    expect(bridgeConfig()).toBeNull();
    g.window = { __TRAIN_BRIDGE__: { token: "abc", base: "/__train__" } };
    expect(bridgeConfig()?.token).toBe("abc");
    delete g.window;
  });
});

/**
 * `backboneVerdict` —— CTC 前置步骤的判据。
 *
 * 它防的是一个**静默**缺陷：`transfer_backbone` 按层名 + 形状搬权重，而
 * conv/bn 的形状不随类别数变，所以拿一个没见过新词的骨干去迁移**照样会打印
 * 「骨干迁移 N 层」**。唯一能发现的办法就是比标签表 —— 也就是这个函数。
 *
 * 判据用差集不用时间戳：mtime 只能说"旧了"，而旧可能只是同一批词多录了几条。
 */
describe("backboneVerdict", () => {
  const base = {
    exists: true,
    trainedAt: 2000,
    datasetAt: 1000,
    seqLen: 32,
  };

  it("数据集里的词骨干全训过 → ok", () => {
    const v = backboneVerdict(
      { ...base, backboneLabels: ["_idle", "you", "eat"], datasetLabels: ["you", "eat"] },
      []
    );
    expect(v.level).toBe("ok");
    expect(v.missing).toEqual([]);
  });

  it("骨干没见过的词被点名 —— 这是整个函数的存在理由", () => {
    const v = backboneVerdict(
      {
        ...base,
        backboneLabels: ["_idle", "you", "eat"],
        datasetLabels: ["you", "eat", "beautiful"],
      },
      []
    );
    expect(v.level).toBe("stale");
    expect(v.missing).toEqual(["beautiful"]);
    // 名字必须出现在文案里：页面上要显示的就是「骨干没见过：beautiful」
    expect(v.detail).toContain("beautiful");
  });

  it("UNTRAINED_WORDS 必须减掉，否则永久挂一条假告警", () => {
    // happy/name/you_pl 在数据集里，但 train_seq.main() 会按同一份清单排掉。
    // 不减的话每次打开这页都写着「骨干没见过 happy」—— 而它本来就不该被训
    const v = backboneVerdict(
      {
        ...base,
        backboneLabels: ["_idle", "you", "eat"],
        datasetLabels: ["you", "eat", "happy", "name", "you_pl"],
      },
      ["happy", "name", "you_pl"]
    );
    expect(v.level).toBe("ok");
  });

  it("数据集比骨干新只降级成附注，不算 stale", () => {
    // 直送完还没重训骨干是常态；同一批词多录几条时迁移仍然有效。
    // 把它判成 stale 会让告警天天在、于是没人再看它
    const v = backboneVerdict(
      {
        ...base,
        trainedAt: 1000,
        datasetAt: 2000,
        backboneLabels: ["_idle", "you", "eat"],
        datasetLabels: ["you", "eat"],
      },
      []
    );
    expect(v.level).toBe("ok");
    expect(v.detail).toContain("数据集比骨干新");
  });

  it("文件不存在 → missing，文案要说出后果", () => {
    const v = backboneVerdict(
      { ...base, exists: false, backboneLabels: null, datasetLabels: ["you"] },
      []
    );
    expect(v.level).toBe("missing");
    // 「随机初始化起跑」这句是这条分支唯一的可操作信息：症状是 loss 不降而不报错
    expect(v.detail).toContain("随机初始化");
  });

  it("桥不可用 / 缺任一份标签表 → unknown，绝不冒充 ok", () => {
    expect(backboneVerdict(null, []).level).toBe("unknown");
    // 还没直送过数据集：比不出差集。这时报 ok 就是在说一句没有依据的保证
    expect(
      backboneVerdict({ ...base, backboneLabels: ["you"], datasetLabels: null }, []).level
    ).toBe("unknown");
    expect(
      backboneVerdict({ ...base, backboneLabels: null, datasetLabels: ["you"] }, []).level
    ).toBe("unknown");
  });

  it("missing 排序稳定 —— 文案不该因为数据集里的标签顺序而变", () => {
    const v = backboneVerdict(
      {
        ...base,
        backboneLabels: ["_idle"],
        datasetLabels: ["zoo", "apple", "mango"],
      },
      []
    );
    expect(v.missing).toEqual(["apple", "mango", "zoo"]);
  });
});
