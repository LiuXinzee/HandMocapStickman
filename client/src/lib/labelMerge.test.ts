/*
 * labelMerge.test —— 合并特征层不可分的类
 *
 * 合并会永久改变模型能输出什么，所以这里锁的是那几条"错了会静默出事"的判据：
 *  1. **不许改传进来的样本** —— 库里的原始标签是换九轴 IMU 后退回去的唯一依据；
 *  2. 关掉时必须是**完全的旁路**，不能悄悄改一点点（那是对照实验的基线）；
 *  3. 合并 id 不能和词表里的真实词撞车（撞了就会把一个真词的样本吞进合并类）；
 *  4. 显示名要在 `getDisplayLabel` 这一层解决 —— 漏一处界面就会露出原始 id；
 *  5. 减少的类别数按**库里真有的成员**算（只有 1 个成员时合并不减类）。
 */
import { describe, expect, it } from "vitest";
import {
  MERGE_GROUPS,
  mergeLabel,
  mergeSamples,
  mergeGroupOf,
  isMergedLabel,
  getMergeGroup,
  classesRemovedBy,
} from "./labelMerge";
import { SIGN_VOCABULARY, getDisplayLabel, IDLE_LABEL } from "./signLanguageVocab";

const rows = (...labels: string[]) =>
  labels.map((primaryLabel, i) => ({ primaryLabel, id: i }));

describe("合并组的定义", () => {
  it("单数只合 你/他，复数三代词合成另一类", () => {
    const sg = MERGE_GROUPS.find((g) => g.members.includes("you"))!;
    const pl = MERGE_GROUPS.find((g) => g.members.includes("we"))!;
    expect(sg.members.sort()).toEqual(["he", "you"]);
    expect(pl.members.sort()).toEqual(["they", "we", "you_pl"]);
    expect(sg.id).not.toBe(pl.id);
  });

  it("「我」不在任何合并组里 —— 它指自己胸口、有接触，分得开", () => {
    // 这条是整次改动的核心断言。「我」要是又被并回去，「我爱你」和「你爱我」
    // 会重新塌成同一个序列，而症状只是"方向偶尔反"，极难定位
    for (const g of MERGE_GROUPS) expect(g.members).not.toContain("i");
    expect(mergeLabel("i")).toBe("i");
  });

  it("每组的默认成员必须是自己的成员，且显式声明（不许按下标取）", () => {
    // 按 members[0] 取默认值的写法在成员表变动时会静默挪位 ——
    // 把「我」拆出去那次就是这么踩的：类型过、测试大半也过，只有翻译悄悄变了
    for (const g of MERGE_GROUPS) {
      expect(g.members).toContain(g.defaultMember);
      if (g.selfMember) expect(g.members).toContain(g.selfMember);
    }
  });

  it("单数组没有 self 成员 —— 模型说朝外指，那就不该翻译成「我」", () => {
    const sg = MERGE_GROUPS.find((g) => g.id === "merged_pron_sg")!;
    expect(sg.selfMember).toBeUndefined();
    expect(sg.defaultMember).toBe("you");
  });

  it("合并 id 不和词表里任何真实词撞车", () => {
    // 撞了的话，那个真词的样本会被静默吞进合并类
    const real = new Set(SIGN_VOCABULARY.map((w) => w.id));
    for (const g of MERGE_GROUPS) {
      expect(real.has(g.id)).toBe(false);
      expect(g.id).not.toBe(IDLE_LABEL);
    }
  });

  it("每个组都写了合并理由（日后别人才不会当 bug 修掉）", () => {
    for (const g of MERGE_GROUPS) {
      expect(g.reason.length).toBeGreaterThan(10);
      expect(g.display.length).toBeGreaterThan(0);
    }
  });

  it("一个原始标签只属于一个组", () => {
    const seen = new Set<string>();
    for (const g of MERGE_GROUPS)
      for (const m of g.members) {
        expect(seen.has(m)).toBe(false);
        seen.add(m);
      }
  });
});

describe("mergeLabel", () => {
  it("你/他 映射到同一个类", () => {
    expect(mergeLabel("you")).toBe(mergeLabel("he"));
  });

  it("单数和复数是两个不同的类（有没有那道弧线是分得开的）", () => {
    expect(mergeLabel("you")).not.toBe(mergeLabel("we"));
  });

  it("不参与合并的词原样返回", () => {
    expect(mergeLabel("hello")).toBe("hello");
    expect(mergeLabel(IDLE_LABEL)).toBe(IDLE_LABEL);
  });

  it("关掉时原样返回（这是对照实验的基线，不能有任何偏差）", () => {
    expect(mergeLabel("you", false)).toBe("you");
    expect(mergeLabel("we", false)).toBe("we");
  });
});

describe("mergeSamples", () => {
  it("不改传进来的样本 —— 原始标签是换硬件后退回去的唯一依据", () => {
    const input = rows("i", "you", "hello");
    const snapshot = input.map((s) => s.primaryLabel);
    mergeSamples(input);
    expect(input.map((s) => s.primaryLabel)).toEqual(snapshot);
  });

  it("五个代词塌成两类，「我」和其余词不动", () => {
    const r = mergeSamples(rows("i", "you", "he", "we", "you_pl", "they", "hello"));
    const labels = new Set(r.samples.map((s) => s.primaryLabel));
    expect(labels.size).toBe(4); // 单数合并、复数合并、i、hello
    expect(labels.has("hello")).toBe(true);
    expect(labels.has("i")).toBe(true); // 「我」原样穿过
    expect(r.merged).toBe(5);
    expect(r.groups).toHaveLength(2);
  });

  it("只报**命中**的组，没出现的成员不算", () => {
    const r = mergeSamples(rows("you", "he", "hello"));
    expect(r.groups).toHaveLength(1);
    expect(r.groups[0].members).toContain("you");
  });

  it("一条都没命中时 merged=0、groups 为空", () => {
    const r = mergeSamples(rows("hello", "eat", IDLE_LABEL));
    expect(r.merged).toBe(0);
    expect(r.groups).toEqual([]);
  });

  it("关掉时原样返回同一个数组引用（完全旁路）", () => {
    const input = rows("you", "he");
    const r = mergeSamples(input, false);
    expect(r.samples).toBe(input);
    expect(r.merged).toBe(0);
  });

  it("保留样本上的其他字段（只动 primaryLabel）", () => {
    const r = mergeSamples([{ primaryLabel: "you", frameCount: 42, keep: "x" }]);
    expect(r.samples[0].frameCount).toBe(42);
    expect(r.samples[0].keep).toBe("x");
  });

  it("空数组不崩", () => {
    expect(mergeSamples([]).samples).toEqual([]);
  });
});

describe("显示名", () => {
  it("合并类显示成摊开的候选，不是原始 id", () => {
    // 模型确实只知道"这是个单数指向"，如实摊开比编一个用户不认识的词诚实
    const id = mergeLabel("you");
    expect(getDisplayLabel(id)).toBe("你/他");
    expect(getDisplayLabel(id)).not.toContain("merged_");
  });

  it("复数组同理", () => {
    expect(getDisplayLabel(mergeLabel("we"))).toBe("我们/你们/他们");
  });

  it("没破坏普通词和 `_idle` 的显示名", () => {
    expect(getDisplayLabel("hello")).toBe("你好");
    expect(getDisplayLabel(IDLE_LABEL)).toBe("—");
  });
});

describe("查询辅助", () => {
  it("isMergedLabel 只对合并出来的 id 为真", () => {
    expect(isMergedLabel(mergeLabel("you"))).toBe(true);
    expect(isMergedLabel("you")).toBe(false);
    expect(isMergedLabel("i")).toBe(false); // 「我」是普通词，不是合并类
    expect(isMergedLabel("hello")).toBe(false);
  });

  it("mergeGroupOf 用原始标签查得到组、用合并 id 查不到", () => {
    expect(mergeGroupOf("he")?.members).toContain("you");
    expect(mergeGroupOf("i")).toBeUndefined(); // 「我」不参与合并
    expect(mergeGroupOf("hello")).toBeUndefined();
  });

  it("getMergeGroup 用合并 id 查得到", () => {
    expect(getMergeGroup(mergeLabel("we"))?.display).toBe("我们/你们/他们");
  });
});

describe("classesRemovedBy", () => {
  it("六个代词都在 → 少 3 个类（你/他 2→1，复数 3→1）", () => {
    // 「我」不再参与合并，所以比以前少减一个类
    const present = ["i", "you", "he", "we", "you_pl", "they", "hello"];
    expect(classesRemovedBy(present)).toBe(3);
  });

  it("只有 1 个成员在库里 → 不减类（合并一个类没有意义）", () => {
    expect(classesRemovedBy(["you", "hello"])).toBe(0);
  });

  it("「我」单独在库里 → 不减类，它本来就不在任何组里", () => {
    expect(classesRemovedBy(["i", "hello"])).toBe(0);
  });

  it("2 个成员 → 少 1 个类", () => {
    expect(classesRemovedBy(["you", "he", "hello"])).toBe(1);
  });

  it("关掉时恒为 0", () => {
    expect(classesRemovedBy(["you", "he"], false)).toBe(0);
  });
});
