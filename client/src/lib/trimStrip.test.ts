import { describe, it, expect } from "vitest";
import {
  frameToX,
  bandFor,
  stripGeometry,
  summarizeStrips,
  sortStripRows,
  uncoveredRunsOf,
  type StripRow,
} from "./trimStrip";
import type { SeqTrimSpan } from "./datasetStore";

const span = (
  startFrame: number,
  endFrame: number,
  extra: Partial<SeqTrimSpan> = {}
): SeqTrimSpan => ({
  startFrame,
  endFrame,
  applied: true,
  reason: "applied",
  keptRatio: 1,
  tactileRan: true,
  ...extra,
});

/**
 * 默认 fixture 的词边界是**占位值** `[0, totalFrames)`，和 `CollectSentence.tsx`
 * 真正写进 IndexedDB 的一模一样。
 *
 * 这一条很重要：这个文件上一版用的是编出来的边界（0-30 / 30-65 / 65-100），
 * 于是「被裁掉的词」那组断言全绿 —— 而线上那个读数恒为 0。测试用了现实中
 * 不存在的输入，就等于没测。要测真边界的路径用下面的 `wordBoundsRow()`。
 */
function row(over: Partial<StripRow> = {}): StripRow {
  return {
    id: 1,
    text: "i is_not sad",
    totalFrames: 100,
    durationMs: 2000,
    visibleRuns: [{ start: 0, end: 100 }],
    visibleFrames: 100,
    trimSpan: span(0, 100),
    segments: [
      { label: "i", startFrame: 0, endFrame: 100 },
      { label: "is_not", startFrame: 0, endFrame: 100 },
      { label: "sad", startFrame: 0, endFrame: 100 },
    ],
    multiRun: false,
    ...over,
  };
}

/** 带真实词边界的条。目前只有孤立词录制是这样；句子要等强制对齐做完 */
function wordBoundsRow(over: Partial<StripRow> = {}): StripRow {
  return row({
    segments: [
      { label: "i", startFrame: 0, endFrame: 30 },
      { label: "is_not", startFrame: 30, endFrame: 65 },
      { label: "sad", startFrame: 65, endFrame: 100 },
    ],
    ...over,
  });
}

describe("frameToX", () => {
  it("线性映射到 [0, width]", () => {
    expect(frameToX(0, 100, 600)).toBe(0);
    expect(frameToX(50, 100, 600)).toBe(300);
    expect(frameToX(100, 100, 600)).toBe(600);
  });

  it("越界的帧下标被夹住而不是画到条外面", () => {
    expect(frameToX(-5, 100, 600)).toBe(0);
    expect(frameToX(500, 100, 600)).toBe(600);
  });

  it("totalFrames 为 0 时给 0 而不是 NaN", () => {
    // NaN 进 SVG 属性会被静默丢弃，整根条不渲染且不报错
    expect(frameToX(3, 0, 600)).toBe(0);
    expect(Number.isNaN(frameToX(3, 0, 600))).toBe(false);
  });
});

describe("bandFor", () => {
  it("半开区间的宽度不含终点帧", () => {
    const b = bandFor(0, 50, 100, 600);
    expect(b.x).toBe(0);
    expect(b.w).toBe(300);
  });

  it("极短的段仍然至少 1 像素宽", () => {
    // 1 帧 / 330 帧 × 600px ≈ 1.8px，但更极端的情况会舍到 0 而彻底看不见 ——
    // 而"有一段极短可见段"恰好说明是掉手，不是全程不可见
    const b = bandFor(50, 51, 6000, 600);
    expect(b.w).toBeGreaterThanOrEqual(1);
  });

  it("零宽区间也给 1 像素（不产生看不见的矩形）", () => {
    expect(bandFor(50, 50, 100, 600).w).toBe(1);
  });
});

describe("stripGeometry", () => {
  it("可见段和保留区间各自成带", () => {
    const g = stripGeometry(row(), 600);
    expect(g.visible).toEqual([{ x: 0, w: 600, covered: true }]);
    expect(g.trim).toEqual({ x: 0, w: 600 });
  });

  it("掉手劈成两段时两段都画出来，空洞留空", () => {
    const g = stripGeometry(
      row({
        visibleRuns: [
          { start: 0, end: 30 },
          { start: 60, end: 100 },
        ],
        visibleFrames: 70,
        multiRun: true,
      }),
      600
    );
    expect(g.visible.map((v) => [v.x, v.w])).toEqual([
      [0, 180],
      [360, 240],
    ]);
  });

  it("整段落在保留区间外的可见段标成 covered:false —— 这就是那个 bug 的样子", () => {
    // 旧的 longest_run 在这条上只留最后一段，前面 30 帧的动作整段被扔掉。
    // 这是不依赖词边界的判据，所以它在真实数据上真的会亮
    const g = stripGeometry(
      row({
        visibleRuns: [
          { start: 0, end: 30 },
          { start: 60, end: 100 },
        ],
        trimSpan: span(60, 100),
        multiRun: true,
      }),
      600
    );
    expect(g.visible.map((v) => v.covered)).toEqual([false, true]);
  });

  it("部分重叠的可见段算 covered", () => {
    // 头尾各切掉几帧是裁剪本来就该做的事。标成"丢了"会把真正整段丢掉的那些淹掉
    const g = stripGeometry(row({ trimSpan: span(20, 100) }), 600);
    expect(g.visible[0].covered).toBe(true);
  });

  it("trimSpan 为 null 时 trim 是 null，可见段一律算 covered", () => {
    // null 是"导出时没传标定，没算过"，不是"全被裁了"。这时候标红会是纯噪声
    const g = stripGeometry(row({ trimSpan: null }), 600);
    expect(g.trim).toBeNull();
    expect(g.visible.every((v) => v.covered)).toBe(true);
  });

  it("词边界是占位值时 segments 给空数组并置位 segmentsArePlaceholder", () => {
    // 画出来会全部堆在最左边（每条都是 x=0、w=整条），比不画更误导
    const g = stripGeometry(row(), 600);
    expect(g.segmentsArePlaceholder).toBe(true);
    expect(g.segments).toEqual([]);
  });

  it("有真实词边界时照常成带", () => {
    const g = stripGeometry(wordBoundsRow(), 600);
    expect(g.segmentsArePlaceholder).toBe(false);
    expect(g.segments.map((s) => s.label)).toEqual(["i", "is_not", "sad"]);
    expect(g.segments[1].x).toBe(180);
  });

  it("只有一个词、且它覆盖整条时也算占位", () => {
    // 孤立词录制就长这样。一个横跨全程的 segment 不含任何边界信息
    const g = stripGeometry(
      row({ segments: [{ label: "sad", startFrame: 0, endFrame: 100 }] }),
      600
    );
    expect(g.segmentsArePlaceholder).toBe(true);
  });

  it("没有 segments 时不算占位（是「没有」，不是「填了假的」）", () => {
    const g = stripGeometry(row({ segments: [] }), 600);
    expect(g.segmentsArePlaceholder).toBe(false);
    expect(g.segments).toEqual([]);
  });
});

describe("uncoveredRunsOf", () => {
  it("挑出完全落在保留区间外的可见段", () => {
    const r = row({
      visibleRuns: [
        { start: 0, end: 20 },
        { start: 25, end: 40 },
        { start: 60, end: 100 },
      ],
      trimSpan: span(60, 100),
    });
    expect(uncoveredRunsOf(r)).toEqual([
      { start: 0, end: 20 },
      { start: 25, end: 40 },
    ]);
  });

  it("紧贴边界（end === startFrame）算在外面", () => {
    // 半开区间：[0,60) 与 [60,100) 一帧都不重叠
    const r = row({ visibleRuns: [{ start: 0, end: 60 }], trimSpan: span(60, 100) });
    expect(uncoveredRunsOf(r)).toHaveLength(1);
  });

  it("trimSpan 为 null 时一段都不算 —— 那是没算过，不是被扔了", () => {
    expect(uncoveredRunsOf(row({ trimSpan: null }))).toEqual([]);
  });
});

describe("summarizeStrips", () => {
  it("统计整段被扔的可见段 / 多段数 / reason 分布 / 平均保留比", () => {
    const rows = [
      row({ id: 1, trimSpan: span(0, 100, { keptRatio: 1 }) }),
      row({
        id: 2,
        visibleRuns: [
          { start: 0, end: 30 },
          { start: 60, end: 100 },
        ],
        trimSpan: span(60, 100, { keptRatio: 0.4 }),
        multiRun: true,
      }),
      row({ id: 3, trimSpan: span(0, 100, { keptRatio: 0.8, reason: "no_run" }) }),
    ];
    const s = summarizeStrips(rows);
    expect(s.total).toBe(3);
    expect(s.multiRun).toBe(1);
    expect(s.uncoveredRuns).toBe(1); // 第 2 条的 [0,30)
    expect(s.uncoveredFrames).toBe(30);
    expect(s.reasons).toEqual({ applied: 2, no_run: 1 });
    expect(s.meanKeptRatio).toBeCloseTo((1 + 0.4 + 0.8) / 3, 6);
  });

  it("部分溢出只计帧数、不计段数", () => {
    // 头尾修剪是正常的。它不该让"验收指标"变红
    const s = summarizeStrips([
      row({ visibleRuns: [{ start: 0, end: 100 }], trimSpan: span(10, 90) }),
    ]);
    expect(s.uncoveredRuns).toBe(0);
    expect(s.uncoveredFrames).toBe(20); // 头 10 + 尾 10
  });

  it("保留区间落在一段内部时两头都算", () => {
    const s = summarizeStrips([
      row({ visibleRuns: [{ start: 20, end: 80 }], trimSpan: span(40, 60) }),
    ]);
    expect(s.uncoveredFrames).toBe(40);
  });

  it("数占位边界的条数", () => {
    const s = summarizeStrips([row({ id: 1 }), wordBoundsRow({ id: 2 })]);
    expect(s.placeholderRows).toBe(1);
  });

  it("trimSpan 为 null 的条计进 notComputed，不参与平均", () => {
    const s = summarizeStrips([row({ id: 1, trimSpan: null }), row({ id: 2 })]);
    expect(s.notComputed).toBe(1);
    expect(s.meanKeptRatio).toBe(1); // 只算了第 2 条
  });

  it("全都没算过时平均保留比是 null 而不是 0", () => {
    // 0 会被读成"全被裁光了"，而真相是"一条都没量过"
    const s = summarizeStrips([row({ trimSpan: null })]);
    expect(s.meanKeptRatio).toBeNull();
    expect(s.uncoveredRuns).toBe(0);
    expect(s.uncoveredFrames).toBe(0);
  });

  it("空列表不炸", () => {
    expect(summarizeStrips([])).toMatchObject({
      total: 0,
      uncoveredRuns: 0,
      uncoveredFrames: 0,
      placeholderRows: 0,
      meanKeptRatio: null,
    });
  });
});

describe("sortStripRows", () => {
  it("有整段被扔的最前、多段次之、正常的最后", () => {
    const rows = [
      row({ id: 10 }),
      row({ id: 11, multiRun: true }),
      row({
        id: 12,
        visibleRuns: [
          { start: 0, end: 30 },
          { start: 60, end: 100 },
        ],
        trimSpan: span(60, 100),
      }),
    ];
    expect(sortStripRows(rows).map((r) => r.id)).toEqual([12, 11, 10]);
  });

  it("同级内按 id 稳定（每次刷新次序不变，才能逐条比对）", () => {
    const rows = [row({ id: 7 }), row({ id: 3 }), row({ id: 5 })];
    expect(sortStripRows(rows).map((r) => r.id)).toEqual([3, 5, 7]);
  });

  it("不修改入参数组", () => {
    const rows = [row({ id: 9 }), row({ id: 1 })];
    sortStripRows(rows);
    expect(rows.map((r) => r.id)).toEqual([9, 1]);
  });
});
