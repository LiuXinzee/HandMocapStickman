/*
 * labelMerge —— 把在特征层不可分的类别合并成一个
 *
 * 为什么要有这个：我/你/他 是同一手型指不同方向，区别几乎纯粹在 yaw。而学生
 * 特征里两条 IMU 通路都对 yaw 不变（sequenceFeatures.ts 开头：相对首帧四元数
 * 消掉绝对朝向、重力投影天生 yaw 不变），所以这三个词在特征空间里是同一个点。
 *
 * 本来还有一条出路 —— 给 yaw 建参考系。实测判死了（yawDrift 探针，399 条）：
 *   静止时 yaw 漂移 P90 15.92°/s → 归零 10s 后累计 159°，而类间距只有 45°
 *   归零点本身 P90 也有 43.4°，已经接近类间距
 * 手套是 ICM-42688 六轴无磁力计，绝对 yaw 零点在硬件层就不存在，软件绕不过去。
 *
 * 所以合并不是"先凑合一下"，是**当前硬件下的正确建模**：三个类抢同一块特征
 * 空间时，softmax 只能按训练集比例随机分配，不但这三个词错，逃逸的概率还会
 * 污染邻近词。合并后这块空间归一个类，剩下 24 类各自的边界反而更干净。
 *
 * 三个设计选择：
 *
 * 1) **训练时重映射，不改库**。和按词排除同一个位置（sequenceModel.ts:475 的
 *    类别表是从传进去的样本现推的）。将来换九轴 IMU，原始标签还在，退得回去。
 *
 * 2) **显示成「我/你/他」而不是「指向(单)」**。模型确实只知道"这是个单数指向"，
 *    把三个候选如实摊开比编一个用户不认识的词诚实，也比在三个里随机挑一个有用。
 *
 * 3) **默认开，但可关**。关掉是有意义的对照实验（想看合并到底帮了多少），
 *    所以留开关；默认开是因为实测数据已经把不合并这条路判死了。
 */

/** 一个合并组：把 `members` 里的原始标签全部映射到 `id` */
export interface MergeGroup {
  /** 合并后的类别 id。带 `merged_` 前缀，避免和词表里的真实词 id 撞车 */
  id: string;
  /** 显示名。摊开候选而不是编新词 —— 模型知道的就是"这几个之一" */
  display: string;
  /** 被合并的原始标签 */
  members: string[];
  /** 为什么合并 —— 报告和界面上都要能看到，免得日后有人当 bug 修掉 */
  reason: string;
}

const YAW_REASON =
  "同手型、区别纯在指向(yaw)；六轴 IMU 无磁力计，绝对 yaw 不可观测" +
  "（实测漂移 P90 15.9°/s，10s 累计 159°，类间距仅 45°）";

export const MERGE_GROUPS: MergeGroup[] = [
  {
    id: "merged_pron_sg",
    display: "我/你/他",
    members: ["i", "you", "he"],
    reason: YAW_REASON,
  },
  {
    id: "merged_pron_pl",
    display: "我们/你们/他们",
    // 复数是"同手型 + 一段横向弧线"，那段弧线在手系里也是同一个旋转，一样简并
    members: ["we", "you_pl", "they"],
    reason: YAW_REASON,
  },
];

/** 原始标签 → 合并组 id。模块级建一次，训练时每条样本都要查 */
const MEMBER_TO_GROUP = new Map<string, string>();
for (const g of MERGE_GROUPS) {
  for (const m of g.members) MEMBER_TO_GROUP.set(m, g.id);
}

const GROUP_BY_ID = new Map(MERGE_GROUPS.map((g) => [g.id, g]));

/** 这个 id 是不是合并出来的类 */
export function isMergedLabel(id: string): boolean {
  return GROUP_BY_ID.has(id);
}

export function getMergeGroup(id: string): MergeGroup | undefined {
  return GROUP_BY_ID.get(id);
}

/** 某个原始标签会被合并到哪个组；不参与合并时返回 undefined */
export function mergeGroupOf(label: string): MergeGroup | undefined {
  const id = MEMBER_TO_GROUP.get(label);
  return id ? GROUP_BY_ID.get(id) : undefined;
}

/**
 * 把一个标签映射到训练用的类别名。
 *
 * `enabled=false` 时原样返回 —— 关掉合并是有意义的对照实验，不该走另一条代码路径
 */
export function mergeLabel(label: string, enabled = true): string {
  if (!enabled) return label;
  return MEMBER_TO_GROUP.get(label) ?? label;
}

/**
 * 合并一批样本的标签。**返回新对象，不改传进来的样本** —— 库里的原始标签
 * 必须留着，否则换了硬件想退回去就只能重采
 */
export function mergeSamples<T extends { primaryLabel: string }>(
  samples: T[],
  enabled = true
): { samples: T[]; merged: number; groups: MergeGroup[] } {
  if (!enabled) return { samples, merged: 0, groups: [] };

  const hit = new Set<string>();
  let merged = 0;
  const out = samples.map((s) => {
    const g = MEMBER_TO_GROUP.get(s.primaryLabel);
    if (!g) return s;
    hit.add(g);
    merged++;
    // segments 里的 label 不动：它是逐段的原始标注，重映射只发生在类别表这一层
    return { ...s, primaryLabel: g };
  });

  return {
    samples: out,
    merged,
    groups: MERGE_GROUPS.filter((g) => hit.has(g.id)),
  };
}

/** 合并后类别数会少多少（只算库里真有的成员，缺的不算） */
export function classesRemovedBy(
  present: Iterable<string>,
  enabled = true
): number {
  if (!enabled) return 0;
  const set = new Set(present);
  let removed = 0;
  for (const g of MERGE_GROUPS) {
    const n = g.members.filter((m) => set.has(m)).length;
    // n 个类变成 1 个，少 n-1 个；库里只有 1 个成员时合并不减类
    if (n > 1) removed += n - 1;
  }
  return removed;
}
