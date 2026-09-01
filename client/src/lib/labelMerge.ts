/*
 * labelMerge —— 把在特征层不可分的类别合并成一个
 *
 * ===== 判据是"差异落在哪个通道上"，不是"都是代词" =====
 *
 * 一开始 我/你/他 三个是合在一起的，理由写的是"同手型指不同方向，纯 yaw 之差"。
 * **那句话对 你/他 成立，对 我 不成立**，现在已经拆开了：
 *
 *   你 = 食指指向对方        ┐ 两个都是朝身体外指，差一个水平角
 *   他 = 食指指向侧前方第三者 ┘ → 绕重力轴的旋转 = 纯 yaw → 合并
 *   我 = 食指指向**自己胸口** → 手收回身体、指尖接触胸部 → 分得开，已拆出
 *
 * 为什么「我」分得开，有两条独立的观测：
 *  1. **接触**。指尖/掌心真的碰到身体，60 路指压 + 72 路掌压里出现一个压力事件，
 *     而"指向空气"永远产生不出来。这条完全不依赖朝向。
 *  2. **pitch / roll**。六轴 IMU 丢的只有 yaw 一个自由度 —— 重力向量把 pitch 和
 *     roll 都钉死了。把手折回胸前和向前平举，这两个角本身就不同。
 * （采集时必须选**有身体接触**的那个打法。「我」在中国手语里有两种打法，选了
 *  "指向自己但不接触"的那种，就只剩第 2 条弱线索，大概率白录。）
 *
 * 而 你 vs 他 只差 yaw，这条路是实测判死的（yawDrift 探针，399 条）：
 *   静止时 yaw 漂移 P90 15.92°/s → 归零 10s 后累计 159°，而类间距只有 45°
 *   归零点本身 P90 也有 43.4°，已经接近类间距
 * 手套是 ICM-42688 六轴无磁力计，绝对 yaw 零点在硬件层就不存在，软件绕不过去。
 *
 * 所以合并不是"先凑合一下"，是**当前硬件下的正确建模**：两个类抢同一块特征
 * 空间时，softmax 只能按训练集比例随机分配，不但这两个词错，逃逸的概率还会
 * 污染邻近词。合并后这块空间归一个类，其余各类的边界反而更干净。
 *
 * ===== 拆开「我」换来了什么 =====
 *
 * 「我爱你」和「你爱我」以前解出来是**同一个序列** [合并类, 爱, 合并类]，方向完全
 * 靠 sentenceGrammar.ts 的规则表猜。现在是 [i, 爱, 合并类] 和 [合并类, 爱, i] ——
 * 动作方向变成量到的。同理 我帮你/你帮我。
 *
 * ===== 复数组为什么还留着「我们」 =====
 *
 * 按上面那条判据，「我们」（指自己 + 一道横向弧线）也该拆出去 —— 它同样有接触。
 * 但复数三个词**一条数据都没有**，更没有按接触变体重录过，现在拆等于凭空断言一个
 * 没验证过的可分性。等真采了再按同一条规则拆。这个不对称是**有意的**，别顺手改齐。
 *
 * ===== 三个设计选择 =====
 *
 * 1) **训练时重映射，不改库**。和按词排除同一个位置（sequenceModel.ts 的类别表
 *    是从传进去的样本现推的）。将来换九轴 IMU，原始标签还在，退得回去。
 *
 * 2) **显示成「你/他」而不是「指向(单)」**。模型确实只知道"这是个朝外的指向"，
 *    把候选如实摊开比编一个用户不认识的词诚实，也比在两个里随机挑一个有用。
 *
 * 3) **默认开，但可关**。关掉是有意义的对照实验（想看合并到底帮了多少），
 *    所以留开关；默认开是因为实测数据已经把 你/他 不合并这条路判死了。
 */

/** 一个合并组：把 `members` 里的原始标签全部映射到 `id` */
export interface MergeGroup {
  /** 合并后的类别 id。带 `merged_` 前缀，避免和词表里的真实词 id 撞车 */
  id: string;
  /** 显示名。摊开候选而不是编新词 —— 模型知道的就是"这几个之一" */
  display: string;
  /** 被合并的原始标签 */
  members: string[];
  /**
   * 顺句时这一组默认翻译成哪个成员（原始词 id）。
   *
   * **必须显式写出来，不能用 `members[0]`。** 以前 sentenceGrammar 就是按下标取的
   * （`opts[0]` / `opts[1]`），结果从组里删掉「我」的那一刻，"默认值"跟着整体挪了
   * 一位、而且不报错 —— 类型过、测试大半也过，只有翻译结果悄悄变了。
   */
  defaultMember: string;
  /**
   * 组里那个"指说话人自己"的成员（没有就不填）。
   *
   * 规则表判断出这个代词位说的是自己时取它（见 sentenceGrammar 的 PronounSlot）。
   * 单数组**没有** —— 「我」已经拆成独立类了，所以单数合并类无论如何都不该
   * 翻译成「我」：模型说"这是个朝外的指向"，那它就不是我。
   */
  selfMember?: string;
  /** 为什么合并 —— 报告和界面上都要能看到，免得日后有人当 bug 修掉 */
  reason: string;
}

const YAW_REASON =
  "同手型、区别纯在指向(yaw)；六轴 IMU 无磁力计，绝对 yaw 不可观测" +
  "（实测漂移 P90 15.9°/s，10s 累计 159°，类间距仅 45°）";

export const MERGE_GROUPS: MergeGroup[] = [
  {
    id: "merged_pron_sg",
    display: "你/他",
    // 「我」不在这里 —— 它指自己胸口、有接触，分得开，是独立类（见文件头）
    members: ["you", "he"],
    defaultMember: "you",
    reason: YAW_REASON,
  },
  {
    id: "merged_pron_pl",
    display: "我们/你们/他们",
    // 复数是"同手型 + 一段横向弧线"，那段弧线在手系里也是同一个旋转，一样简并。
    // 「我们」按单数那条判据本该拆出去，但复数一条数据都没有 —— 见文件头，
    // 这个不对称是有意的
    members: ["we", "you_pl", "they"],
    // 原来是 `you_pl` —— 和单数组取第二人称是对称的，但 `you_pl` 在
    // `UNTRAINED_WORDS` 里（sentenceTemplates.ts），也就是说这一组的默认翻译
    // 指向一个**模型从来没训过的词**。取 `they`：`we` 是 selfMember，规则表判出
    // "说的是自己"时已经会取它，再拿它当默认值就等于 selfMember 永远不起作用
    defaultMember: "they",
    selfMember: "we",
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
