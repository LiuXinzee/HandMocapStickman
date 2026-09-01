/*
 * 手语词汇库 — 中国手语常用词汇
 * 初始 31 个，2026-08-14 追加 14 个（代词 6、判断 2、动作 4、名词/疑问 2），
 * 2026-08-29 追加 4 个（好看/笑/像/太阳，为三句新句型），共 49 个
 *
 * ===== 2026-08-30：11 条 description 由采集者本人更正 =====
 *
 * 更正的词：i / love / hello / sad / know / listen / speak / beautiful / sun /
 * is / is_not。原描述是按"标准中国手语常见打法"写的，与采集者实际打的手势不符。
 * 这些 description 是采集页给人看的提示语，写错的代价是照它录一批错手势 ——
 * 而且分析数据时如果拿它当真值，会得出完全跑偏的结论（已经发生过一次）。
 *
 * ⚠ **库里已有的录制不一定符合更正后的描述。** 这两件事是分开的：
 * description 改对了 ≠ 数据跟着变了。已确认对不上的一例：`hello` 的 15 条录制
 * 全在 2026-08-13，按弯折特征分段核对，它的后半段离「竖起大拇指」（`is` 的手型）
 * 越来越远（全段 2.25 → 后 40% 3.07 → 后 25% 2.98），也就是**那批录制里没有
 * "然后比划大拇指"这一段**，还是旧口径的「食指中指向前点头」。同批还有
 * angry / goodbye / happy / help / sorry / study / thank_you / welcome / work / sad
 * 在同一个嫌疑里（都是 08-13 那批）。这些词要真正修好只能重录。
 *
 * `sad` 查过了，**数据是对的**（2026-08-30）。曾经怀疑那 15 条不是握拳，依据是
 * 拇指压力接近 0（生气 5.1，难过 0.1）—— 但采集者说明难过是**虚握**、不是紧握，
 * 虚握本来就不该有拇指压力。同理"拇指相对其它四指 −91"也是虚握该有的样子
 * （拇指顺着食指搭着、比蜷起的四指直）。把 sad 从嫌疑名单里撤掉。
 *
 * 这一段留着是因为**它记录了一个容易重犯的推理错误**：拿"标准打法"去解释传感器
 * 读数，读数不符就断定数据错了。正确的顺序是先问采集者实际怎么打。
 * 顺带一个实测事实：这批 sad 录制**完全可学** —— 模型在模拟实时滑窗下 170/170
 * 全对、置信度 0.994。实测互认是 15 条样本的泛化缺口，不是数据错。
 *
 * 谢谢 / 难过 的实时互认改用推理时的物理量闸门解决，见 `fistGate.ts`。
 *
 * 仍未经采集者确认的只剩 `resemble`（下面 2026-08-29 那段的警告对它依然有效）。
 * `smile` 已于 2026-08-30 更正并确认：**单手**，不是原来写的双手。
 *
 * dynamic 字段区分静态手型词与带轨迹的动态词：
 * - 静态词（数字、"停"、"家"…）单帧就能判别
 * - 动态词（"再见"左右摆动、"来"招手…）必须看整段序列
 * 序列模型统一处理两者（静态词 = 帧间无变化的序列），dynamic 只用于
 * 采集页分组显示和提示用户该录多久。
 */

import { getMergeGroup } from "./labelMerge";

export interface SignWord {
  id: string;
  label: string;        // 中文词汇
  pinyin: string;       // 拼音
  category: string;     // 分类
  description: string;  // 手势描述
  /** 是否为带轨迹的动态词（需要时序模型才能区分） */
  dynamic?: boolean;
}

/**
 * 空闲/过渡伪类标签。
 *
 * 滑窗推理下模型对任意窗口都会强行输出某个词，没有这一类的话，手放松、
 * 动作过渡、手移动到起始位置时会持续乱吐词。采集时需要专门录这一类
 * （建议样本数为单词类的 2~3 倍），推理时命中它就丢弃、不进翻译历史。
 */
export const IDLE_LABEL = "_idle";

/** 空闲类的展示名（不在 SIGN_VOCABULARY 中，getWordById 查不到） */
export const IDLE_DISPLAY_LABEL = "—";

export const SIGN_CATEGORIES = [
  { id: "greeting", label: "问候", color: "#00f0ff" },
  { id: "number", label: "数字", color: "#00e5a0" },
  { id: "daily", label: "日常", color: "#f59e0b" },
  { id: "emotion", label: "情感", color: "#ff2d7b" },
  { id: "action", label: "动作", color: "#a855f7" },
] as const;

export const SIGN_VOCABULARY: SignWord[] = [
  // 问候类
  // ⚠ 08-13 那 15 条录制不含"竖大拇指"这一段，与本描述不符（见文件头）
  { id: "hello", label: "你好", pinyin: "nǐ hǎo", category: "greeting", description: "食指先向前指，然后比划大拇指（两段）", dynamic: true },
  // 2026-08-30 采集者更正：原写「右手手心向下，从嘴边向前伸出」，是完全另一个手势。
  // 更正后它与 hello / beautiful 的第二段**同型**（都是竖大拇指）—— 这不是巧合，
  // 是 compoundWords.ts 那两条规则存在的根本原因
  { id: "thank_you", label: "谢谢", pinyin: "xiè xiè", category: "greeting", description: "竖起大拇指，大拇指向下压一次", dynamic: true },
  { id: "sorry", label: "对不起", pinyin: "duì bù qǐ", category: "greeting", description: "右手握拳放在胸前，轻轻拍打", dynamic: true },
  { id: "goodbye", label: "再见", pinyin: "zài jiàn", category: "greeting", description: "右手掌心向外，左右摆动", dynamic: true },
  { id: "please", label: "请", pinyin: "qǐng", category: "greeting", description: "右手掌心向上，向前伸出", dynamic: true },
  { id: "welcome", label: "欢迎", pinyin: "huān yíng", category: "greeting", description: "双手掌心向上，向外展开", dynamic: true },

  // 数字类（全部为静态手型）
  { id: "num_0", label: "零", pinyin: "líng", category: "number", description: "拇指和食指圈成O形" },
  { id: "num_1", label: "一", pinyin: "yī", category: "number", description: "伸出食指" },
  { id: "num_2", label: "二", pinyin: "èr", category: "number", description: "伸出食指和中指" },
  { id: "num_3", label: "三", pinyin: "sān", category: "number", description: "伸出食指、中指、无名指" },
  { id: "num_4", label: "四", pinyin: "sì", category: "number", description: "伸出四指，拇指弯曲" },
  { id: "num_5", label: "五", pinyin: "wǔ", category: "number", description: "五指张开" },
  { id: "num_6", label: "六", pinyin: "liù", category: "number", description: "伸出拇指和小指" },
  { id: "num_7", label: "七", pinyin: "qī", category: "number", description: "拇指、食指、中指捏在一起" },
  { id: "num_8", label: "八", pinyin: "bā", category: "number", description: "拇指和食指伸开成L形" },
  { id: "num_9", label: "九", pinyin: "jiǔ", category: "number", description: "食指弯曲成钩状" },
  { id: "num_10", label: "十", pinyin: "shí", category: "number", description: "食指交叉成十字" },

  // 日常类
  { id: "eat", label: "吃", pinyin: "chī", category: "daily", description: "手指捏在一起送向嘴边", dynamic: true },
  { id: "drink", label: "喝", pinyin: "hē", category: "daily", description: "拇指和小指伸出，送向嘴边", dynamic: true },
  { id: "sleep", label: "睡觉", pinyin: "shuì jiào", category: "daily", description: "手掌贴在脸侧，头微倾" },
  { id: "home", label: "家", pinyin: "jiā", category: "daily", description: "双手指尖相触成屋顶状" },
  { id: "work", label: "工作", pinyin: "gōng zuò", category: "daily", description: "双手握拳交替上下运动", dynamic: true },
  { id: "study", label: "学习", pinyin: "xué xí", category: "daily", description: "一手做翻书动作", dynamic: true },

  // 情感类
  { id: "happy", label: "高兴", pinyin: "gāo xìng", category: "emotion", description: "双手在脸旁向上展开", dynamic: true },
  // 「虚握」两个字是要紧的：紧握会让拇指吃力，而拇指有没有吃力正是 fistGate.ts
  // 用来把它和「谢谢」分开的判据。照"紧握"录会把那条闸门废掉
  { id: "sad", label: "难过", pinyin: "nán guò", category: "emotion", description: "一只手虚握成拳（不用紧握），拳心朝内在胸口划一圈", dynamic: true },
  { id: "love", label: "爱", pinyin: "ài", category: "emotion", description: "一只手比划大拇指，另一只手抚摸这个大拇指", dynamic: true },
  { id: "angry", label: "生气", pinyin: "shēng qì", category: "emotion", description: "双手握拳向上举", dynamic: true },

  // 动作类
  { id: "go", label: "去", pinyin: "qù", category: "action", description: "食指向前指", dynamic: true },
  { id: "come", label: "来", pinyin: "lái", category: "action", description: "手掌向内招手", dynamic: true },
  { id: "help", label: "帮助", pinyin: "bāng zhù", category: "action", description: "一手托起另一手", dynamic: true },
  { id: "stop", label: "停", pinyin: "tíng", category: "action", description: "手掌向前伸出" },
  { id: "see", label: "看", pinyin: "kàn", category: "action", description: "食指中指分开成 V 形，从眼睛旁向前伸出", dynamic: true },
  { id: "listen", label: "听", pinyin: "tīng", category: "action", description: "一只手手指并拢、微微弯曲，掌心朝内放在耳后" },
  { id: "speak", label: "说", pinyin: "shuō", category: "action", description: "食指和中指并拢伸直，在嘴边转动两圈", dynamic: true },
  { id: "know", label: "认识", pinyin: "rèn shi", category: "action", description: "两只手都伸出食指和中指，两只手相对", dynamic: true },

  /*
   * 代词与判断词（2026-08-14 追加）。
   *
   * 分类**沿用现有的语义五类**——现有分类是语义轴（问候/数字/日常/情感/动作），
   * 代词、疑问词是词性轴，两者不是同一把尺子，硬加词性分类会让筛选结果互相重叠。
   * 所以这里一律落在最接近的「日常」，等哪天筛选真要换成词性轴时一起改。
   *
   * 单复数是**同一个手型 + 一段横向弧线**的区别（我→我们、你→你们、他→他们），
   * 静态单帧模型区分不了这三对，必须走时序模型 —— 所以复数三个 dynamic: true，
   * 单数三个是纯指向、静态。这也是为什么这批词最好在 /collect-seq 采。
   */
  { id: "i", label: "我", pinyin: "wǒ", category: "daily", description: "五指指向自己胸口" },
  { id: "you", label: "你", pinyin: "nǐ", category: "daily", description: "食指指向对方" },
  { id: "he", label: "他", pinyin: "tā", category: "daily", description: "食指指向侧前方的第三者" },
  { id: "we", label: "我们", pinyin: "wǒ men", category: "daily", description: "食指指自己，再横向划一道弧（复数）", dynamic: true },
  { id: "you_pl", label: "你们", pinyin: "nǐ men", category: "daily", description: "食指指对方，再横向划一道弧（复数）", dynamic: true },
  { id: "they", label: "他们", pinyin: "tā men", category: "daily", description: "食指指侧前方，再横向划一道弧（复数）", dynamic: true },
  /*
   * 是/不是 是**同一个手型**（食指伸直、中指缠绕食指），只差运动方向：
   * 「是」从后往前点一下，「不是」左右摇晃。
   *
   * 后果是这一对在弯折通道上天然不可分（实测 Fisher 0.36），全部判别信息都在运动里。
   * 而这副手套的包 2 是 144B 的旧版，**没有加速度计和陀螺仪**（见 gloveProtocol.ts
   * 的帧格式说明，加速度只在 296B 帧里才有），运动信息只有 IMU 四元数一条通道。
   * 好消息是这条通道足够：实测「是」的首尾净旋转 |1.6|（点一下不回原点），
   * 「不是」|0.28|（摇回中心），差 6 倍。前提是四元数的参考帧正确 ——
   * 而参考帧取的是裁剪后第一帧，裁剪落点一晃这两个词就一起废掉。
   */
  { id: "is", label: "是", pinyin: "shì", category: "daily", description: "食指伸直、中指缠绕食指，从后往前轻轻点一下", dynamic: true },
  { id: "is_not", label: "不是", pinyin: "bú shì", category: "daily", description: "食指伸直、中指缠绕食指，左右摇晃", dynamic: true },
  { id: "name", label: "名字", pinyin: "míng zi", category: "daily", description: "右手食指在左手掌心做写字状", dynamic: true },
  { id: "what", label: "什么", pinyin: "shén me", category: "daily", description: "手掌向上摊开，左右轻摆", dynamic: true },

  /*
   * 2026-08-29 追加。为三句新句型：
   *   你叫什么名字   ["you","name","what"]        —— 词都是现成的，只是把 name 训回来
   *   你真好看       ["you","beautiful"]
   *   你笑起来像太阳 ["you","smile","resemble","sun"]
   *
   * ⚠ 这四条 description 原本是按标准中国手语常见打法写的，不是从词典抄的。
   * 2026-08-30 采集者确认了其中两条，另两条**仍未确认**：
   *   `beautiful`（好看）  ✅ 已更正 —— 实际是"食指中指划过鼻子，再竖大拇指"，
   *                        原来写的"脸前绕圈"是错的
   *   `sun`（太阳）        ✅ 已更正 —— 实际是"捏圆从右到左滑动"，
   *                        既不举过头顶也没有"张开表光芒"
   *   `smile`（笑）        ✅ 已更正 —— 实际是**单手**、拇指与食指张开搭在两个嘴角，
 *                        原来写的"双手食指"手数和手型都错
   *   `resemble`（像）     ❓ 未确认。很多地方直接借用「一样/相同」的手势
   *                        （双手食指并拢），没有独立的「像」—— 要是实际打的是
   *                        「一样」，那这个词更该叫 `same`，句型也要跟着改
   * 选定哪种打法都行，**关键是 20 条录制内部一致** —— 同一类里混两种打法，
   * 症状是那个词的准确率莫名偏低（`name` 就是这么废掉的，见 sentenceTemplates.ts）。
   *
   * 分类沿用上面那条约定（语义五类，词性不另开一轴）：`smile` 落「情感」——
   * 它就是「高兴」的外显；`beautiful`/`resemble`/`sun` 三个在五类里都没有对应的，
   * 一律落兜底的「日常」。
   */
  // 好看 与 你好 都是「伸指手势 → 竖大拇指」的两段结构 —— 结构相同是这两个词
  // 容易互相混淆的直接原因，不需要靠特征分析就能看出来
  { id: "beautiful", label: "好看", pinyin: "hǎo kàn", category: "daily", description: "食指和中指伸出，沿鼻子从眉心划向鼻尖，然后伸出大拇指（两段）", dynamic: true },
  // 2026-08-30 采集者更正：原写「双手食指分别在两侧嘴角」，手数和手型都是错的。
  // 实际是**单手**、拇指与食指张开（≈ `num_8` 的 L 形手型）分别搭在两个嘴角。
  // 更正与数据一致：15 条录制的左手活动量 0.039、右手 0.113，本来就是单手录的。
  { id: "smile", label: "笑", pinyin: "xiào", category: "emotion", description: "单手，食指和大拇指张开，分别放在两侧嘴角，向上翘动", dynamic: true },
  { id: "resemble", label: "像", pinyin: "xiàng", category: "daily", description: "双手食指伸出、平行向前，由分开到并拢相靠（表示两者相似）", dynamic: true }, // ⚠ 未经采集者确认
  { id: "sun", label: "太阳", pinyin: "tài yáng", category: "daily", description: "大拇指和食指捏成圆形，从右向左滑动", dynamic: true },
];

export function getWordById(id: string): SignWord | undefined {
  return SIGN_VOCABULARY.find((w) => w.id === id);
}

export function getWordsByCategory(category: string): SignWord[] {
  return SIGN_VOCABULARY.filter((w) => w.category === category);
}

export function getCategoryColor(category: string): string {
  return SIGN_CATEGORIES.find((c) => c.id === category)?.color ?? "#556677";
}

/** 带轨迹的动态词 */
export function getDynamicWords(): SignWord[] {
  return SIGN_VOCABULARY.filter((w) => w.dynamic);
}

/** 单帧即可判别的静态词 */
export function getStaticWords(): SignWord[] {
  return SIGN_VOCABULARY.filter((w) => !w.dynamic);
}

/**
 * **翻译输出**用的显示名 —— 与 `getDisplayLabel` 的区别只在合并类上。
 *
 * `getDisplayLabel("merged_pron_sg")` 给的是「你/他」，那在采集页和训练页是对的：
 * 那里要让人知道这一类**覆盖**哪几个词。但翻译结果里摊开候选没有用 ——
 * 用户看到的是一句话，「你/他 好看」不是中文。所以这里落到组里的 `defaultMember`
 * 上（单数组是 `you` → 「你」）。
 *
 * ⚠ 这与 `labelMerge.ts` 文件头「设计选择 2」写的"显示成『你/他』而不是在两个里
 * 随机挑一个"**有意相反**，是使用者明确要求的（"你/他 在 UI 上统一识别为你"）。
 * 别为了和那段注释一致而改回去 —— 那段说的是采集/训练页的口径，这里是翻译页。
 * 真要区分 你/他 只能换九轴 IMU，届时合并组本身会消失，这个函数会自动退化成
 * `getDisplayLabel`。
 */
export function getTranslationLabel(id: string): string {
  if (id === IDLE_LABEL) return IDLE_DISPLAY_LABEL;
  const merged = getMergeGroup(id);
  if (merged) return getWordById(merged.defaultMember)?.label ?? merged.display;
  return getWordById(id)?.label ?? id;
}

/** 合并类 → 组里默认那个成员的原始 id；不是合并类时原样返回。查 category/描述用 */
export function resolveToMember(id: string): string {
  return getMergeGroup(id)?.defaultMember ?? id;
}

/** 显示名：空闲伪类不在词表里，单独处理 */
export function getDisplayLabel(id: string): string {
  if (id === IDLE_LABEL) return IDLE_DISPLAY_LABEL;
  // 合并类同样不在词表里。放在这里而不是各个界面各自判一次 —— 漏一处就会
  // 在某个角落露出 `merged_pron_sg` 这种原始 id
  const merged = getMergeGroup(id);
  if (merged) return merged.display;
  return getWordById(id)?.label ?? id;
}
