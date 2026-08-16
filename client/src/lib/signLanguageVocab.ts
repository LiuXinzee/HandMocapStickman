/*
 * 手语词汇库 — 中国手语常用词汇
 * 初始 31 个，2026-08-14 追加 14 个（代词 6、判断 2、动作 4、名词/疑问 2），共 45 个
 *
 * dynamic 字段区分静态手型词与带轨迹的动态词：
 * - 静态词（数字、"停"、"家"…）单帧就能判别
 * - 动态词（"再见"左右摆动、"来"招手…）必须看整段序列
 * 序列模型统一处理两者（静态词 = 帧间无变化的序列），dynamic 只用于
 * 采集页分组显示和提示用户该录多久。
 */

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
  { id: "hello", label: "你好", pinyin: "nǐ hǎo", category: "greeting", description: "右手伸出食指和中指，向前点头", dynamic: true },
  { id: "thank_you", label: "谢谢", pinyin: "xiè xiè", category: "greeting", description: "右手手心向下，从嘴边向前伸出", dynamic: true },
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
  { id: "sad", label: "难过", pinyin: "nán guò", category: "emotion", description: "食指从眼角向下划", dynamic: true },
  { id: "love", label: "爱", pinyin: "ài", category: "emotion", description: "双手交叉放在胸前" },
  { id: "angry", label: "生气", pinyin: "shēng qì", category: "emotion", description: "双手握拳向上举", dynamic: true },

  // 动作类
  { id: "go", label: "去", pinyin: "qù", category: "action", description: "食指向前指", dynamic: true },
  { id: "come", label: "来", pinyin: "lái", category: "action", description: "手掌向内招手", dynamic: true },
  { id: "help", label: "帮助", pinyin: "bāng zhù", category: "action", description: "一手托起另一手", dynamic: true },
  { id: "stop", label: "停", pinyin: "tíng", category: "action", description: "手掌向前伸出" },
  { id: "see", label: "看", pinyin: "kàn", category: "action", description: "食指中指分开成 V 形，从眼睛旁向前伸出", dynamic: true },
  { id: "listen", label: "听", pinyin: "tīng", category: "action", description: "食指指向耳朵（或手拢在耳后）" },
  { id: "speak", label: "说", pinyin: "shuō", category: "action", description: "食指在嘴边向前转动", dynamic: true },
  { id: "know", label: "认识", pinyin: "rèn shi", category: "action", description: "食指点太阳穴，再向前伸出", dynamic: true },

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
  { id: "i", label: "我", pinyin: "wǒ", category: "daily", description: "食指指向自己胸口" },
  { id: "you", label: "你", pinyin: "nǐ", category: "daily", description: "食指指向对方" },
  { id: "he", label: "他", pinyin: "tā", category: "daily", description: "食指指向侧前方的第三者" },
  { id: "we", label: "我们", pinyin: "wǒ men", category: "daily", description: "食指指自己，再横向划一道弧（复数）", dynamic: true },
  { id: "you_pl", label: "你们", pinyin: "nǐ men", category: "daily", description: "食指指对方，再横向划一道弧（复数）", dynamic: true },
  { id: "they", label: "他们", pinyin: "tā men", category: "daily", description: "食指指侧前方，再横向划一道弧（复数）", dynamic: true },
  { id: "is", label: "是", pinyin: "shì", category: "daily", description: "右手竖起拇指，向下点动一次", dynamic: true },
  { id: "is_not", label: "不是", pinyin: "bú shì", category: "daily", description: "食指左右摆动（否定），再做「是」的拇指点动", dynamic: true },
  { id: "name", label: "名字", pinyin: "míng zi", category: "daily", description: "右手食指在左手掌心做写字状", dynamic: true },
  { id: "what", label: "什么", pinyin: "shén me", category: "daily", description: "手掌向上摊开，左右轻摆", dynamic: true },
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

/** 显示名：空闲伪类不在词表里，单独处理 */
export function getDisplayLabel(id: string): string {
  if (id === IDLE_LABEL) return IDLE_DISPLAY_LABEL;
  return getWordById(id)?.label ?? id;
}
