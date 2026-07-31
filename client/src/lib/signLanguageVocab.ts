/*
 * 手语词汇库 — 中国手语常用静态手势词汇
 * 初始支持 30 个常用词汇，可逐步扩展
 */

export interface SignWord {
  id: string;
  label: string;        // 中文词汇
  pinyin: string;       // 拼音
  category: string;     // 分类
  description: string;  // 手势描述
}

export const SIGN_CATEGORIES = [
  { id: "greeting", label: "问候", color: "#00f0ff" },
  { id: "number", label: "数字", color: "#00e5a0" },
  { id: "daily", label: "日常", color: "#f59e0b" },
  { id: "emotion", label: "情感", color: "#ff2d7b" },
  { id: "action", label: "动作", color: "#a855f7" },
] as const;

export const SIGN_VOCABULARY: SignWord[] = [
  // 问候类
  { id: "hello", label: "你好", pinyin: "nǐ hǎo", category: "greeting", description: "右手伸出食指和中指，向前点头" },
  { id: "thank_you", label: "谢谢", pinyin: "xiè xiè", category: "greeting", description: "右手手心向下，从嘴边向前伸出" },
  { id: "sorry", label: "对不起", pinyin: "duì bù qǐ", category: "greeting", description: "右手握拳放在胸前，轻轻拍打" },
  { id: "goodbye", label: "再见", pinyin: "zài jiàn", category: "greeting", description: "右手掌心向外，左右摆动" },
  { id: "please", label: "请", pinyin: "qǐng", category: "greeting", description: "右手掌心向上，向前伸出" },
  { id: "welcome", label: "欢迎", pinyin: "huān yíng", category: "greeting", description: "双手掌心向上，向外展开" },

  // 数字类
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
  { id: "eat", label: "吃", pinyin: "chī", category: "daily", description: "手指捏在一起送向嘴边" },
  { id: "drink", label: "喝", pinyin: "hē", category: "daily", description: "拇指和小指伸出，送向嘴边" },
  { id: "sleep", label: "睡觉", pinyin: "shuì jiào", category: "daily", description: "手掌贴在脸侧，头微倾" },
  { id: "home", label: "家", pinyin: "jiā", category: "daily", description: "双手指尖相触成屋顶状" },
  { id: "work", label: "工作", pinyin: "gōng zuò", category: "daily", description: "双手握拳交替上下运动" },
  { id: "study", label: "学习", pinyin: "xué xí", category: "daily", description: "一手做翻书动作" },

  // 情感类
  { id: "happy", label: "高兴", pinyin: "gāo xìng", category: "emotion", description: "双手在脸旁向上展开" },
  { id: "sad", label: "难过", pinyin: "nán guò", category: "emotion", description: "食指从眼角向下划" },
  { id: "love", label: "爱", pinyin: "ài", category: "emotion", description: "双手交叉放在胸前" },
  { id: "angry", label: "生气", pinyin: "shēng qì", category: "emotion", description: "双手握拳向上举" },

  // 动作类
  { id: "go", label: "去", pinyin: "qù", category: "action", description: "食指向前指" },
  { id: "come", label: "来", pinyin: "lái", category: "action", description: "手掌向内招手" },
  { id: "help", label: "帮助", pinyin: "bāng zhù", category: "action", description: "一手托起另一手" },
  { id: "stop", label: "停", pinyin: "tíng", category: "action", description: "手掌向前伸出" },
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
