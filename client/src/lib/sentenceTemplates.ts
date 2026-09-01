/*
 * sentenceTemplates —— 句型表的 **TypeScript 侧唯一来源**。
 *
 * 这张表原本有三份：`python_train/synth_sentences.py` 的 `SENTENCE_TEMPLATES`、
 * `sentenceGrammar.test.ts` 里手抄的一份、以及（阶段 4 之后）采集页要用的一份。
 * 三份互相不认识，加句型时漏一处的后果是静默的：
 *   - Python 加了、TS 没加 → 训了但顺不出汉语（结果退化成原样拼词）
 *   - TS 加了、Python 没加 → 采了几十条，训练时那个句型对不上任何合成句型
 *
 * 现在收成两份（TS 一份 + Python 一份），并且由 `sentenceTemplates.test.ts`
 * **真的读 synth_sentences.py 逐条比对**锁住。
 * （`sentenceGrammar.test.ts` 从前声称能抓 Python 侧漂移，其实抓不到 ——
 *  它比的是自己手抄的那份常量。那份已删，改成 import 这里。）
 *
 * 语序按**手语**写，不是汉语：疑问词放句尾、没有「的」、「是」很多时候不打。
 * 顺成汉语是 `sentenceGrammar.ts` 的活，这里只管标签序列。
 *
 * 标签是**未合并**的原始 id（i/you/he 分开写）。模型看到的是合并后的类
 * （见 `labelMerge.ts`），但表里必须写原始 id —— 它同时是"这一句该打什么"的说明，
 * 而「我 爱 你」和「你 爱 我」是两个不同的动作，写成合并类就分不出来了。
 */

/**
 * ===== 本轮不训练的词 =====
 *
 * 这几个词的录制**一条都不进训练**：`TrainSequence` 的默认排除清单取自这里，
 * `SENTENCE_TEMPLATES` 里也不允许出现（有测试锁着）。Python 侧同名常量在
 * `python_train/synth_sentences.py`，由 `sentenceTemplates.test.ts` 逐条比对。
 *
 * 在这之前是**两套词表并存**：浏览器词模型 22 类、句子模型 23 类 —— 句子模型会
 * 输出词模型从来没训过的词。症状只是"演示时偶尔蹦一个奇怪的词"，不报错，
 * 合成 WER 也看不出来。
 *
 * 两个词各自的理由不同，**在类别表上的后果也不同**：
 *   `happy`   只有 3 条录制。合成句里同 3 条反复出现，学到的是这 3 条的噪声
 *   `you_pl`  与 we/they 只差 yaw，合并后本来就不是一个独立类（见 `labelMerge.ts`）
 *
 * `happy` 合并后还是自己 → 类别表**少一类**。`you_pl` 合并进 `merged_pron_pl`，
 * 而 we/they 还在 → **那个类不会消失**，只是不再拿 you_pl 的录制去训它。
 * 所以过滤必须"先合并、再看这个类还有没有活着的成员"，不能拿名字直接从类别表里删。
 *
 * ⚠ 这是**过滤不是删除**：录制还在库里，将来想训回来（比如 `happy` 补够条数）
 * 只需把它从这个清单里去掉，再把对应句型加回两边的表。
 *
 * ===== `name` 已于 2026-08-29 移出本清单 =====
 *
 * 它当初进来的理由是"17 条里 4 条裁完不足 250ms（`too_short` 整条不裁）→
 * 同一类里两种时间包络"。现在要训「你叫什么名字」，所以把它训回来。
 *
 * ⚠ **旧的 17 条录制必须先在浏览器里删掉再重录**，不能直接放行 —— 那 4 条坏包络
 * 还在库里，放行等于把当初的问题原样请回来，而症状只是「名字」这个词准确率偏低，
 * 训练日志上看不出来。删除入口在 `/train-seq` 的数据体检区（录制存在
 * IndexedDB 里，改代码删不掉）。重录在 `/collect-seq`。
 */
export const UNTRAINED_WORDS: readonly string[] = ["happy", "you_pl"];

/**
 * 全部句型，与 `python_train/synth_sentences.py` 的 `SENTENCE_TEMPLATES` 逐条相同。
 * 改这里必须同时改那边（有测试锁着）。
 *
 * 66 → 53 句：含 `name` 的 7 条 + 含 `happy` 的 3 条按 `UNTRAINED_WORDS` 删掉，
 * 三条"词堆"删掉（见下面 `BATCH1_TEMPLATES` 的说明），`["you_pl","listen"]`
 * 换成 `["they","listen"]`。
 *
 * 53 → 56 句（2026-08-29）：加回 `["you","name","what"]`（`name` 训回来了），
 * 新增 `["you","beautiful"]` 与 `["you","smile","resemble","sun"]`。
 * 后两句用到四个新词（见 `signLanguageVocab.ts` 末尾那批），**一条录制都还没有** ——
 * 合成时 `synthesize_sentences` 会打印「跳过 … —— 缺 …」把它们丢掉，
 * 那不是报错，是提醒还没采。含 name 的另外 6 条没加回来：`name` 这一轮只服务
 * 「你叫什么名字」这一句，句型加多了等于把采集预算摊薄。
 *
 * ⚠ **`is`（是）现在一个句型都没用到了** —— 它原来只出现在「我名字是X」这两句里。
 * 库里 17 条 `is` 录制仍会进词模型（单词识别里「是」是个正常的词），但句子模型
 * 会有一个从未在任何训练句里出现过的类。`train_seq.py --ctc` 会把这种"有类无句型"
 * 的词打印出来，别当它是噪声：要么给它编一句合理的手语句子，要么把它也加进
 * `UNTRAINED_WORDS`。这里**故意没有替你决定** —— 27 词的表里造不出既通顺、
 * 又真的会打「是」的句子。
 */
export const SENTENCE_TEMPLATES: readonly (readonly string[])[] = [
  // 问答
  // 2026-08-29 加回：`name` 已移出 UNTRAINED_WORDS。语序是手语的（疑问词句尾），
  // 顺成汉语靠 sentenceGrammar 的「问名字」规则 —— 那条规则一直都在，没删过
  ["you", "name", "what"],
  ["you", "eat", "what"],
  ["you", "speak", "what"],
  ["you", "work", "what"],
  ["you", "study", "what"],
  // 寒暄
  ["hello", "you"],
  ["goodbye", "you"],
  ["thank_you", "you"],
  ["sorry", "i"],
  ["welcome", "you"],
  ["welcome", "you", "we"],
  // 主谓宾
  ["i", "love", "you"],
  ["you", "love", "i"],
  ["i", "listen", "you"],
  ["i", "help", "you"],
  ["you", "help", "i"],
  ["i", "speak", "you"],
  // 否定
  ["i", "is_not", "sad"],
  ["i", "is_not", "angry"],
  ["i", "is_not", "eat"],
  ["you", "is_not", "listen"],
  // 陈述
  ["i", "sad"],
  ["i", "eat"],
  ["i", "drink"],
  ["i", "work"],
  ["i", "study"],
  ["we", "study"],
  ["we", "work"],
  ["they", "study"],
  // 原来是 ["you_pl","listen"]。换成 they 而不是删掉：复数类（merged_pron_pl）
  // 本来就只有合成数据支撑，少一句就更薄
  ["they", "listen"],
  ["he", "is_not", "work"],
  ["i", "thank_you", "you"],
  ["i", "help", "you", "study"],
  ["you", "eat", "drink"],
  // 赞美 / 比喻（2026-08-29 追加）。四个新词一条录制都还没有，合成阶段会先跳过
  ["you", "beautiful"],
  // 全表第二条 4 词句（此前只有 ["i","help","you","study"]）。4 词是最吃 32 个输出帧
  // 预算的一档（T=128 池化两次），多一条真实数据支撑对那一档是纯赚
  ["you", "smile", "resemble", "sun"],
  // 无主语句（手语里主语常省略）。这一组在 synth_sentences.py 里是为了压类先验加的
  ["eat", "what"],
  ["drink", "what"],
  ["work", "what"],
  ["study", "what"],
  ["speak", "what"],
  ["listen", "what"],
  ["hello", "welcome"],
  ["hello", "thank_you"],
  ["goodbye", "thank_you"],
  ["help", "thank_you"],
  ["sorry", "goodbye"],
  ["is_not", "angry"],
  ["is_not", "sad"],
  ["is_not", "eat"],
  ["is_not", "drink"],
  ["is_not", "work"],
  ["eat", "drink"],
  ["listen", "speak"],
  ["love", "eat"],
  ["love", "study"],
];

/** 每个句型建议采集的条数。20 条才够划出训练/验证两半而验证集不至于只剩 1~2 条 */
export const RECOMMENDED_PER_TEMPLATE = 20;

/**
 * 第一批真实采集的 15 句。**必须是 `SENTENCE_TEMPLATES` 的子集**（有测试锁着）。
 *
 * ===== 为什么只采这些、每句 20 条 =====
 *
 * 第一批要回答的问题是"真实协同发音到底能不能学到"，不是"覆盖全表"。
 * 剩下 41 句仍有合成数据兜着。15 句 × 20 条 ≈ 50 分钟净打，
 * 而 56 句摊薄到每句 5 条等于把预算平均分配到"每句都不够"。
 *
 * ⚠ **新加的 3 句要先补词级录制才能采句子。** 见本注释末尾「2026-08-29 追加的三句」。
 *
 * ===== 分组的判据是"有没有猜的成分"，不是"有没有代词" =====
 *
 * 你/他 合并成一类之后模型**永远**分不出这两个（六轴 IMU 观测不到绝对 yaw，
 * 见 labelMerge.ts），落到这个类上的位置只能由 `sentenceGrammar.ts` 的规则表给
 * 一个默认值。**但「我」不是** —— 它指自己胸口、有接触，已经是独立类，模型直接
 * 认出来。所以：
 *   - 没有猜的成分（无代词，或只含「我」）→ 输出的每个字都是模型真认出来的，
 *     是**唯一能"全对/全错"二分判断**的验收材料
 *   - 有猜的成分（含 你/他 或复数）→ 验证合并类在真实数据上是否仍然吞掉别的词
 *
 * ===== 那三条"词堆"已经删了 =====
 *
 * 原来凑够无代词句子的方式是从全表里挑无代词的组合，挑出来这三条：
 *   ["study","work"]             学习工作
 *   ["welcome","study","work"]   欢迎学习工作
 *   ["help","study","thank_you"] 帮助学习谢谢
 * **这三条根本不是句子**，是词的堆叠。采集页会把它当"请打这句话"提示、演示时也会
 * 读出来，读出来就是坏的。它们留在 `SENTENCE_TEMPLATES` 的理由曾经是**压类先验**
 * （`_PRIOR_WARN_SHARE = 0.35`，而最高频类曾是 `merged_pron_sg` 占 41%，那个状态下
 * CTC 一遇到没把握的输入就无脑输出代词）。
 *
 * **「我」拆成独立类之后这条理由不成立了。** 在 66 句的表上实测过：
 *   拆分前：最高频类 merged_pron_sg 26.5%
 *   拆分后：merged_pron_sg 13.6%、i 13.0%（同一批词位被劈成两半）
 *   拿掉那三条 → 最高频类 14.3%、代词词位合计 31.2%
 * 也就是说删掉它们最高频类只从 13.6% 涨到 14.3%，离 35% 的报警线还有一倍多余量。
 * 41% 那个数字是旧句型表 + 我/你/他 同一类时代的，现在引用它是错的。
 * （删掉 name/happy 那 10 句之后先验会再变一次，训练时 `_report_prior` 会现场量。）
 *
 * ===== 本批与上一版的差别 =====
 *
 * 上一版 12 句里有 6 句含 `name`/`happy`，随 `UNTRAINED_WORDS` 一起作废。补上的是：
 *   `["work","what"]` / `["hello","thank_you"]` / `["is_not","angry"]` / `["i","is_not","sad"]`
 *     —— 顶掉 `["name","what"]` / `["i","name","is"]` / `["is_not","happy"]` / `["i","is_not","happy"]`
 *   `["you","eat","what"]` —— 顶掉 `["you","name","what"]`
 *   `["we","study"]` —— 顶掉 `["hello","you","name","what"]`，顺带补掉下面「已知缺口 1」
 *
 * ===== 已知缺口 =====
 *
 * 1. `merged_pron_pl`（我们/你们/他们）原来**一条真实数据都没有**。这一版拿
 *    `["we","study"]` 占了一格，算是最低限度的覆盖 —— 一句 20 条，只够看"复数类
 *    在真实数据上会不会整类塌掉"，不够看组内表现（组内本来也分不开）。
 * 2. **4 词句原来只剩 `["i","help","you","study"]` 一条**（删掉含 name 的两句之后
 *    全表里就只有它了）。4 词是最吃 32 个输出帧预算的一档（T=128 池化两次）。
 *    2026-08-29 加的 `["you","smile","resemble","sun"]` 是第二条 —— 缺口缓解了一半，
 *    但那句的四个词还一条录制都没有，所以在补齐词级录制之前这条缺口照旧。
 * 3. **同一个类在一句里出现两次的情形，本表已经不覆盖了。** 以前
 *    `["i","love","you"]` 解出来是 `[sg, love, sg]`，正好能测"CTC 把中间没有 blank
 *    隔开的两次出现折成一次"这个失败模式。「我」拆出去之后它变成 `[i, love, sg]`，
 *    而全表里所有双代词句都是「我 + 你」的组合 —— 重复类的情形整张表都消失了。
 *    这是拆类的**副作用**，不是遗漏：真实句子里确实不再有重复类。要专门测这条
 *    只能造 `["you","love","he"]` 这种表里没有的序列，那属于解码器单测
 *    （`ctcDecode.test.ts`）的活，不该占采集预算。
 *
 * ===== 2026-08-29 追加的三句 =====
 *
 *   ["you","name","what"]              你叫什么名字
 *   ["you","beautiful"]                你真好看
 *   ["you","smile","resemble","sun"]   你笑起来像太阳
 *
 * **采句子之前必须先补词级录制**，顺序不能反 —— 三个前置条件都是静默失败的：
 *
 * 1. **类别表来自导出的 `primaryLabel` 去重**，而句子样本的 primaryLabel 只是
 *    **第一个词**（`datasetStore.ts`）。所以 `beautiful`/`smile`/`resemble`/`sun`
 *    只出现在句中的话**根本不会有类**，`build_ctc_xy` 会把它们从目标序列里
 *    悄悄丢掉（只打一行 ⚠️），参考答案就是错的，而 WER 只是"略差一点"。
 * 2. **合成句是从词录制交叉淡入拼出来的。** 没有词录制，这两句在合成阶段
 *    会被 `synthesize_sentences` 打印「跳过 … —— 缺 …」直接丢掉。
 * 3. **CTC 起跑要迁移词骨干** `out/student.keras`（`transfer_backbone`）。
 *    conv/bn 的形状不随类别数变，所以骨干没见过新词也照样打印「迁移成功」——
 *    症状只是那几个词 WER 特别差。`/train-sentence` 顶部那一栏（Band ⓪）
 *    就是为这件事加的，它按标签差集判，会点名说"骨干没见过 beautiful"。
 *
 * 于是完整顺序是：`/collect-seq` 补 4 个新词 + 重录 `name` → `/train-seq` 训词模型
 * （产出骨干）→ `/collect-sentence` 采这三句 → `/train-seq` 直送 → `/train-sentence`。
 *
 * ⚠ `name` 的旧 17 条**要先删再录**（见文件顶部 `UNTRAINED_WORDS` 那段）。
 */
export const BATCH1_TEMPLATES: readonly (readonly string[])[] = [
  // ——— 没有猜的成分 8 句：每个字都是模型认出来的，可以直接演示 ———
  // 前 4 条不含任何代词
  ["eat", "what"],
  ["work", "what"],
  ["hello", "thank_you"],
  ["is_not", "angry"],
  // 后 4 条只含「我」。「我」是独立类（指自己胸口、有接触，不是纯 yaw 之差，
  // 见 labelMerge.ts），所以这几句里**没有任何一位是规则表猜的** —— 和"无代词"
  // 在验收上等价
  ["i", "eat"],
  ["i", "study"],
  ["i", "is_not", "eat"],
  ["i", "is_not", "sad"],
  // ——— 有猜的成分 4 句：验证合并类在真实数据上会不会吞掉别的词 ———
  // [i, love, sg]：拆开「我」之后这句和「你爱我」不再是同一个序列，
  // 方向是量到的；剩下要猜的只有宾语是「你」还是「他」
  ["i", "love", "you"],
  ["you", "eat", "what"],
  // 复数类唯一的真实数据（见「已知缺口 1」）
  ["we", "study"],
  // [i, help, sg, study]：一句里两个代词位、类别不同（一个独立类一个合并类），
  // 解码器把它们混起来的话会少词 —— 专门留着测这个
  ["i", "help", "you", "study"],
  // ——— 2026-08-29 追加的三句（见上面注释末尾）———
  // 三句都以「你」开头，也就是三句都落在 merged_pron_sg 上 → 都归"有猜的成分"。
  // 「你」这一位模型永远分不出你/他，是 sentenceGrammar 的规则表按 other 给的默认值；
  // 这三句的规则都写死了 other，所以显示上是对的，但**它不是模型认出来的**
  ["you", "name", "what"],
  ["you", "beautiful"],
  ["you", "smile", "resemble", "sun"],
];

/** 句型的稳定 key：标签序列用空格连起来。用于按句型计数与查询 */
export function templateKey(labels: readonly string[]): string {
  return labels.join(" ");
}
