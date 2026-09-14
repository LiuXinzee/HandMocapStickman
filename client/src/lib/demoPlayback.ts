/*
 * demoPlayback — 把库里的**真录制**取出来，做成能直接喂手模的动作片段。
 *
 * 用在翻译页的离线演示（`translateDemo.ts`）：点「演示」时不只逐词跳字，手模也跟着
 * 比划对应的手势。动作来源是 IndexedDB 里录过的 `SequenceSample`，不是合成的曲线
 * —— 这是明确选定的方案，代价与理由见下面「诚实性」那一段。
 *
 * ===== 为什么要单独一层，而不是在页面里直接读库 =====
 *
 * 从库里那条记录到手模能用的一帧之间，有三道**都能静默出错**的换算：
 *
 *  1. **弯折在 137 维里的指序左右相反**。`leftSensor` 存的是重排后的 137
 *     （`SEQ_SENSOR_N`），弯折固定在 60~64，但左手那五路是 [小指…拇指]、右手是
 *     [拇指…小指]。按下标直接取会得到一只"拇指和小指互换"的手 —— 看着像手型不准，
 *     不像取错了。所以一律走 `canonicalBendMapped`。
 *  2. **归一化必须与实时一致**。实时是 `bendRatios`（两点内插，未标定时退 0.42 柔和
 *     预览）。回放另抄一份就会与同一页上的实时手型系统性不同。所以走
 *     `bendRatiosFromCanonical`，与 `bendRatios` 共用同一段数学。
 *  3. **录制的绝对朝向不可用**。六轴无磁力计，yaw 零点是每次上电的随机数
 *     （`imuHealth.ts`），录制那天的零点和今天 localStorage 里存的零位没有关系。
 *     照搬 `applyOrientationCalib` 会得到一只朝向随机偏掉几十度的手。
 *     解法见下面 `buildTrack`：参考帧取**这条录制自己的首帧**。
 *
 * ===== 诚实性：这里播的是真数据，但"真"不等于"对" =====
 *
 * 库里 08-13 那批录制与词表描述不符（`signLanguageVocab.ts` 文件头，已确认的是
 * `hello`）。演示脚本里 `hello` 和 `thank_you` 都落在这批里，也就是说**演示会
 * 一本正经地播两个已知/疑似打错的手势**。
 *
 * 这件事不能只留在注释里 —— 看的人不会去读源码。所以 `DemoClipSet.caveats` 把它
 * 作为数据带出来。要真修只能重录。
 *
 * ⚠ 但它**不再画在画面上**：那条橙带在真给人看演示的时候是反效果的，观众读不懂
 * "与词表描述不符"，只看见产品自己挂了个警告。所以现在分两路走（见
 * `describeDemoClips` / `demoClipQualityNote`）：
 *  - 看的人需要知道的（某个词库里没有录制 → 手模不动、没标定 → 握拳不成形）
 *    仍然画在画面上，因为那些**看起来像故障**，不解释就会被当成程序卡了；
 *  - 录制本身对不对，是给做演示的人自己看的，进 console.warn，不上屏。
 * 别把它合回一条：合回去就只剩"全显示"或"全不显示"两个选项，而这两件事
 * 面向的是不同的人。
 *
 * 合成样本（`origin === "synthesized"`）一律不用：它们的 IMU 是单帧复制出来的
 * （见 `yawDrift.ts` 里那条 `origin !== "recorded"` 的过滤），播出来是一只
 * **完全不动的手**，比没有动作更容易被当成"功能坏了"。
 */
import {
  bendRatiosFromCanonical,
  canonicalBendMapped,
  loadBendRange,
  HAND_LEFT,
  HAND_RIGHT,
  type BendRange,
  type HandKey,
} from "./bendRange";
import {
  getSequencesByLabel,
  isSentenceSample,
  SEQ_IMU_N,
  SEQ_SENSOR_N,
  type SequenceSample,
} from "./datasetStore";
import {
  getTranslationLabel,
  recordingCaveat,
  resolveToMember,
  SUSPECT_BATCH_DATE,
} from "./signLanguageVocab";
import {
  applyOrientationCalib,
  loadOrientationCalib,
  type Quat,
} from "./orientationCalib";
import { detectSignSpan, DEFAULT_TRIM } from "./sequenceTrim";
import type { BendRanges } from "./dominantHand";

/** 一只手在一个片段里的整条轨迹。列存，与 `SequenceSample` 同风格 */
export interface DemoHandTrack {
  /** [T*4] 姿态 [w,x,y,z]，已换算成"相对本片段首帧"（见 `buildTrack`） */
  quats: Float32Array;
  /** [T*5] canonical 拇指→小指 的 0~1 弯曲度，口径与实时 `bendRatios` 一致 */
  curls: Float32Array;
}

export interface DemoWordClip {
  /** 脚本里的词 id（可能是合并类，如 `merged_pron_sg`） */
  wordId: string;
  /** 实际取的录制标签（合并类摊到 `defaultMember`，如 `you`） */
  sourceLabel: string;
  /** 来源样本 id，用于回查是哪一条 */
  sampleId: number | null;
  frameCount: number;
  /** [T] 相对本片段起点的 ms（裁剪起点已减掉） */
  timesMs: Float32Array;
  durationMs: number;
  left: DemoHandTrack | null;
  right: DemoHandTrack | null;
  /** 这条录制可不可信，见文件头「诚实性」 */
  caveat: "wrong" | "suspect" | null;
  /** 这个标签在库里有几条可用录制（选了其中一条，见 `pickRepresentative`） */
  candidateCount: number;
}

export interface DemoClipCaveat {
  wordId: string;
  sourceLabel: string;
  kind: "wrong" | "suspect";
}

export interface DemoClipSet {
  /** 按脚本里的词 id 索引 */
  clips: Map<string, DemoWordClip>;
  /** 库里没有可用录制的词 id（这些词手模不动，字照出） */
  missing: string[];
  caveats: DemoClipCaveat[];
  /**
   * 有没有拿到弯折两点标定。没有时所有手指只弯到约 42%（柔和预览），握拳不成形
   * —— 与实时手模在未标定下的表现完全一致，但演示里没有"去第 1 步"这个出口，
   * 所以要能说出来。
   */
  bendCalibrated: { left: boolean; right: boolean };
}

function currentBendRanges(): BendRanges {
  return { LH: loadBendRange("LH"), RH: loadBendRange("RH") };
}

/**
 * 从一个标签的所有录制里挑一条来播。
 *
 * 取**时长的中位数那条**，两个理由：
 *  - 躲开两头的异常条 —— 最短的那条常是按早了/录漏了，最长的那条常是手举着发呆；
 *  - **确定性**。同一次演示反复点开，播的必须是同一条。随机挑会让人以为
 *    "识别结果不稳"，而这一页的每个动效都在被当成产品行为看。
 *
 * 排序的第二关键字是 id：时长相同时也要定死是哪一条。
 */
export function pickRepresentative(
  samples: SequenceSample[]
): SequenceSample | null {
  const usable = samples.filter(
    (s) =>
      // 合成样本的 IMU 是单帧复制的，播出来不动（文件头）
      s.origin === "recorded" &&
      // `getSequencesByLabel` 查的是 primaryLabel，而句子录制的 primaryLabel
      // 是它的第一个词 —— 不挡掉的话「你 叫 什么 名字」会冒充一条 `你`
      !isSentenceSample(s) &&
      s.frameCount >= 2 &&
      s.durationMs > 0
  );
  if (usable.length === 0) return null;
  const sorted = usable
    .slice()
    .sort((a, b) => a.durationMs - b.durationMs || (a.id ?? 0) - (b.id ?? 0));
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

/** 取第 t 帧的四元数；全零 = 那一帧没写进去（`datasetStore` 的约定） */
function quatAt(imu: Float32Array, t: number): Quat | null {
  const o = t * SEQ_IMU_N;
  if (o + 3 >= imu.length) return null;
  const q: Quat = [imu[o], imu[o + 1], imu[o + 2], imu[o + 3]];
  return Math.hypot(q[0], q[1], q[2], q[3]) > 1e-6 ? q : null;
}

/**
 * 一只手的轨迹。
 *
 * **参考帧取本片段的首帧，不是 localStorage 里的零位。**
 *
 * 这是这个模块里唯一一处刻意偏离实时链路的地方，理由是绝对朝向在这里不存在：
 * 手套是六轴无磁力计，yaw 零点是每次上电的随机值（`imuHealth.ts:4-7`）。
 * 存在 localStorage 里的 `reference` 是**某一天某一次上电**下按的零位，而这条录制
 * 是另一天另一次上电录的 —— 拿前者去减后者，等于减掉一个随机数，手模会整体
 * 偏掉几十度，而且每个词偏的量还不一样（录制横跨多个会话）。
 *
 * 取首帧则天然消掉这一项：`q0⁻¹ ⊗ q_t` 描述的是"相对起势姿态转了多少"，与上电
 * 零点无关。这也正是训练特征那边的口径（`sequenceFeatures.ts` 开头：相对首帧的
 * 四元数消掉绝对朝向），所以看到的动作和模型看到的是同一个东西。
 *
 * 代价：每个词都从"单位旋转"（= 标定约定的竖立、手心朝自己）起手，起手朝向丢了。
 * 手语的信息主要在**手型和轨迹**上，这个代价是可接受的；真正需要绝对朝向的
 * 只有我/你/他 那三个词，而它们本来就因为这个原因被合并成了一类。
 *
 * `axisMap` 照旧套用：它修的是"IMU 装在手套里的轴向"，与哪次上电无关，同一副
 * 手套录的和现在戴的是同一个矩阵。没标定过就没有，那时候是把机体轴当模型轴用
 * —— 与实时链路在未标定下的行为一致。
 */
function buildTrack(
  imu: Float32Array | null,
  sensor: Uint8Array | null,
  handKey: HandKey,
  sensorType: number,
  range: BendRange | null,
  startFrame: number,
  endFrame: number
): DemoHandTrack | null {
  if (!imu && !sensor) return null;
  const T = endFrame - startFrame;
  const quats = new Float32Array(T * 4);
  const curls = new Float32Array(T * 5);

  // 只借 axisMap，reference 换成本片段首帧（见上）
  const axisMap = loadOrientationCalib(handKey)?.axisMap;
  let reference: Quat | null = null;
  if (imu) {
    for (let t = startFrame; t < endFrame && !reference; t++) {
      reference = quatAt(imu, t);
    }
  }

  // 上一帧的姿态：这一帧 IMU 缺了就沿用，而不是跳回单位旋转（那会是一下抽动）
  let hold: Quat = [1, 0, 0, 0];
  for (let i = 0; i < T; i++) {
    const t = startFrame + i;
    if (imu && reference) {
      const raw = quatAt(imu, t);
      if (raw) hold = applyOrientationCalib(raw, { reference, axisMap });
    }
    quats[i * 4] = hold[0];
    quats[i * 4 + 1] = hold[1];
    quats[i * 4 + 2] = hold[2];
    quats[i * 4 + 3] = hold[3];

    if (sensor && (t + 1) * SEQ_SENSOR_N <= sensor.length) {
      const ratios = bendRatiosFromCanonical(
        canonicalBendMapped(sensor, sensorType, t * SEQ_SENSOR_N),
        range
      );
      for (let k = 0; k < 5; k++) curls[i * 5 + k] = ratios[k];
    }
  }
  return { quats, curls };
}

/** 一条样本 → 一个可播片段。裁掉两头的"举着等按键" */
export function buildWordClip(
  wordId: string,
  sourceLabel: string,
  sample: SequenceSample,
  ranges: BendRanges,
  candidateCount: number
): DemoWordClip | null {
  /*
   * 裁剪走**和训练完全一样的那套判据**（`detectSignSpan`），不另写一套：
   * 录制是「按下 → 抬手 → 打手势 → 停住 → 松手」，不裁的话每个词前面都有
   * 半秒到一秒半的呆站，5 句连播就是十几秒的死时间。
   *
   * 这里是词的口径（`DEFAULT_TRIM` 就是），只把 `ranges` 补上 —— 第三层
   * （触觉静止段）是唯一会裁**尾巴**的一层，而演示里尾巴上的呆站最扎眼。
   * 没有弯折标定时那一层不跑（那是它自己的守则，"判不出来就不裁"），
   * 于是演示会带上呆站 —— 这是"宁可慢一点，也不静默切错手势"。
   */
  const span = detectSignSpan(sample, { ...DEFAULT_TRIM, ranges });
  const startFrame = span.startFrame;
  const endFrame = span.endFrame;
  const T = endFrame - startFrame;
  if (T < 2) return null;

  const ts = sample.timestamps;
  const timesMs = new Float32Array(T);
  const t0 = ts[startFrame];
  for (let i = 0; i < T; i++) timesMs[i] = ts[startFrame + i] - t0;
  const durationMs = timesMs[T - 1];
  if (!(durationMs > 0)) return null;

  const left = buildTrack(
    sample.leftImu,
    sample.leftSensor,
    "LH",
    HAND_LEFT,
    ranges.LH ?? null,
    startFrame,
    endFrame
  );
  const right = buildTrack(
    sample.rightImu,
    sample.rightSensor,
    "RH",
    HAND_RIGHT,
    ranges.RH ?? null,
    startFrame,
    endFrame
  );
  // 两只手都没数据的录制没有可播的东西，当作库里没有这个词
  if (!left && !right) return null;

  return {
    wordId,
    sourceLabel,
    sampleId: sample.id ?? null,
    frameCount: T,
    timesMs,
    durationMs,
    left,
    right,
    caveat: recordingCaveat(sourceLabel),
    candidateCount,
  };
}

/**
 * 把脚本里用到的词都从库里取出来。
 *
 * 合并类先摊到成员上（`resolveToMember`）：`merged_pron_sg` 自己**一条录制都没有**
 * —— 它是 `labelMerge` 造出来的类，库里只有 `you` / `he`。不摊的话
 * `getSequencesByLabel("merged_pron_sg")` 返回空，而演示脚本里每句都有它，
 * 结果是"5 句里手模一多半时间不动"，看起来像功能没做完。
 *
 * 去重：`merged_pron_sg` 在 5 句里出现 5 次，只查一次库、5 次共用同一条录制
 * —— 也就是同一个词每次都比划同一个动作。这是对的：它本来就是同一个词。
 */
export async function loadDemoClips(
  wordIds: readonly string[]
): Promise<DemoClipSet> {
  const ranges = currentBendRanges();
  const clips = new Map<string, DemoWordClip>();
  const missing: string[] = [];
  const caveats: DemoClipCaveat[] = [];

  // `Array.from` 而不是 for..of Set：tsconfig 的 target 不支持迭代 Set
  for (const wordId of Array.from(new Set(wordIds))) {
    const sourceLabel = resolveToMember(wordId);
    let clip: DemoWordClip | null = null;
    try {
      const samples = await getSequencesByLabel(sourceLabel);
      const pick = pickRepresentative(samples);
      if (pick) {
        const usableCount = samples.filter(
          (s) => s.origin === "recorded" && !isSentenceSample(s)
        ).length;
        clip = buildWordClip(wordId, sourceLabel, pick, ranges, usableCount);
      }
    } catch {
      // 库打不开（隐私模式 / 版本冲突）不该让演示整个失败：字还是能演的，
      // 手模不动而已。所以这里吞掉，靠 missing 如实报出来
      clip = null;
    }
    if (clip) {
      clips.set(wordId, clip);
      if (clip.caveat) {
        caveats.push({ wordId, sourceLabel, kind: clip.caveat });
      }
    } else {
      missing.push(wordId);
    }
  }

  return {
    clips,
    missing,
    caveats,
    bendCalibrated: { left: !!ranges.LH, right: !!ranges.RH },
  };
}

/**
 * 手模动作那一行要说的话 —— 没有可说的就返回 null。
 *
 * 放在库这一层而不是页面里：这段话的**内容**是由数据决定的（哪几个词缺录制），
 * 页面只负责把它画出来。写在页面里就会变成"改了名单忘了改文案"。
 *
 * 这里**只说"看起来像故障、其实不是"的那几件事**：
 *  - 缺录制：手模不动，不是卡了；
 *  - 没标定：手指弯不到底，握拳不成形。
 * 录制本身准不准不在这条里 —— 见 `demoClipQualityNote` 和文件头那段。
 */
export function describeDemoClips(set: DemoClipSet): string | null {
  const parts: string[] = [];

  if (set.missing.length > 0) {
    parts.push(
      `「${set.missing.map((id) => getTranslationLabel(id)).join("、")}」库里没有录制，` +
        `手模在这几个词上不动（不是卡住）`
    );
  }
  if (!set.bendCalibrated.left && !set.bendCalibrated.right) {
    parts.push("没有弯折两点标定，手指只弯到约 42%，握拳不会成形");
  }
  return parts.length > 0 ? parts.join("；") + "。" : null;
}

/**
 * 录制**质量**存疑的那几条 —— 给做演示的人自己看的，不上屏（进 console.warn）。
 *
 * 两种存疑分开说，意味着的事不一样：
 *  - 已确认打错：这个手势不能照着学；
 *  - 同批嫌疑：没逐条查过，别当范本。
 *
 * 为什么不上屏：见文件头那段。简单说，观众读不懂"与词表描述不符"，
 * 只会看见产品自己挂了个橙色警告。而这条信息的受众本来就是做演示的人。
 */
export function demoClipQualityNote(set: DemoClipSet): string | null {
  const parts: string[] = [];
  const name = (c: DemoClipCaveat) => getTranslationLabel(c.wordId);

  const wrong = set.caveats.filter((c) => c.kind === "wrong");
  if (wrong.length > 0) {
    parts.push(
      `手模动作是库里的真录制，但「${wrong.map(name).join("、")}」那几条` +
        `与词表描述不符（${SUSPECT_BATCH_DATE} 那批，已核对），别照着学`
    );
  }
  const suspect = set.caveats.filter((c) => c.kind === "suspect");
  if (suspect.length > 0) {
    parts.push(
      `「${suspect.map(name).join("、")}」是同批录的、未逐条核对，同样别当范本`
    );
  }
  return parts.length > 0 ? parts.join("；") + "。" : null;
}

/** 每个词的手势时长，喂给 `buildDemoTimeline` 让字的节奏跟着动作走 */
export function clipDurations(set: DemoClipSet): Map<string, number> {
  const out = new Map<string, number>();
  set.clips.forEach((clip, wordId) => out.set(wordId, clip.durationMs));
  return out;
}

/**
 * 片段里 `offsetMs` 落在第几帧 —— **取最近帧，不插值**。
 *
 * 栅格是 50Hz（20ms），rAF 是 60Hz，最近帧的抖动上界是半帧 10ms；而 `HandModel`
 * 的 `useFrame` 本来就在朝目标姿态做指数插值（时间常数约 110ms），10ms 的台阶
 * 从那一头看不出来。为此再写一套四元数 slerp 是白写。
 */
export function frameAt(clip: DemoWordClip, offsetMs: number): number {
  const T = clip.frameCount;
  if (offsetMs <= 0) return 0;
  if (offsetMs >= clip.durationMs) return T - 1;
  // 线性扫。T 是 50~200 的量级、每帧两只手各扫一次，比 rAF 里其它开销小得多
  let i = 0;
  while (i + 1 < T && clip.timesMs[i + 1] <= offsetMs) i++;
  // 落在 i 与 i+1 之间，取近的那个
  if (i + 1 < T) {
    const a = offsetMs - clip.timesMs[i];
    const b = clip.timesMs[i + 1] - offsetMs;
    if (b < a) return i + 1;
  }
  return i;
}

/** 一帧的姿态与手型，形状与 `HandDrive` 对齐（那边只读这两项 + hasData） */
export interface DemoHandPose {
  quaternion: [number, number, number, number];
  curl: number[];
}

export function poseAt(
  clip: DemoWordClip,
  side: "left" | "right",
  frame: number
): DemoHandPose | null {
  const track = side === "left" ? clip.left : clip.right;
  if (!track) return null;
  const q = frame * 4;
  const c = frame * 5;
  return {
    quaternion: [
      track.quats[q],
      track.quats[q + 1],
      track.quats[q + 2],
      track.quats[q + 3],
    ],
    curl: [
      track.curls[c],
      track.curls[c + 1],
      track.curls[c + 2],
      track.curls[c + 3],
      track.curls[c + 4],
    ],
  };
}
