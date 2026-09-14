/*
 * StepNav — 四步流程的「我在第几步 / 手套还在不在 / 下一步去哪」
 *
 * 装在六个流程页的 header 右侧。零 props：步号与下一步都从路由查表得来，
 * 手套状态直接读全应用唯一的 GloveContext（连接入口只有 /mocap 一处）。
 *
 * 两件事是这个组件存在的理由：
 * 1) 每一步做完有出口，不用退回首页再点下一步；
 * 2) 左右手**分开**显示。以前各页头上只有一个笼统的 GLOVE OFF，
 *    少插一只手看不出来，录到一半才发现右手全程是空的。
 *
 * /train-skeleton 是旁支、不在流程里，查表落空就渲染 null。
 */
import { Link, useLocation } from "wouter";
import { ArrowRight, Hand } from "lucide-react";
import { useGloves } from "@/contexts/GloveContext";

interface NextLink {
  href: string;
  label: string;
  /** 主推路径高亮；次要路径走暗色 */
  primary?: boolean;
}

interface StepEntry {
  step: number;
  title: string;
  next: NextLink[];
}

const TOTAL_STEPS = 4;

/*
 * 文案约定：**每一条都必须带「静态」「时序」或「句子」**。
 *
 * 第 2、3 步有三条链路，走错一条不会报错，只会训出一个空的或喂错的模型：
 *
 *   静态单帧   MLP   `/collect`          + `/train`
 *   时序滑窗   TCN   `/collect-seq`      + `/train-seq`
 *   连续句子   CTC   `/collect-sentence` + `/train-sentence`
 *
 * 之前这里写的是「采集」「训练」「去训练手语模型」「再去采集」，站在页面上看不出
 * 自己在哪条链路上，所以标题也一并写死成"静态采集/时序训练"。用词与 `/translate`
 * 的 MODE 开关保持一致，不要再引入第四套叫法。
 *
 * ⚠ **句子那条不是并列的第三条，是 Y 形。** 句子训练同时吃句子录制和时序词录制
 * （类别表从数据集全部标签推、合成句由词录制拼出来，见 train_seq.py 的 train_ctc）。
 * 反过来不成立：孤立词训练会用 `isSentenceSample` 把句子样本滤掉。
 * 所以句子采集页的主推出口是 `/train-sentence`，**不能是 `/train-seq`** ——
 * 点过去训不到任何句子数据，而页面上看不出这件事。
 */
const STEPS: Record<string, StepEntry> = {
  // 三条链路的入口都列出来。少列一条的表现是"那条路在这页上根本进不去，
  // 得退回首页再找" —— 体检做完就是要去采集，这里是最常被点的地方
  /*
   * 顺序是**静态 → 时序 → 句子**，按链路复杂度递增排，和首页流程图第 2 步里三条
   * 的排法一致。primary 高亮给时序（最常用的那条），但它不排第一 —— 高亮和顺序
   * 是两件事，为了把常用项挪到最左而打乱三条的固定次序，换页时就得重新找。
   */
  "/mocap": {
    step: 1,
    title: "准备",
    next: [
      { href: "/collect", label: "去静态采集" },
      { href: "/collect-seq", label: "去时序采集", primary: true },
      { href: "/collect-sentence", label: "去句子采集" },
    ],
  },
  "/collect": {
    step: 2,
    title: "静态采集",
    next: [{ href: "/train", label: "去静态训练", primary: true }],
  },
  "/collect-seq": {
    step: 2,
    title: "时序采集",
    next: [
      { href: "/train-seq", label: "去时序训练", primary: true },
      { href: "/collect-sentence", label: "去句子采集" },
    ],
  },
  /*
   * 句子采集与 /collect-seq 同属第 2 步、同一个 store，区别只在一条录制里装几个词
   * （`isSentenceSample`）。所以不给它单开一步 —— 编号变了会让人以为"孤立词采完
   * 才能采句子"。
   *
   * 但**出口不一样**：句子录制只喂 `/train-sentence`。这里原来主推 `/train-seq`，
   * 那是错的（见文件头的 ⚠）。次要出口留 `/collect-seq`，因为句子训练确实还要
   * 词录制才跑得起来 —— 采完句子发现训不动，多半是词数据不够。
   */
  "/collect-sentence": {
    step: 2,
    title: "句子采集",
    next: [
      { href: "/train-sentence", label: "去句子训练", primary: true },
      { href: "/collect-seq", label: "去时序采集" },
    ],
  },
  "/train": {
    step: 3,
    title: "静态训练",
    next: [{ href: "/translate", label: "去使用", primary: true }],
  },
  "/train-seq": {
    step: 3,
    title: "时序训练",
    next: [{ href: "/translate", label: "去使用", primary: true }],
  },
  /*
   * 表里少了这一条的表现是：整个 StepNav 在这页上渲染 null（查表落空那一支），
   * 于是页面既没有步号也没有出口 —— 而它看起来只是"这页头上比别的页空一点"。
   */
  /*
   * `?mode=sentence`：/translate 落地默认那一档是按"哪个模型更新"选的，而它只认识
   * 静态和时序 —— 句子模型不在 IndexedDB 里（Python 导出的构建产物），查不到。
   * 不带这个参数的话，从句子训练点「去使用」会落到时序滑窗档上。
   */
  "/train-sentence": {
    step: 3,
    title: "句子训练",
    next: [{ href: "/translate?mode=sentence", label: "去使用", primary: true }],
  },
  /*
   * 最后一步没有出口。`next: []` 是有意的，不是漏填 —— 表里保留这一条，
   * STEP 4/4 和左右手两个点才还在（查表落空的话整个组件渲染 null）。
   *
   * 这里以前挂着「再去时序采集」。往回走不该由流程条推：想加数据的人本来就会
   * 从首页或侧栏进采集页，而一个指回第 2 步的箭头会让第 4 步看起来还没走完。
   */
  "/translate": {
    step: 4,
    title: "使用",
    next: [],
  },
};

function HandDot({
  label,
  connected,
  fps,
}: {
  label: string;
  connected: boolean;
  fps: number;
}) {
  return (
    <span
      className="font-mono text-[10px]"
      style={{ color: connected ? "var(--hud-ok)" : "var(--hud-dim)" }}
      title={connected ? `${label} 已连接 ${fps}Hz` : `${label} 未连接`}
    >
      {label}
      <span className="ml-0.5">{connected ? fps : "—"}</span>
    </span>
  );
}

export default function StepNav() {
  const [location] = useLocation();
  const { left, right } = useGloves();

  const entry = STEPS[location];
  if (!entry) return null;

  const anyConnected = left.isConnected || right.isConnected;
  // 第 1 步本身就是连接页，不给"去第 1 步"的链接
  const showConnectHint = entry.step > 1 && !anyConnected;

  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[9px] text-[var(--hud-dim)] tracking-wider whitespace-nowrap">
        STEP{" "}
        <span className="text-[var(--hud-accent)]">
          {entry.step}/{TOTAL_STEPS}
        </span>{" "}
        {entry.title}
      </span>

      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <Hand
          className="w-3 h-3"
          style={{ color: anyConnected ? "var(--hud-ok)" : "var(--hud-dim)" }}
        />
        <HandDot
          label="LH"
          connected={left.isConnected}
          fps={left.gloveFps}
        />
        <HandDot
          label="RH"
          connected={right.isConnected}
          fps={right.gloveFps}
        />
      </span>

      {showConnectHint && (
        <Link
          href="/mocap"
          className="font-mono text-[10px] text-[var(--hud-warn)] hover:underline whitespace-nowrap"
        >
          ← 去第 1 步连接
        </Link>
      )}

      {entry.next.map((n) => (
        <Link
          key={n.href}
          href={n.href}
          className={`px-2 py-1 rounded-sm border font-mono text-[10px] flex items-center gap-1 whitespace-nowrap transition-colors ${
            n.primary
              ? "border-[var(--hud-accent)] text-[var(--hud-accent)] hover:bg-[var(--hud-line)]"
              : "border-[var(--hud-line)] text-[var(--hud-dim)] hover:text-[var(--hud-accent)]"
          }`}
        >
          {n.label}
          <ArrowRight className="w-3 h-3" />
        </Link>
      ))}
    </div>
  );
}
