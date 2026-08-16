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
 * 文案约定：**每一条都必须带「静态」或「时序」**。
 *
 * 第 2、3 步各有两条互不相干的链路——静态单帧（MLP，`/collect` + `/train`）与
 * 时序滑窗（TCN，`/collect-seq` + `/train-seq`）。两者数据集分开、模型分开、
 * localStorage 键也分开，走错一条不会报错，只会训出一个空的或喂错的模型。
 * 之前这里写的是「采集」「训练」「去训练手语模型」「再去采集」，
 * 站在页面上看不出自己在哪条链路上，所以标题也一并写死成"静态采集/时序训练"。
 * 用词与 `/translate` 的 MODE 开关保持一致，不要再引入第三套叫法。
 */
const STEPS: Record<string, StepEntry> = {
  "/mocap": {
    step: 1,
    title: "准备",
    next: [
      { href: "/collect-seq", label: "去时序采集", primary: true },
      { href: "/collect", label: "去静态采集" },
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
    next: [{ href: "/train-seq", label: "去时序训练", primary: true }],
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
  "/translate": {
    step: 4,
    title: "使用",
    next: [{ href: "/collect-seq", label: "再去时序采集" }],
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
      style={{ color: connected ? "#00e5a0" : "#556677" }}
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
      <span className="font-mono text-[9px] text-[#556677] tracking-wider whitespace-nowrap">
        STEP{" "}
        <span className="text-[#00f0ff]">
          {entry.step}/{TOTAL_STEPS}
        </span>{" "}
        {entry.title}
      </span>

      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <Hand
          className="w-3 h-3"
          style={{ color: anyConnected ? "#00e5a0" : "#556677" }}
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
          className="font-mono text-[10px] text-[#f59e0b] hover:underline whitespace-nowrap"
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
              ? "border-[#00f0ff]/50 text-[#00f0ff] hover:bg-[#00f0ff]/10"
              : "border-[#00f0ff]/15 text-[#556677] hover:text-[#00f0ff]"
          }`}
        >
          {n.label}
          <ArrowRight className="w-3 h-3" />
        </Link>
      ))}
    </div>
  );
}
