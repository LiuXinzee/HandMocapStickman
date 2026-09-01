/*
 * SampleQc —— 录完一条之后的**质检**显示件，孤立词采集页与句子采集页共用。
 *
 * 这些件原来是 `SequenceCollect.tsx` 的文件内私有函数。搬出来是因为句子采集页
 * （`CollectSentence.tsx`）要显示同样的东西，而"照抄一份"在质检这件事上特别糟：
 * 抄的那份改了门限、少了一行提示，采集时人看到的就是"这条没问题"，
 * 而废样本会一路进训练集 —— 它不报错，只让某个词莫名变差。
 *
 * 这里只放**纯显示 + 纯计算**，不碰采集流程（孤立词是空格键起停、句子是自动收句，
 * 两套流程本来就不一样，合起来只会互相牵制）。
 */
import { useMemo } from "react";
import {
  analyzeSequenceImu,
  worstVerdict,
  type ImuHealthReport,
  type ImuVerdict,
} from "@/lib/imuHealth";
import type { SequenceSample } from "@/lib/datasetStore";

/** verdict → 中文 / 颜色。ok 用青色（与其他指标一致），warn 琥珀，bad 品红 */
export const IMU_VERDICT_TEXT: Record<ImuVerdict, string> = {
  ok: "正常",
  warn: "可疑",
  bad: "不可用",
  unknown: "未知",
};
export const IMU_VERDICT_COLOR: Record<ImuVerdict, string> = {
  ok: "#00f0ff",
  warn: "#f59e0b",
  bad: "#ff2d7b",
  unknown: "#556677",
};
const IMU_RANK: Record<ImuVerdict, number> = { ok: 0, unknown: 1, warn: 2, bad: 3 };

export function Metric({
  label,
  value,
  warn,
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div>
      <div className="text-[#556677] uppercase text-[8px]">{label}</div>
      <div style={{ color: warn ? "#f59e0b" : "#00f0ff" }}>{value}</div>
    </div>
  );
}

/**
 * 运动能量折线。静止误录会是一条贴地直线——这是最快的质检手段，
 * 比事后看混淆矩阵早了整整一轮训练。
 */
export function EnergyChart({ energy }: { energy: Float32Array }) {
  const W = 600;
  const H = 60;
  const max = Math.max(...Array.from(energy), 1e-6);
  const pts = Array.from(energy)
    .map((v, i) => {
      const x = (i / Math.max(1, energy.length - 1)) * W;
      const y = H - (v / max) * (H - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const nearlyStatic = max < 0.002;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: H }}
        preserveAspectRatio="none"
      >
        <polyline
          points={pts}
          fill="none"
          stroke={nearlyStatic ? "#f59e0b" : "#00f0ff"}
          strokeWidth="1.5"
        />
      </svg>
      {nearlyStatic && (
        <div className="text-[9px] text-[#f59e0b] font-mono mt-1">
          几乎没有运动 —— 如果这是动态词，很可能录废了
        </div>
      )}
    </div>
  );
}

export interface SampleImuVerdict {
  verdict: ImuVerdict;
  /** 两只手里**更差**的那份报告，用来显示倾角数值和中文原因 */
  detail: ImuHealthReport;
}

/**
 * 这条样本的 IMU 健康度。录制器停止时已 post-hoc 算好写进样本；
 * 万一拿到的是没有该字段的旧样本，就地从 leftImu/rightImu 重算 —— 数据本来就都在。
 *
 * 两只手取更差的那份：一只手的陀螺废了，这条样本的四元数通道就已经不能用了。
 */
export function sampleImuVerdict(
  sample: SequenceSample | null
): SampleImuVerdict | null {
  if (!sample) return null;
  const h =
    sample.imuHealth ??
    {
      // 没戴的那只手给 null，而不是让它产出一份 unknown 报告去污染结论
      left: sample.leftImu
        ? analyzeSequenceImu(sample.leftImu, sample.frameCount, sample.timestamps)
        : null,
      right: sample.rightImu
        ? analyzeSequenceImu(sample.rightImu, sample.frameCount, sample.timestamps)
        : null,
    };
  const reports = [h.left, h.right].filter(Boolean) as ImuHealthReport[];
  if (reports.length === 0) return null;
  const detail = reports.reduce((worst, r) =>
    IMU_RANK[r.verdict] > IMU_RANK[worst.verdict] ||
    (IMU_RANK[r.verdict] === IMU_RANK[worst.verdict] &&
      r.tiltInconsistencyDeg > worst.tiltInconsistencyDeg)
      ? r
      : worst
  );
  return { verdict: worstVerdict(h.left, h.right), detail };
}

/** `sampleImuVerdict` 的 memo 版（样本没换就不重算，重算要扫整段 IMU） */
export function useSampleImuVerdict(
  sample: SequenceSample | null
): SampleImuVerdict | null {
  return useMemo(() => sampleImuVerdict(sample), [sample]);
}

/**
 * IMU 陀螺漂移那一行。**这一行存在的理由**：漂移状态下录的序列，帧数、时长、
 * 运动能量曲线全都正常，四元数通道却已经废了 —— 不显示出来就没有任何一处能发现。
 */
export function ImuHealthLine({ imu }: { imu: SampleImuVerdict }) {
  return (
    <div className="mt-2 pt-2 border-t border-[#00f0ff]/10 font-mono">
      <div className="text-[10px]" style={{ color: IMU_VERDICT_COLOR[imu.verdict] }}>
        IMU 陀螺漂移：{IMU_VERDICT_TEXT[imu.verdict]}
        {imu.detail.usableFrames > 0 && (
          <span className="text-[#556677] ml-2">
            倾角偏差 {imu.detail.tiltInconsistencyDeg.toFixed(1)}° · 可用帧{" "}
            {imu.detail.usableFrames}
          </span>
        )}
      </div>
      {imu.verdict !== "ok" && (
        <div className="text-[9px] text-[#556677] mt-0.5 leading-relaxed">
          {imu.detail.reason}
          {imu.verdict === "bad" && (
            <span className="text-[#ff2d7b]">
              {" "}
              建议删掉重录：把手套放平不动，短按主控按键做陀螺校准后再录。
            </span>
          )}
        </div>
      )}
    </div>
  );
}
