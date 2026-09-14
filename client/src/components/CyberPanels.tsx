/*
 * CyberPanels — 训练页共用的 HUD 小件。
 *
 * 原先住在 `TrainSequence.tsx` 里。`TrainSentence.tsx` 要用同一套视觉语言，
 * 所以**移**过来而不是复制 —— 复制成两份的话，改一处配色就会出现两页不一致，
 * 而这类不一致没有任何测试会发现。
 */
import type React from "react";

export function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 pb-1 border-b border-[#1677ff]/15">
        <div className="w-1 h-3 bg-[var(--hud-accent)] rounded-full shadow-[0_0_4px_rgba(0,240,255,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[var(--hud-accent)] font-mono">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

export function DataRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex justify-between text-[10px] font-mono">
      <span className="text-[var(--hud-dim)]">{label}</span>
      <span style={{ color }}>{value}</span>
    </div>
  );
}

export function ParamInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  isFloat = false,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  isFloat?: boolean;
}) {
  return (
    <div className="flex items-center justify-between text-[10px] font-mono">
      <span className="text-[var(--hud-dim)]">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => {
          const v = isFloat
            ? parseFloat(e.target.value)
            : parseInt(e.target.value);
          if (!isNaN(v) && v >= min && v <= max) onChange(v);
        }}
        min={min}
        max={max}
        step={step}
        className="w-16 bg-[var(--hud-track)] border border-[#1677ff]/20 rounded-sm px-1.5 py-0.5 text-[var(--hud-accent)] text-center text-[10px]"
      />
    </div>
  );
}

export function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="cyber-panel p-2 rounded-sm text-center">
      <div className="text-[8px] font-mono text-[var(--hud-dim)] uppercase">
        {label}
      </div>
      <div
        className="text-sm font-bold font-mono mt-0.5"
        /* 浅色底不做辉光：白底描不出光晕，只会糊出一圈脏边 */
        style={{ color }}
      >
        {value}
      </div>
    </div>
  );
}

/** 勾选开关。ParamInput 的布尔版，两页的 flag 都用它 */
export function ToggleRow({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <label className="flex items-start justify-between gap-2 text-[10px] font-mono cursor-pointer">
      <span className="text-[var(--hud-dim)] leading-relaxed">
        {label}
        {hint && (
          <span className="block text-[9px] text-[var(--hud-faint)]">{hint}</span>
        )}
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 accent-[var(--hud-accent)] shrink-0"
      />
    </label>
  );
}
