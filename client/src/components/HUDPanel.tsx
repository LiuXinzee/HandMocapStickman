/*
 * HUDPanel — 赛博朋克 HUD 数据面板
 * DESIGN: Cyberpunk HUD — 霓虹色数据面板
 */
import type { HandResult } from "@/hooks/useHandTracking";
import { useEffect, useRef, useState } from "react";

interface HUDPanelProps {
  fps: number;
  isRunning: boolean;
  isLoading: boolean;
  handResults: HandResult | null;
}

export default function HUDPanel({
  fps,
  isRunning,
  isLoading,
  handResults,
}: HUDPanelProps) {
  const handsDetected = handResults?.landmarks.length ?? 0;
  const handedness = handResults?.handedness ?? [];
  const surfaces = handResults?.surfaces ?? [];

  // FPS 历史记录（用于迷你图表）
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);
  const fpsRef = useRef(fps);
  fpsRef.current = fps;

  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setFpsHistory(prev => {
        const next = [...prev, fpsRef.current];
        return next.length > 30 ? next.slice(-30) : next;
      });
    }, 500);
    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* 系统状态 */}
      <div className="cyber-panel p-3 rounded-sm">
        <PanelHeader
          label="System Status"
          statusColor={
            isRunning ? "var(--hud-accent)" : isLoading ? "var(--hud-warn)" : "var(--hud-err)"
          }
        />
        <div className="space-y-1.5 font-mono text-[11px] mt-2">
          <DataRow
            label="STATUS"
            value={isRunning ? "ACTIVE" : isLoading ? "LOADING" : "STANDBY"}
            valueColor={
              isRunning ? "var(--hud-accent)" : isLoading ? "var(--hud-warn)" : "var(--hud-err)"
            }
          />
          <DataRow label="FPS" value={isRunning ? fps.toString() : "--"} />
          <DataRow
            label="MODE"
            value={
              handResults?.detectionSource === "glove-enhanced"
                ? "GLOVE BOOST"
                : "STANDARD"
            }
            valueColor={
              handResults?.detectionSource === "glove-enhanced"
                ? "var(--hud-warn)"
                : "var(--hud-dim)"
            }
          />
          <DataRow label="MODEL" value="MEDIAPIPE V2" />
          <DataRow label="COMPLEXITY" value="1 (FULL)" />
          <DataRow label="MAX HANDS" value="2" />
        </div>

        {/* FPS 迷你图 */}
        {isRunning && fpsHistory.length > 2 && (
          <div className="mt-2 h-8 flex items-end gap-px">
            {fpsHistory.slice(-20).map((f, i) => {
              const maxFps = Math.max(...fpsHistory.slice(-20), 1);
              const height = (f / maxFps) * 100;
              return (
                <div
                  key={i}
                  className="flex-1 rounded-t-sm"
                  style={{
                    height: `${Math.max(height, 5)}%`,
                    backgroundColor:
                      f > 20
                        ? "rgba(0, 240, 255, 0.4)"
                        : f > 10
                          ? "rgba(245, 158, 11, 0.4)"
                          : "rgba(255, 45, 123, 0.4)",
                    transition: "height 0.3s ease-out",
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* 检测结果 */}
      <div className="cyber-panel p-3 rounded-sm">
        <PanelHeader label="Detection" />
        <div className="space-y-1.5 font-mono text-[11px] mt-2">
          <DataRow
            label="HANDS"
            value={handsDetected.toString()}
            valueColor={handsDetected > 0 ? "var(--hud-accent)" : "var(--hud-faint)"}
          />
          {handedness.map((h, i) => (
            <div key={i} className="space-y-1.5">
              <DataRow
                label={`HAND ${i + 1}`}
                value={h.toUpperCase()}
                valueColor="var(--hud-ok)"
              />
              {handResults && handResults.gloveConfidences[i] >= 0.45 && (
                <DataRow
                  label={`SIDE ${i + 1}`}
                  value={surfaceLabel(surfaces[i])}
                  valueColor={surfaces[i] === "palm" ? "var(--hud-text)" : "var(--hud-violet)"}
                />
              )}
            </div>
          ))}
          <DataRow
            label="JOINTS"
            value={handsDetected > 0 ? `${handsDetected * 21}` : "0"}
          />
          <DataRow
            label="BONES"
            value={handsDetected > 0 ? `${handsDetected * 20}` : "0"}
          />
        </div>
      </div>

      {/* 关节坐标 */}
      {handResults && handResults.landmarks.length > 0 && (
        <div className="cyber-panel p-3 rounded-sm">
          <PanelHeader label="Fingertip Coords" />
          <div className="space-y-1 font-mono text-[10px] mt-2">
            {[
              { name: "THUMB", idx: 4, color: "var(--hud-accent)" },
              { name: "INDEX", idx: 8, color: "var(--hud-ok)" },
              { name: "MIDDLE", idx: 12, color: "var(--hud-violet)" },
              { name: "RING", idx: 16, color: "var(--hud-warn)" },
              { name: "PINKY", idx: 20, color: "var(--hud-err)" },
            ].map(({ name, idx, color }) => {
              const lm = handResults.landmarks[0][idx];
              return (
                <div key={name} className="flex justify-between items-center">
                  <span style={{ color }} className="w-14">
                    {name}
                  </span>
                  <span className="text-[var(--hud-dim)] tabular-nums">
                    {(1 - lm.x).toFixed(2)} {lm.y.toFixed(2)} {lm.z.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 颜色图例 */}
      <div className="cyber-panel p-3 rounded-sm">
        <PanelHeader label="Joint Map" />
        <div className="space-y-1.5 text-[10px] font-mono mt-2">
          {[
            { name: "THUMB", color: "var(--hud-accent)", joints: "0-4" },
            { name: "INDEX", color: "var(--hud-ok)", joints: "5-8" },
            { name: "MIDDLE", color: "var(--hud-violet)", joints: "9-12" },
            { name: "RING", color: "var(--hud-warn)", joints: "13-16" },
            { name: "PINKY", color: "var(--hud-err)", joints: "17-20" },
          ].map(({ name, color, joints }) => (
            <div key={name} className="flex items-center gap-2">
              <div
                className="w-3 h-0.5 rounded-full"
                style={{
                  backgroundColor: color,
                }}
              />
              <span style={{ color }}>{name}</span>
              <span className="text-[var(--hud-faint)] ml-auto">{joints}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function surfaceLabel(surface: HandResult["surfaces"][number] | undefined) {
  if (surface === "palm") return "PALM / 银色";
  if (surface === "back") return "BACK / 黑色";
  return "UNKNOWN";
}

function PanelHeader({
  label,
  statusColor,
}: {
  label: string;
  statusColor?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      {statusColor && (
        <div
          className="w-1.5 h-1.5 rounded-full"
          style={{
            backgroundColor: statusColor,
            boxShadow: `0 0 6px ${statusColor}`,
          }}
        />
      )}
      <span className="cyber-text text-[10px] uppercase tracking-widest">
        {label}
      </span>
    </div>
  );
}

function DataRow({
  label,
  value,
  valueColor = "var(--hud-dim)",
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[var(--hud-faint)]">{label}</span>
      <span className="tabular-nums" style={{ color: valueColor }}>
        {value}
      </span>
    </div>
  );
}
