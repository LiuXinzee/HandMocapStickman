/*
 * GlovePanel — 手套连接状态与传感器数据面板
 * DESIGN: Cyberpunk HUD 风格
 *
 * 使用 sensorMapping.ts 中的索引映射，将原始 256 字节数据
 * 按照规格书定义的物理位置重新排列后显示。
 */
import { GloveFrame } from "@/hooks/useGloveSerial";
import {
  remapSensorData,
  getBendValues,
  getFingerPressures,
  LEFT_HAND_REGIONS,
  RIGHT_HAND_REGIONS,
} from "@/lib/sensorMapping";
import { useMemo } from "react";

interface GlovePanelProps {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  gloveFps: number;
  gloveFrameCount: number;
  latestFrame: GloveFrame | null;
  onConnect: () => void;
  onDisconnect: () => void;
  /** 面板标题，默认 "GLOVE DATA"（双手场景用 "左手 GLOVE" / "右手 GLOVE"） */
  title?: string;
  /** 连接按钮文案，默认 "连接手套" */
  connectLabel?: string;
}

export default function GlovePanel({
  isConnected,
  isConnecting,
  error,
  gloveFps,
  gloveFrameCount,
  latestFrame,
  onConnect,
  onDisconnect,
  title = "GLOVE DATA",
  connectLabel = "连接手套",
}: GlovePanelProps) {
  // 重映射后的传感器数据 (137点物理顺序)
  const mappedData = useMemo(() => {
    if (!latestFrame) return null;
    return remapSensorData(latestFrame.sensor_data, latestFrame.hand);
  }, [latestFrame]);

  // 各手指压力
  const fingerPressures = useMemo(() => {
    if (!latestFrame) return null;
    return getFingerPressures(latestFrame.sensor_data, latestFrame.hand);
  }, [latestFrame]);

  // 弯折值
  const bendValues = useMemo(() => {
    if (!latestFrame) return null;
    return getBendValues(latestFrame.sensor_data, latestFrame.hand);
  }, [latestFrame]);

  // 传感器摘要（基于有效传感点）
  const sensorSummary = useMemo(() => {
    if (!mappedData) return null;
    const sum = mappedData.reduce((a, b) => a + b, 0);
    const avg = sum / mappedData.length;
    const max = Math.max(...mappedData);
    const active = mappedData.filter((v) => v > 20).length;
    return { avg: avg.toFixed(1), max, active };
  }, [mappedData]);

  // 四元数显示
  const quatStr = useMemo(() => {
    if (!latestFrame) return null;
    const [w, x, y, z] = latestFrame.quaternion;
    return `w=${w.toFixed(3)} x=${x.toFixed(3)} y=${y.toFixed(3)} z=${z.toFixed(3)}`;
  }, [latestFrame]);

  // 区域标签
  const regions = latestFrame?.hand === 0x01 ? LEFT_HAND_REGIONS : RIGHT_HAND_REGIONS;

  return (
    <div className="space-y-2">
      <PanelHeader title={title} />

      {/* 连接状态 */}
      <div className="space-y-1.5">
        <DataRow
          label="STATUS"
          value={
            isConnected
              ? "CONNECTED"
              : isConnecting
              ? "CONNECTING..."
              : "DISCONNECTED"
          }
          color={isConnected ? "var(--hud-ok)" : isConnecting ? "var(--hud-warn)" : "var(--hud-dim)"}
        />
        {isConnected && latestFrame && (
          <>
            <DataRow label="HAND" value={latestFrame.handLabel} color="var(--hud-wrist)" />
            <DataRow label="FPS" value={String(gloveFps)} color="var(--hud-accent)" />
            <DataRow
              label="FRAMES"
              value={gloveFrameCount.toLocaleString()}
              color="var(--hud-accent)"
            />
            <DataRow label="VALID PTS" value="137/256" color="var(--hud-dim)" />
          </>
        )}
      </div>

      {/* 连接/断开按钮 */}
      <div className="pt-1">
        {!isConnected ? (
          <button
            onClick={onConnect}
            disabled={isConnecting}
            className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5 disabled:opacity-50"
          >
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{ backgroundColor: isConnecting ? "var(--hud-warn)" : "var(--hud-ok)" }}
            />
            {isConnecting ? "连接中..." : connectLabel}
          </button>
        ) : (
          <button
            onClick={onDisconnect}
            className="w-full cyber-btn cyber-btn-accent px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--hud-err)]" />
            断开手套
          </button>
        )}
      </div>

      {/* 错误信息 */}
      {error && (
        <div className="text-[9px] text-[var(--hud-err)] font-mono px-1 py-1 border border-[#e11d48]/20 rounded-sm bg-[#e11d48]/5">
          {error}
        </div>
      )}

      {/* 手指压力 */}
      {isConnected && fingerPressures && (
        <div className="space-y-1.5 pt-1 border-t border-[#1677ff]/10">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            Finger Pressure
          </div>
          <FingerBar label="拇指" value={fingerPressures.thumb} color="var(--hud-f1)" />
          <FingerBar label="食指" value={fingerPressures.index} color="var(--hud-f2)" />
          <FingerBar label="中指" value={fingerPressures.middle} color="var(--hud-f3)" />
          <FingerBar label="无名" value={fingerPressures.ring} color="var(--hud-f4)" />
          <FingerBar label="小指" value={fingerPressures.pinky} color="var(--hud-f5)" />
        </div>
      )}

      {/* 弯折传感器 */}
      {isConnected && bendValues && (
        <div className="space-y-1.5 pt-1 border-t border-[#1677ff]/10">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            Bend Sensors
          </div>
          {latestFrame?.hand === 0x01 ? (
            <>
              <FingerBar label="小指弯" value={bendValues[0]} color="var(--hud-f5)" />
              <FingerBar label="无名弯" value={bendValues[1]} color="var(--hud-f4)" />
              <FingerBar label="中指弯" value={bendValues[2]} color="var(--hud-f3)" />
              <FingerBar label="食指弯" value={bendValues[3]} color="var(--hud-f2)" />
              <FingerBar label="拇指弯" value={bendValues[4]} color="var(--hud-f1)" />
            </>
          ) : (
            <>
              <FingerBar label="拇指弯" value={bendValues[0]} color="var(--hud-f1)" />
              <FingerBar label="食指弯" value={bendValues[1]} color="var(--hud-f2)" />
              <FingerBar label="中指弯" value={bendValues[2]} color="var(--hud-f3)" />
              <FingerBar label="无名弯" value={bendValues[3]} color="var(--hud-f4)" />
              <FingerBar label="小指弯" value={bendValues[4]} color="var(--hud-f5)" />
            </>
          )}
        </div>
      )}

      {/* 传感器摘要 */}
      {isConnected && sensorSummary && (
        <div className="space-y-1.5 pt-1 border-t border-[#1677ff]/10">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            Sensor Summary (137pts)
          </div>
          <DataRow label="AVG" value={sensorSummary.avg} color="var(--hud-accent)" />
          <DataRow label="MAX" value={String(sensorSummary.max)} color="var(--hud-err)" />
          <DataRow
            label="ACTIVE"
            value={`${sensorSummary.active}/137`}
            color="var(--hud-ok)"
          />
        </div>
      )}

      {/* 四元数 */}
      {isConnected && quatStr && (
        <div className="space-y-1 pt-1 border-t border-[#1677ff]/10">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            IMU Quaternion
          </div>
          <div
            className="text-[9px] font-mono break-all"
            style={{ color: "var(--hud-soft)" }}
          >
            {quatStr}
          </div>
        </div>
      )}

      {/* 加速度 / 姿态角（仅带加速度手套 296B 帧） */}
      {isConnected && latestFrame?.acceleration && (
        <div className="space-y-1 pt-1 border-t border-[#1677ff]/10">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
            Acceleration
          </div>
          <div className="text-[9px] font-mono break-all" style={{ color: "var(--hud-soft)" }}>
            {`x=${latestFrame.acceleration[0].toFixed(3)} y=${latestFrame.acceleration[1].toFixed(3)} z=${latestFrame.acceleration[2].toFixed(3)}`}
          </div>
          {latestFrame.attitude && (
            <>
              <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider">
                Attitude (deg)
              </div>
              <div className="text-[9px] font-mono break-all" style={{ color: "var(--hud-soft)" }}>
                {`yaw=${latestFrame.attitude[0].toFixed(1)} roll=${latestFrame.attitude[1].toFixed(1)} pitch=${latestFrame.attitude[2].toFixed(1)}`}
              </div>
            </>
          )}
        </div>
      )}

      {/* 区域热力图 */}
      {isConnected && mappedData && (
        <div className="pt-1 border-t border-[#1677ff]/10">
          <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider mb-1">
            Region Heatmap
          </div>
          <RegionHeatmap data={mappedData} regions={regions} />
        </div>
      )}
    </div>
  );
}

/** 按区域显示的热力条 */
function RegionHeatmap({
  data,
  regions,
}: {
  data: number[];
  regions: typeof LEFT_HAND_REGIONS;
}) {
  return (
    <div className="space-y-1">
      {regions.map((region) => {
        const regionData = data.slice(region.startIdx, region.endIdx);
        const avg =
          regionData.reduce((a, b) => a + b, 0) / regionData.length;
        const max = Math.max(...regionData);
        const intensity = Math.min(avg / 150, 1);

        return (
          <div key={region.name} className="flex items-center gap-1.5">
            <span
              className="text-[8px] font-mono w-8 text-right"
              style={{ color: region.color }}
            >
              {region.name}
            </span>
            <div className="flex-1 h-2 bg-[var(--hud-page)] rounded-sm overflow-hidden relative">
              <div
                className="h-full rounded-sm transition-all duration-100"
                style={{
                  width: `${intensity * 100}%`,
                  backgroundColor: region.color,
                  opacity: 0.7 + intensity * 0.3,
                }}
              />
            </div>
            <span className="text-[8px] font-mono w-6 text-right" style={{ color: "var(--hud-dim)" }}>
              {max}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** 手指压力条 */
function FingerBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const intensity = Math.min(value / 200, 1);
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="text-[8px] font-mono w-8 text-right"
        style={{ color }}
      >
        {label}
      </span>
      <div className="flex-1 h-1.5 bg-[var(--hud-page)] rounded-sm overflow-hidden">
        <div
          className="h-full rounded-sm transition-all duration-100"
          style={{
            width: `${intensity * 100}%`,
            backgroundColor: color,
            opacity: 0.7 + intensity * 0.3,
          }}
        />
      </div>
      <span className="text-[8px] font-mono w-5 text-right" style={{ color: "var(--hud-dim)" }}>
        {Math.round(value)}
      </span>
    </div>
  );
}

function PanelHeader({ title }: { title: string }) {
  return (
    <div className="flex items-center gap-2 pb-1 border-b border-[#1677ff]/15">
      <div className="w-1 h-3 bg-[var(--hud-ok)] rounded-full shadow-[0_0_4px_rgba(0,229,160,0.6)]" />
      <span
        className="text-[10px] font-bold tracking-widest"
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          color: "var(--hud-ok)",
        }}
      >
        {title}
      </span>
    </div>
  );
}

function DataRow({
  label,
  value,
  color = "var(--hud-accent)",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between text-[10px] font-mono">
      <span style={{ color: "var(--hud-dim)" }}>{label}</span>
      {/* 浅色底不做辉光：白底描不出光晕，只会糊出一圈脏边 */}
      <span style={{ color }}>{value}</span>
    </div>
  );
}
