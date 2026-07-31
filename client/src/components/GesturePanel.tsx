/*
 * GesturePanel — 手势识别结果显示面板
 * 显示当前检测到的手势和手指弯曲角度
 */
import type { HandResult } from "@/hooks/useHandTracking";
import { detectGesture, getFingerAngles } from "@/lib/gestureDetector";
import { useMemo } from "react";

interface GesturePanelProps {
  handResults: HandResult | null;
}

const FINGER_COLORS: Record<string, string> = {
  thumb: "#00f0ff",
  index: "#00e5a0",
  middle: "#a855f7",
  ring: "#f59e0b",
  pinky: "#ff2d7b",
};

const FINGER_LABELS: Record<string, string> = {
  thumb: "拇指",
  index: "食指",
  middle: "中指",
  ring: "无名指",
  pinky: "小指",
};

export default function GesturePanel({ handResults }: GesturePanelProps) {
  const gestureData = useMemo(() => {
    if (!handResults || handResults.landmarks.length === 0) return null;
    const landmarks = handResults.landmarks[0];
    const gesture = detectGesture(landmarks);
    const angles = getFingerAngles(landmarks);
    return { gesture, angles };
  }, [handResults]);

  if (!gestureData) return null;

  const { gesture, angles } = gestureData;

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* 手势识别 */}
      <div className="cyber-panel p-3 rounded-sm">
        <div className="flex items-center gap-2 mb-2">
          <span className="cyber-text text-[10px] uppercase tracking-widest">
            Gesture
          </span>
        </div>
        <div className="text-center py-2">
          <div
            className="text-lg font-bold"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "#00f0ff",
              textShadow: "0 0 12px rgba(0,240,255,0.5)",
            }}
          >
            {gesture.label}
          </div>
          <div className="text-[10px] text-[#556677] font-mono mt-1">
            {gesture.gesture} · {(gesture.confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* 手指弯曲角度 */}
      <div className="cyber-panel p-3 rounded-sm">
        <div className="flex items-center gap-2 mb-2">
          <span className="cyber-text text-[10px] uppercase tracking-widest">
            Flex Angles
          </span>
        </div>
        <div className="space-y-2">
          {Object.entries(angles).map(([finger, angle]) => {
            const color = FINGER_COLORS[finger];
            const normalizedAngle = Math.min(angle / 180, 1);
            return (
              <div key={finger}>
                <div className="flex justify-between items-center mb-0.5">
                  <span
                    className="text-[10px] font-mono"
                    style={{ color }}
                  >
                    {FINGER_LABELS[finger]}
                  </span>
                  <span className="text-[10px] font-mono text-[#8899aa]">
                    {angle.toFixed(0)}°
                  </span>
                </div>
                <div className="h-1 bg-[#1a1a2e] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-150"
                    style={{
                      width: `${normalizedAngle * 100}%`,
                      backgroundColor: color,
                      boxShadow: `0 0 6px ${color}60`,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
