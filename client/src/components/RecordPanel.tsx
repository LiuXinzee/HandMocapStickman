/*
 * RecordPanel — 同步录制控制面板
 * DESIGN: Cyberpunk HUD 风格
 * 功能: 开始/停止录制、显示录制状态、导出数据
 */
import { Circle, Download, Square, Trash2 } from "lucide-react";
import { useState } from "react";

interface RecordPanelProps {
  isRecording: boolean;
  recordingDuration: number;
  videoFrameCount: number;
  gloveFrameCount: number;
  gloveConnected: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onExportCSV: () => void;
  onExportJSON: () => void;
  onClear: () => void;
}

export default function RecordPanel({
  isRecording,
  recordingDuration,
  videoFrameCount,
  gloveFrameCount,
  gloveConnected,
  onStartRecording,
  onStopRecording,
  onExportCSV,
  onExportJSON,
  onClear,
}: RecordPanelProps) {
  const [showExportMenu, setShowExportMenu] = useState(false);

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const frac = Math.floor((ms % 1000) / 100);
    return `${minutes.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}.${frac}`;
  };

  const hasData = videoFrameCount > 0 || gloveFrameCount > 0;

  return (
    <div className="space-y-2">
      <PanelHeader title="SYNC RECORD" />

      {/* 录制状态 */}
      <div className="space-y-1.5">
        <DataRow
          label="STATUS"
          value={isRecording ? "● REC" : hasData ? "STOPPED" : "IDLE"}
          color={isRecording ? "#ff2d7b" : hasData ? "#f59e0b" : "#556677"}
        />
        {(isRecording || hasData) && (
          <>
            <DataRow
              label="DURATION"
              value={formatDuration(recordingDuration)}
              color="#00f0ff"
            />
            <DataRow
              label="VIDEO"
              value={`${videoFrameCount} frames`}
              color="#00f0ff"
            />
            <DataRow
              label="GLOVE"
              value={`${gloveFrameCount} frames`}
              color={gloveConnected ? "#00e5a0" : "#556677"}
            />
            {isRecording && videoFrameCount > 0 && (
              <DataRow
                label="V-FPS"
                value={`~${Math.round(
                  videoFrameCount / (recordingDuration / 1000 || 1)
                )}`}
                color="#556677"
              />
            )}
            {isRecording && gloveFrameCount > 0 && (
              <DataRow
                label="G-FPS"
                value={`~${Math.round(
                  gloveFrameCount / (recordingDuration / 1000 || 1)
                )}`}
                color="#556677"
              />
            )}
          </>
        )}
      </div>

      {/* 同步指示 */}
      {isRecording && (
        <div className="flex items-center gap-2 py-1 px-2 border border-[#ff2d7b]/30 rounded-sm bg-[#ff2d7b]/5">
          <div className="w-2 h-2 rounded-full bg-[#ff2d7b] animate-pulse shadow-[0_0_8px_rgba(255,45,123,0.6)]" />
          <span className="text-[9px] font-mono text-[#ff2d7b]">
            同步录制中...
          </span>
          {!gloveConnected && (
            <span className="text-[8px] font-mono text-[#f59e0b] ml-auto">
              仅视频
            </span>
          )}
        </div>
      )}

      {/* 控制按钮 */}
      <div className="flex gap-1.5 pt-1">
        {!isRecording ? (
          <button
            onClick={onStartRecording}
            className="flex-1 cyber-btn px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            style={{
              borderColor: "rgba(255, 45, 123, 0.4)",
              color: "#ff2d7b",
            }}
          >
            <Circle className="w-3 h-3 fill-current" />
            录制
          </button>
        ) : (
          <button
            onClick={onStopRecording}
            className="flex-1 cyber-btn cyber-btn-accent px-2 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
          >
            <Square className="w-3 h-3 fill-current" />
            停止
          </button>
        )}

        {hasData && !isRecording && (
          <>
            <div className="relative">
              <button
                onClick={() => setShowExportMenu(!showExportMenu)}
                className="cyber-btn px-2 py-1.5 rounded-sm text-[10px] flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
                导出
              </button>
              {showExportMenu && (
                <div
                  className="absolute bottom-full left-0 mb-1 w-32 py-1 rounded-sm border border-[#00f0ff]/30 z-50"
                  style={{ backgroundColor: "rgba(10, 14, 26, 0.98)" }}
                >
                  <button
                    onClick={() => {
                      onExportCSV();
                      setShowExportMenu(false);
                    }}
                    className="w-full text-left px-3 py-1.5 text-[10px] font-mono text-[#8899aa] hover:text-[#00f0ff] hover:bg-[#00f0ff]/5 transition-colors"
                  >
                    同步 CSV
                  </button>
                  <button
                    onClick={() => {
                      onExportJSON();
                      setShowExportMenu(false);
                    }}
                    className="w-full text-left px-3 py-1.5 text-[10px] font-mono text-[#8899aa] hover:text-[#00f0ff] hover:bg-[#00f0ff]/5 transition-colors"
                  >
                    原始 JSON
                  </button>
                </div>
              )}
            </div>
            <button
              onClick={onClear}
              className="cyber-btn px-2 py-1.5 rounded-sm text-[10px] flex items-center gap-1"
              title="清除数据"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </>
        )}
      </div>

      {/* 提示信息 */}
      {!isRecording && !hasData && (
        <div className="text-[9px] font-mono text-[#334455] leading-relaxed px-1">
          点击录制按钮开始同步采集视频关键点和手套传感器数据。
          {!gloveConnected && (
            <span className="text-[#f59e0b]">
              {" "}
              未连接手套，将仅录制视频数据。
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function PanelHeader({ title }: { title: string }) {
  return (
    <div className="flex items-center gap-2 pb-1 border-b border-[#00f0ff]/15">
      <div className="w-1 h-3 bg-[#ff2d7b] rounded-full shadow-[0_0_4px_rgba(255,45,123,0.6)]" />
      <span
        className="text-[10px] font-bold tracking-widest"
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          color: "#ff2d7b",
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
  color = "#00f0ff",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between text-[10px] font-mono">
      <span style={{ color: "#556677" }}>{label}</span>
      <span style={{ color, textShadow: `0 0 6px ${color}40` }}>{value}</span>
    </div>
  );
}
