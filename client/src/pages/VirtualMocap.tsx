/*
 * VirtualMocap — 仅触觉驱动的虚拟火柴人动捕页面
 * DESIGN: Cyberpunk HUD 风格
 *
 * 功能:
 * 1. 连接手套，获取实时触觉数据
 * 2. 用骨架回归模型预测手部关键点坐标
 * 3. 在 Canvas 上渲染赛博朋克风格的火柴人骨架动画
 * 4. 无需摄像头，仅靠手套即可驱动
 */
import { useGloveSerial } from "@/hooks/useGloveSerial";
import {
  predictSkeleton,
  isSkeletonModelLoaded,
  loadSkeletonModelFromSaved,
} from "@/lib/skeletonModel";
import { getLatestSkeletonModel } from "@/lib/datasetStore";
import { FINGER_CONNECTION_GROUPS } from "@/hooks/useHandTracking";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Bone,
  Hand,
  Wifi,
  WifiOff,
  Activity,
} from "lucide-react";

const FINGER_COLORS: Record<string, string> = {
  thumb: "#00f0ff",
  index: "#00e5a0",
  middle: "#a855f7",
  ring: "#f59e0b",
  pinky: "#ff2d7b",
  palm: "#00f0ff",
};

const FINGER_GLOW_COLORS: Record<string, string> = {
  thumb: "rgba(0, 240, 255, 0.35)",
  index: "rgba(0, 229, 160, 0.35)",
  middle: "rgba(168, 85, 247, 0.35)",
  ring: "rgba(245, 158, 11, 0.35)",
  pinky: "rgba(255, 45, 123, 0.35)",
  palm: "rgba(0, 240, 255, 0.2)",
};

const CANVAS_W = 640;
const CANVAS_H = 480;

export default function VirtualMocap() {
  const {
    isConnected,
    isConnecting,
    error: gloveError,
    gloveFps,
    gloveFrameCount,
    latestFrame,
    latestFrameRef: gloveFrameRef, // 全速更新的 ref
    connect: connectGlove,
    disconnect: disconnectGlove,
  } = useGloveSerial();

  const [modelReady, setModelReady] = useState(isSkeletonModelLoaded());
  const [modelName, setModelName] = useState("");
  const [loadMsg, setLoadMsg] = useState("");
  const [predFps, setPredFps] = useState(0);
  const [confidence, setConfidence] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const prevLandmarksRef = useRef<{ x: number; y: number; z: number }[] | null>(
    null
  );
  const fpsCountRef = useRef({ count: 0, lastTime: performance.now() });
  // 使用 useGloveSerial 暴露的全速 ref，不再自己维护

  // 自动加载最新骨架模型
  useEffect(() => {
    if (!isSkeletonModelLoaded()) {
      (async () => {
        setLoadMsg("正在加载骨架模型...");
        const saved = await getLatestSkeletonModel();
        if (saved) {
          try {
            await loadSkeletonModelFromSaved(saved);
            setModelReady(true);
            setModelName(saved.name);
            setLoadMsg("");
          } catch (e: any) {
            setLoadMsg(`模型加载失败: ${e.message}`);
          }
        } else {
          setLoadMsg("未找到骨架模型，请先训练");
        }
      })();
    } else {
      setModelReady(true);
    }
  }, []);

  // 平滑关键点
  const smoothLandmarks = useCallback(
    (current: { x: number; y: number; z: number }[]) => {
      const prev = prevLandmarksRef.current;
      if (!prev || prev.length !== current.length) {
        prevLandmarksRef.current = current;
        return current;
      }
      const alpha = 0.45; // 稍微更平滑
      const smoothed = current.map((lm, i) => ({
        x: prev[i].x + alpha * (lm.x - prev[i].x),
        y: prev[i].y + alpha * (lm.y - prev[i].y),
        z: prev[i].z + alpha * (lm.z - prev[i].z),
      }));
      prevLandmarksRef.current = smoothed;
      return smoothed;
    },
    []
  );

  // Canvas 渲染循环
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;

    const render = () => {
      const time = Date.now() / 1000;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

      // 背景
      ctx.fillStyle = "#0a0e1a";
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      drawGrid(ctx, CANVAS_W, CANVAS_H);
      drawScanLines(ctx, CANVAS_W, CANVAS_H);

      const frame = gloveFrameRef.current;
      let predicted = false;

      if (frame && modelReady && isSkeletonModelLoaded()) {
        const landmarks = predictSkeleton(
          frame.mapped_data,
          frame.quaternion
        );

        if (landmarks && landmarks.length === 21) {
          const smoothed = smoothLandmarks(landmarks);
          drawSkeleton(ctx, smoothed, CANVAS_W, CANVAS_H);
          drawJoints(ctx, smoothed, CANVAS_W, CANVAS_H);
          drawPalmCrosshair(ctx, smoothed, CANVAS_W, CANVAS_H);
          predicted = true;

          // 计算置信度（基于关键点分布的合理性）
          const spread = computeSpread(smoothed);
          setConfidence(Math.min(100, Math.max(0, spread)));

          // FPS 计算
          fpsCountRef.current.count++;
          const now = performance.now();
          if (now - fpsCountRef.current.lastTime > 1000) {
            setPredFps(fpsCountRef.current.count);
            fpsCountRef.current.count = 0;
            fpsCountRef.current.lastTime = now;
          }
        }
      }

      if (!predicted) {
        drawWaitingState(ctx, CANVAS_W, CANVAS_H, time);
      }

      drawHUDCorners(ctx, CANVAS_W, CANVAS_H);

      // HUD 标签
      ctx.save();
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = "rgba(245, 158, 11, 0.6)";
      ctx.fillText("VIRTUAL MOCAP — TACTILE ONLY", 12, 18);
      if (predicted) {
        ctx.fillStyle = "rgba(0, 229, 160, 0.5)";
        ctx.fillText(`PRED FPS: ${predFps}`, CANVAS_W - 100, 18);
      }
      ctx.restore();

      animRef.current = requestAnimationFrame(render);
    };

    render();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [modelReady, smoothLandmarks, predFps]);

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: "#0a0e1a" }}
    >
      {/* 顶部导航 */}
      <header className="h-12 flex items-center justify-between px-4 border-b border-[#00f0ff]/15 shrink-0">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[#00f0ff]/20" />
          <span className="text-xs font-bold tracking-widest text-[#f59e0b] font-mono">
            VIRTUAL MOCAP
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            TACTILE → SKELETON RENDERING
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          {modelReady && (
            <span className="text-[#f59e0b] flex items-center gap-1">
              <Bone className="w-3 h-3" />
              MODEL: {modelName || "LOADED"}
            </span>
          )}
          <span
            className={`flex items-center gap-1 ${
              isConnected ? "text-[#00e5a0]" : "text-[#556677]"
            }`}
          >
            {isConnected ? (
              <Wifi className="w-3 h-3" />
            ) : (
              <WifiOff className="w-3 h-3" />
            )}
            GLOVE
          </span>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 主画布 */}
        <div className="flex-1 flex items-center justify-center p-4">
          <div
            className="relative border border-[#f59e0b]/20 rounded-sm overflow-hidden"
            style={{
              boxShadow: "0 0 30px rgba(245,158,11,0.1), inset 0 0 30px rgba(10,14,26,0.5)",
            }}
          >
            <canvas
              ref={canvasRef}
              className="block"
              style={{
                width: `${CANVAS_W}px`,
                height: `${CANVAS_H}px`,
                imageRendering: "auto",
              }}
            />
            {/* 模式标签 */}
            <div className="absolute top-2 left-2 px-2 py-0.5 rounded-sm bg-[#f59e0b]/15 border border-[#f59e0b]/30">
              <span className="text-[9px] font-mono text-[#f59e0b]">
                <Hand className="w-3 h-3 inline mr-1" />
                GLOVE-ONLY MODE
              </span>
            </div>
          </div>
        </div>

        {/* 右侧面板 */}
        <div className="w-64 border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-4 shrink-0">
          {/* 手套连接 */}
          <Section title="GLOVE CONNECTION">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[#556677]">STATUS</span>
              <span
                className={isConnected ? "text-[#00e5a0]" : "text-[#ff2d7b]"}
              >
                {isConnected
                  ? "CONNECTED"
                  : isConnecting
                  ? "CONNECTING..."
                  : "DISCONNECTED"}
              </span>
            </div>
            {isConnected && (
              <>
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-[#556677]">GLOVE FPS</span>
                  <span className="text-[#00f0ff]">{gloveFps}</span>
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-[#556677]">FRAMES</span>
                  <span className="text-[#8899aa]">{gloveFrameCount}</span>
                </div>
              </>
            )}
            {gloveError && (
              <p className="text-[8px] text-[#ff2d7b]">{gloveError}</p>
            )}
            <button
              onClick={isConnected ? disconnectGlove : connectGlove}
              disabled={isConnecting}
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{
                borderColor: isConnected
                  ? "rgba(255,45,123,0.3)"
                  : "rgba(0,229,160,0.3)",
              }}
            >
              {isConnected ? (
                <>
                  <WifiOff className="w-3 h-3" /> 断开手套
                </>
              ) : (
                <>
                  <Wifi className="w-3 h-3" /> 连接手套
                </>
              )}
            </button>
          </Section>

          {/* 推理状态 */}
          <Section title="PREDICTION">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[#556677]">MODEL</span>
              <span
                className={modelReady ? "text-[#00e5a0]" : "text-[#ff2d7b]"}
              >
                {modelReady ? "READY" : "NOT LOADED"}
              </span>
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[#556677]">PRED FPS</span>
              <span className="text-[#f59e0b]">{predFps}</span>
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[#556677]">CONFIDENCE</span>
              <span className="text-[#da77f2]">{confidence.toFixed(0)}%</span>
            </div>
            {/* 置信度条 */}
            <div className="h-1.5 bg-[#1a2030] rounded-full overflow-hidden border border-[#f59e0b]/10">
              <div
                className="h-full rounded-full transition-all duration-200"
                style={{
                  width: `${confidence}%`,
                  background:
                    confidence > 70
                      ? "#00e5a0"
                      : confidence > 40
                      ? "#f59e0b"
                      : "#ff2d7b",
                  boxShadow: `0 0 6px ${
                    confidence > 70
                      ? "rgba(0,229,160,0.5)"
                      : confidence > 40
                      ? "rgba(245,158,11,0.5)"
                      : "rgba(255,45,123,0.5)"
                  }`,
                }}
              />
            </div>
            {loadMsg && (
              <p className="text-[8px] text-[#f59e0b]">{loadMsg}</p>
            )}
          </Section>

          {/* 传感器概览 */}
          {isConnected && latestFrame && (
            <Section title="SENSOR OVERVIEW">
              <MiniHeatmap data={latestFrame.mapped_data} />
              <div className="text-[8px] font-mono text-[#334455] text-center mt-1">
                137 mapped sensors
              </div>
            </Section>
          )}

          {/* 快捷导航 */}
          <Section title="NAVIGATION">
            <Link
              href="/train-skeleton"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{ borderColor: "rgba(245,158,11,0.3)" }}
            >
              <Activity className="w-3 h-3" />
              骨架训练
            </Link>
            <Link
              href="/collect"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5 mt-1.5"
            >
              数据采集
            </Link>
            <Link
              href="/translate"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5 mt-1.5"
            >
              手语翻译
            </Link>
          </Section>
        </div>
      </div>
    </div>
  );
}

// ===== 辅助组件 =====

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 pb-1 border-b border-[#f59e0b]/15">
        <div className="w-1 h-3 bg-[#f59e0b] rounded-full shadow-[0_0_4px_rgba(245,158,11,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[#f59e0b] font-mono">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function MiniHeatmap({ data }: { data: number[] }) {
  const cols = 14;
  const rows = Math.ceil(data.length / cols);
  return (
    <div
      className="grid gap-px"
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
    >
      {data.slice(0, rows * cols).map((v, i) => {
        const norm = Math.min(v / 200, 1);
        const r = Math.round(norm * 245);
        const g = Math.round((1 - norm) * 158 + norm * 45);
        const b = Math.round((1 - norm) * 11 + norm * 123);
        return (
          <div
            key={i}
            className="aspect-square rounded-[1px]"
            style={{
              backgroundColor: `rgb(${r}, ${g}, ${b})`,
              opacity: 0.3 + norm * 0.7,
            }}
          />
        );
      })}
    </div>
  );
}

// ===== Canvas 绘制函数 =====

function computeSpread(landmarks: { x: number; y: number; z: number }[]): number {
  // 基于关键点分布范围估算置信度
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  // 合理的手部范围大约 0.1-0.5
  const spread = (rangeX + rangeY) / 2;
  if (spread < 0.02) return 10; // 太集中，可能不准
  if (spread > 0.8) return 20; // 太分散，可能不准
  return Math.min(100, spread * 300);
}

function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const step = 50;
  ctx.strokeStyle = "rgba(245, 158, 11, 0.03)";
  ctx.lineWidth = 0.5;
  for (let x = 0; x < w; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(245, 158, 11, 0.06)";
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}

function drawScanLines(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = "rgba(245, 158, 11, 0.008)";
  for (let y = 0; y < h; y += 3) {
    ctx.fillRect(0, y, w, 1);
  }
}

function drawWaitingState(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  time: number
) {
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.12;

  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(245, 158, 11, 0.5)";
  ctx.lineWidth = 2;
  const angle1 = time * 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, angle1, angle1 + 1.2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(255, 45, 123, 0.4)";
  const angle2 = -time * 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.6, angle2, angle2 + 0.8);
  ctx.stroke();

  ctx.fillStyle = `rgba(245, 158, 11, ${0.4 + 0.2 * Math.sin(time * 2)})`;
  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.textAlign = "center";
  ctx.fillText("WAITING FOR GLOVE DATA...", cx, cy + radius + 35);

  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillStyle = "rgba(85, 102, 119, 0.6)";
  ctx.fillText("请连接手套并确保骨架模型已加载", cx, cy + radius + 55);

  ctx.restore();
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  w: number,
  h: number
) {
  Object.entries(FINGER_CONNECTION_GROUPS).forEach(([finger, connections]) => {
    const color = FINGER_COLORS[finger];
    const glowColor = FINGER_GLOW_COLORS[finger];

    connections.forEach(([from, to]) => {
      const x1 = landmarks[from].x * w;
      const y1 = landmarks[from].y * h;
      const x2 = landmarks[to].x * w;
      const y2 = landmarks[to].y * h;

      ctx.save();
      ctx.strokeStyle = glowColor;
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.restore();
    });
  });
}

function drawJoints(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  w: number,
  h: number
) {
  landmarks.forEach((lm, idx) => {
    const x = lm.x * w;
    const y = lm.y * h;

    let color = "#f59e0b";
    if (idx <= 4) color = FINGER_COLORS.thumb;
    else if (idx <= 8) color = FINGER_COLORS.index;
    else if (idx <= 12) color = FINGER_COLORS.middle;
    else if (idx <= 16) color = FINGER_COLORS.ring;
    else color = FINGER_COLORS.pinky;

    const isTip = [4, 8, 12, 16, 20].includes(idx);
    const isWrist = idx === 0;
    const radius = isWrist ? 7 : isTip ? 5.5 : 3.5;

    ctx.save();
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 4);
    gradient.addColorStop(0, color + "50");
    gradient.addColorStop(0.5, color + "15");
    gradient.addColorStop(1, color + "00");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius * 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.beginPath();
    ctx.arc(x - radius * 0.25, y - radius * 0.25, radius * 0.3, 0, Math.PI * 2);
    ctx.fill();

    if (isTip) {
      ctx.strokeStyle = color + "60";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  });
}

function drawPalmCrosshair(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  w: number,
  h: number
) {
  const palmIndices = [0, 5, 9, 13, 17];
  const cx =
    (palmIndices.reduce((sum, i) => sum + landmarks[i].x, 0) /
      palmIndices.length) *
    w;
  const cy =
    (palmIndices.reduce((sum, i) => sum + landmarks[i].y, 0) /
      palmIndices.length) *
    h;

  const size = 18;
  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.35)";
  ctx.lineWidth = 0.8;

  ctx.beginPath();
  ctx.moveTo(cx - size, cy);
  ctx.lineTo(cx - 5, cy);
  ctx.moveTo(cx + 5, cy);
  ctx.lineTo(cx + size, cy);
  ctx.moveTo(cx, cy - size);
  ctx.lineTo(cx, cy - 5);
  ctx.moveTo(cx, cy + 5);
  ctx.lineTo(cx, cy + size);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 2, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(245, 158, 11, 0.15)";
  ctx.beginPath();
  ctx.arc(cx, cy, size + 5, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

function drawHUDCorners(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const cornerSize = 25;
  const offset = 6;
  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.3)";
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.moveTo(offset, offset + cornerSize);
  ctx.lineTo(offset, offset);
  ctx.lineTo(offset + cornerSize, offset);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(w - offset - cornerSize, offset);
  ctx.lineTo(w - offset, offset);
  ctx.lineTo(w - offset, offset + cornerSize);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(offset, h - offset - cornerSize);
  ctx.lineTo(offset, h - offset);
  ctx.lineTo(offset + cornerSize, h - offset);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(w - offset - cornerSize, h - offset);
  ctx.lineTo(w - offset, h - offset);
  ctx.lineTo(w - offset, h - offset - cornerSize);
  ctx.stroke();

  ctx.restore();
}
