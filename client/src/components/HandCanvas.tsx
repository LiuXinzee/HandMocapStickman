/*
 * HandCanvas — 赛博朋克 HUD 风格手部骨架渲染
 * 在 Canvas 上绘制霓虹色手部关键点和骨架连线
 * DESIGN: Cyberpunk HUD — 霓虹发光骨架 + 网格背景 + HUD 角标
 */
import {
  FINGER_CONNECTION_GROUPS,
  type HandResult,
} from "@/hooks/useHandTracking";
import { useEffect, useRef } from "react";

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

const CANVAS_FRAME_INTERVAL_MS = 1000 / 30;

interface HandCanvasProps {
  handResults: HandResult | null;
  videoWidth: number;
  videoHeight: number;
  showVideo?: boolean;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

export default function HandCanvas({
  handResults,
  videoWidth,
  videoHeight,
  showVideo = true,
  videoRef,
}: HandCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prevLandmarksRef = useRef<
    Record<string, { x: number; y: number; z: number }[]>
  >({});
  const latestHandResultsRef = useRef(handResults);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    latestHandResultsRef.current = handResults;
  }, [handResults]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    // 停止之前的动画循环
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }

    let lastRenderAt = -Infinity;
    const render = (timestamp = performance.now()) => {
      if (timestamp - lastRenderAt < CANVAS_FRAME_INTERVAL_MS) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }
      lastRenderAt = timestamp;
      const time = timestamp / 1000;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 绘制视频帧
      if (showVideo && videoRef?.current && videoRef.current.readyState >= 2) {
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        ctx.fillStyle = "rgba(10, 14, 26, 0.4)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      } else {
        ctx.fillStyle = "#0a0e1a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawGrid(ctx, canvas.width, canvas.height);
      }

      drawScanLines(ctx, canvas.width, canvas.height);

      const currentResults = latestHandResultsRef.current;
      if (!currentResults || currentResults.landmarks.length === 0) {
        drawWaitingState(ctx, canvas.width, canvas.height, time);
        prevLandmarksRef.current = {};
      } else {
        currentResults.landmarks.forEach((landmarks, handIdx) => {
          const mirrored = landmarks.map(lm => ({
            x: 1 - lm.x,
            y: lm.y,
            z: lm.z,
          }));
          const handKey =
            currentResults.trackingIds[handIdx] ||
            currentResults.handedness[handIdx] ||
            `hand-${handIdx}`;
          const smoothed = smoothLandmarks(mirrored, handKey);
          drawSkeleton(ctx, smoothed, canvas.width, canvas.height);
          drawJoints(ctx, smoothed, canvas.width, canvas.height);
          drawPalmCrosshair(ctx, smoothed, canvas.width, canvas.height);
          drawHandLabel(
            ctx,
            smoothed,
            canvas.width,
            canvas.height,
            currentResults.handedness[handIdx],
            currentResults.surfaces[handIdx],
            currentResults.surfaceConfidences[handIdx] ?? 0
          );
        });
      }

      drawHUDCorners(ctx, canvas.width, canvas.height);
      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [videoWidth, videoHeight, showVideo, videoRef]);

  function smoothLandmarks(
    current: { x: number; y: number; z: number }[],
    handKey: string
  ) {
    const prev = prevLandmarksRef.current[handKey];
    if (!prev || prev.length !== current.length) {
      prevLandmarksRef.current[handKey] = current;
      return current;
    }
    const alpha = 0.55;
    const smoothed = current.map((lm, i) => ({
      x: prev[i].x + alpha * (lm.x - prev[i].x),
      y: prev[i].y + alpha * (lm.y - prev[i].y),
      z: prev[i].z + alpha * (lm.z - prev[i].z),
    }));
    prevLandmarksRef.current[handKey] = smoothed;
    return smoothed;
  }

  function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
    const step = 50;
    ctx.strokeStyle = "rgba(0, 240, 255, 0.04)";
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
    ctx.strokeStyle = "rgba(0, 240, 255, 0.08)";
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  }

  function drawScanLines(ctx: CanvasRenderingContext2D, w: number, h: number) {
    ctx.fillStyle = "rgba(0, 240, 255, 0.012)";
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

    // 外圈
    ctx.strokeStyle = "rgba(0, 240, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    // 内圈
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.6, 0, Math.PI * 2);
    ctx.stroke();

    // 旋转弧线 1
    ctx.strokeStyle = "rgba(0, 240, 255, 0.5)";
    ctx.lineWidth = 2;
    const angle1 = time * 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, angle1, angle1 + 1.2);
    ctx.stroke();

    // 旋转弧线 2（反向）
    ctx.strokeStyle = "rgba(255, 45, 123, 0.4)";
    const angle2 = -time * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.6, angle2, angle2 + 0.8);
    ctx.stroke();

    // 脉冲圆
    const pulseRadius = radius * (1 + 0.15 * Math.sin(time * 3));
    ctx.strokeStyle = `rgba(0, 240, 255, ${0.1 + 0.05 * Math.sin(time * 3)})`;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.arc(cx, cy, pulseRadius, 0, Math.PI * 2);
    ctx.stroke();

    // 提示文字
    ctx.fillStyle = `rgba(0, 240, 255, ${0.4 + 0.2 * Math.sin(time * 2)})`;
    ctx.font = '13px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.fillText("SCANNING FOR HANDS...", cx, cy + radius + 35);

    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = "rgba(85, 102, 119, 0.6)";
    ctx.fillText("请将手放入摄像头视野范围内", cx, cy + radius + 55);

    ctx.restore();
  }

  function drawSkeleton(
    ctx: CanvasRenderingContext2D,
    landmarks: { x: number; y: number; z: number }[],
    w: number,
    h: number
  ) {
    Object.entries(FINGER_CONNECTION_GROUPS).forEach(
      ([finger, connections]) => {
        const color = FINGER_COLORS[finger];
        const glowColor = FINGER_GLOW_COLORS[finger];

        connections.forEach(([from, to]) => {
          const x1 = landmarks[from].x * w;
          const y1 = landmarks[from].y * h;
          const x2 = landmarks[to].x * w;
          const y2 = landmarks[to].y * h;

          ctx.save();
          // 发光层
          ctx.strokeStyle = glowColor;
          ctx.lineWidth = 10;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();

          // 主线条
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();

          // 高亮中心线
          ctx.strokeStyle = color + "cc";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();

          ctx.restore();
        });
      }
    );
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

      let color = "#00f0ff";
      if (idx <= 4) color = FINGER_COLORS.thumb;
      else if (idx <= 8) color = FINGER_COLORS.index;
      else if (idx <= 12) color = FINGER_COLORS.middle;
      else if (idx <= 16) color = FINGER_COLORS.ring;
      else color = FINGER_COLORS.pinky;

      const isTip = [4, 8, 12, 16, 20].includes(idx);
      const isWrist = idx === 0;
      const radius = isWrist ? 7 : isTip ? 5.5 : 3.5;

      ctx.save();

      // 外发光
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 4);
      gradient.addColorStop(0, color + "50");
      gradient.addColorStop(0.5, color + "15");
      gradient.addColorStop(1, color + "00");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius * 4, 0, Math.PI * 2);
      ctx.fill();

      // 内核
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      // 白色高光
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.beginPath();
      ctx.arc(
        x - radius * 0.25,
        y - radius * 0.25,
        radius * 0.3,
        0,
        Math.PI * 2
      );
      ctx.fill();

      // 指尖标记环
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
    ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
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

    ctx.strokeStyle = "rgba(0, 240, 255, 0.15)";
    ctx.beginPath();
    ctx.arc(cx, cy, size + 5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  function drawHandLabel(
    ctx: CanvasRenderingContext2D,
    landmarks: { x: number; y: number; z: number }[],
    w: number,
    h: number,
    handedness: string | undefined,
    surface: HandResult["surfaces"][number] | undefined,
    confidence: number
  ) {
    const palmIndices = [0, 5, 9, 13, 17];
    const x =
      (palmIndices.reduce((sum, i) => sum + landmarks[i].x, 0) /
        palmIndices.length) *
      w;
    const y =
      (palmIndices.reduce((sum, i) => sum + landmarks[i].y, 0) /
        palmIndices.length) *
      h;
    const handText =
      handedness === "Left" ? "左手" : handedness === "Right" ? "右手" : "手部";
    const surfaceText =
      surface === "palm"
        ? "手心"
        : surface === "back"
          ? "手背"
          : "正反面识别中";
    const color =
      surface === "palm"
        ? "#d8e2e8"
        : surface === "back"
          ? "#a855f7"
          : "#f59e0b";

    ctx.save();
    ctx.font = '11px "JetBrains Mono", monospace';
    const confidenceText =
      surface === "unknown" || !surface
        ? ""
        : ` ${Math.round(confidence * 100)}%`;
    const label = `${handText} · ${surfaceText}${confidenceText}`;
    const labelWidth = ctx.measureText(label).width + 16;
    const boxX = Math.min(Math.max(6, x + 24), w - labelWidth - 6);
    const boxY = Math.min(Math.max(20, y - 30), h - 24);
    ctx.fillStyle = "rgba(10, 14, 26, 0.86)";
    ctx.fillRect(boxX, boxY - 15, labelWidth, 22);
    ctx.strokeStyle = `${color}88`;
    ctx.lineWidth = 1;
    ctx.strokeRect(boxX, boxY - 15, labelWidth, 22);
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, boxX + 8, boxY - 4);
    ctx.restore();
  }

  function drawHUDCorners(ctx: CanvasRenderingContext2D, w: number, h: number) {
    const cornerSize = 25;
    const offset = 6;
    ctx.save();
    ctx.strokeStyle = "rgba(0, 240, 255, 0.3)";
    ctx.lineWidth = 1.5;

    // 左上
    ctx.beginPath();
    ctx.moveTo(offset, offset + cornerSize);
    ctx.lineTo(offset, offset);
    ctx.lineTo(offset + cornerSize, offset);
    ctx.stroke();

    // 右上
    ctx.beginPath();
    ctx.moveTo(w - offset - cornerSize, offset);
    ctx.lineTo(w - offset, offset);
    ctx.lineTo(w - offset, offset + cornerSize);
    ctx.stroke();

    // 左下
    ctx.beginPath();
    ctx.moveTo(offset, h - offset - cornerSize);
    ctx.lineTo(offset, h - offset);
    ctx.lineTo(offset + cornerSize, h - offset);
    ctx.stroke();

    // 右下
    ctx.beginPath();
    ctx.moveTo(w - offset - cornerSize, h - offset);
    ctx.lineTo(w - offset, h - offset);
    ctx.lineTo(w - offset, h - offset - cornerSize);
    ctx.stroke();

    ctx.restore();
  }

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full object-contain"
      style={{ imageRendering: "auto" }}
    />
  );
}
