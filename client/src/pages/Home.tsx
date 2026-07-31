/*
 * Home — 赛博朋克 HUD 手部动作捕捉主界面
 * DESIGN: Cyberpunk HUD — 深色沉浸式布局
 *
 * 功能:
 * 1. 摄像头手部关键点检测 (MediaPipe Hands)
 * 2. 手套 WebSocket 桥接连接 (触觉传感器)
 * 3. 同步录制与数据导出
 */
import GesturePanel from "@/components/GesturePanel";
import GlovePanel from "@/components/GlovePanel";
import HandCanvas from "@/components/HandCanvas";
import HUDPanel from "@/components/HUDPanel";
import RecordPanel from "@/components/RecordPanel";
import { useDualGloveSerial } from "@/hooks/useDualGloveSerial";
import { useHandTracking } from "@/hooks/useHandTracking";
import { useSyncRecorder } from "@/hooks/useSyncRecorder";
import {
  Camera,
  CameraOff,
  Eye,
  EyeOff,
  Maximize,
  Minimize,
  PanelRightClose,
  PanelRightOpen,
  Database,
  Brain,
  MessageSquare,
  Bone,
  Hand,
} from "lucide-react";
import { Link } from "wouter";
import { useCallback, useEffect, useRef, useState } from "react";

const HERO_BG =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663331800787/fHYEVehh6kc4x7HNL7px46/hero-bg-gbyGpPuyfHg6jNpdxkYd78.webp";
const HAND_DEMO =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663331800787/fHYEVehh6kc4x7HNL7px46/hand-skeleton-demo-RLuzXBfnbi7PPAb9guYRWF.webp";

export default function Home() {
  // ===== 手部追踪 =====
  const {
    videoRef,
    isLoading,
    isRunning,
    error,
    handResults,
    fps,
    loadingStatus,
    startTracking,
    stopTracking,
  } = useHandTracking({ maxHands: 2 });

  // ===== 同步录制 =====
  const recorder = useSyncRecorder();

  // ===== 手套双手连接 (Web Serial API 直连，296B 带加速度) =====
  const {
    left: gloveLeft,
    right: gloveRight,
    anyConnected: gloveConnected,
  } = useDualGloveSerial({
    baudRate: 921600,
    onLeftFrame: (frame) => recorder.recordGloveFrame(frame),
    onRightFrame: (frame) => recorder.recordGloveFrame(frame),
  });
  // 双手总帧率（用于顶栏概览）
  const gloveFps = gloveLeft.gloveFps + gloveRight.gloveFps;

  // 录制视频帧（在 handResults 变化时）
  const prevResultsRef = useRef(handResults);
  useEffect(() => {
    if (recorder.isRecording && handResults !== prevResultsRef.current) {
      recorder.recordVideoFrame(handResults);
      prevResultsRef.current = handResults;
    }
  }, [handResults, recorder.isRecording]);

  // ===== UI 状态 =====
  const [showVideo, setShowVideo] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const mainRef = useRef<HTMLDivElement>(null);

  // Canvas 尺寸
  const [canvasSize, setCanvasSize] = useState({ width: 1280, height: 720 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const w = rect.width - 16;
        const h = Math.min((w * 9) / 16, rect.height - 60);
        const finalW = (h * 16) / 9;
        setCanvasSize({
          width: Math.round(Math.min(w, finalW)),
          height: Math.round(h),
        });
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [showSidebar]);

  const toggleFullscreen = useCallback(() => {
    if (!mainRef.current) return;
    if (!document.fullscreenElement) {
      mainRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  // ===== 导出处理 =====
  const handleExportCSV = useCallback(() => {
    const csv = recorder.exportSyncedCSV();
    if (!csv) return;
    downloadFile(csv, `mocap_sync_${getTimestamp()}.csv`, "text/csv");
  }, [recorder]);

  const handleExportJSON = useCallback(() => {
    const json = recorder.exportRawJSON();
    if (!json) return;
    downloadFile(json, `mocap_raw_${getTimestamp()}.json`, "application/json");
  }, [recorder]);

  // ===== 启动页 =====
  if (!isRunning && !isLoading) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden"
        style={{
          backgroundImage: `url(${HERO_BG})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      >
        <div className="absolute inset-0 bg-[#0a0e1a]/70 pointer-events-none" />
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,240,255,0.015) 2px, rgba(0,240,255,0.015) 3px)",
          }}
        />

        <div className="relative z-10 flex flex-col items-center gap-8 max-w-2xl px-6">
          {/* Logo */}
          <div className="flex flex-col items-center gap-5">
            <div className="w-32 h-32 rounded-full border border-[#00f0ff]/30 flex items-center justify-center relative overflow-hidden group">
              <img
                src={HAND_DEMO}
                alt="Hand Skeleton"
                className="w-28 h-28 object-cover rounded-full opacity-80 group-hover:opacity-100 transition-opacity duration-300"
              />
              <div className="absolute inset-0 border border-[#00f0ff]/20 rounded-full animate-[pulse-glow_2s_ease-in-out_infinite] pointer-events-none" />
              <div
                className="absolute inset-0 rounded-full pointer-events-none"
                style={{
                  background:
                    "radial-gradient(circle, transparent 60%, rgba(0,240,255,0.08) 100%)",
                }}
              />
            </div>

            <h1
              className="text-4xl md:text-5xl font-bold tracking-tight text-center"
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                color: "#00f0ff",
                textShadow:
                  "0 0 30px rgba(0,240,255,0.3), 0 0 60px rgba(0,240,255,0.1)",
              }}
            >
              HAND MOCAP
            </h1>
            <div className="text-center space-y-1">
              <p
                className="text-sm md:text-base leading-relaxed"
                style={{
                  fontFamily: "'Space Grotesk', sans-serif",
                  color: "rgba(136, 153, 170, 0.9)",
                }}
              >
                实时手部动作捕捉 + 触觉手套同步采集系统
              </p>
              <p
                className="text-xs"
                style={{ color: "rgba(0, 240, 255, 0.5)" }}
              >
                21 关键点 · 256 传感器 · IMU 四元数 · 同步录制
              </p>
            </div>
          </div>

          {/* 功能卡片 */}
          <div className="grid grid-cols-4 gap-3 w-full max-w-xl">
            {[
              { label: "关键点", value: "21", desc: "指关节定位" },
              { label: "传感器", value: "256", desc: "触觉采集" },
              { label: "帧率", value: "100+", desc: "手套Hz" },
              { label: "同步", value: "✓", desc: "时间戳对齐" },
            ].map(({ label, value, desc }) => (
              <div
                key={label}
                className="cyber-panel p-3 rounded-sm text-center group hover:border-[#00f0ff]/50 transition-all duration-200"
              >
                <div
                  className="text-xl font-bold"
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "#00f0ff",
                    textShadow: "0 0 10px rgba(0,240,255,0.4)",
                  }}
                >
                  {value}
                </div>
                <div
                  className="text-[9px] uppercase tracking-widest mt-1"
                  style={{ color: "#556677" }}
                >
                  {label}
                </div>
                <div className="text-[9px] mt-0.5" style={{ color: "#445566" }}>
                  {desc}
                </div>
              </div>
            ))}
          </div>

          {/* 启动按钮 */}
          <button
            onClick={() => {
              console.log("[UI] Start button clicked");
              startTracking();
            }}
            className="cyber-btn px-10 py-3.5 rounded-sm text-sm flex items-center gap-3 group relative overflow-hidden z-20"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-[#00f0ff]/0 via-[#00f0ff]/10 to-[#00f0ff]/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 pointer-events-none" />
            <Camera className="w-4 h-4" />
            <span>启动摄像头</span>
          </button>

          {error && (
            <div className="cyber-panel p-3 rounded-sm border-[#ff2d7b]/50 max-w-md">
              <p className="text-[#ff2d7b] text-xs font-mono">{error}</p>
            </div>
          )}

          {/* 手语识别系统导航 */}
          <div className="w-full max-w-xl">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider text-center mb-2">
              Sign Language Recognition System
            </div>
            <div className="grid grid-cols-3 gap-2">
              <Link href="/collect" className="cyber-panel p-3 rounded-sm text-center group hover:border-[#00f0ff]/50 transition-all duration-200 block">
                <Database className="w-5 h-5 mx-auto text-[#00f0ff] mb-1" />
                <div className="text-[10px] text-[#8899aa] group-hover:text-[#ccd6e0] transition-colors">数据采集</div>
              </Link>
              <Link href="/train" className="cyber-panel p-3 rounded-sm text-center group hover:border-[#00e5a0]/50 transition-all duration-200 block">
                <Brain className="w-5 h-5 mx-auto text-[#00e5a0] mb-1" />
                <div className="text-[10px] text-[#8899aa] group-hover:text-[#ccd6e0] transition-colors">手语训练</div>
              </Link>
              <Link href="/translate" className="cyber-panel p-3 rounded-sm text-center group hover:border-[#ff2d7b]/50 transition-all duration-200 block">
                <MessageSquare className="w-5 h-5 mx-auto text-[#ff2d7b] mb-1" />
                <div className="text-[10px] text-[#8899aa] group-hover:text-[#ccd6e0] transition-colors">手语翻译</div>
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
              <Link href="/train-skeleton" className="cyber-panel p-3 rounded-sm text-center group hover:border-[#f59e0b]/50 transition-all duration-200 block">
                <Bone className="w-5 h-5 mx-auto text-[#f59e0b] mb-1" />
                <div className="text-[10px] text-[#8899aa] group-hover:text-[#ccd6e0] transition-colors">骨架训练</div>
                <div className="text-[8px] text-[#334455]">触觉→骨架回归</div>
              </Link>
              <Link href="/mocap" className="cyber-panel p-3 rounded-sm text-center group hover:border-[#da77f2]/50 transition-all duration-200 block">
                <Hand className="w-5 h-5 mx-auto text-[#da77f2] mb-1" />
                <div className="text-[10px] text-[#8899aa] group-hover:text-[#ccd6e0] transition-colors">虚拟动捕</div>
                <div className="text-[8px] text-[#334455]">仅手套驱动火柴人</div>
              </Link>
            </div>
          </div>

          {/* 底部提示 */}
          <div className="text-center space-y-1">
            <p className="text-[10px] text-[#334455] font-mono">
              POWERED BY MEDIAPIPE HANDS · TENSORFLOW.JS · WEB SERIAL API
            </p>
            <p className="text-[10px] text-[#2a3a4a] font-mono">
              需要浏览器摄像头权限 · 手套通过 Web Serial 直连（Chrome/Edge）
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ===== 加载中 =====
  if (isLoading) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center"
        style={{ backgroundColor: "#0a0e1a" }}
      >
        <div className="flex flex-col items-center gap-6">
          <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-full border-2 border-[#00f0ff]/20" />
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-[#00f0ff] animate-spin" />
            <div className="absolute inset-3 rounded-full border border-[#00f0ff]/10" />
            <div
              className="absolute inset-3 rounded-full border border-transparent border-b-[#ff2d7b]/60 animate-spin"
              style={{
                animationDirection: "reverse",
                animationDuration: "1.5s",
              }}
            />
          </div>
          <div className="text-center space-y-2">
            <p className="cyber-text text-sm tracking-wider">
              INITIALIZING SYSTEM
            </p>
            <div className="space-y-1">
              <LoadingStep
                text="Requesting camera access..."
                done={loadingStatus !== "camera"}
                active={loadingStatus === "camera"}
              />
              <LoadingStep
                text="Loading MediaPipe model..."
                done={loadingStatus === "warmup" || loadingStatus === "ready"}
                active={loadingStatus === "model"}
                pending={loadingStatus === "camera"}
              />
              <LoadingStep
                text="Warming up detection engine..."
                done={loadingStatus === "ready"}
                active={loadingStatus === "warmup"}
                pending={
                  loadingStatus === "camera" || loadingStatus === "model"
                }
              />
            </div>
            <p className="text-[10px] text-[#334455] font-mono mt-4">
              首次加载模型约需 5-15 秒，请耐心等待...
            </p>
            {loadingStatus === "model" && (
              <p className="text-[10px] text-[#556677] font-mono">
                正在从 CDN 下载模型文件 (~5MB)...
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ===== 主界面（运行中） =====
  return (
    <div
      ref={mainRef}
      className="h-screen flex flex-col overflow-hidden select-none"
      style={{ backgroundColor: "#0a0e1a" }}
    >
      {/* 顶部状态栏 */}
      <header
        className="h-10 flex items-center justify-between px-4 border-b border-[#00f0ff]/15 shrink-0"
        style={{ backgroundColor: "rgba(10, 14, 26, 0.95)" }}
      >
        <div className="flex items-center gap-3">
          <span
            className="text-xs font-bold tracking-widest"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "#00f0ff",
              textShadow: "0 0 8px rgba(0,240,255,0.4)",
            }}
          >
            HAND MOCAP
          </span>
          <div className="w-px h-4 bg-[#00f0ff]/20" />
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] shadow-[0_0_6px_rgba(0,240,255,0.8)] animate-pulse" />
            <span className="text-[10px] text-[#00f0ff]/80 font-mono">
              LIVE
            </span>
          </div>
          {/* 手套连接指示 */}
          {gloveConnected && (
            <>
              <div className="w-px h-4 bg-[#00f0ff]/20" />
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#00e5a0] shadow-[0_0_6px_rgba(0,229,160,0.8)] animate-pulse" />
                <span className="text-[10px] text-[#00e5a0]/80 font-mono">
                  GLOVE
                </span>
              </div>
            </>
          )}
          {/* 录制指示 */}
          {recorder.isRecording && (
            <>
              <div className="w-px h-4 bg-[#00f0ff]/20" />
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#ff2d7b] shadow-[0_0_6px_rgba(255,45,123,0.8)] animate-pulse" />
                <span className="text-[10px] text-[#ff2d7b]/80 font-mono">
                  REC
                </span>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-5 text-[10px] font-mono text-[#556677]">
          <span>
            FPS: <span className="text-[#00f0ff]">{fps}</span>
          </span>
          <span>
            HANDS:{" "}
            <span className="text-[#00f0ff]">
              {handResults?.landmarks.length ?? 0}
            </span>
          </span>
          {gloveConnected && (
            <span>
              GLOVE: <span className="text-[#00e5a0]">{gloveFps}Hz</span>
            </span>
          )}
        </div>
      </header>

      {/* 主内容区 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 主视窗 */}
        <div className="flex-1 flex flex-col relative" ref={containerRef}>
          {/* Canvas 区域 */}
          <div className="flex-1 flex items-center justify-center p-2 relative">
            <div
              className="relative border border-[#00f0ff]/20 rounded-sm overflow-hidden"
              style={{
                width: canvasSize.width,
                height: canvasSize.height,
                maxWidth: "100%",
                maxHeight: "100%",
                boxShadow:
                  "0 0 30px rgba(0,240,255,0.05), inset 0 0 30px rgba(0,240,255,0.02)",
              }}
            >
              <HandCanvas
                handResults={handResults}
                videoWidth={canvasSize.width}
                videoHeight={canvasSize.height}
                showVideo={showVideo}
                videoRef={videoRef}
              />

              {/* 左上角时间戳 */}
              <div className="absolute top-2 left-3 text-[9px] font-mono text-[#00f0ff]/40 pointer-events-none">
                <TimeDisplay />
              </div>

              {/* 右上角分辨率 */}
              <div className="absolute top-2 right-3 text-[9px] font-mono text-[#00f0ff]/40 pointer-events-none">
                {canvasSize.width}x{canvasSize.height}
              </div>

              {/* 录制指示器（左下角） */}
              {recorder.isRecording && (
                <div className="absolute bottom-2 left-3 flex items-center gap-1.5 pointer-events-none">
                  <div className="w-2 h-2 rounded-full bg-[#ff2d7b] animate-pulse shadow-[0_0_8px_rgba(255,45,123,0.8)]" />
                  <span className="text-[10px] font-mono text-[#ff2d7b]">
                    REC{" "}
                    {formatDurationShort(recorder.recordingDuration)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* 底部控制栏 */}
          <div className="h-12 flex items-center justify-center gap-2 px-4 border-t border-[#00f0ff]/10 shrink-0">
            <ControlButton
              icon={
                showVideo ? (
                  <Eye className="w-3.5 h-3.5" />
                ) : (
                  <EyeOff className="w-3.5 h-3.5" />
                )
              }
              label={showVideo ? "隐藏视频" : "显示视频"}
              onClick={() => setShowVideo(!showVideo)}
              active={showVideo}
            />
            <ControlButton
              icon={
                isFullscreen ? (
                  <Minimize className="w-3.5 h-3.5" />
                ) : (
                  <Maximize className="w-3.5 h-3.5" />
                )
              }
              label={isFullscreen ? "退出全屏" : "全屏"}
              onClick={toggleFullscreen}
            />
            <ControlButton
              icon={
                showSidebar ? (
                  <PanelRightClose className="w-3.5 h-3.5" />
                ) : (
                  <PanelRightOpen className="w-3.5 h-3.5" />
                )
              }
              label={showSidebar ? "隐藏面板" : "显示面板"}
              onClick={() => setShowSidebar(!showSidebar)}
              active={showSidebar}
            />
            <div className="w-px h-5 bg-[#00f0ff]/15 mx-1" />
            <button
              onClick={stopTracking}
              className="cyber-btn cyber-btn-accent px-4 py-1.5 rounded-sm text-[11px] flex items-center gap-2"
            >
              <CameraOff className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">停止捕捉</span>
            </button>
          </div>
        </div>

        {/* 右侧数据面板 */}
        {showSidebar && (
          <aside
            className="w-60 border-l border-[#00f0ff]/15 overflow-y-auto shrink-0"
            style={{
              backgroundColor: "rgba(10, 14, 26, 0.95)",
              scrollbarWidth: "thin",
              scrollbarColor: "#00f0ff30 transparent",
            }}
          >
            <div className="p-3 space-y-4">
              {/* 同步录制面板 */}
              <RecordPanel
                isRecording={recorder.isRecording}
                recordingDuration={recorder.recordingDuration}
                videoFrameCount={recorder.videoFrameCount}
                gloveFrameCount={recorder.gloveFrameCount}
                gloveConnected={gloveConnected}
                onStartRecording={recorder.startRecording}
                onStopRecording={recorder.stopRecording}
                onExportCSV={handleExportCSV}
                onExportJSON={handleExportJSON}
                onClear={recorder.clear}
              />

              {/* 左手手套面板 */}
              <GlovePanel
                title="左手 GLOVE"
                connectLabel="连接左手"
                isConnected={gloveLeft.isConnected}
                isConnecting={gloveLeft.isConnecting}
                error={gloveLeft.error}
                gloveFps={gloveLeft.gloveFps}
                gloveFrameCount={gloveLeft.gloveFrameCount}
                latestFrame={gloveLeft.latestFrame}
                onConnect={gloveLeft.connect}
                onDisconnect={gloveLeft.disconnect}
              />

              {/* 右手手套面板 */}
              <GlovePanel
                title="右手 GLOVE"
                connectLabel="连接右手"
                isConnected={gloveRight.isConnected}
                isConnecting={gloveRight.isConnecting}
                error={gloveRight.error}
                gloveFps={gloveRight.gloveFps}
                gloveFrameCount={gloveRight.gloveFrameCount}
                latestFrame={gloveRight.latestFrame}
                onConnect={gloveRight.connect}
                onDisconnect={gloveRight.disconnect}
              />

              {/* 系统状态面板 */}
              <HUDPanel
                fps={fps}
                isRunning={isRunning}
                isLoading={isLoading}
                handResults={handResults}
              />

              {/* 手势面板 */}
              <GesturePanel handResults={handResults} />
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

// ===== 辅助组件 =====

function ControlButton({
  icon,
  label,
  onClick,
  active,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center gap-1.5 ${
        active ? "bg-[#00f0ff]/15 border-[#00f0ff]/40" : ""
      }`}
      title={label}
    >
      {icon}
      <span className="hidden md:inline">{label}</span>
    </button>
  );
}

function LoadingStep({
  text,
  done,
  active,
  pending,
}: {
  text: string;
  done?: boolean;
  active?: boolean;
  pending?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px] font-mono">
      {done ? (
        <span className="text-[#00e5a0]">✓</span>
      ) : active ? (
        <span className="text-[#f59e0b] animate-pulse">●</span>
      ) : pending ? (
        <span className="text-[#556677]">○</span>
      ) : (
        <span className="text-[#556677]">○</span>
      )}
      <span
        className={
          done
            ? "text-[#556677]"
            : active
            ? "text-[#8899aa]"
            : "text-[#334455]"
        }
      >
        {text}
      </span>
    </div>
  );
}

function TimeDisplay() {
  const [time, setTime] = useState(
    new Date().toLocaleTimeString("en-US", { hour12: false })
  );
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date().toLocaleTimeString("en-US", { hour12: false }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);
  return <>{time}</>;
}

// ===== 工具函数 =====

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function getTimestamp(): string {
  const now = new Date();
  return `${now.getFullYear()}${(now.getMonth() + 1)
    .toString()
    .padStart(2, "0")}${now.getDate().toString().padStart(2, "0")}_${now
    .getHours()
    .toString()
    .padStart(2, "0")}${now.getMinutes().toString().padStart(2, "0")}${now
    .getSeconds()
    .toString()
    .padStart(2, "0")}`;
}

function formatDurationShort(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${secs
    .toString()
    .padStart(2, "0")}`;
}
