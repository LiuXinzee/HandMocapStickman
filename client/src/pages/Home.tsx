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
import { useGloveFrames, useGloves } from "@/contexts/GloveContext";
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
  Video,
  Waves,
} from "lucide-react";
import { Link } from "wouter";
import { useCallback, useEffect, useRef, useState } from "react";

/*
 * 封面原来铺一张深蓝赛博背景图（CloudFront 上那张 hero-bg.webp），上面盖 70% 的
 * #f4f6fa。全站翻成浅色皮之后那个组合只剩缺点：合成出来是一片去饱和的 #a8adb5 灰，
 * 比 --hud-page 暗一大截，白卡片（cyber-panel）等于浮在灰底上；图里的青色高光还会
 * 从雾里透出来，跟蓝色强调打架。现在底色直接用 --hud-page，纹理改成 CSS 画
 * —— 网格与 .hud-stage 同款，顺带省掉一次 1920×1080 的外链请求。
 */
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

  // ===== 手套双手连接 =====
  // 连接住在 GloveProvider 里（全应用唯一），首页与四个流程页共享同一个串口。
  // 首页仍保留连接按钮：它不是编号步骤，是"随手看一眼原始数据"的入口。
  const {
    left: gloveLeft,
    right: gloveRight,
    anyConnected: gloveConnected,
  } = useGloves();
  useGloveFrames(
    (frame) => recorder.recordGloveFrame(frame),
    (frame) => recorder.recordGloveFrame(frame)
  );
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
        style={{ backgroundColor: "var(--hud-page)" }}
      >
        {/* 浅网格：和 3D 视口的 .hud-stage 同一套画法、同一个 28px 步长 */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(148,163,184,0.14) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.14) 1px, transparent 1px)",
            backgroundSize: "28px 28px",
          }}
        />
        {/* logo/标题背后一团柔和蓝光，替掉原来那层青色雾 */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(60% 45% at 50% 32%, rgba(22,119,255,0.10), transparent 70%)",
          }}
        />

        <div className="relative z-10 flex flex-col items-center gap-8 max-w-3xl px-6">
          {/* Logo */}
          <div className="flex flex-col items-center gap-5">
            <div className="w-32 h-32 rounded-full border border-[#1677ff]/30 flex items-center justify-center relative overflow-hidden group">
              <img
                src={HAND_DEMO}
                alt="Hand Skeleton"
                className="w-28 h-28 object-cover rounded-full opacity-80 group-hover:opacity-100 transition-opacity duration-300"
              />
              <div className="absolute inset-0 border border-[#1677ff]/20 rounded-full animate-[pulse-glow_2s_ease-in-out_infinite] pointer-events-none" />
              <div
                className="absolute inset-0 rounded-full pointer-events-none"
                style={{
                  background:
                    "radial-gradient(circle, transparent 60%, rgba(22,119,255,0.10) 100%)",
                }}
              />
            </div>

            <h1
              className="text-4xl md:text-5xl font-bold tracking-tight text-center"
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                color: "var(--hud-accent)",
                // 浅底上"辉光"是不成立的（发不出光，只会糊）。改成一层很淡的蓝色
                // 落影，作用是把标题从网格上托起来，不是让它发亮
                textShadow: "0 2px 12px rgba(22,119,255,0.18)",
              }}
            >
              HAND MOCAP
            </h1>
            <div className="text-center space-y-1">
              <p
                className="text-sm md:text-base leading-relaxed"
                style={{
                  fontFamily: "'Space Grotesk', sans-serif",
                  color: "var(--hud-soft)",
                }}
              >
                实时手部动作捕捉 + 触觉手套同步采集系统
              </p>
              <p
                className="text-xs"
                style={{ color: "var(--hud-accent)" }}
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
                className="cyber-panel p-3 rounded-sm text-center group hover:border-[#1677ff]/50 transition-all duration-200"
              >
                <div
                  className="text-xl font-bold"
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "var(--hud-accent)",
                  }}
                >
                  {value}
                </div>
                <div
                  className="text-[9px] uppercase tracking-widest mt-1"
                  style={{ color: "var(--hud-dim)" }}
                >
                  {label}
                </div>
                <div className="text-[9px] mt-0.5" style={{ color: "var(--hud-faint)" }}>
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
            <div className="absolute inset-0 bg-gradient-to-r from-[#1677ff]/0 via-[#1677ff]/10 to-[#1677ff]/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 pointer-events-none" />
            <Camera className="w-4 h-4" />
            <span>启动摄像头</span>
          </button>

          {error && (
            <div className="cyber-panel p-3 rounded-sm border-[#e11d48]/50 max-w-md">
              <p className="text-[var(--hud-err)] text-xs font-mono">{error}</p>
            </div>
          )}

          {/* 手语识别系统导航 —— 按实际操作顺序排成递进流程，而不是按模块罗列。
              静态词与动态词是两条平行支线：②里选哪个采，③就用对应的那个训。
              /train-skeleton 不在这条链路上（它只驱动 /mocap 的火柴人），单列为旁支。

              ⚠ **句子那条不是第三条平行支线，是 Y 形。** 句子训练同时吃两份数据：
              句子录制（真实连续手语）**和**时序词录制。三处硬依赖：
                1. 类别表从数据集全部标签推（train_seq.py 的 merged_class_table）；
                2. 合成句子由孤立词录制交叉淡化拼出来（synth_sentences.py），
                   45 条真实句根本训不动一个 CTC；
                3. 数据不够时 train_seq.py 直接 SystemExit。
              反过来不成立：孤立词训练会把句子样本滤掉（sequenceModel.ts 的
              isSentenceSample）。所以颜色上让时序采集同时连着两条训练，
              而句子采集只连句子训练。 */}
          <div className="w-full">
            <div className="text-[9px] font-mono text-[var(--hud-dim)] uppercase tracking-wider text-center mb-3">
              Sign Language Recognition System · 按顺序操作
            </div>

            <div className="flex flex-col md:flex-row md:items-stretch gap-1.5">
              <FlowStep n="1" title="准备" hint="先插手套">
                <StepLink
                  href="/mocap"
                  icon={<Hand className="w-3.5 h-3.5 text-[var(--hud-wrist)] shrink-0" />}
                  label="手套体检"
                  sub="IMU 自检 · 弯折标定"
                  hoverCls="hover:border-[#c026d3]/60"
                />
              </FlowStep>

              <FlowArrow />

              <FlowStep n="2" title="采集数据" hint="手套 + 摄像头">
                <StepLink
                  href="/collect"
                  icon={<Database className="w-3.5 h-3.5 text-[var(--hud-accent)] shrink-0" />}
                  label="静态采集"
                  sub="单帧手型"
                  hoverCls="hover:border-[#1677ff]/60"
                />
                {/* sub 里写"两条训练都要"是因为这件事完全反直觉：多数人以为
                    句子训练只吃句子录制。它是句子训练的必要输入，缺了训不动 */}
                <StepLink
                  href="/collect-seq"
                  icon={<Video className="w-3.5 h-3.5 text-[var(--hud-violet)] shrink-0" />}
                  label="时序采集"
                  sub="动态词 · 两条训练都要"
                  hoverCls="hover:border-[#7c3aed]/60"
                />
                {/* 句子采集与时序采集共用一个 store、一条时序链路，但它只喂 CTC
                    那条路（孤立词训练会把句子样本滤掉）。紫色标的就是这条路 */}
                <StepLink
                  href="/collect-sentence"
                  icon={<Waves className="w-3.5 h-3.5 text-[var(--hud-violet)] shrink-0" />}
                  label="句子采集"
                  sub="连续手语 · 只喂句子训练"
                  hoverCls="hover:border-[#7c3aed]/60"
                />
              </FlowStep>

              <FlowArrow />

              <FlowStep n="3" title="训练模型" hint="离线，可拔手套">
                {/* 青色，与 /train 页自己的标题色一致。原来这里是绿色（--hud-ok），
                    而绿色在本项目里表示"已连接/已就绪"，不是链路色 */}
                <StepLink
                  href="/train"
                  icon={<Brain className="w-3.5 h-3.5 text-[var(--hud-accent)] shrink-0" />}
                  label="静态训练"
                  sub="配静态采集 · 单帧 MLP"
                  hoverCls="hover:border-[#1677ff]/60"
                />
                <StepLink
                  href="/train-seq"
                  icon={<Waves className="w-3.5 h-3.5 text-[var(--hud-violet)] shrink-0" />}
                  label="时序训练"
                  sub="配时序采集 · TCN 含蒸馏"
                  hoverCls="hover:border-[#7c3aed]/60"
                />
                {/*
                  也是紫色。约定是**紫＝时序链路、青＝静态链路**
                  （见 CollectSentence.tsx 头部那条注释），句子属于时序链路
                  —— 同一个 store、同一套特征，区别只在一条录制里装几个词。
                  不要拿颜色去区分"句子 vs 单词"：那会把紫色的含义从"链路"
                  偷偷改成"子分支"，而 /collect-seq、/train-seq、/translate
                  的 MODE 开关全都按前一个含义在用它。

                  Y 形关系（句子训练还要吃时序采集的词录制）在界面上只由第 2 步
                  那两条 sub 承载 —— "两条训练都要" / "只喂句子训练"。这里曾经
                  有一段讲清 Y 形的说明文字，删掉了；别再往流程图下面加。
                */}
                <StepLink
                  href="/train-sentence"
                  icon={<Brain className="w-3.5 h-3.5 text-[var(--hud-violet)] shrink-0" />}
                  label="句子训练"
                  sub="CTC 连续手语 · 本机 Python"
                  hoverCls="hover:border-[#7c3aed]/60"
                />
              </FlowStep>

              <FlowArrow />

              <FlowStep n="4" title="使用" hint="只要手套">
                <StepLink
                  href="/translate"
                  icon={<MessageSquare className="w-3.5 h-3.5 text-[var(--hud-err)] shrink-0" />}
                  label="手语翻译"
                  sub="纯触觉推理，不开摄像头"
                  hoverCls="hover:border-[#e11d48]/60"
                />
              </FlowStep>
            </div>

            {/* 旁支：不在识别链路上，放在流程外面免得被当成必经步骤 */}
            <div className="mt-2 flex items-center gap-2">
              <span className="text-[8px] font-mono text-[var(--hud-faint)] uppercase tracking-wider shrink-0">
                旁支
              </span>
              <div className="h-px flex-1 bg-[#1677ff]/10" />
              <Link
                href="/train-skeleton"
                className="rounded-sm border border-[#d97706]/20 hover:border-[#d97706]/60 px-2 py-1 flex items-center gap-1.5 transition-colors shrink-0"
              >
                <Bone className="w-3.5 h-3.5 text-[var(--hud-warn)] shrink-0" />
                <span className="text-[10px] text-[var(--hud-soft)]">骨架训练</span>
              </Link>
              <span className="text-[8px] text-[var(--hud-faint)] hidden sm:inline">
                触觉→骨架回归，只驱动手套体检页的火柴人，不参与识别
              </span>
            </div>
          </div>

          {/* 底部提示 */}
          <div className="text-center space-y-1">
            <p className="text-[10px] text-[var(--hud-faint)] font-mono">
              POWERED BY MEDIAPIPE HANDS · TENSORFLOW.JS · WEB SERIAL API
            </p>
            <p className="text-[10px] text-[var(--hud-faint)] font-mono">
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
        style={{ backgroundColor: "var(--hud-page)" }}
      >
        <div className="flex flex-col items-center gap-6">
          <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-full border-2 border-[#1677ff]/20" />
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-[var(--hud-accent)] animate-spin" />
            <div className="absolute inset-3 rounded-full border border-[#1677ff]/10" />
            <div
              className="absolute inset-3 rounded-full border border-transparent border-b-[#e11d48]/60 animate-spin"
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
            <p className="text-[10px] text-[var(--hud-faint)] font-mono mt-4">
              首次加载模型约需 5-15 秒，请耐心等待...
            </p>
            {loadingStatus === "model" && (
              <p className="text-[10px] text-[var(--hud-dim)] font-mono">
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
      style={{ backgroundColor: "var(--hud-page)" }}
    >
      {/* 顶部状态栏 */}
      <header
        className="h-10 flex items-center justify-between px-4 border-b border-[#1677ff]/15 shrink-0"
        style={{ backgroundColor: "rgba(10, 14, 26, 0.95)" }}
      >
        <div className="flex items-center gap-3">
          <span
            className="text-xs font-bold tracking-widest"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--hud-accent)",
              textShadow: "0 0 8px rgba(0,240,255,0.4)",
            }}
          >
            HAND MOCAP
          </span>
          <div className="w-px h-4 bg-[#1677ff]/20" />
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--hud-accent)] shadow-[0_0_6px_rgba(0,240,255,0.8)] animate-pulse" />
            <span className="text-[10px] text-[#1677ff]/80 font-mono">
              LIVE
            </span>
          </div>
          {/* 手套连接指示 */}
          {gloveConnected && (
            <>
              <div className="w-px h-4 bg-[#1677ff]/20" />
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[var(--hud-ok)] shadow-[0_0_6px_rgba(0,229,160,0.8)] animate-pulse" />
                <span className="text-[10px] text-[#16a34a]/80 font-mono">
                  GLOVE
                </span>
              </div>
            </>
          )}
          {/* 录制指示 */}
          {recorder.isRecording && (
            <>
              <div className="w-px h-4 bg-[#1677ff]/20" />
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[var(--hud-err)] shadow-[0_0_6px_rgba(255,45,123,0.8)] animate-pulse" />
                <span className="text-[10px] text-[#e11d48]/80 font-mono">
                  REC
                </span>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-5 text-[10px] font-mono text-[var(--hud-dim)]">
          <span>
            FPS: <span className="text-[var(--hud-accent)]">{fps}</span>
          </span>
          <span>
            HANDS:{" "}
            <span className="text-[var(--hud-accent)]">
              {handResults?.landmarks.length ?? 0}
            </span>
          </span>
          {gloveConnected && (
            <span>
              GLOVE: <span className="text-[var(--hud-ok)]">{gloveFps}Hz</span>
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
              className="relative border border-[#1677ff]/20 rounded-sm overflow-hidden"
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
              <div className="absolute top-2 left-3 text-[9px] font-mono text-[#1677ff]/40 pointer-events-none">
                <TimeDisplay />
              </div>

              {/* 右上角分辨率 */}
              <div className="absolute top-2 right-3 text-[9px] font-mono text-[#1677ff]/40 pointer-events-none">
                {canvasSize.width}x{canvasSize.height}
              </div>

              {/* 录制指示器（左下角） */}
              {recorder.isRecording && (
                <div className="absolute bottom-2 left-3 flex items-center gap-1.5 pointer-events-none">
                  <div className="w-2 h-2 rounded-full bg-[var(--hud-err)] animate-pulse shadow-[0_0_8px_rgba(255,45,123,0.8)]" />
                  <span className="text-[10px] font-mono text-[var(--hud-err)]">
                    REC{" "}
                    {formatDurationShort(recorder.recordingDuration)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* 底部控制栏 */}
          <div className="h-12 flex items-center justify-center gap-2 px-4 border-t border-[#1677ff]/10 shrink-0">
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
            <div className="w-px h-5 bg-[#1677ff]/15 mx-1" />
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
            className="w-60 border-l border-[#1677ff]/15 overflow-y-auto shrink-0"
            style={{
              backgroundColor: "rgba(10, 14, 26, 0.95)",
              scrollbarWidth: "thin",
              scrollbarColor: "var(--hud-line-strong) transparent",
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

/** 启动页流程图的一个步骤格子：序号 + 标题 + 一句前提 + 若干个可点的模块 */
function FlowStep({
  n,
  title,
  hint,
  children,
}: {
  n: string;
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex-1 cyber-panel rounded-sm p-2 flex flex-col gap-1.5 min-w-0">
      <div className="flex items-center gap-1.5">
        <span
          className="w-4 h-4 rounded-full border border-[#1677ff]/40 text-[var(--hud-accent)] flex items-center justify-center shrink-0 text-[9px]"
          style={{ fontFamily: "'JetBrains Mono', monospace" }}
        >
          {n}
        </span>
        <span className="text-[11px] text-[var(--hud-text)] truncate">{title}</span>
      </div>
      <div className="text-[8px] text-[var(--hud-faint)] pl-[22px] -mt-1">{hint}</div>
      <div className="flex flex-col gap-1">{children}</div>
    </div>
  );
}

/** 步骤之间的箭头：窄屏竖排朝下，宽屏横排朝右 */
function FlowArrow() {
  return (
    <div className="flex items-center justify-center text-[#1677ff]/25 shrink-0 text-[10px] leading-none">
      <span className="md:hidden">▼</span>
      <span className="hidden md:inline">▶</span>
    </div>
  );
}

/**
 * 流程里可点击的模块入口。
 * hoverCls 必须传字面量类名——Tailwind 是静态扫描的，拼出来的类名不会被生成。
 */
function StepLink({
  href,
  icon,
  label,
  sub,
  hoverCls,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
  sub: string;
  hoverCls: string;
}) {
  return (
    <Link
      href={href}
      className={`rounded-sm border border-[#1677ff]/15 bg-[#1677ff]/[0.03] px-2 py-1.5 block transition-colors group ${hoverCls}`}
    >
      <div className="flex items-center gap-1.5 min-w-0">
        {icon}
        <span className="text-[10px] text-[var(--hud-soft)] group-hover:text-[var(--hud-text)] transition-colors truncate">
          {label}
        </span>
      </div>
      <div className="text-[8px] text-[var(--hud-faint)] pl-[20px] leading-tight">
        {sub}
      </div>
    </Link>
  );
}

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
        active ? "bg-[#1677ff]/15 border-[#1677ff]/40" : ""
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
        <span className="text-[var(--hud-ok)]">✓</span>
      ) : active ? (
        <span className="text-[var(--hud-warn)] animate-pulse">●</span>
      ) : pending ? (
        <span className="text-[var(--hud-dim)]">○</span>
      ) : (
        <span className="text-[var(--hud-dim)]">○</span>
      )}
      <span
        className={
          done
            ? "text-[var(--hud-dim)]"
            : active
            ? "text-[var(--hud-soft)]"
            : "text-[var(--hud-faint)]"
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
