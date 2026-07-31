/*
 * DataCollect — 手语数据采集标注页面（视觉+触觉同步）
 * DESIGN: Cyberpunk HUD 风格
 *
 * 功能:
 * 1. 选择目标手语词汇
 * 2. 同时启动摄像头（视觉骨架）+ 手套（触觉传感器）
 * 3. 每次采集同步录制 21 关键点 + 137 传感器 + 四元数
 * 4. 支持批量采集模式
 *
 * 修复:
 * - startCollecting 中 stopCollecting 闭包引用问题
 * - setInterval 内 async 异常被静默吞掉
 * - 添加详细 console.log 调试日志
 */
import { useDualGloveSerial } from "@/hooks/useDualGloveSerial";
import { useHandTracking } from "@/hooks/useHandTracking";
import HandCanvas from "@/components/HandCanvas";
import {
  SIGN_VOCABULARY,
  SIGN_CATEGORIES,
  getCategoryColor,
  type SignWord,
} from "@/lib/signLanguageVocab";
import {
  addSample,
  getDatasetStats,
  deleteSamplesByLabel,
  type DatasetStats,
  type TrainingSample,
} from "@/lib/datasetStore";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Camera,
  Database,
  Hand,
  Play,
  Square,
  Trash2,
  Zap,
} from "lucide-react";
import HandSkeletonAnim from "@/components/HandSkeletonAnim";

export default function DataCollect() {
  // 手套双手连接
  const {
    left: gloveLeft,
    right: gloveRight,
    isSupported: gloveSupported,
    anyConnected: gloveConnected,
    disconnectAll,
  } = useDualGloveSerial({ baudRate: 921600 });
  const gloveConnecting = gloveLeft.isConnecting || gloveRight.isConnecting;
  const gloveError = gloveLeft.error || gloveRight.error;
  const gloveFps = gloveLeft.gloveFps + gloveRight.gloveFps;
  // 预览用（优先展示右手，其次左手）
  const latestFrame = gloveRight.latestFrame ?? gloveLeft.latestFrame;

  // 摄像头 + 手部检测（双手）
  const {
    videoRef,
    isLoading: cameraLoading,
    isRunning: cameraRunning,
    error: cameraError,
    handResults,
    handResultsRef: handResultsDirectRef, // 直接从 MediaPipe 回调更新的 ref，不依赖 React 渲染
    fps: cameraFps,
    startTracking,
    stopTracking,
  } = useHandTracking({ maxHands: 2 });

  // 状态
  const [selectedWord, setSelectedWord] = useState<SignWord | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [isCollecting, setIsCollecting] = useState(false);
  const [collectCount, setCollectCount] = useState(0);
  const [targetCount, setTargetCount] = useState(20);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [message, setMessage] = useState<string>("");
  const [collectInterval, setCollectInterval] = useState(100);

  const collectTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const collectCountRef = useRef(0);
  const selectedWordRef = useRef<SignWord | null>(null);
  const targetCountRef = useRef(20);

  // 保持最新引用
  useEffect(() => {
    selectedWordRef.current = selectedWord;
  }, [selectedWord]);

  useEffect(() => {
    targetCountRef.current = targetCount;
  }, [targetCount]);

  // 加载统计信息
  const refreshStats = useCallback(async () => {
    try {
      const s = await getDatasetStats();
      setStats(s);
      console.log(
        "[DataCollect] Stats refreshed:",
        s.totalSamples,
        "samples,",
        s.labels.length,
        "classes"
      );
    } catch (err) {
      console.error("[DataCollect] Failed to refresh stats:", err);
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  // 两个设备都就绪
  const bothReady = gloveConnected && cameraRunning;

  // 从当前左右手手套帧 + 视频手部关键点构建一个双手训练样本
  const buildDualSample = useCallback(
    (label: string): TrainingSample | null => {
      const lf = gloveLeft.latestFrameRef.current;
      const rf = gloveRight.latestFrameRef.current;
      const hands = handResultsDirectRef.current;

      // handResults 已在检测出口校正为真实左右手，可直接与对应手套配对。
      const findLandmarks = (which: "Left" | "Right") => {
        if (!hands?.landmarks || hands.landmarks.length === 0) return [];
        const idx = hands.handedness?.findIndex(h => h === which) ?? -1;
        const lm = idx >= 0 ? hands.landmarks[idx] : undefined;
        return lm ? lm.map(p => ({ x: p.x, y: p.y, z: p.z })) : [];
      };

      const left = lf
        ? {
            sensor_data: [...lf.mapped_data],
            quaternion: [...lf.quaternion] as [number, number, number, number],
            landmarks: findLandmarks("Left"),
          }
        : null;
      const right = rf
        ? {
            sensor_data: [...rf.mapped_data],
            quaternion: [...rf.quaternion] as [number, number, number, number],
            landmarks: findLandmarks("Right"),
          }
        : null;

      if (!left && !right) return null;
      return { label, left, right, timestamp: Date.now() };
    },
    [gloveLeft.latestFrameRef, gloveRight.latestFrameRef, handResultsDirectRef]
  );

  // 停止采集 — 使用 ref 避免闭包问题
  const doStopCollecting = useCallback(() => {
    console.log("[DataCollect] Stopping collection...");
    if (collectTimerRef.current) {
      clearInterval(collectTimerRef.current);
      collectTimerRef.current = null;
    }
    setIsCollecting(false);
    refreshStats();
  }, [refreshStats]);

  // 开始采集
  const startCollecting = useCallback(() => {
    if (!selectedWord || !bothReady) {
      console.warn(
        "[DataCollect] Cannot start: selectedWord=",
        selectedWord?.id,
        "bothReady=",
        bothReady
      );
      return;
    }
    console.log(
      "[DataCollect] Starting collection for:",
      selectedWord.label,
      "target:",
      targetCount,
      "interval:",
      collectInterval,
      "ms"
    );

    setIsCollecting(true);
    setCollectCount(0);
    collectCountRef.current = 0;
    setMessage(`正在采集 "${selectedWord.label}" — 请保持手势稳定...`);

    let skipCount = 0;
    let successCount = 0;

    collectTimerRef.current = setInterval(() => {
      const word = selectedWordRef.current;
      const target = targetCountRef.current;

      // 检查是否已达到目标
      if (collectCountRef.current >= target) {
        console.log("[DataCollect] Target reached, stopping.");
        doStopCollecting();
        setMessage(
          `✓ 已完成 "${word?.label}" 采集 ${target} 个样本（视觉+触觉）`
        );
        return;
      }

      if (!word) {
        console.warn("[DataCollect] No word selected");
        return;
      }

      const sample = buildDualSample(word.id);
      // 至少一只手套有数据才采集
      if (!sample) {
        skipCount++;
        if (skipCount % 20 === 0) {
          console.warn(
            "[DataCollect] No glove frame available, skipped",
            skipCount,
            "times"
          );
        }
        return;
      }

      // 异步存储，但不阻塞定时器
      addSample(sample)
        .then(() => {
          successCount++;
          collectCountRef.current += 1;
          const currentCount = collectCountRef.current;
          setCollectCount(currentCount);

          if (successCount <= 3 || successCount % 10 === 0) {
            console.log(
              "[DataCollect] Sample saved:",
              successCount,
              "/",
              target,
              "| L:",
              sample.left ? "✓" : "✗",
              "R:",
              sample.right ? "✓" : "✗"
            );
          }

          if (currentCount >= target) {
            console.log(
              "[DataCollect] Collection complete:",
              currentCount,
              "samples"
            );
            doStopCollecting();
            setMessage(
              `✓ 已完成 "${word.label}" 采集 ${target} 个样本（视觉+触觉）`
            );
          }
        })
        .catch(err => {
          console.error("[DataCollect] Failed to save sample:", err);
          setMessage(`⚠ 存储失败: ${err.message || err}`);
        });
    }, collectInterval);
  }, [
    selectedWord,
    bothReady,
    targetCount,
    collectInterval,
    doStopCollecting,
    buildDualSample,
  ]);

  // 停止采集（UI 按钮用）
  const stopCollecting = doStopCollecting;

  // 单次采集
  const collectOnce = useCallback(async () => {
    if (!selectedWord || !bothReady) return;

    const sample = buildDualSample(selectedWord.id);
    if (!sample) {
      setMessage("⚠ 请确保至少一只手套已连接并有数据");
      return;
    }

    try {
      await addSample(sample);
      setCollectCount(prev => prev + 1);
      refreshStats();
      console.log(
        "[DataCollect] Single sample saved for:",
        selectedWord.label,
        "| L:",
        sample.left ? "✓" : "✗",
        "R:",
        sample.right ? "✓" : "✗"
      );
      setMessage(`✓ 已采集 1 个 "${selectedWord.label}" 样本（视觉+触觉）`);
    } catch (err: any) {
      console.error("[DataCollect] Failed to save single sample:", err);
      setMessage(`⚠ 存储失败: ${err.message || err}`);
    }
  }, [selectedWord, bothReady, refreshStats, buildDualSample]);

  // 删除某个词汇的所有样本
  const handleDeleteLabel = useCallback(
    async (label: string) => {
      if (!confirm(`确定删除所有 "${label}" 的样本？`)) return;
      await deleteSamplesByLabel(label);
      refreshStats();
      setMessage(`已删除 "${label}" 的所有样本`);
    },
    [refreshStats]
  );

  // 过滤词汇
  const filteredWords =
    selectedCategory === "all"
      ? SIGN_VOCABULARY
      : SIGN_VOCABULARY.filter(w => w.category === selectedCategory);

  // 获取视频尺寸
  const videoWidth = videoRef.current?.videoWidth || 640;
  const videoHeight = videoRef.current?.videoHeight || 480;

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
          <span
            className="text-xs font-bold tracking-widest"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "#00f0ff",
            }}
          >
            DATA COLLECTION
          </span>
          <span className="text-[9px] text-[#556677] font-mono ml-2">
            VISION + TACTILE SYNC
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          {/* 摄像头状态 */}
          {cameraRunning ? (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <Camera className="w-3 h-3" />
              CAM {cameraFps}fps
            </span>
          ) : (
            <span className="text-[#556677] flex items-center gap-1">
              <Camera className="w-3 h-3" />
              CAM OFF
            </span>
          )}
          {/* 手套状态 */}
          {gloveConnected ? (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-[#00e5a0] animate-pulse" />
              GLOVE {gloveFps}Hz
            </span>
          ) : (
            <span className="text-[#556677]">GLOVE OFF</span>
          )}
          <span className="text-[#556677]">
            SAMPLES:{" "}
            <span className="text-[#00f0ff]">{stats?.totalSamples ?? 0}</span>
          </span>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 左侧：词汇选择 */}
        <div
          className="w-64 border-r border-[#00f0ff]/15 overflow-y-auto shrink-0 p-3 space-y-3"
          style={{
            scrollbarWidth: "thin",
            scrollbarColor: "#00f0ff30 transparent",
          }}
        >
          {/* 分类筛选 */}
          <div className="space-y-1.5">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              Category Filter
            </div>
            <div className="flex flex-wrap gap-1">
              <CategoryChip
                label="全部"
                color="#00f0ff"
                active={selectedCategory === "all"}
                onClick={() => setSelectedCategory("all")}
              />
              {SIGN_CATEGORIES.map(cat => (
                <CategoryChip
                  key={cat.id}
                  label={cat.label}
                  color={cat.color}
                  active={selectedCategory === cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                />
              ))}
            </div>
          </div>

          {/* 词汇列表 */}
          <div className="space-y-1">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              Vocabulary ({filteredWords.length})
            </div>
            {filteredWords.map(word => {
              const count = stats?.labelCounts[word.id] ?? 0;
              const isSelected = selectedWord?.id === word.id;
              return (
                <button
                  key={word.id}
                  onClick={() => setSelectedWord(word)}
                  className={`w-full text-left px-2.5 py-2 rounded-sm border transition-all duration-150 ${
                    isSelected
                      ? "border-[#00f0ff]/60 bg-[#00f0ff]/10"
                      : "border-[#00f0ff]/10 hover:border-[#00f0ff]/30 bg-transparent"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-1.5 h-1.5 rounded-full"
                        style={{
                          backgroundColor: getCategoryColor(word.category),
                        }}
                      />
                      <span className="text-[11px] text-[#ccd6e0]">
                        {word.label}
                      </span>
                      <span className="text-[9px] text-[#556677]">
                        {word.pinyin}
                      </span>
                    </div>
                    <span
                      className="text-[9px] font-mono"
                      style={{
                        color:
                          count >= targetCount
                            ? "#00e5a0"
                            : count > 0
                              ? "#f59e0b"
                              : "#334455",
                      }}
                    >
                      {count}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* 中间：采集控制 + 视频预览 */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* 设备连接区 */}
          {!bothReady && (
            <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-6">
              <div className="text-center space-y-2 mb-4">
                <h2
                  className="text-lg font-bold"
                  style={{
                    color: "#00f0ff",
                    fontFamily: "'Space Grotesk', sans-serif",
                  }}
                >
                  设备连接
                </h2>
                <p className="text-[11px] text-[#8899aa]">
                  需要同时连接摄像头和触觉手套才能开始采集
                </p>
              </div>

              <div className="flex gap-6">
                {/* 摄像头卡片 */}
                <div className="cyber-panel p-4 rounded-sm w-56 space-y-3 text-center">
                  <Camera
                    className="w-10 h-10 mx-auto"
                    style={{ color: cameraRunning ? "#00e5a0" : "#556677" }}
                  />
                  <div className="text-[11px] font-mono text-[#8899aa]">
                    摄像头 (视觉骨架)
                  </div>
                  {cameraRunning ? (
                    <div className="text-[10px] text-[#00e5a0] font-mono">
                      ✓ 已启动 ({cameraFps}fps)
                    </div>
                  ) : cameraLoading ? (
                    <div className="text-[10px] text-[#f59e0b] font-mono animate-pulse">
                      加载中...
                    </div>
                  ) : (
                    <>
                      {cameraError && (
                        <div className="text-[9px] text-[#ff2d7b]">
                          {cameraError}
                        </div>
                      )}
                      <button
                        onClick={startTracking}
                        disabled={cameraLoading}
                        className="cyber-btn px-4 py-1.5 rounded-sm text-[10px] flex items-center gap-1.5 mx-auto"
                      >
                        <Camera className="w-3 h-3" />
                        启动摄像头
                      </button>
                    </>
                  )}
                </div>

                {/* 手套卡片 */}
                <div className="cyber-panel p-4 rounded-sm w-56 space-y-3 text-center">
                  <Hand
                    className="w-10 h-10 mx-auto"
                    style={{ color: gloveConnected ? "#00e5a0" : "#556677" }}
                  />
                  <div className="text-[11px] font-mono text-[#8899aa]">
                    触觉手套 (左/右手)
                  </div>
                  {!gloveSupported && (
                    <div className="text-[9px] text-[#ff2d7b]">
                      请使用 Chrome/Edge
                    </div>
                  )}
                  {gloveError && (
                    <div className="text-[9px] text-[#ff2d7b]">
                      {gloveError}
                    </div>
                  )}
                  <div className="flex flex-col gap-2">
                    <button
                      onClick={
                        gloveLeft.isConnected
                          ? gloveLeft.disconnect
                          : gloveLeft.connect
                      }
                      disabled={gloveLeft.isConnecting || !gloveSupported}
                      className="cyber-btn px-4 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
                    >
                      <Zap className="w-3 h-3" />
                      {gloveLeft.isConnected
                        ? `左手 ✓ ${gloveLeft.gloveFps}Hz (断开)`
                        : gloveLeft.isConnecting
                          ? "连接中..."
                          : "连接左手"}
                    </button>
                    <button
                      onClick={
                        gloveRight.isConnected
                          ? gloveRight.disconnect
                          : gloveRight.connect
                      }
                      disabled={gloveRight.isConnecting || !gloveSupported}
                      className="cyber-btn px-4 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
                    >
                      <Zap className="w-3 h-3" />
                      {gloveRight.isConnected
                        ? `右手 ✓ ${gloveRight.gloveFps}Hz (断开)`
                        : gloveRight.isConnecting
                          ? "连接中..."
                          : "连接右手"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 设备就绪 - 采集界面 */}
          {bothReady && (
            <div className="flex-1 flex overflow-hidden">
              {/* 视频预览区 */}
              <div className="flex-1 relative">
                <div className="absolute inset-0">
                  <HandCanvas
                    handResults={handResults}
                    videoWidth={videoWidth}
                    videoHeight={videoHeight}
                    showVideo={true}
                    videoRef={videoRef}
                  />
                </div>

                {/* 叠加：采集状态 */}
                {isCollecting && (
                  <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10">
                    <div className="bg-[#ff2d7b]/90 px-4 py-1.5 rounded-sm flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                      <span className="text-[11px] font-mono text-white font-bold">
                        REC {collectCount}/{targetCount}
                      </span>
                    </div>
                  </div>
                )}

                {/* 叠加：手部检测状态 */}
                <div className="absolute bottom-3 left-3 z-10">
                  <div
                    className="px-2 py-1 rounded-sm text-[9px] font-mono"
                    style={{
                      backgroundColor: "rgba(10, 14, 26, 0.8)",
                      color: handResults ? "#00e5a0" : "#f59e0b",
                      border: `1px solid ${handResults ? "#00e5a040" : "#f59e0b40"}`,
                    }}
                  >
                    {handResults
                      ? `✓ HAND DETECTED (${handResults.landmarks.length})`
                      : "⚠ NO HAND IN VIEW"}
                  </div>
                </div>

                {/* 叠加：数据流状态调试 */}
                <div className="absolute bottom-3 right-3 z-10">
                  <div
                    className="px-2 py-1 rounded-sm text-[9px] font-mono"
                    style={{
                      backgroundColor: "rgba(10, 14, 26, 0.8)",
                      color: "#556677",
                      border: "1px solid rgba(85, 102, 119, 0.3)",
                    }}
                  >
                    GLOVE:{" "}
                    {latestFrame
                      ? `✓ ${latestFrame.mapped_data.filter(v => v > 0).length} active`
                      : "✗ null"}
                    {" | "}
                    VISION:{" "}
                    {handResults?.landmarks?.[0]
                      ? `✓ ${handResults.landmarks[0].length}pts`
                      : "✗"}
                  </div>
                </div>
              </div>

              {/* 右侧采集控制面板 */}
              <div className="w-72 border-l border-[#00f0ff]/15 p-3 space-y-4 overflow-y-auto shrink-0">
                {/* 当前词汇 */}
                {selectedWord ? (
                  <div className="space-y-3">
                    {/* 骨架动画 */}
                    <div className="flex justify-center">
                      <div className="cyber-panel p-2 rounded-sm">
                        <HandSkeletonAnim wordId={selectedWord.id} size={120} />
                      </div>
                    </div>
                    <div className="text-center">
                      <div
                        className="text-3xl font-bold"
                        style={{
                          fontFamily: "'Space Grotesk', sans-serif",
                          color: getCategoryColor(selectedWord.category),
                          textShadow: `0 0 15px ${getCategoryColor(selectedWord.category)}40`,
                        }}
                      >
                        {selectedWord.label}
                      </div>
                      <p className="text-[11px] text-[#8899aa] mt-1">
                        {selectedWord.description}
                      </p>
                      <p className="text-[9px] text-[#556677] font-mono mt-1">
                        已采集: {stats?.labelCounts[selectedWord.id] ?? 0} /{" "}
                        {targetCount}
                      </p>
                    </div>

                    {/* 进度条 */}
                    {isCollecting && (
                      <div className="space-y-1">
                        <div className="h-2 bg-[#1a2030] rounded-full overflow-hidden border border-[#00f0ff]/20">
                          <div
                            className="h-full rounded-full transition-all duration-200"
                            style={{
                              width: `${Math.min((collectCount / targetCount) * 100, 100)}%`,
                              background:
                                "linear-gradient(90deg, #00f0ff, #00e5a0)",
                              boxShadow: "0 0 10px rgba(0,240,255,0.5)",
                            }}
                          />
                        </div>
                        <div className="flex justify-between text-[8px] font-mono text-[#556677]">
                          <span>{collectCount} 帧</span>
                          <span>{targetCount} 目标</span>
                        </div>
                      </div>
                    )}

                    {/* 采集按钮 */}
                    <div className="flex flex-col gap-2">
                      {!isCollecting ? (
                        <>
                          <button
                            onClick={startCollecting}
                            disabled={!handResults}
                            className="w-full cyber-btn px-4 py-2 rounded-sm text-[11px] flex items-center justify-center gap-2 disabled:opacity-40"
                          >
                            <Play className="w-4 h-4" />
                            批量采集 ({targetCount}帧)
                          </button>
                          <button
                            onClick={collectOnce}
                            disabled={!handResults}
                            className="w-full cyber-btn px-4 py-2 rounded-sm text-[11px] flex items-center justify-center gap-2 disabled:opacity-40"
                            style={{ borderColor: "rgba(0, 229, 160, 0.3)" }}
                          >
                            <Zap className="w-3.5 h-3.5" />
                            单次采集
                          </button>
                          {!handResults && (
                            <p className="text-[9px] text-[#f59e0b] text-center">
                              请将手放入摄像头视野
                            </p>
                          )}
                        </>
                      ) : (
                        <button
                          onClick={stopCollecting}
                          className="w-full cyber-btn cyber-btn-accent px-4 py-2 rounded-sm text-[11px] flex items-center justify-center gap-2"
                        >
                          <Square className="w-4 h-4" />
                          停止采集
                        </button>
                      )}
                    </div>

                    {/* 参数设置 */}
                    {!isCollecting && (
                      <div className="space-y-2 pt-2 border-t border-[#00f0ff]/10">
                        <div className="text-[9px] font-mono text-[#556677] uppercase">
                          Parameters
                        </div>
                        <div className="flex items-center justify-between text-[10px] font-mono">
                          <span className="text-[#8899aa]">目标数量</span>
                          <input
                            type="number"
                            value={targetCount}
                            onChange={e =>
                              setTargetCount(
                                Math.max(5, parseInt(e.target.value) || 20)
                              )
                            }
                            className="w-14 bg-[#1a2030] border border-[#00f0ff]/20 rounded-sm px-1.5 py-0.5 text-[#00f0ff] text-center"
                          />
                        </div>
                        <div className="flex items-center justify-between text-[10px] font-mono">
                          <span className="text-[#8899aa]">间隔(ms)</span>
                          <input
                            type="number"
                            value={collectInterval}
                            onChange={e =>
                              setCollectInterval(
                                Math.max(50, parseInt(e.target.value) || 100)
                              )
                            }
                            className="w-14 bg-[#1a2030] border border-[#00f0ff]/20 rounded-sm px-1.5 py-0.5 text-[#00f0ff] text-center"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 space-y-2">
                    <Database className="w-10 h-10 mx-auto text-[#556677]" />
                    <p className="text-[11px] text-[#8899aa]">
                      请在左侧选择要采集的手语词汇
                    </p>
                  </div>
                )}

                {/* 传感器预览 */}
                {latestFrame && (
                  <div className="cyber-panel p-2 rounded-sm">
                    <div className="text-[8px] font-mono text-[#556677] uppercase tracking-wider mb-1">
                      Tactile Preview
                    </div>
                    <MiniSensorGrid data={latestFrame.mapped_data} />
                  </div>
                )}

                {/* 消息 */}
                {message && (
                  <div className="text-[10px] font-mono text-[#00e5a0] text-center">
                    {message}
                  </div>
                )}

                {/* 双手连接/断开 */}
                <div className="pt-2 border-t border-[#00f0ff]/10 space-y-2">
                  <div className="flex gap-2">
                    <button
                      onClick={
                        gloveLeft.isConnected
                          ? gloveLeft.disconnect
                          : gloveLeft.connect
                      }
                      disabled={gloveLeft.isConnecting || !gloveSupported}
                      className="flex-1 text-[9px] font-mono py-1 border rounded-sm transition-colors"
                      style={{
                        color: gloveLeft.isConnected ? "#00e5a0" : "#556677",
                        borderColor: gloveLeft.isConnected
                          ? "#00e5a040"
                          : "#55667733",
                      }}
                    >
                      {gloveLeft.isConnected
                        ? `左手✓${gloveLeft.gloveFps}Hz`
                        : "连接左手"}
                    </button>
                    <button
                      onClick={
                        gloveRight.isConnected
                          ? gloveRight.disconnect
                          : gloveRight.connect
                      }
                      disabled={gloveRight.isConnecting || !gloveSupported}
                      className="flex-1 text-[9px] font-mono py-1 border rounded-sm transition-colors"
                      style={{
                        color: gloveRight.isConnected ? "#00e5a0" : "#556677",
                        borderColor: gloveRight.isConnected
                          ? "#00e5a040"
                          : "#55667733",
                      }}
                    >
                      {gloveRight.isConnected
                        ? `右手✓${gloveRight.gloveFps}Hz`
                        : "连接右手"}
                    </button>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={stopTracking}
                      className="flex-1 text-[9px] text-[#556677] hover:text-[#ff2d7b] transition-colors font-mono py-1 border border-[#556677]/20 rounded-sm"
                    >
                      关闭摄像头
                    </button>
                    <button
                      onClick={disconnectAll}
                      className="flex-1 text-[9px] text-[#556677] hover:text-[#ff2d7b] transition-colors font-mono py-1 border border-[#556677]/20 rounded-sm"
                    >
                      断开手套
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 右侧：数据集概览 */}
        <div
          className="w-56 border-l border-[#00f0ff]/15 overflow-y-auto shrink-0 p-3 space-y-3"
          style={{
            scrollbarWidth: "thin",
            scrollbarColor: "#00f0ff30 transparent",
          }}
        >
          <div className="flex items-center gap-2 pb-1 border-b border-[#00f0ff]/15">
            <Database className="w-3 h-3 text-[#00f0ff]" />
            <span className="text-[10px] font-bold tracking-widest text-[#00f0ff] font-mono">
              DATASET
            </span>
          </div>

          <div className="space-y-1">
            <DataRow
              label="TOTAL"
              value={String(stats?.totalSamples ?? 0)}
              color="#00f0ff"
            />
            <DataRow
              label="CLASSES"
              value={String(stats?.labels.length ?? 0)}
              color="#00e5a0"
            />
            <DataRow
              label="TARGET"
              value={`${targetCount}/class`}
              color="#556677"
            />
            <DataRow
              label="FEATURES"
              value="408D (126V+282T)"
              color="#da77f2"
            />
          </div>

          {/* 各词汇采集进度 */}
          <div className="space-y-1 pt-2 border-t border-[#00f0ff]/10">
            <div className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
              Per-Class Progress
            </div>
            {SIGN_VOCABULARY.filter(
              w => (stats?.labelCounts[w.id] ?? 0) > 0
            ).map(word => {
              const count = stats?.labelCounts[word.id] ?? 0;
              const pct = Math.min((count / targetCount) * 100, 100);
              return (
                <div key={word.id} className="space-y-0.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[9px] text-[#8899aa]">
                      {word.label}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[9px] font-mono text-[#556677]">
                        {count}
                      </span>
                      <button
                        onClick={() => handleDeleteLabel(word.id)}
                        className="text-[#556677] hover:text-[#ff2d7b] transition-colors"
                        title="删除该词汇所有样本"
                      >
                        <Trash2 className="w-2.5 h-2.5" />
                      </button>
                    </div>
                  </div>
                  <div className="h-1 bg-[#1a2030] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${pct}%`,
                        backgroundColor:
                          pct >= 100
                            ? "#00e5a0"
                            : pct > 50
                              ? "#f59e0b"
                              : "#00f0ff",
                      }}
                    />
                  </div>
                </div>
              );
            })}
            {(stats?.labels.length ?? 0) === 0 && (
              <p className="text-[9px] text-[#334455] italic">暂无数据</p>
            )}
          </div>

          {/* 导航链接 */}
          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-1.5">
            <Link
              href="/train"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            >
              前往训练 →
            </Link>
            <Link
              href="/translate"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
              style={{ borderColor: "rgba(0, 229, 160, 0.3)" }}
            >
              前往翻译 →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

// ===== 辅助组件 =====

function CategoryChip({
  label,
  color,
  active,
  onClick,
}: {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="px-2 py-0.5 rounded-sm text-[9px] font-mono border transition-all duration-150"
      style={{
        borderColor: active ? `${color}80` : `${color}20`,
        backgroundColor: active ? `${color}15` : "transparent",
        color: active ? color : "#556677",
      }}
    >
      {label}
    </button>
  );
}

function MiniSensorGrid({ data }: { data: number[] }) {
  const fingerColors = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#4dabf7"];

  return (
    <div className="space-y-1">
      {/* 手指压力 5x12 */}
      <div className="space-y-[2px]">
        <div className="text-[7px] font-mono text-[#445566]">FINGERS</div>
        {[0, 1, 2, 3, 4].map(finger => (
          <div key={finger} className="flex gap-[1px]">
            {Array.from({ length: 12 }, (_, i) => {
              const idx = finger * 12 + i;
              const val = data[idx] ?? 0;
              const intensity = Math.min(val / 200, 1);
              return (
                <div
                  key={i}
                  className="w-[10px] h-[6px] rounded-[1px]"
                  style={{
                    backgroundColor: fingerColors[finger],
                    opacity: 0.15 + intensity * 0.85,
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>
      {/* 弯折 */}
      <div className="flex gap-[2px]">
        {[0, 1, 2, 3, 4].map(i => {
          const val = data[60 + i] ?? 0;
          const intensity = Math.min(val / 200, 1);
          return (
            <div
              key={i}
              className="flex-1 h-[6px] rounded-[1px]"
              style={{
                backgroundColor: "#da77f2",
                opacity: 0.15 + intensity * 0.85,
              }}
            />
          );
        })}
      </div>
      {/* 手掌 */}
      <div className="space-y-[1px]">
        {Array.from({ length: 8 }, (_, r) => (
          <div key={r} className="flex gap-[1px]">
            {Array.from({ length: 9 }, (_, c) => {
              const idx = 65 + r * 9 + c;
              const val = data[idx] ?? 0;
              const intensity = Math.min(val / 200, 1);
              return (
                <div
                  key={c}
                  className="w-[10px] h-[6px] rounded-[1px]"
                  style={{
                    backgroundColor:
                      intensity < 0.3
                        ? "#00f0ff"
                        : intensity < 0.6
                          ? "#00e5a0"
                          : "#ff2d7b",
                    opacity: 0.15 + intensity * 0.85,
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>
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
