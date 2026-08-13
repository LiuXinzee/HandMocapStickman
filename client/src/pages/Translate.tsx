/*
 * Translate — 实时手语翻译页面
 * DESIGN: Cyberpunk HUD 风格
 *
 * 功能:
 * 1. 连接手套后实时推理
 * 2. 显示识别结果（大字体 + 置信度）
 * 3. 历史翻译记录（句子拼接）
 * 4. Top-K 候选词显示
 */
import { useDualGloveSerial } from "@/hooks/useDualGloveSerial";
import {
  predict,
  isModelLoaded,
  getLoadedLabels,
  loadModelFromSaved,
} from "@/lib/signLanguageModel";
import { getLatestModel } from "@/lib/datasetStore";
import { getWordById, getCategoryColor } from "@/lib/signLanguageVocab";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "wouter";
import {
  ArrowLeft,
  Brain,
  Hand,
  MessageSquare,
  Trash2,
  Volume2,
  Zap,
} from "lucide-react";

interface TranslationEntry {
  word: string;
  label: string;
  confidence: number;
  timestamp: number;
}

export default function Translate() {
  // 手套双手连接
  const { left: gloveLeft, right: gloveRight, isSupported, anyConnected, disconnectAll } =
    useDualGloveSerial({ baudRate: 921600 });
  const isConnected = anyConnected;
  const isConnecting = gloveLeft.isConnecting || gloveRight.isConnecting;
  const gloveError = gloveLeft.error || gloveRight.error;
  const gloveFps = gloveLeft.gloveFps + gloveRight.gloveFps;

  // 状态
  const [modelReady, setModelReady] = useState(isModelLoaded());
  const [currentPrediction, setCurrentPrediction] = useState<{
    label: string;
    word: string;
    confidence: number;
    allProbabilities: Array<{ label: string; probability: number }>;
  } | null>(null);
  const [history, setHistory] = useState<TranslationEntry[]>([]);
  const [isTranslating, setIsTranslating] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [smoothingWindow, setSmoothingWindow] = useState(5); // 平滑窗口
  const [message, setMessage] = useState("");

  // 平滑缓冲区
  const predictionBufferRef = useRef<string[]>([]);
  const lastAddedWordRef = useRef<string>("");
  const lastAddedTimeRef = useRef<number>(0);
  const translateIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const confidenceThresholdRef = useRef(confidenceThreshold);
  const smoothingWindowRef = useRef(smoothingWindow);

  // 保持 ref 与 state 同步
  useEffect(() => {
    confidenceThresholdRef.current = confidenceThreshold;
  }, [confidenceThreshold]);
  useEffect(() => {
    smoothingWindowRef.current = smoothingWindow;
  }, [smoothingWindow]);

  // 自动加载最新模型
  useEffect(() => {
    if (!isModelLoaded()) {
      getLatestModel().then((model) => {
        if (model) {
          loadModelFromSaved(model).then(() => {
            setModelReady(true);
            setMessage(`✓ 已自动加载模型 "${model.name}"`);
          });
        }
      });
    } else {
      setModelReady(true);
    }
  }, []);

  // 推理循环 — 使用 ref 读取最新帧，避免闭包陷阱
  useEffect(() => {
    if (!isTranslating || !isConnected || !modelReady) return;

    console.log("[Translate] Starting inference loop...");

    translateIntervalRef.current = setInterval(() => {
      // 从左右手全速 ref 读取，构建双手触觉输入
      const lf = gloveLeft.latestFrameRef.current;
      const rf = gloveRight.latestFrameRef.current;
      if (!lf && !rf) {
        return;
      }
      const leftInput = lf
        ? { sensor_data: lf.mapped_data, quaternion: lf.quaternion }
        : null;
      const rightInput = rf
        ? { sensor_data: rf.mapped_data, quaternion: rf.quaternion }
        : null;

      const result = predict(leftInput, rightInput);
      if (!result) {
        console.warn("[Translate] predict() returned null");
        return;
      }

      const word = getWordById(result.label);
      setCurrentPrediction({
        label: result.label,
        word: word?.label ?? result.label,
        confidence: result.confidence,
        allProbabilities: result.allProbabilities.slice(0, 5),
      });

      // 平滑处理：连续 N 帧相同结果才确认
      const window = smoothingWindowRef.current;
      predictionBufferRef.current.push(result.label);
      if (predictionBufferRef.current.length > window) {
        predictionBufferRef.current.shift();
      }

      // 检查是否稳定
      if (predictionBufferRef.current.length >= window) {
        const allSame = predictionBufferRef.current.every(
          (l) => l === result.label
        );
        const now = Date.now();
        const threshold = confidenceThresholdRef.current;
        if (
          allSame &&
          result.confidence >= threshold &&
          (result.label !== lastAddedWordRef.current ||
            now - lastAddedTimeRef.current > 2000) // 同一个词至少间隔2秒
        ) {
          // 确认识别结果
          const entry: TranslationEntry = {
            word: word?.label ?? result.label,
            label: result.label,
            confidence: result.confidence,
            timestamp: now,
          };
          setHistory((prev) => [...prev, entry]);
          lastAddedWordRef.current = result.label;
          lastAddedTimeRef.current = now;
          predictionBufferRef.current = [];
        }
      }
    }, 100); // 10Hz 推理频率

    return () => {
      console.log("[Translate] Stopping inference loop.");
      if (translateIntervalRef.current) {
        clearInterval(translateIntervalRef.current);
        translateIntervalRef.current = null;
      }
    };
  }, [isTranslating, isConnected, modelReady]); // 不再依赖 latestFrame/confidenceThreshold/smoothingWindow

  // 开始/停止翻译
  const toggleTranslation = useCallback(() => {
    if (isTranslating) {
      setIsTranslating(false);
      setCurrentPrediction(null);
      predictionBufferRef.current = [];
    } else {
      setIsTranslating(true);
    }
  }, [isTranslating]);

  // 清除历史
  const clearHistory = useCallback(() => {
    setHistory([]);
    lastAddedWordRef.current = "";
  }, []);

  // 组合翻译文本
  const translatedText = history.map((h) => h.word).join(" ");

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: "#0a0e1a" }}
    >
      {/* 顶部导航 */}
      <header className="min-h-12 flex flex-wrap items-center justify-between gap-2 px-4 py-2 border-b border-[#00f0ff]/15 shrink-0">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="cyber-btn px-2 py-1 rounded-sm text-[10px] flex items-center gap-1"
          >
            <ArrowLeft className="w-3 h-3" />
            返回
          </Link>
          <div className="w-px h-5 bg-[#00f0ff]/20" />
          <span className="text-xs font-bold tracking-widest text-[#00f0ff] font-mono">
            SIGN LANGUAGE TRANSLATOR
          </span>
        </div>
        <nav
          className="flex items-center border border-[#00f0ff]/20 rounded-sm p-0.5 font-mono text-[10px]"
          aria-label="识别模式"
        >
          <Link
            href="/translate"
            aria-current="page"
            className="px-3 py-1 bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30 rounded-sm"
          >
            静态识别
          </Link>
          <Link
            href="/translate-dynamic"
            className="px-3 py-1 text-[#667788] hover:text-[#00f0ff] transition-colors"
          >
            动态识别
          </Link>
        </nav>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          {modelReady && (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <Brain className="w-3 h-3" />
              MODEL
            </span>
          )}
          {isConnected && (
            <span className="text-[#00e5a0] flex items-center gap-1">
              <Hand className="w-3 h-3" />
              GLOVE {gloveFps}Hz
            </span>
          )}
          {isTranslating && (
            <span className="text-[#ff2d7b] flex items-center gap-1 animate-pulse">
              <Volume2 className="w-3 h-3" />
              LIVE
            </span>
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 主内容区 */}
        <div className="flex-1 flex flex-col">
          {/* 翻译输出区 */}
          <div className="flex-1 flex flex-col items-center justify-center p-8 space-y-8">
            {/* 模型未加载 */}
            {!modelReady && (
              <div className="text-center space-y-3">
                <Brain className="w-16 h-16 mx-auto text-[#334455]" />
                <p className="text-sm text-[#556677]">请先训练并加载模型</p>
                <Link
                  href="/train"
                  className="cyber-btn px-4 py-2 rounded-sm text-xs inline-flex items-center gap-2"
                >
                  前往训练 →
                </Link>
              </div>
            )}

            {/* 手套未连接 */}
            {modelReady && !isConnected && (
              <div className="text-center space-y-4">
                <Hand className="w-16 h-16 mx-auto text-[#556677]" />
                <p className="text-sm text-[#8899aa]">连接手套开始翻译</p>
                {gloveError && (
                  <p className="text-[10px] text-[#ff2d7b]">{gloveError}</p>
                )}
                <div className="flex items-center gap-2 justify-center">
                  <button
                    onClick={gloveLeft.connect}
                    disabled={gloveLeft.isConnecting || !isSupported || gloveLeft.isConnected}
                    className="cyber-btn px-5 py-2.5 rounded-sm text-xs flex items-center gap-2 disabled:opacity-50"
                  >
                    <Zap className="w-4 h-4" />
                    {gloveLeft.isConnected ? "左手 ✓" : gloveLeft.isConnecting ? "连接中..." : "连接左手"}
                  </button>
                  <button
                    onClick={gloveRight.connect}
                    disabled={gloveRight.isConnecting || !isSupported || gloveRight.isConnected}
                    className="cyber-btn px-5 py-2.5 rounded-sm text-xs flex items-center gap-2 disabled:opacity-50"
                  >
                    <Zap className="w-4 h-4" />
                    {gloveRight.isConnected ? "右手 ✓" : gloveRight.isConnecting ? "连接中..." : "连接右手"}
                  </button>
                </div>
                <p className="text-[10px] text-[#556677]">连接任一只手即可翻译；双手手语请两只都连</p>
              </div>
            )}

            {/* 就绪状态 - 可以翻译 */}
            {modelReady && isConnected && (
              <>
                {/* 当前识别结果 */}
                <div className="text-center space-y-4">
                  {currentPrediction ? (
                    <>
                      <div
                        className="text-7xl font-bold transition-all duration-300"
                        style={{
                          fontFamily: "'Space Grotesk', sans-serif",
                          color:
                            currentPrediction.confidence >= confidenceThreshold
                              ? getCategoryColor(
                                  getWordById(currentPrediction.label)?.category ?? ""
                                )
                              : "#556677",
                          textShadow:
                            currentPrediction.confidence >= confidenceThreshold
                              ? `0 0 30px ${getCategoryColor(
                                  getWordById(currentPrediction.label)?.category ?? ""
                                )}40`
                              : "none",
                          opacity:
                            currentPrediction.confidence >= 0.5
                              ? 1
                              : 0.4,
                        }}
                      >
                        {currentPrediction.word}
                      </div>
                      {/* 置信度条 */}
                      <div className="w-48 mx-auto space-y-1">
                        <div className="h-2 bg-[#1a2030] rounded-full overflow-hidden border border-[#00f0ff]/20">
                          <div
                            className="h-full rounded-full transition-all duration-200"
                            style={{
                              width: `${currentPrediction.confidence * 100}%`,
                              backgroundColor:
                                currentPrediction.confidence >= confidenceThreshold
                                  ? "#00e5a0"
                                  : currentPrediction.confidence >= 0.5
                                  ? "#f59e0b"
                                  : "#ff2d7b",
                            }}
                          />
                        </div>
                        <div className="text-[10px] font-mono text-[#556677] text-center">
                          置信度: {(currentPrediction.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </>
                  ) : isTranslating ? (
                    <div className="space-y-2">
                      <div className="text-4xl text-[#334455] animate-pulse">
                        ...
                      </div>
                      <p className="text-[10px] text-[#556677] font-mono">
                        等待手势输入
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <MessageSquare className="w-12 h-12 mx-auto text-[#334455]" />
                      <p className="text-sm text-[#556677]">
                        点击"开始翻译"进入实时识别模式
                      </p>
                    </div>
                  )}
                </div>

                {/* 控制按钮 */}
                <div className="flex items-center gap-3">
                  <button
                    onClick={toggleTranslation}
                    className={`cyber-btn px-6 py-2.5 rounded-sm text-xs flex items-center gap-2 ${
                      isTranslating ? "cyber-btn-accent" : ""
                    }`}
                  >
                    {isTranslating ? (
                      <>
                        <div className="w-2 h-2 rounded-full bg-[#ff2d7b] animate-pulse" />
                        停止翻译
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-4 h-4" />
                        开始翻译
                      </>
                    )}
                  </button>
                </div>

                {/* 翻译历史文本 */}
                {history.length > 0 && (
                  <div className="w-full max-w-2xl">
                    <div className="cyber-panel p-4 rounded-sm">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[9px] font-mono text-[#556677] uppercase tracking-wider">
                          Translation Output
                        </span>
                        <button
                          onClick={clearHistory}
                          className="text-[#556677] hover:text-[#ff2d7b] transition-colors"
                          title="清除"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                      <p
                        className="text-lg leading-relaxed"
                        style={{
                          fontFamily: "'Space Grotesk', sans-serif",
                          color: "#ccd6e0",
                        }}
                      >
                        {translatedText}
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* 消息 */}
            {message && (
              <div className="text-[10px] font-mono text-[#00e5a0]">
                {message}
              </div>
            )}
          </div>
        </div>

        {/* 右侧面板 */}
        <div className="w-60 border-l border-[#00f0ff]/15 overflow-y-auto p-3 space-y-4 shrink-0">
          {/* Top-K 候选 */}
          {currentPrediction && isTranslating && (
            <Section title="CANDIDATES">
              <div className="space-y-1">
                {currentPrediction.allProbabilities.map((p, i) => {
                  const word = getWordById(p.label);
                  return (
                    <div
                      key={p.label}
                      className="flex items-center justify-between text-[10px] font-mono"
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-[#556677] w-3">{i + 1}.</span>
                        <span
                          style={{
                            color:
                              p.probability >= confidenceThreshold
                                ? "#ccd6e0"
                                : "#556677",
                          }}
                        >
                          {word?.label ?? p.label}
                        </span>
                      </div>
                      <span
                        style={{
                          color:
                            p.probability >= confidenceThreshold
                              ? "#00e5a0"
                              : p.probability >= 0.3
                              ? "#f59e0b"
                              : "#334455",
                        }}
                      >
                        {(p.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {/* 参数设置 */}
          <Section title="SETTINGS">
            <div className="space-y-2">
              <div className="space-y-1">
                <div className="flex justify-between text-[9px] font-mono text-[#556677]">
                  <span>置信度阈值</span>
                  <span className="text-[#00f0ff]">
                    {(confidenceThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="0.95"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) =>
                    setConfidenceThreshold(parseFloat(e.target.value))
                  }
                  className="w-full h-1 bg-[#1a2030] rounded-full appearance-none cursor-pointer"
                  style={{ accentColor: "#00f0ff" }}
                />
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-[9px] font-mono text-[#556677]">
                  <span>平滑窗口</span>
                  <span className="text-[#00f0ff]">{smoothingWindow} 帧</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="15"
                  step="1"
                  value={smoothingWindow}
                  onChange={(e) =>
                    setSmoothingWindow(parseInt(e.target.value))
                  }
                  className="w-full h-1 bg-[#1a2030] rounded-full appearance-none cursor-pointer"
                  style={{ accentColor: "#00f0ff" }}
                />
              </div>
            </div>
          </Section>

          {/* 模型信息 */}
          {modelReady && (
            <Section title="MODEL INFO">
              <div className="text-[9px] font-mono text-[#556677] space-y-0.5">
                <p>
                  Classes:{" "}
                  <span className="text-[#00f0ff]">
                    {getLoadedLabels().length}
                  </span>
                </p>
                <p>
                  Vocab:{" "}
                  <span className="text-[#8899aa]">
                    {getLoadedLabels()
                      .map((l) => getWordById(l)?.label ?? l)
                      .slice(0, 8)
                      .join(", ")}
                    {getLoadedLabels().length > 8 ? "..." : ""}
                  </span>
                </p>
              </div>
            </Section>
          )}

          {/* 翻译历史 */}
          {history.length > 0 && (
            <Section title="HISTORY">
              <div className="space-y-0.5 max-h-40 overflow-y-auto">
                {history
                  .slice(-20)
                  .reverse()
                  .map((entry, i) => (
                    <div
                      key={i}
                      className="flex justify-between text-[9px] font-mono"
                    >
                      <span className="text-[#8899aa]">{entry.word}</span>
                      <span className="text-[#556677]">
                        {(entry.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          {/* 导航 */}
          <div className="pt-3 border-t border-[#00f0ff]/10 space-y-1.5">
            <Link
              href="/collect"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            >
              ← 采集数据
            </Link>
            <Link
              href="/train"
              className="w-full cyber-btn px-3 py-1.5 rounded-sm text-[10px] flex items-center justify-center gap-1.5"
            >
              ← 训练模型
            </Link>
          </div>

          {/* 双手连接/断开控制 */}
          {isConnected && (
            <div className="flex flex-col gap-1 pt-2">
              <div className="flex items-center justify-center gap-2 text-[10px] font-mono">
                <button
                  onClick={gloveLeft.isConnected ? gloveLeft.disconnect : gloveLeft.connect}
                  className={gloveLeft.isConnected ? "text-[#00e5a0] hover:text-[#ff2d7b]" : "text-[#556677] hover:text-[#00e5a0]"}
                >
                  左手 {gloveLeft.isConnected ? `✓${gloveLeft.gloveFps}Hz(断开)` : "○(连接)"}
                </button>
                <span className="text-[#334455]">|</span>
                <button
                  onClick={gloveRight.isConnected ? gloveRight.disconnect : gloveRight.connect}
                  className={gloveRight.isConnected ? "text-[#00e5a0] hover:text-[#ff2d7b]" : "text-[#556677] hover:text-[#00e5a0]"}
                >
                  右手 {gloveRight.isConnected ? `✓${gloveRight.gloveFps}Hz(断开)` : "○(连接)"}
                </button>
              </div>
              <button
                onClick={disconnectAll}
                className="w-full text-[10px] text-[#556677] hover:text-[#ff2d7b] transition-colors font-mono text-center"
              >
                全部断开
              </button>
            </div>
          )}
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
      <div className="flex items-center gap-2 pb-1 border-b border-[#00f0ff]/15">
        <div className="w-1 h-3 bg-[#00f0ff] rounded-full shadow-[0_0_4px_rgba(0,240,255,0.6)]" />
        <span className="text-[10px] font-bold tracking-widest text-[#00f0ff] font-mono">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}
