/*
 * useHandTracking — MediaPipe Hands 手部关键点检测 Hook
 * 封装摄像头获取、MediaPipe Hands 初始化、实时检测循环
 *
 * 修复：
 * 1. 使用 script 标签加载 MediaPipe 而非动态 import（避免 ESM 兼容问题）
 * 2. video 元素独立于 React 渲染周期
 * 3. 串行检测循环，await 每帧完成
 * 4. 详细的 console.log 便于调试
 * 5. 加载状态细分，便于 UI 反馈
 */
import {
  classifyGloveSurface,
  enhanceGloveFrame,
  type EnhancedGloveFrame,
  type GloveSurfaceClassification,
  type GloveVisionRegion,
} from "@/lib/gloveVision";
import {
  mapRoiLandmarksToFrame,
  mergeHandDetections,
  type HandDetectionCandidate,
} from "@/lib/handDetectionFusion";
import { HandIdentityTracker } from "@/lib/handIdentityTracker";
import { stabilizeHandSurface } from "@/lib/handSurfaceStabilizer";
import {
  excludeDetectedGloveRegions,
  GloveRoiTracker,
} from "@/lib/gloveRoiTracker";
import {
  MediaPipeHandsRunner,
  type MediaPipeHandsLike,
} from "@/lib/mediaPipeHandsRunner";
import { useCallback, useEffect, useRef, useState } from "react";

const MEDIAPIPE_HANDS_ASSET_ROOT = "/vendor/mediapipe-hands";

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

export const FINGER_GROUPS: Record<string, number[]> = {
  thumb: [0, 1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

export const FINGER_CONNECTION_GROUPS: Record<string, [number, number][]> = {
  thumb: [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
  ],
  index: [
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
  ],
  middle: [
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
  ],
  ring: [
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
  ],
  pinky: [
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
  ],
  palm: [
    [5, 9],
    [9, 13],
    [13, 17],
  ],
};

export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export type HandSurface = "palm" | "back" | "unknown";
export type HandDetectionSource = "standard" | "glove-enhanced";

export interface HandResult {
  landmarks: HandLandmark[][];
  handedness: string[];
  /** 与 landmarks 同序，用于避免双手标签偶发重复时互相污染平滑状态。 */
  trackingIds: string[];
  /** 手套朝向：银色面为掌心，纯黑面为手背。裸手或证据不足时为 unknown。 */
  surfaces: HandSurface[];
  /** 与 surfaces 同序，范围 0..1。 */
  surfaceConfidences: number[];
  /** 与 landmarks 同序，表示当前目标为视觉增强手套的可能性，范围 0..1。 */
  gloveConfidences: number[];
  /** 与 landmarks 同序，记录每只手来自整帧还是手套 ROI 补检。 */
  detectionSources: HandDetectionSource[];
  /** 当前结果来自原始画面还是手套增强画面。 */
  detectionSource: HandDetectionSource;
}

interface UseHandTrackingOptions {
  maxHands?: number;
  minDetectionConfidence?: number;
  minTrackingConfidence?: number;
  /** 原始画面连续漏检时，启用明暗手套轮廓和边缘增强重试。 */
  gloveEnhancement?: boolean;
}

export type LoadingStatus =
  | "idle"
  | "camera"
  | "model"
  | "warmup"
  | "ready"
  | "error";

interface UseHandTrackingReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isLoading: boolean;
  isRunning: boolean;
  error: string | null;
  handResults: HandResult | null;
  handResultsRef: React.RefObject<HandResult | null>; // 直接在回调中更新，不依赖 React 渲染周期
  fps: number;
  loadingStatus: LoadingStatus;
  startTracking: () => void;
  stopTracking: () => void;
}

// 通过 script 标签加载 MediaPipe
function loadMediaPipeScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    // 如果已经加载过
    if ((window as any).Hands) {
      resolve();
      return;
    }
    const existing = document.querySelector("script[data-mediapipe-hands]");
    if (existing) {
      // 等待已有 script 加载完成
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () =>
        reject(new Error("Script load failed"))
      );
      if ((window as any).Hands) resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = `${MEDIAPIPE_HANDS_ASSET_ROOT}/hands.js`;
    script.setAttribute("data-mediapipe-hands", "true");
    script.crossOrigin = "anonymous";
    script.onload = () => {
      console.log("[HandMocap] MediaPipe script loaded");
      resolve();
    };
    script.onerror = () => {
      reject(new Error("Failed to load local MediaPipe Hands script"));
    };
    document.head.appendChild(script);
  });
}

const DETECTION_FRAME_WIDTH = 640;
const ANALYSIS_FRAME_WIDTH = 416;
const MAIN_DETECTION_INTERVAL_MS = 1000 / 30;
const ROI_DETECTION_SIZE = 256;
const ROI_RETRY_INTERVAL_MS = 180;
const ROI_RESULT_TTL_MS = 680;
const RESULT_HOLD_MS = 360;
const SURFACE_ANALYSIS_INTERVAL_MS = 125;
const UI_RESULT_INTERVAL_MS = 66;

interface MediaPipeHandedness {
  label?: string;
  score?: number;
}

interface MediaPipeHandsResults {
  multiHandLandmarks?: Array<Array<{ x: number; y: number; z?: number }>>;
  multiHandedness?: MediaPipeHandedness[];
}

interface MediaPipeHandsInstance
  extends MediaPipeHandsLike<MediaPipeHandsResults, HTMLCanvasElement> {
  setOptions(options: {
    maxNumHands: number;
    modelComplexity: number;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
  }): void;
  initialize(): Promise<void>;
}

type HandsRunner = MediaPipeHandsRunner<
  MediaPipeHandsResults,
  HTMLCanvasElement
>;

function drawVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  maxWidth: number
): boolean {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (sourceWidth <= 0 || sourceHeight <= 0) return false;

  const width = Math.min(maxWidth, sourceWidth);
  const height = Math.max(1, Math.round((width * sourceHeight) / sourceWidth));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext("2d");
  if (!context) return false;
  context.drawImage(video, 0, 0, width, height);
  return true;
}

function readVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  maxWidth: number
): ImageData | null {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (sourceWidth <= 0 || sourceHeight <= 0) return null;

  const width = Math.min(maxWidth, sourceWidth);
  const height = Math.max(1, Math.round((width * sourceHeight) / sourceWidth));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(video, 0, 0, width, height);
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

function prepareEnhancedGloveFrame(
  frame: ImageData,
  canvas: HTMLCanvasElement,
  maxHands: number,
  preferredRegions: ReadonlyArray<GloveVisionRegion> = []
): EnhancedGloveFrame | null {
  const enhanced = enhanceGloveFrame(
    frame.data,
    frame.width,
    frame.height,
    maxHands,
    preferredRegions
  );
  if (!enhanced.found || enhanced.regions.length === 0) return null;

  if (canvas.width !== frame.width || canvas.height !== frame.height) {
    canvas.width = frame.width;
    canvas.height = frame.height;
  }
  const context = canvas.getContext("2d");
  if (!context) return null;
  const outputFrame = context.createImageData(frame.width, frame.height);
  outputFrame.data.set(enhanced.pixels);
  context.putImageData(outputFrame, 0, 0);
  return enhanced;
}

function prepareRoiCanvas(
  source: HTMLCanvasElement,
  target: HTMLCanvasElement,
  region: GloveVisionRegion
): boolean {
  if (
    region.width <= 0 ||
    region.height <= 0 ||
    Math.abs(region.width - region.height) > 1
  ) {
    return false;
  }

  if (
    target.width !== ROI_DETECTION_SIZE ||
    target.height !== ROI_DETECTION_SIZE
  ) {
    target.width = ROI_DETECTION_SIZE;
    target.height = ROI_DETECTION_SIZE;
  }
  const context = target.getContext("2d");
  if (!context) return false;
  context.fillStyle = "rgb(238, 238, 238)";
  context.fillRect(0, 0, target.width, target.height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    target.width,
    target.height
  );
  return true;
}

function classifyVisibleHands(
  frame: ImageData | null,
  hands: HandLandmark[][],
  handedness: ReadonlyArray<string>
): GloveSurfaceClassification[] {
  if (!frame) {
    return hands.map(() => ({
      surface: "unknown",
      isGlove: false,
      silverRatio: 0,
      darkRatio: 0,
      confidence: 0,
      lightRatio: 0,
    }));
  }

  return hands.map((landmarks, index) =>
    classifyGloveSurface(
      frame.data,
      frame.width,
      frame.height,
      landmarks,
      handedness[index]
    )
  );
}

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function physicalHandedness(label: string): string {
  // MediaPipe Hands 假设输入已做自拍镜像；实际送入的是未镜像 video，需交换标签。
  if (label === "Left") return "Right";
  if (label === "Right") return "Left";
  return label;
}

function parseMediaPipeCandidates(
  results: MediaPipeHandsResults,
  source: HandDetectionCandidate["source"],
  region?: GloveVisionRegion,
  frameWidth?: number,
  frameHeight?: number,
  regionIndex?: number,
  regionKey?: string
): HandDetectionCandidate[] {
  const hands = results.multiHandLandmarks ?? [];
  return hands.flatMap((hand, index) => {
    const localLandmarks = hand.map(landmark => ({
      x: landmark.x,
      y: landmark.y,
      z: landmark.z ?? 0,
    }));
    const landmarks =
      source === "glove-roi"
        ? mapRoiLandmarksToFrame(
            localLandmarks,
            region,
            frameWidth ?? 0,
            frameHeight ?? 0
          )
        : localLandmarks;
    if (landmarks.length !== 21) return [];

    const metadata = results.multiHandedness?.[index];
    return [
      {
        landmarks,
        handedness: physicalHandedness(metadata?.label ?? "Unknown"),
        score: Number.isFinite(metadata?.score) ? metadata!.score! : 0,
        source,
        regionIndex,
        regionKey,
      },
    ];
  });
}

function regionsAreNearDuplicates(
  first: GloveVisionRegion,
  second: GloveVisionRegion
): boolean {
  const overlapWidth = Math.max(
    0,
    Math.min(first.x + first.width, second.x + second.width) -
      Math.max(first.x, second.x)
  );
  const overlapHeight = Math.max(
    0,
    Math.min(first.y + first.height, second.y + second.height) -
      Math.max(first.y, second.y)
  );
  const overlap = overlapWidth * overlapHeight;
  const smallerArea = Math.min(
    first.width * first.height,
    second.width * second.height
  );
  if (overlap / Math.max(smallerArea, 1) >= 0.72) return true;

  const firstCenterX = first.x + first.width * 0.5;
  const firstCenterY = first.y + first.height * 0.5;
  const secondCenterX = second.x + second.width * 0.5;
  const secondCenterY = second.y + second.height * 0.5;
  return (
    Math.hypot(firstCenterX - secondCenterX, firstCenterY - secondCenterY) <=
    Math.min(first.width, second.width) * 0.2
  );
}

function contourRegionKey(
  region: GloveVisionRegion,
  frameWidth: number,
  frameHeight: number
): string {
  const centerX = (region.x + region.width * 0.5) / frameWidth;
  const centerY = (region.y + region.height * 0.5) / frameHeight;
  const relativeSize = region.width / Math.min(frameWidth, frameHeight);
  return `contour:${Math.round(centerX * 12)}:${Math.round(centerY * 10)}:${Math.round(relativeSize * 8)}`;
}

function regionContainsPalmCenter(
  region: GloveVisionRegion,
  candidate: HandDetectionCandidate,
  frameWidth: number,
  frameHeight: number
): boolean {
  const palmIndices = [0, 5, 9, 13, 17] as const;
  let centerX = 0;
  let centerY = 0;
  for (const index of palmIndices) {
    centerX += candidate.landmarks[index].x;
    centerY += candidate.landmarks[index].y;
  }
  centerX = (centerX / palmIndices.length) * frameWidth;
  centerY = (centerY / palmIndices.length) * frameHeight;
  return (
    centerX >= region.x &&
    centerX <= region.x + region.width &&
    centerY >= region.y &&
    centerY <= region.y + region.height
  );
}

async function disposeRunnerSafely(runner: HandsRunner): Promise<void> {
  try {
    await runner.dispose();
  } catch (error) {
    console.warn("[HandMocap] Model cleanup warning:", error);
  }
}

export function useHandTracking(
  options: UseHandTrackingOptions = {}
): UseHandTrackingReturn {
  const {
    maxHands = 2,
    minDetectionConfidence = 0.55,
    minTrackingConfidence = 0.5,
    gloveEnhancement = true,
  } = options;
  const handLimit = Number.isFinite(maxHands)
    ? Math.min(2, Math.max(1, Math.trunc(maxHands)))
    : 2;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const internalVideoRef = useRef<HTMLVideoElement | null>(null);
  const primaryRunnerRef = useRef<HandsRunner | null>(null);
  const roiRunnerRef = useRef<HandsRunner | null>(null);
  const runningRef = useRef<boolean>(false);
  const startingRef = useRef<boolean>(false);
  const streamRef = useRef<MediaStream | null>(null);
  const handResultsRefInternal = useRef<HandResult | null>(null);
  const rawCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const enhancedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const roiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef(0);
  const teardownPromiseRef = useRef<Promise<void>>(Promise.resolve());
  const surfaceTracksRef = useRef(
    new Map<
      string,
      { score: number; surface: HandSurface; gloveConfidence: number }
    >()
  );

  const [isLoading, setIsLoading] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [handResults, setHandResults] = useState<HandResult | null>(null);
  const [fps, setFps] = useState(0);
  const [loadingStatus, setLoadingStatus] = useState<LoadingStatus>("idle");

  const fpsCounterRef = useRef({ frames: 0, lastTime: performance.now() });

  const queueRunnerDisposal = (runners: Array<HandsRunner | null>) => {
    const uniqueRunners = Array.from(
      new Set(runners.filter(Boolean))
    ) as HandsRunner[];
    if (uniqueRunners.length === 0) return;
    teardownPromiseRef.current = teardownPromiseRef.current
      .catch(() => undefined)
      .then(async () => {
        await Promise.all(uniqueRunners.map(disposeRunnerSafely));
      });
  };

  const startTracking = useCallback(() => {
    if (startingRef.current || runningRef.current) return;
    startingRef.current = true;
    const sessionId = ++sessionRef.current;
    // 使用非 async 包装，确保 setState 立即生效
    setIsLoading(true);
    setError(null);
    setLoadingStatus("camera");

    console.log("[HandMocap] Start tracking triggered");

    // 延迟执行 async 逻辑
    setTimeout(() => {
      if (sessionRef.current === sessionId) doStartTracking(sessionId);
    }, 50);
  }, []);

  const doStartTracking = async (sessionId: number) => {
    let localStream: MediaStream | null = null;
    let localVideo: HTMLVideoElement | null = null;
    let localPrimaryHands: MediaPipeHandsInstance | null = null;
    let localRoiHands: MediaPipeHandsInstance | null = null;
    let localPrimaryRunner: HandsRunner | null = null;
    let localRoiRunner: HandsRunner | null = null;
    const isActive = () => sessionRef.current === sessionId;

    const cleanupLocalResources = () => {
      const runners: Array<HandsRunner | null> = [];
      if (
        localPrimaryRunner &&
        primaryRunnerRef.current !== localPrimaryRunner
      ) {
        runners.push(localPrimaryRunner);
      }
      if (localRoiRunner && roiRunnerRef.current !== localRoiRunner) {
        runners.push(localRoiRunner);
      }
      queueRunnerDisposal(runners);
      if (localPrimaryHands && !localPrimaryRunner) {
        void localPrimaryHands.close().catch(() => undefined);
      }
      if (localRoiHands && !localRoiRunner) {
        void localRoiHands.close().catch(() => undefined);
      }
      localStream?.getTracks().forEach(track => track.stop());
      if (localVideo) {
        localVideo.srcObject = null;
        localVideo.remove();
      }
      if (streamRef.current === localStream) streamRef.current = null;
      if (internalVideoRef.current === localVideo) {
        internalVideoRef.current = null;
      }
      if (videoRef.current === localVideo) videoRef.current = null;
    };

    try {
      await teardownPromiseRef.current.catch(() => undefined);
      if (!isActive()) {
        cleanupLocalResources();
        return;
      }

      // 1. 获取摄像头
      console.log("[HandMocap] Requesting camera access...");
      setLoadingStatus("camera");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      });
      localStream = stream;
      if (!isActive()) {
        cleanupLocalResources();
        return;
      }
      streamRef.current = stream;
      console.log("[HandMocap] Camera access granted");

      // 创建内部 video 元素
      const video = document.createElement("video");
      video.setAttribute("playsinline", "true");
      video.setAttribute("autoplay", "true");
      video.muted = true;
      video.style.position = "fixed";
      video.style.top = "-9999px";
      video.style.left = "-9999px";
      video.style.width = "1px";
      video.style.height = "1px";
      video.style.opacity = "0.01";
      document.body.appendChild(video);
      localVideo = video;
      internalVideoRef.current = video;

      video.srcObject = stream;
      await video.play();
      if (!isActive()) {
        cleanupLocalResources();
        return;
      }
      videoRef.current = video;
      console.log("[HandMocap] Video playing, readyState:", video.readyState);

      // 2. 加载 MediaPipe Hands
      console.log("[HandMocap] Loading MediaPipe model...");
      setLoadingStatus("model");

      await loadMediaPipeScript();
      if (!isActive()) {
        cleanupLocalResources();
        return;
      }

      const HandsClass = (window as any).Hands as
        | (new (options: {
            locateFile: (file: string) => string;
          }) => MediaPipeHandsInstance)
        | undefined;
      if (!HandsClass) {
        throw new Error("MediaPipe Hands class not found after script load");
      }

      const createHands = () =>
        new HandsClass({
          locateFile: (file: string) => `${MEDIAPIPE_HANDS_ASSET_ROOT}/${file}`,
        });

      const primaryHands = createHands();
      localPrimaryHands = primaryHands;
      primaryHands.setOptions({
        maxNumHands: handLimit,
        modelComplexity: 1,
        minDetectionConfidence,
        minTrackingConfidence,
      });
      const primaryRunner = new MediaPipeHandsRunner(primaryHands, {
        resultTimeoutMs: 2_000,
      });
      localPrimaryRunner = primaryRunner;

      rawCanvasRef.current = document.createElement("canvas");
      analysisCanvasRef.current = document.createElement("canvas");
      enhancedCanvasRef.current = document.createElement("canvas");
      roiCanvasRef.current = document.createElement("canvas");
      surfaceTracksRef.current.clear();
      fpsCounterRef.current = { frames: 0, lastTime: performance.now() };

      // 3. 初始化整帧模型，再尝试初始化独立的手套 ROI 补检模型。
      console.log("[HandMocap] Warming up primary model...");
      setLoadingStatus("warmup");
      await primaryHands.initialize();
      if (!isActive()) {
        cleanupLocalResources();
        return;
      }

      if (gloveEnhancement) {
        try {
          const roiHands = createHands();
          localRoiHands = roiHands;
          roiHands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: Math.min(minDetectionConfidence, 0.4),
            minTrackingConfidence: Math.min(
              0.75,
              Math.max(minTrackingConfidence, 0.6)
            ),
          });
          const roiRunner = new MediaPipeHandsRunner(roiHands, {
            resultTimeoutMs: 2_000,
          });
          localRoiRunner = roiRunner;
          console.log("[HandMocap] Warming up glove ROI model...");
          await roiHands.initialize();
          if (!isActive()) {
            cleanupLocalResources();
            return;
          }
        } catch (roiError) {
          console.warn(
            "[HandMocap] Glove ROI model unavailable; continuing with primary model:",
            roiError
          );
          if (localRoiRunner) {
            await disposeRunnerSafely(localRoiRunner);
          } else if (localRoiHands) {
            await localRoiHands.close().catch(() => undefined);
          }
          localRoiRunner = null;
          localRoiHands = null;
        }
      }

      console.log("[HandMocap] Models ready!");
      primaryRunnerRef.current = primaryRunner;
      roiRunnerRef.current = localRoiRunner;
      runningRef.current = true;
      startingRef.current = false;
      setLoadingStatus("ready");
      setIsLoading(false);
      setIsRunning(true);

      let roiFallbackBusy = false;
      let lastRoiAttemptAt = -Infinity;
      let lastPublishedAt = -Infinity;
      let lastSurfaceAnalysisAt = -Infinity;
      let lastUiPublishAt = -Infinity;
      let lastUiSignature = "";
      let underHandLimitFrames = 0;
      let mergedSplitCursor = 0;
      let roiGeneration = 0;
      const identityTracker = new HandIdentityTracker<
        HandDetectionCandidate["source"]
      >();
      const gloveRoiTracker = new GloveRoiTracker();
      let roiCacheEntries: Array<{
        candidate: HandDetectionCandidate;
        capturedAt: number;
      }> = [];

      const publishCandidates = (
        candidates: HandDetectionCandidate[],
        frame: ImageData | null,
        now: number
      ) => {
        if (!isActive()) return;
        const stableFrame = identityTracker.update({
          landmarks: candidates.map(candidate => candidate.landmarks),
          handedness: candidates.map(candidate => candidate.handedness),
          detectionSources: candidates.map(candidate => candidate.source),
        });
        const stableCandidates = stableFrame.inputIndices.map(
          inputIndex => candidates[inputIndex]
        );

        if (stableCandidates.length === 0) {
          gloveRoiTracker.update([], [], now);
          if (
            handResultsRefInternal.current &&
            now - lastPublishedAt >= RESULT_HOLD_MS
          ) {
            handResultsRefInternal.current = null;
            lastUiSignature = "";
            setHandResults(null);
          }
          return;
        }

        const landmarks = stableCandidates.map(
          candidate => candidate.landmarks
        );
        const handedness = stableFrame.handedness;
        const trackingIds = stableFrame.trackingIds;
        const classifications = frame
          ? classifyVisibleHands(frame, landmarks, handedness)
          : null;
        const surfaces: HandSurface[] = [];
        const surfaceConfidences: number[] = [];
        const gloveConfidences: number[] = [];
        const roiTrackingIds: string[] = [];
        const roiTrackingLandmarks: HandLandmark[][] = [];

        stableCandidates.forEach((candidate, index) => {
          const key = trackingIds[index];
          const previous = surfaceTracksRef.current.get(key);
          let gloveConfidence =
            previous?.gloveConfidence ??
            (candidate.source === "glove-roi" ? 0.6 : 0);
          const classification = classifications?.[index];
          const { score, surface } = stabilizeHandSurface(
            previous,
            classification?.surface,
            classification?.confidence ?? 0
          );
          if (classification) {
            const materialRatio = clamp01(
              classification.darkRatio +
                classification.silverRatio +
                classification.lightRatio
            );
            const measuredGloveConfidence = classification.isGlove
              ? clamp01(0.5 + (materialRatio - 0.35) * 0.9)
              : clamp01((materialRatio - 0.2) * 0.8);
            gloveConfidence = previous
              ? previous.gloveConfidence * 0.55 + measuredGloveConfidence * 0.45
              : measuredGloveConfidence;
          }
          surfaceTracksRef.current.set(key, {
            score,
            surface,
            gloveConfidence,
          });

          surfaces.push(surface);
          surfaceConfidences.push(
            surface === "unknown" ? 0 : Math.max(Math.abs(score), 0.35)
          );
          gloveConfidences.push(gloveConfidence);
          if (
            gloveEnhancement &&
            (candidate.source === "glove-roi" || gloveConfidence >= 0.25)
          ) {
            roiTrackingIds.push(key);
            roiTrackingLandmarks.push(candidate.landmarks);
          }
        });
        gloveRoiTracker.update(roiTrackingIds, roiTrackingLandmarks, now);

        const detectionSources = stableCandidates.map<HandDetectionSource>(
          candidate =>
            candidate.source === "glove-roi" ? "glove-enhanced" : "standard"
        );
        const result: HandResult = {
          landmarks,
          handedness,
          trackingIds,
          surfaces,
          surfaceConfidences,
          gloveConfidences,
          detectionSources,
          detectionSource: detectionSources.includes("glove-enhanced")
            ? "glove-enhanced"
            : "standard",
        };
        lastPublishedAt = now;
        handResultsRefInternal.current = result;
        const uiSignature = `${handedness.join(",")}|${detectionSources.join(",")}`;
        if (
          uiSignature !== lastUiSignature ||
          now - lastUiPublishAt >= UI_RESULT_INTERVAL_MS
        ) {
          lastUiSignature = uiSignature;
          lastUiPublishAt = now;
          setHandResults(result);
        }
      };

      const runRoiFallback = async (
        frame: ImageData,
        primaryCandidates: HandDetectionCandidate[],
        generation: number,
        capturedAt: number
      ) => {
        const roiRunner = localRoiRunner;
        const enhancedCanvas = enhancedCanvasRef.current;
        const roiCanvas = roiCanvasRef.current;
        if (!roiRunner || roiRunner.isBroken || !enhancedCanvas || !roiCanvas) {
          return;
        }

        const predictedRegions = gloveRoiTracker.predict(
          frame.width,
          frame.height,
          capturedAt
        );
        const enhanced = prepareEnhancedGloveFrame(
          frame,
          enhancedCanvas,
          handLimit,
          predictedRegions
        );
        if (!enhanced) {
          // Keep recent ROI results and history until their TTLs expire. A
          // side-on or occluded glove can have no usable contour for one pass.
          return;
        }

        const indexedRegions = enhanced.regions.map((region, regionIndex) => ({
          region,
          regionIndex,
          regionKey: contourRegionKey(region, frame.width, frame.height),
        }));
        const neededRegionCount = Math.max(
          0,
          handLimit - primaryCandidates.length
        );
        const missingPredictions = excludeDetectedGloveRegions(
          predictedRegions,
          primaryCandidates,
          frame.width,
          frame.height
        );
        const selectedRegions: Array<{
          region: GloveVisionRegion;
          regionIndex: number;
          regionKey: string;
        }> = missingPredictions.slice(0, neededRegionCount).map(region => ({
          region,
          regionIndex: 100 + predictedRegions.indexOf(region),
          regionKey: `history:${region.trackingId}`,
        }));

        let contourCandidates = indexedRegions;
        if (indexedRegions.length > 2) {
          const splitIndex = 1 + (mergedSplitCursor++ % 2);
          contourCandidates = [indexedRegions[0], indexedRegions[splitIndex]];
        }
        if (primaryCandidates.length > 0) {
          contourCandidates = contourCandidates.filter(
            ({ region }) =>
              !primaryCandidates.some(candidate =>
                regionContainsPalmCenter(
                  region,
                  candidate,
                  frame.width,
                  frame.height
                )
              )
          );
        }
        for (const candidate of contourCandidates) {
          if (selectedRegions.length >= neededRegionCount) break;
          if (
            selectedRegions.some(selected =>
              regionsAreNearDuplicates(selected.region, candidate.region)
            )
          ) {
            continue;
          }
          selectedRegions.push(candidate);
        }

        const roiCandidates: HandDetectionCandidate[] = [];
        for (const { region, regionIndex, regionKey } of selectedRegions) {
          if (
            !isActive() ||
            generation !== roiGeneration ||
            roiRunner.isBroken
          ) {
            return;
          }
          if (!prepareRoiCanvas(enhancedCanvas, roiCanvas, region)) continue;

          const roiResults = await roiRunner.detect(roiCanvas);
          if (!isActive() || generation !== roiGeneration) return;
          roiCandidates.push(
            ...parseMediaPipeCandidates(
              roiResults,
              "glove-roi",
              region,
              frame.width,
              frame.height,
              regionIndex,
              regionKey
            )
          );
        }

        if (
          isActive() &&
          generation === roiGeneration &&
          performance.now() - capturedAt <= ROI_RESULT_TTL_MS
        ) {
          const detectedRegionKeys = new Set(
            roiCandidates.map(
              candidate =>
                candidate.regionKey ?? `index:${candidate.regionIndex ?? -1}`
            )
          );
          roiCacheEntries = roiCacheEntries.filter(
            entry =>
              capturedAt - entry.capturedAt <= ROI_RESULT_TTL_MS &&
              !detectedRegionKeys.has(
                entry.candidate.regionKey ??
                  `index:${entry.candidate.regionIndex ?? -1}`
              )
          );
          roiCacheEntries.push(
            ...roiCandidates.map(candidate => ({ candidate, capturedAt }))
          );
        }
      };

      // 4. 整帧持续跟踪，ROI 模型在后台低频补齐漏掉的手套。
      const detectLoop = async () => {
        while (runningRef.current && isActive()) {
          if (video.readyState < 2) {
            await new Promise(resolve => setTimeout(resolve, 100));
            continue;
          }
          if (roiFallbackBusy) {
            await new Promise(resolve => setTimeout(resolve, 8));
            continue;
          }

          const iterationStartedAt = performance.now();
          try {
            const rawCanvas = rawCanvasRef.current;
            if (
              !rawCanvas ||
              !drawVideoFrame(video, rawCanvas, DETECTION_FRAME_WIDTH)
            ) {
              await new Promise<void>(resolve =>
                requestAnimationFrame(() => resolve())
              );
              continue;
            }

            const primaryResults = await primaryRunner.detect(rawCanvas);
            if (!isActive()) break;

            const counter = fpsCounterRef.current;
            counter.frames++;
            const now = performance.now();
            if (now - counter.lastTime >= 1000) {
              setFps(counter.frames);
              counter.frames = 0;
              counter.lastTime = now;
            }

            const primaryCandidates = parseMediaPipeCandidates(
              primaryResults,
              "standard"
            );

            if (primaryCandidates.length >= handLimit) {
              underHandLimitFrames = 0;
              if (roiCacheEntries.length > 0) roiGeneration++;
              roiCacheEntries = [];
            } else {
              underHandLimitFrames++;
              roiCacheEntries = roiCacheEntries.filter(
                entry => now - entry.capturedAt <= ROI_RESULT_TTL_MS
              );
            }

            const merged = mergeHandDetections(
              primaryCandidates,
              [...roiCacheEntries]
                .sort((first, second) => second.capturedAt - first.capturedAt)
                .map(entry => entry.candidate),
              handLimit
            );
            const shouldAnalyzeSurface =
              merged.length > 0 &&
              now - lastSurfaceAnalysisAt >= SURFACE_ANALYSIS_INTERVAL_MS;
            const shouldTryRoi =
              gloveEnhancement &&
              primaryCandidates.length < handLimit &&
              underHandLimitFrames >= 3 &&
              localRoiRunner &&
              !localRoiRunner.isBroken &&
              now - lastRoiAttemptAt >= ROI_RETRY_INTERVAL_MS;
            const analysisCanvas = analysisCanvasRef.current;
            const analysisFrame =
              analysisCanvas && (shouldAnalyzeSurface || shouldTryRoi)
                ? readVideoFrame(video, analysisCanvas, ANALYSIS_FRAME_WIDTH)
                : null;
            if (shouldAnalyzeSurface && analysisFrame) {
              lastSurfaceAnalysisAt = now;
            }
            publishCandidates(
              merged,
              shouldAnalyzeSurface ? analysisFrame : null,
              now
            );

            if (shouldTryRoi && analysisFrame) {
              const generation = roiGeneration;
              const capturedAt = performance.now();
              roiFallbackBusy = true;
              void runRoiFallback(
                analysisFrame,
                primaryCandidates,
                generation,
                capturedAt
              )
                .catch(error => {
                  console.warn("[HandMocap] Glove ROI frame warning:", error);
                })
                .finally(() => {
                  lastRoiAttemptAt = performance.now();
                  roiFallbackBusy = false;
                });
            }
          } catch (error: any) {
            console.warn("[HandMocap] Frame error:", error?.message || error);
            if (primaryRunner.isBroken) break;
          }

          const remainingFrameBudget = Math.max(
            0,
            MAIN_DETECTION_INTERVAL_MS -
              (performance.now() - iterationStartedAt)
          );
          await new Promise(resolve =>
            setTimeout(resolve, remainingFrameBudget)
          );
        }

        if (isActive() && primaryRunner.isBroken) {
          sessionRef.current++;
          runningRef.current = false;
          startingRef.current = false;
          if (primaryRunnerRef.current === primaryRunner) {
            primaryRunnerRef.current = null;
          }
          if (roiRunnerRef.current === localRoiRunner) {
            roiRunnerRef.current = null;
          }
          queueRunnerDisposal([primaryRunner, localRoiRunner]);
          localStream?.getTracks().forEach(track => track.stop());
          if (streamRef.current === localStream) streamRef.current = null;
          if (localVideo) {
            localVideo.srcObject = null;
            localVideo.remove();
          }
          if (internalVideoRef.current === localVideo) {
            internalVideoRef.current = null;
          }
          if (videoRef.current === localVideo) videoRef.current = null;
          rawCanvasRef.current = null;
          analysisCanvasRef.current = null;
          enhancedCanvasRef.current = null;
          roiCanvasRef.current = null;
          handResultsRefInternal.current = null;
          setIsRunning(false);
          setIsLoading(false);
          setHandResults(null);
          setLoadingStatus("error");
          setError("手部检测模型运行失败，请停止后重新启动识别。");
        }
      };

      void detectLoop();
    } catch (err: any) {
      if (!isActive()) {
        cleanupLocalResources();
        return;
      }
      console.error("[HandMocap] Error:", err);
      startingRef.current = false;
      setIsLoading(false);
      setLoadingStatus("error");

      if (err.name === "NotAllowedError") {
        setError("摄像头权限被拒绝，请在浏览器设置中允许访问摄像头后重试。");
      } else if (
        err.name === "NotFoundError" ||
        err.name === "NotReadableError"
      ) {
        setError("未检测到摄像头设备，请确保摄像头已连接且未被其他程序占用。");
      } else if (err.name === "OverconstrainedError") {
        setError("摄像头不支持请求的分辨率，正在尝试降低要求...");
        // 可以在这里重试低分辨率
      } else if (
        err.message?.includes("Script") ||
        err.message?.includes("fetch") ||
        err.message?.includes("CDN")
      ) {
        setError(
          "模型文件加载失败，请检查网络连接后刷新页面重试。可能需要科学上网。"
        );
      } else {
        setError(`初始化失败: ${err.message || "未知错误"}。请刷新页面重试。`);
      }

      cleanupLocalResources();
      rawCanvasRef.current = null;
      analysisCanvasRef.current = null;
      enhancedCanvasRef.current = null;
      roiCanvasRef.current = null;
    }
  };

  const stopTracking = useCallback(() => {
    console.log("[HandMocap] Stopping tracking...");
    sessionRef.current++;
    runningRef.current = false;
    startingRef.current = false;

    const primaryRunner = primaryRunnerRef.current;
    const roiRunner = roiRunnerRef.current;
    primaryRunnerRef.current = null;
    roiRunnerRef.current = null;
    queueRunnerDisposal([primaryRunner, roiRunner]);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (internalVideoRef.current) {
      internalVideoRef.current.srcObject = null;
      internalVideoRef.current.remove();
      internalVideoRef.current = null;
    }
    videoRef.current = null;
    rawCanvasRef.current = null;
    analysisCanvasRef.current = null;
    enhancedCanvasRef.current = null;
    roiCanvasRef.current = null;
    handResultsRefInternal.current = null;
    surfaceTracksRef.current.clear();

    setIsRunning(false);
    setIsLoading(false);
    setError(null);
    setHandResults(null);
    setFps(0);
    setLoadingStatus("idle");
  }, []);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      sessionRef.current++;
      runningRef.current = false;
      startingRef.current = false;
      const primaryRunner = primaryRunnerRef.current;
      const roiRunner = roiRunnerRef.current;
      primaryRunnerRef.current = null;
      roiRunnerRef.current = null;
      queueRunnerDisposal([primaryRunner, roiRunner]);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      if (internalVideoRef.current) {
        internalVideoRef.current.srcObject = null;
        internalVideoRef.current.remove();
        internalVideoRef.current = null;
      }
      videoRef.current = null;
      rawCanvasRef.current = null;
      analysisCanvasRef.current = null;
      enhancedCanvasRef.current = null;
      roiCanvasRef.current = null;
      handResultsRefInternal.current = null;
    };
  }, []);

  return {
    videoRef,
    isLoading,
    isRunning,
    error,
    handResults,
    handResultsRef: handResultsRefInternal,
    fps,
    loadingStatus,
    startTracking,
    stopTracking,
  };
}
