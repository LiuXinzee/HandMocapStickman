/*
 * useGloveSerial — Web Serial API 手套直连 Hook（单口/单手）
 * 直接在浏览器中通过 Web Serial API 读取一个串口的数据，无需 Python 桥接服务。
 *
 * 帧解析逻辑统一在 @/lib/gloveProtocol 的 GloveParser 中，供本 Hook 与
 * useDualGloveSerial 复用。协议详见 gloveProtocol.ts（支持 272B/296B，
 * 296B 帧含加速度与姿态角）。
 *
 * 双手场景：用 useDualGloveSerial（内部对左右手各起一个本 Hook 实例）。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  GloveParser,
  SENSOR_TYPE_MAP,
  type GloveFrame,
  type GloveType2Length,
} from "@/lib/gloveProtocol";
import {
  closeSerialResources,
  runSerialPortTransition,
} from "@/lib/serialPortLifecycle";

// 向后兼容：历史上这些从本模块导出，其它文件仍从这里引用
export { SENSOR_TYPE_MAP };
export type { GloveFrame };

interface UseGloveSerialOptions {
  baudRate?: number;
  targetFps?: number; // 目标帧率（限制 setState 频率），默认 30
  onFrame?: (frame: GloveFrame) => void;
  /** 指定该口的手别（0x01=左/0x02=右），驱动 remap；不设则用固件 sensor_type */
  handType?: number;
  /** 固定 type-2 数据段长度；不设置时自动识别 144/168。 */
  type2Len?: GloveType2Length;
}

interface UseGloveSerialReturn {
  isConnected: boolean;
  isConnecting: boolean;
  isSupported: boolean;
  error: string | null;
  latestFrame: GloveFrame | null;
  latestFrameRef: React.RefObject<GloveFrame | null>; // 全速更新的 ref，供采集/推理使用
  gloveFps: number;
  gloveFrameCount: number;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

export function useGloveSerial(
  options: UseGloveSerialOptions = {}
): UseGloveSerialReturn {
  const {
    baudRate = 921600,
    targetFps = 30,
    onFrame,
    handType,
    type2Len,
  } = options;

  const portRef = useRef<SerialPort | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(
    null
  );
  const readLoopPromiseRef = useRef<Promise<void> | null>(null);
  const runningRef = useRef(false);
  const connectionAttemptRef = useRef(0);
  const connectingRef = useRef(false);
  const mountedRef = useRef(true);
  const disconnectPromiseRef = useRef<Promise<void> | null>(null);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestFrame, setLatestFrame] = useState<GloveFrame | null>(null);
  const [gloveFps, setGloveFps] = useState(0);
  const [gloveFrameCount, setGloveFrameCount] = useState(0);

  // 全速帧 ref（不受 throttle 影响，供采集/推理直接读取）
  const latestFrameRefInternal = useRef<GloveFrame | null>(null);

  // 帧率限制：setState 最多每 targetFps 次/秒
  const throttleIntervalMs = 1000 / targetFps;
  const lastSetStateTimeRef = useRef(0);

  // FPS 计算（显示实际硬件帧率）
  const fpsRef = useRef({ count: 0, lastTime: performance.now() });

  // 解析器（每帧回调）
  const parserRef = useRef<GloveParser | null>(null);

  const isSupported = typeof navigator !== "undefined" && "serial" in navigator;

  const handleFrame = useCallback(
    (frame: GloveFrame) => {
      if (!mountedRef.current) return;

      // 全速更新 ref
      latestFrameRefInternal.current = frame;

      // FPS 计算
      const counter = fpsRef.current;
      counter.count++;
      const now = performance.now();
      if (now - counter.lastTime >= 1000) {
        setGloveFps(counter.count);
        counter.count = 0;
        counter.lastTime = now;
      }

      // 全速回调（采集/推理/录制用）
      onFrameRef.current?.(frame);

      // 限频 setState（限制 React 重渲染）
      if (now - lastSetStateTimeRef.current >= throttleIntervalMs) {
        setLatestFrame(frame);
        setGloveFrameCount(frame.frame_id);
        lastSetStateTimeRef.current = now;
      }
    },
    [throttleIntervalMs]
  );

  const disconnectInternal = useCallback((clearError: boolean) => {
    connectionAttemptRef.current++;
    runningRef.current = false;
    connectingRef.current = false;
    parserRef.current = null;
    latestFrameRefInternal.current = null;

    const resourceCleanup = closeSerialResources(
      readerRef,
      portRef,
      readLoopPromiseRef
    );
    const previousCleanup = disconnectPromiseRef.current;
    const cleanup = previousCleanup
      ? Promise.allSettled([previousCleanup, resourceCleanup]).then(results => {
          const latestResult = results[1];
          if (latestResult.status === "rejected") throw latestResult.reason;
        })
      : resourceCleanup;
    disconnectPromiseRef.current = cleanup;
    const clearCleanup = () => {
      if (disconnectPromiseRef.current === cleanup) {
        disconnectPromiseRef.current = null;
      }
    };
    void cleanup.then(clearCleanup, clearCleanup);
    if (clearError) {
      void cleanup.catch(closeError => {
        console.error("[GloveSerial] Close error:", closeError);
        if (mountedRef.current) {
          setError("串口关闭失败，端口可能仍被占用；请再次连接以重试释放。");
        }
      });
    }

    if (mountedRef.current) {
      setIsConnected(false);
      setIsConnecting(false);
      setLatestFrame(null);
      setGloveFps(0);
      setGloveFrameCount(0);
      if (clearError) setError(null);
    }

    return cleanup;
  }, []);

  const disconnect = useCallback(
    () => disconnectInternal(true),
    [disconnectInternal]
  );

  const readLoop = useCallback(
    async (
      reader: ReadableStreamDefaultReader<Uint8Array>,
      parser: GloveParser,
      attemptId: number
    ) => {
      while (runningRef.current && connectionAttemptRef.current === attemptId) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value && value.length > 0) {
          parser.push(value);
        }
      }
    },
    []
  );

  const connect = useCallback(async () => {
    if (!isSupported) {
      if (mountedRef.current) {
        setError(
          "当前浏览器不支持 Web Serial API，请使用 Chrome 或 Edge 浏览器。"
        );
      }
      return;
    }
    if (
      connectingRef.current ||
      runningRef.current ||
      portRef.current !== null
    ) {
      return;
    }

    const attemptId = ++connectionAttemptRef.current;
    connectingRef.current = true;
    if (mountedRef.current) {
      setIsConnecting(true);
      setError(null);
    }

    try {
      const port = await navigator.serial.requestPort();
      const session = await runSerialPortTransition(async () => {
        if (connectionAttemptRef.current !== attemptId || !mountedRef.current) {
          return null;
        }

        portRef.current = port;
        await port.open({
          baudRate,
          dataBits: 8,
          stopBits: 1,
          parity: "none",
          bufferSize: 4096,
        });

        if (connectionAttemptRef.current !== attemptId || !mountedRef.current) {
          if (portRef.current === port) {
            portRef.current = null;
            try {
              await port.close();
            } catch {
              // A queued disconnect may already be closing this port.
            }
          }
          return null;
        }
        if (!port.readable) {
          throw new Error("串口不可读");
        }

        const reader = port.readable.getReader();
        const parser = new GloveParser({
          handType,
          type2Len,
          onFrame: handleFrame,
        });
        readerRef.current = reader;
        parserRef.current = parser;
        runningRef.current = true;
        return { reader, parser };
      });

      if (
        !session ||
        connectionAttemptRef.current !== attemptId ||
        !mountedRef.current
      ) {
        return;
      }

      // 重置计数/节流
      fpsRef.current = { count: 0, lastTime: performance.now() };
      lastSetStateTimeRef.current = 0;

      const loopPromise = readLoop(session.reader, session.parser, attemptId);
      readLoopPromiseRef.current = loopPromise;
      void loopPromise.then(
        () => {
          if (readLoopPromiseRef.current === loopPromise) {
            readLoopPromiseRef.current = null;
          }
          if (
            connectionAttemptRef.current === attemptId &&
            runningRef.current
          ) {
            void disconnectInternal(false);
          }
        },
        (readError: unknown) => {
          if (readLoopPromiseRef.current === loopPromise) {
            readLoopPromiseRef.current = null;
          }
          if (
            connectionAttemptRef.current === attemptId &&
            runningRef.current
          ) {
            console.error("[GloveSerial] Read error:", readError);
            if (mountedRef.current) {
              const message =
                readError instanceof Error
                  ? readError.message
                  : String(readError);
              setError(`串口读取错误: ${message}`);
            }
            void disconnectInternal(false);
          }
        }
      );

      if (mountedRef.current) {
        setIsConnected(true);
      }
    } catch (e: unknown) {
      if (connectionAttemptRef.current !== attemptId || !mountedRef.current) {
        return;
      }
      console.error("[GloveSerial] Connect error:", e);
      const cleanup = disconnectInternal(false);
      const serialError = e as { name?: string; message?: string };

      if (serialError.name === "NotFoundError") {
        setError("未选择串口设备，请重试并在弹窗中选择手套对应的 COM 口。");
      } else if (serialError.message?.includes("already open")) {
        setError(
          "串口已被占用，请关闭其他使用该串口的程序（如 glove_all_v3）后重试。"
        );
      } else {
        setError(`连接失败: ${serialError.message ?? "未知错误"}`);
      }
      await cleanup;
    } finally {
      if (connectionAttemptRef.current === attemptId) {
        connectingRef.current = false;
        if (mountedRef.current) setIsConnecting(false);
      }
    }
  }, [
    baudRate,
    disconnectInternal,
    handleFrame,
    handType,
    isSupported,
    readLoop,
    type2Len,
  ]);

  useEffect(() => {
    mountedRef.current = true;
    const handlePageHide = () => {
      void disconnectInternal(false);
    };
    window.addEventListener("pagehide", handlePageHide);

    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      mountedRef.current = false;
      void disconnectInternal(false);
    };
  }, [disconnectInternal]);

  return {
    isConnected,
    isConnecting,
    isSupported,
    error,
    latestFrame,
    latestFrameRef: latestFrameRefInternal,
    gloveFps,
    gloveFrameCount,
    connect,
    disconnect,
  };
}
