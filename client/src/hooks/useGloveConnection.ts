/*
 * useGloveConnection — 手套 WebSocket 桥接连接 Hook
 * 连接本地 Python 桥接服务，接收手套传感器数据
 *
 * 数据格式:
 * {
 *   type: "glove_frame",
 *   timestamp: number,     // Unix 时间戳(秒)
 *   hand: number,          // 传感器类型
 *   sensor_data: number[], // 256个传感器值
 *   quaternion: number[],  // [w, x, y, z]
 *   frame_id: number       // 帧序号
 * }
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface GloveFrame {
  type: "glove_frame";
  timestamp: number;
  hand: number;
  sensor_data: number[];
  quaternion: [number, number, number, number];
  acceleration: [number, number, number] | null;
  attitude: [number, number, number] | null;
  frame_id: number;
}

export interface GloveStatus {
  connected: boolean;
  fps: number;
  frameCount: number;
  errorCount: number;
  port: string;
  baudrate: number;
}

interface UseGloveConnectionOptions {
  wsUrl?: string;
  autoConnect?: boolean;
  onFrame?: (frame: GloveFrame) => void;
}

interface UseGloveConnectionReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  latestFrame: GloveFrame | null;
  gloveFps: number;
  gloveFrameCount: number;
  connect: () => void;
  disconnect: () => void;
}

export function useGloveConnection(
  options: UseGloveConnectionOptions = {}
): UseGloveConnectionReturn {
  const {
    wsUrl = "ws://localhost:8765",
    autoConnect = false,
    onFrame,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestFrame, setLatestFrame] = useState<GloveFrame | null>(null);
  const [gloveFps, setGloveFps] = useState(0);
  const [gloveFrameCount, setGloveFrameCount] = useState(0);

  // FPS 计算
  const fpsRef = useRef({ count: 0, lastTime: performance.now() });

  const connect = useCallback(() => {
    // 如果已连接或正在连接，跳过
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    setIsConnecting(true);
    setError(null);
    console.log("[Glove] Connecting to", wsUrl);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[Glove] WebSocket connected");
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "glove_frame") {
            const frame = data as GloveFrame;
            setLatestFrame(frame);
            setGloveFrameCount(frame.frame_id);

            // FPS 计算
            const counter = fpsRef.current;
            counter.count++;
            const now = performance.now();
            if (now - counter.lastTime >= 1000) {
              setGloveFps(counter.count);
              counter.count = 0;
              counter.lastTime = now;
            }

            // 回调
            if (onFrameRef.current) {
              onFrameRef.current(frame);
            }
          } else if (data.type === "connected") {
            console.log("[Glove] Bridge confirmed:", data.message);
          }
        } catch (e) {
          // 忽略解析错误
        }
      };

      ws.onerror = (e) => {
        console.warn("[Glove] WebSocket error");
        setError("WebSocket 连接错误");
      };

      ws.onclose = (e) => {
        console.log("[Glove] WebSocket closed, code:", e.code);
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null;

        if (e.code !== 1000) {
          // 非正常关闭，尝试重连
          setError("连接已断开，5秒后重试...");
          reconnectTimerRef.current = setTimeout(() => {
            setError(null);
            connect();
          }, 5000);
        }
      };
    } catch (e: any) {
      console.error("[Glove] Connection failed:", e);
      setIsConnecting(false);
      setError(`连接失败: ${e.message}`);
    }
  }, [wsUrl]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(1000);
      wsRef.current = null;
    }
    setIsConnected(false);
    setIsConnecting(false);
    setError(null);
    setLatestFrame(null);
    setGloveFps(0);
  }, []);

  // 自动连接
  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => {
      disconnect();
    };
  }, []);

  return {
    isConnected,
    isConnecting,
    error,
    latestFrame,
    gloveFps,
    gloveFrameCount,
    connect,
    disconnect,
  };
}
