/*
 * useDualGloveSerial — 双手手套同时连接 Hook
 *
 * 手语需要左右手协同，硬件为两个 COM 口各一只手（实测 COM63/COM64）。
 * 本 Hook 内部对左右手各起一个 useGloveSerial 实例：
 *   - 左手实例 handType=0x01，右手实例 handType=0x02
 *   - 该 handType 驱动 remapSensorData 与 handLabel，不依赖固件 sensor_type 字节
 *
 * 连接方式：分别调用 left.connect() / right.connect()，各弹一次串口选择窗，
 * 选对应的 CH343 口即可（一个 COM 口同一时刻只能被一个进程占用）。
 */
import { useCallback } from "react";
import { useGloveSerial } from "./useGloveSerial";
import type { GloveFrame, GloveType2Length } from "@/lib/gloveProtocol";

export interface DualGloveOptions {
  baudRate?: number;
  targetFps?: number;
  /** 固定 type-2 数据段长度；不设置时自动识别 144/168。 */
  type2Len?: GloveType2Length;
  onLeftFrame?: (frame: GloveFrame) => void;
  onRightFrame?: (frame: GloveFrame) => void;
}

export type HandChannel = ReturnType<typeof useGloveSerial>;

export interface UseDualGloveSerialReturn {
  left: HandChannel;
  right: HandChannel;
  /** 是否支持 Web Serial（两手一致） */
  isSupported: boolean;
  /** 任一只手已连接 */
  anyConnected: boolean;
  /** 两只手都已连接 */
  bothConnected: boolean;
  /** 断开两只手 */
  disconnectAll: () => Promise<void>;
}

export function useDualGloveSerial(
  options: DualGloveOptions = {}
): UseDualGloveSerialReturn {
  const {
    baudRate = 921600,
    targetFps = 30,
    type2Len,
    onLeftFrame,
    onRightFrame,
  } = options;

  const left = useGloveSerial({
    baudRate,
    targetFps,
    type2Len,
    handType: 0x01,
    onFrame: onLeftFrame,
  });

  const right = useGloveSerial({
    baudRate,
    targetFps,
    type2Len,
    handType: 0x02,
    onFrame: onRightFrame,
  });

  const disconnectLeft = left.disconnect;
  const disconnectRight = right.disconnect;
  const disconnectAll = useCallback(async () => {
    await Promise.allSettled([disconnectLeft(), disconnectRight()]);
  }, [disconnectLeft, disconnectRight]);

  return {
    left,
    right,
    isSupported: left.isSupported,
    anyConnected: left.isConnected || right.isConnected,
    bothConnected: left.isConnected && right.isConnected,
    disconnectAll,
  };
}
