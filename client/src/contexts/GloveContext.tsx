/*
 * GloveContext — 全应用唯一的手套连接
 *
 * 为什么要有这个文件：
 * 一个 COM 口同一时刻只能被一个持有者打开。以前 Home / DataCollect / SequenceCollect /
 * VirtualMocap / Translate 各自调一次 useDualGloveSerial，换路由时上一页的 hook 连同它
 * 持有的 reader 一起被丢弃、但串口还开着（useGloveSerial 没有卸载清理），下一页再
 * port.open() 就会撞上 "already open"。表现就是"每走一步都要重连手套"，
 * 于是第 1 步做的 IMU 零位与弯折标定全部白做。
 *
 * 解法：把连接提到 <Router/> 之上，全应用只调一次 useDualGloveSerial。
 *
 * 性能：provider 每秒 setState 30~60 次（左右手各按 targetFps 节流），但 children 是
 * App 里创建的稳定元素引用，React 会跳过整棵子树 —— 只有真正 useGloves() 的组件重渲染，
 * 也就是今天本来就在重渲染的那几个页面。/train、/train-seq 不订阅就完全不受影响。
 *
 * 帧回调：各页要的回调不一样（录制推帧 / 推理喂窗口 / 采集缓存），所以 provider 自己持
 * 一个监听者 Set，传给 useDualGloveSerial 的是稳定的分发函数。页面用 useGloveFrames()
 * 注册，卸载自动注销；回调存在 ref 里，所以调用方写不写 useCallback 都不会反复重订阅。
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
} from "react";
import {
  useDualGloveSerial,
  type UseDualGloveSerialReturn,
} from "@/hooks/useDualGloveSerial";
import type { GloveFrame } from "@/lib/gloveProtocol";

type FrameListener = (frame: GloveFrame) => void;

interface GloveContextValue extends UseDualGloveSerialReturn {
  /** 内部用：注册一对帧回调，返回注销函数。页面请用 useGloveFrames() */
  subscribe: (onLeft?: FrameListener, onRight?: FrameListener) => () => void;
}

const GloveContext = createContext<GloveContextValue | undefined>(undefined);

export function GloveProvider({ children }: { children: React.ReactNode }) {
  const leftListeners = useRef(new Set<FrameListener>());
  const rightListeners = useRef(new Set<FrameListener>());

  // 稳定的分发函数：身份不变，所以 useGloveSerial 里的 onFrameRef 不会有抖动
  const onLeftFrame = useCallback((frame: GloveFrame) => {
    leftListeners.current.forEach(fn => fn(frame));
  }, []);
  const onRightFrame = useCallback((frame: GloveFrame) => {
    rightListeners.current.forEach(fn => fn(frame));
  }, []);

  const dual = useDualGloveSerial({
    baudRate: 921600,
    onLeftFrame,
    onRightFrame,
  });

  const subscribe = useCallback(
    (onLeft?: FrameListener, onRight?: FrameListener) => {
      if (onLeft) leftListeners.current.add(onLeft);
      if (onRight) rightListeners.current.add(onRight);
      return () => {
        if (onLeft) leftListeners.current.delete(onLeft);
        if (onRight) rightListeners.current.delete(onRight);
      };
    },
    []
  );

  const value = useMemo<GloveContextValue>(
    () => ({ ...dual, subscribe }),
    [dual, subscribe]
  );

  return (
    <GloveContext.Provider value={value}>{children}</GloveContext.Provider>
  );
}

/** 取全应用唯一的手套连接。返回结构与 useDualGloveSerial 一致。 */
export function useGloves(): UseDualGloveSerialReturn {
  const ctx = useContext(GloveContext);
  if (!ctx) {
    throw new Error("useGloves must be used within GloveProvider");
  }
  return ctx;
}

/**
 * 订阅全速帧回调（不受 targetFps 节流影响，采集/推理用）。
 * 回调存在 ref 里，每次渲染刷新，所以传内联闭包也不会反复重订阅。
 */
export function useGloveFrames(
  onLeft?: FrameListener,
  onRight?: FrameListener
): void {
  const ctx = useContext(GloveContext);
  if (!ctx) {
    throw new Error("useGloveFrames must be used within GloveProvider");
  }
  const { subscribe } = ctx;

  const leftRef = useRef(onLeft);
  const rightRef = useRef(onRight);
  leftRef.current = onLeft;
  rightRef.current = onRight;

  useEffect(() => {
    return subscribe(
      frame => leftRef.current?.(frame),
      frame => rightRef.current?.(frame)
    );
  }, [subscribe]);
}
