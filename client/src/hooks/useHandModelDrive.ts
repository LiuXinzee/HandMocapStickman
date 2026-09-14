/*
 * useHandModelDrive — 用手套原始数据驱动一只 3D 手模，**只读标定、不做标定**。
 *
 * 为什么单独抽一个 hook：`/mocap`（第 1 步）里的 `useHandCheck` 是"标定 + 自检 + 驱动"
 * 三件事的合体，它必须让标定在运行中随时变（用户刚点完"捕捉握拳"，比例条要立刻跟上）。
 * 而其他页面（`/translate` 等）只是想看手动没动、手型对不对，标定入口不在那里
 * ——连接和标定全流程只在第 1 步做一次。所以这里刻意做成：
 *
 *   - 挂载时从 localStorage **惰性读一次**弯折两点与朝向标定，之后不再重读。
 *     （标定改了要回第 1 步，回来时这一页已重新挂载，自然读到新值。）
 *   - 自带 rAF 循环填 `driveRef`，**全程不触发一次 React 重渲染**。
 *     手套 100Hz，任何走 state 的写法都会把整页每秒重渲染上百次；
 *     `HandModel` 在自己的 `useFrame` 里读这个 ref，两边都不经过 React。
 *
 * 驱动公式与 `/mocap` 完全一致（同一份标定、同一套函数），所以两页看到的手型必然相同：
 *   quaternion = applyOrientationCalib(frame.quaternion, 朝向标定)   // 未标定则原样返回
 *   curl       = bendRatios(frame.sensor_data, frame.hand, 弯折两点)  // 未标定则柔和预览
 *
 * 注意 `bendRatios` 必须传 `frame.hand`：左右手在 137 维里的五指顺序是**相反**的
 * （`sensorMapping.ts:30` vs `:66`），自己按下标取值会把左手弄错。
 *
 * 唯一的例外是 `playbackRef`（离线演示回放）：它顶掉手套数据，且**已经在上游算完**
 * ——不在这里再套一次标定。见那个参数上的注释。
 */
import { useEffect, useRef, type RefObject } from "react";
import type { HandChannel } from "@/hooks/useDualGloveSerial";
import { makeHandDrive, type HandDrive } from "@/components/HandModel";
import {
  bendRatios,
  isCalibrated,
  loadBendRange,
  type BendRange,
  type HandKey,
} from "@/lib/bendRange";
import {
  applyOrientationCalib,
  loadOrientationCalib,
  type OrientationCalib,
} from "@/lib/orientationCalib";

export interface HandModelDrive {
  driveRef: RefObject<HandDrive>;
  /** 这只手是否已做过弯折两点标定（未标定时手模只到柔和预览，UI 该标出来） */
  bendCalibrated: boolean;
  /** 这只手是否已做过朝向标定（未标定时手模朝向大概率与实手对不上） */
  orientCalibrated: boolean;
}

/**
 * @param playbackRef 非空时**顶掉手套数据**，手模照它摆（翻译页的离线演示用它播
 *   库里的真录制，见 `demoPlayback.ts`）。
 *
 *   做成"传进来一个 ref、由本 hook 的循环去读"，而不是让调用方直接写 `driveRef`：
 *   `driveRef` 必须只有一个写者。两边各自 rAF 写同一个对象的话，谁最后写的取决于
 *   两个回调的注册顺序，表现是手模在演示动作和"没数据"之间每帧闪一次 —— 而这
 *   两个循环一个在页面里、一个在 hook 里，看代码时根本不在一起。
 */
export function useHandModelDrive(
  channel: HandChannel,
  handKey: HandKey,
  playbackRef?: RefObject<HandDrive | null>
): HandModelDrive {
  const driveRef = useRef<HandDrive>(makeHandDrive());
  // handKey 在一个实例里是常量（useDualGloveSerial 把 handType 写死成 0x01/0x02），
  // 所以惰性读一次就够，不用监听换手
  // 用一个 loaded 标志，而不是判 `=== null` —— 没标定过时 load 就是返回 null，
  // 按 null 判会变成"每次渲染都去读一遍 localStorage"
  const loadedRef = useRef(false);
  const bendRangeRef = useRef<BendRange | null>(null);
  const orientRef = useRef<OrientationCalib | null>(null);
  if (!loadedRef.current) {
    loadedRef.current = true;
    bendRangeRef.current = loadBendRange(handKey);
    orientRef.current = loadOrientationCalib(handKey);
  }

  const frameRef = channel.latestFrameRef;

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const frame = frameRef.current;
      const played = playbackRef?.current;
      if (played) {
        // 回放优先。片段里的姿态与手型已经在 demoPlayback 里换算好了
        // （朝向按"相对片段首帧"、弯折走与实时同一份 bendRatios 数学），
        // 这里不能再套一次标定 —— 套第二次就是把零位减两遍
        driveRef.current.quaternion = played.quaternion;
        driveRef.current.curl = played.curl;
        driveRef.current.hasData = true;
      } else if (frame) {
        driveRef.current.quaternion = applyOrientationCalib(
          frame.quaternion,
          orientRef.current
        );
        driveRef.current.curl = bendRatios(
          frame.sensor_data,
          frame.hand,
          bendRangeRef.current
        );
        driveRef.current.hasData = true;
      } else {
        // 手套断开：停在静止姿态而不是冻在最后一帧，否则会以为还连着
        driveRef.current.hasData = false;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [frameRef, playbackRef]);

  return {
    driveRef,
    // 只捕了一个点（只有张开、没有握拳）不算标定完成，`bendRatios` 那时仍走柔和预览
    bendCalibrated: isCalibrated(bendRangeRef.current),
    orientCalibrated: orientRef.current !== null,
  };
}
