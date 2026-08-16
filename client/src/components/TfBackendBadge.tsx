/*
 * TfBackendBadge — 这一轮训练/推理跑在 GPU 还是 CPU 上。
 *
 * 浏览器里的 tfjs 默认挑 `webgl` 后端（= 显卡跑），显卡驱动/上下文拿不到时会**静默**
 * 回落到 `cpu`。回落之后什么都不报错，只是慢十几倍——训练看起来"卡住不动"其实是在 CPU 上算。
 * 页面上没有任何地方能看出这件事，所以在两个训练页把后端直接标出来。
 *
 * 必须等 `tf.ready()`：在那之前 `getBackend()` 可能还没选好，读到的是空。
 */
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function TfBackendBadge() {
  const [backend, setBackend] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    tf.ready().then(() => {
      if (!cancelled) setBackend(tf.getBackend() || "unknown");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!backend) return null;
  const gpu = backend === "webgl" || backend === "webgpu";

  return (
    <span
      className="font-mono text-[9px] whitespace-nowrap"
      style={{ color: gpu ? "#00e5a0" : "#f59e0b" }}
      title={
        gpu
          ? `tfjs 后端 ${backend}：训练与推理跑在显卡上`
          : `tfjs 后端 ${backend}：没拿到 WebGL，正在用 CPU 训练，会慢十几倍`
      }
    >
      {gpu ? `GPU · ${backend}` : `CPU · ${backend}`}
    </span>
  );
}
