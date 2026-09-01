"""
把 keras 模型的**权重**导成浏览器能直接吃的 weights.bin + weights.json。

===== 为什么不用 export_to_tfjs.py =====

tensorflowjs 的转换器在**本机(Windows)装不上**,而且不是版本问题:
`tensorflowjs/converters/tf_saved_model_conversion_v2.py` 无条件
`import tensorflow_decision_forests`,而 TF-DF 只发 Linux/macOS 轮子(官方文档写明
Windows 要走 WSL)。桩掉它之后下一关是 tfjs 用的是 tf_keras(Keras 2),读不了
Keras 3 存的 `.keras`。两个坑叠起来,不值得为一个只有 8 层的网络去趟。

===== 这条路为什么更稳 =====

`client/src/lib/sequenceModel.ts` 里的 `tcnBackbone` 已经用 tfjs 把**同一个结构**
搭出来了(这个仓库本来就要求两边逐层同构 —— 见 tcn_backbone 的注释)。所以浏览器
不需要"从 JSON 反序列化结构",它自己搭,只需要把权重填进去。

好处是错配会**立刻炸**:逐层按名字取、逐个张量比形状,不一致就抛错并说清是哪一层。
走转换器时结构不兼容的典型症状是 "Unknown layer" 或者更糟 —— 加载成功但权重
对错了位,线上表现为"能跑、有置信度、全错"。

===== 格式 =====

weights.bin  : 所有张量按 weights.json 里的顺序,float32 小端,首尾相接
weights.json : {
    "format": "seq-weights-1.0",
    "layers": [{"name": "student_b1_conv", "weights": [{"shape": [5,294,64], "offset": 0, "count": 94080}]}],
    "meta": {...}          # 原样带上 *_meta.json,labels 顺序就是输出下标
  }

offset 是**元素**下标不是字节 —— JS 侧 `new Float32Array(buf, byteOffset, count)` 要求
byteOffset 是 4 的倍数,用元素下标乘 4 天然满足,不会踩对齐。

用法:
    python export_weights.py                                  # 句子模型
    python export_weights.py --model out/student.keras --meta out/student_meta.json \
                             --out ../client/public/models/seq_student
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def export(model_path: Path, meta_path: Path, out_dir: Path) -> dict:
    import keras

    if not model_path.exists():
        raise SystemExit(f"找不到 {model_path},先跑 train_seq.py")

    model = keras.models.load_model(model_path, compile=False)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    layers: list[dict] = []
    offset = 0
    for layer in model.layers:
        ws = layer.get_weights()
        if not ws:
            continue  # 激活/池化/dropout 没有权重,不占位
        entries = []
        for w in ws:
            a = np.ascontiguousarray(w, dtype=np.float32)
            entries.append({"shape": list(a.shape), "offset": offset, "count": int(a.size)})
            chunks.append(a.reshape(-1))
            offset += int(a.size)
        layers.append({"name": layer.name, "weights": entries})

    if not chunks:
        raise SystemExit("模型里一个权重都没有,不对劲")

    blob = np.concatenate(chunks)
    assert len(blob) == offset
    (out_dir / "weights.bin").write_bytes(blob.tobytes())

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        print(f"⚠️  没找到 {meta_path} —— 浏览器少了 labels 就无法把输出下标翻回词。")

    # 导出时刻(epoch 毫秒)。孤立词那一档浏览器里同时存在两个模型:这个部署产物,
    # 和用户在 /train-sequence 页面内训出来、存在 IndexedDB 的那个。选型规则是
    # "谁更新用谁"(与 Translate 里静态/时序之间的既有约定一致),而 SavedModel 的
    # createdAt 也是 epoch 毫秒 —— 两者可直接比。
    # 用**导出时刻**而不是训练时刻:导出才是"这份权重进浏览器"的时刻,重新导一次
    # 就该重新赢过页面内那个。
    meta["exportedAt"] = int(time.time() * 1000)

    manifest = {"format": "seq-weights-1.0", "layers": layers, "meta": meta}
    (out_dir / "weights.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"blob": blob, "manifest": manifest, "model": model}


def verify(model, out_dir: Path, seq_len: int, frame_dim: int) -> None:
    """
    回读一遍:按 manifest 把权重塞回一个**新建**的同结构模型,比对两者输出。

    这不是形式主义 —— 它验证的正是浏览器要做的那件事(自己搭结构 + 按名字填权重)。
    在 Python 这边验过,浏览器那边对不上就只可能是 tfjs 的层实现或权重顺序不同,
    排查范围小很多。
    """
    import keras

    from train_seq import build_student

    manifest = json.loads((out_dir / "weights.json").read_text(encoding="utf-8"))
    blob = np.frombuffer((out_dir / "weights.bin").read_bytes(), dtype=np.float32)
    n_cls = len(manifest["meta"].get("labels", []))
    is_ctc = bool(manifest["meta"].get("ctc"))
    fresh = build_student(n_cls, seq_len, manifest["meta"].get("backbone", "tcn"), ctc=is_ctc)

    by_name = {l.name: l for l in fresh.layers}
    for entry in manifest["layers"]:
        layer = by_name.get(entry["name"])
        if layer is None:
            raise SystemExit(f"新建模型里没有层 {entry['name']} —— 结构不同构")
        ws = [
            blob[w["offset"] : w["offset"] + w["count"]].reshape(w["shape"])
            for w in entry["weights"]
        ]
        have = [tuple(t.shape) for t in layer.get_weights()]
        want = [tuple(t.shape) for t in ws]
        if have != want:
            raise SystemExit(f"层 {entry['name']} 形状不符: 期望 {have} 得到 {want}")
        layer.set_weights(ws)

    rng = np.random.default_rng(0)
    x = rng.random((4, seq_len, frame_dim), dtype=np.float32)
    a = model.predict(x, verbose=0)
    b = fresh.predict(x, verbose=0)
    diff = float(np.abs(a - b).max())
    if diff > 1e-5:
        raise SystemExit(f"回读后输出不一致,最大差 {diff:.3g} —— 权重顺序或形状错了")
    print(f"  回读校验通过(最大输出差 {diff:.2g})")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="keras 权重 → 浏览器 weights.bin/json")
    ap.add_argument("--model", default="out/sentence_student.keras")
    ap.add_argument("--meta", default="out/sentence_student_meta.json")
    ap.add_argument("--out", default="../client/public/models/seq_sentence")
    ap.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    res = export(Path(args.model), Path(args.meta), out_dir)
    meta = res["manifest"]["meta"]
    n_w = sum(len(l["weights"]) for l in res["manifest"]["layers"])
    print(f"已导出 {len(res['manifest']['layers'])} 层 / {n_w} 个张量 / "
          f"{len(res['blob'])} 个 float32({len(res['blob'])*4/1024:.0f} KB)")
    print(f"  → {out_dir}/weights.bin + weights.json")

    if args.verify:
        verify(res["model"], out_dir, meta.get("seqLen", 32), meta.get("frameDim", 294))

    labels = meta.get("labels", [])
    print(f"\n标签 {len(labels)} 类(顺序即输出下标):\n  {', '.join(labels)}")
    if meta.get("ctc"):
        print(f"blank 下标 {meta.get('blankIndex')}(= 类别数,在末位)")


if __name__ == "__main__":
    main()
