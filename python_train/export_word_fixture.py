"""
生成**孤立词**模型的 Python↔浏览器数值对照 fixture。

===== 为什么单独一个脚本 =====

`export_fixture.py` 是句子(CTC)专用的:它的 main 里从 `build_ctc_xy` 到
`greedy_decode` 到 `blankIndex` 全是 CTC 概念,合成句型表还要与训练时一致。
孤立词没有 blank、没有句型、head 是 GAP+Dense 而不是 frame-wise ——
硬塞进同一个 main 会让两条路互相绑住,改一条得同时想另一条。

===== 这份 fixture 验什么 =====

只验一件事:**同一个输入,keras 和 tfjs 的输出逐位一致**。

孤立词模型从今天起也走"浏览器自己搭结构 + 填权重"这条路(`wordModel.ts`),于是继承
了句子模型那边同一类失配 —— tfjs 与 keras 的层实现不完全一致:
  - BatchNormalization 的 epsilon 默认值(keras 3 是 1e-3)
  - conv1d `padding='same'` 在偶数 kernel 下左右补零不对称
  - **GlobalAveragePooling1D 的轴**(孤立词 head 独有,句子模型没有这一层)

这些都不会抛异常,只让输出偏一点 —— 偏一点就足够让 29 类的 argmax 换人,
线上表现是"能跑、有置信度、词全错"。只比对 labels/shape 抓不到它。

===== 为什么还要存 student_fc 的输出 =====

ramp 输入远在真实特征分布之外,模型对它**必然满置信**:实测各种 mod
(97/31/13/7/251/181)下最大概率都在 0.9985~1.0,熵近 0。softmax 饱和之后,
BN epsilon 用错这类偏差在概率上可能只差 1e-7 —— 逐位断言照样通过,
测试形同没写。换 mod 解决不了,因为饱和的原因是输入离分布太远,不是输入不够激励。

所以真正的断言落在 `student_fc` 的输出上(64 维,relu,**没有** softmax)。
它在三个 conv 块 + BN + GAP 之后,上游任何层实现失配都会经过它,
而且不被压扁 —— 数值偏差原样体现。`probs` 仍然存,但它只是顺带。

用的是确定性输入(ramp),不用随机数:

    x[t][d] = ((t * frameDim + d) % 97) / 97

两边逐字实现同一个公式,不涉及浮点 RNG 的跨语言复现(那件事很难做对:JS 的
imul/ToInt32 语义要在 Python 里手工掩位)。取 97 是质数且与 frameDim=294 互质,
保证同一时刻各通道不同、相邻时刻不重复,conv 核会被真正激励到 ——
常数输入(全 0/全 1)测不出核里权重排布错位。

**不存真实样本。** 那会变成在测特征构建(`build_features` vs
`buildSequenceFeatures`)而不是测权重加载,是另一个风险、该另一个测试管;
而且要把 32×294 的输入原样写进 JSON 才能让两边喂同一份,文件会涨到几百 KB。
句子模型那边的 `samples` 也只喂给解码器,没参与前向对照。

用法:
    python export_word_fixture.py            # 默认读 out/student.keras
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAMP_MOD = 97


def ramp_input(seq_len: int, frame_dim: int) -> np.ndarray:
    """与 export_fixture.py 的 ramp_input 同一个式子。两处必须一致。"""
    n = seq_len * frame_dim
    x = (np.arange(n, dtype=np.int64) % RAMP_MOD).astype(np.float32) / RAMP_MOD
    return x.reshape(1, seq_len, frame_dim)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="生成孤立词模型的 JS 侧对照 fixture")
    ap.add_argument("--model", default="out/student.keras")
    ap.add_argument("--meta", default="out/student_meta.json")
    ap.add_argument("--out", default="../client/src/lib/wordModelFixture.json")
    args = ap.parse_args()

    import keras

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    classes: list[str] = meta["labels"]
    seq_len: int = meta["seqLen"]
    frame_dim: int = meta["frameDim"]
    model = keras.models.load_model(args.model, compile=False)

    xr = ramp_input(seq_len, frame_dim)
    pr = model.predict(xr, verbose=0)[0]
    if pr.shape != (len(classes),):
        raise SystemExit(
            f"输出形状 {pr.shape} 与类别数 {len(classes)} 不符 —— "
            f"孤立词 head 应该把时间维塌缩掉,给的是不是 CTC 模型?"
        )
    total = float(pr.sum())
    print(f"ramp 前向 {pr.shape},概率和 {total:.6f}")
    top = int(np.argmax(pr))

    # 中间层输出。层名写死是有意的:改了 head 的层名,这里会立刻 KeyError 而不是
    # 静默少存一个字段(少存的表现是 JS 侧那条最灵敏的断言被 skip 掉)
    fc_layer = model.get_layer("student_fc")
    fc_model = keras.Model(model.inputs, fc_layer.output)
    fc = fc_model.predict(xr, verbose=0)[0]
    nz = int((fc > 0).sum())
    print(f"student_fc {fc.shape},relu 后非零 {nz}/{len(fc)},幅度 {fc.max():.4f}")
    if nz == 0:
        raise SystemExit(
            "student_fc 输出全被 relu 归零 —— 这条对照测不出任何东西,"
            "换个 ramp 公式或改用别的中间层"
        )

    fixture = {
        "_comment": (
            "由 python_train/export_word_fixture.py 生成,不要手改。"
            "probs 是 softmax 之后的概率(isolatedWordHead 带 softmax,没有 logits)。"
            "ramp.formula 见该脚本的 ramp_input(),JS 侧要逐字实现同一个公式。"
        ),
        "labels": classes,
        "seqLen": seq_len,
        "frameDim": frame_dim,
        "numClasses": len(classes),
        "provenance": {
            # JS 侧用它对账:export_weights.py 把同一个 meta 抄进 weights.json,
            # 两边对不上说明模型重新导过、fixture 没重生成。
            # 少了这一条的表现是 wordModel.test 报 maxDiff 很大、提示指向
            # "tfjs 与 keras 层实现不一致",而真正的原因是 fixture 过期。
            "trainMeta": {
                k: meta.get(k)
                for k in ("valAccuracy", "numSequences", "numTrain", "numVal", "epochsRun")
            },
        },
        "ramp": {
            "formula": f"x[t][d] = ((t * {frame_dim} + d) % {RAMP_MOD}) / {RAMP_MOD}",
            "mod": RAMP_MOD,
            # softmax 之后。ramp 输入下必然饱和(见文件头),所以这一项灵敏度低
            "probs": np.round(pr, 6).tolist(),
            "argmax": top,
            "argmaxLabel": classes[top],
            # **主断言在这里**:student_fc 的 64 维 relu 输出,没被 softmax 压扁。
            # 存 6 位小数,与 probs 一致;幅度是 O(1) 量级,6 位足够分辨层实现差异
            "fc": np.round(fc, 6).tolist(),
            "fcLayer": "student_fc",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    print(f"已写 {out}({out.stat().st_size / 1024:.0f} KB)")
    print(f"ramp argmax → {classes[top]} ({pr[top]:.4f})")
    print(f"标签 {len(classes)} 类,valAccuracy {meta.get('valAccuracy')}")


if __name__ == "__main__":
    main()
