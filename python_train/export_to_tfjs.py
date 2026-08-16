"""
把 train_seq.py 训好的 keras 模型转成浏览器能加载的 tfjs LayersModel。

产物:
    out/tfjs_student/model.json + group1-shard*.bin
    out/tfjs_student/meta.json          # 标签、seqLen、backbone、frameDim

浏览器怎么用:
`/train-seq` 页面训练时是把权重存进 IndexedDB 的 models store。要用 Python 训的模型,
把 out/tfjs_student/ 整个目录放到 client/public/models/seq_student/ 下,然后在
sequenceModel.ts 里用 `tf.loadLayersModel('/models/seq_student/model.json')` 加载,
并把 meta.json 的 labels 传给 setActiveSequenceModel —— **标签顺序必须原样带过去**,
label 数组的下标就是 softmax 的输出下标,顺序错了每个词都会翻译成另一个词。

用法:
    python export_to_tfjs.py                       # 转 out/student.keras
    python export_to_tfjs.py --model out/teacher.keras --out out/tfjs_teacher
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def convert(model_path: Path, out_dir: Path) -> None:
    if not model_path.exists():
        raise SystemExit(f"找不到 {model_path},先跑 train_seq.py")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 用 python -m 而不是裸 tensorflowjs_converter:后者要求 Scripts/ 在 PATH 上,
    # venv 没激活时会莫名其妙地 "command not found"
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        str(model_path),
        str(out_dir),
    ]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(
            "转换失败。若报 'Unknown layer',多半是 tensorflowjs 与 tensorflow 版本不配套,"
            "按 requirements.txt 的版本装。"
        )
    print(proc.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description="keras → tfjs LayersModel")
    ap.add_argument("--model", default="out/student.keras")
    ap.add_argument("--meta", default="out/student_meta.json")
    ap.add_argument("--out", default="out/tfjs_student")
    args = ap.parse_args()

    out_dir = Path(args.out)
    convert(Path(args.model), out_dir)

    meta_src = Path(args.meta)
    if meta_src.exists():
        shutil.copy(meta_src, out_dir / "meta.json")
        meta = json.loads(meta_src.read_text(encoding="utf-8"))
        print(f"\n标签({len(meta.get('labels', []))} 类,顺序即 softmax 下标):")
        print("  " + ", ".join(meta.get("labels", [])))
    else:
        print(f"\n⚠️  没找到 {meta_src},meta.json 没有一起导出。"
              "浏览器侧缺了 labels 就无法把输出下标翻译回词。")

    shards = sorted(out_dir.glob("*.bin"))
    total = sum(p.stat().st_size for p in shards)
    print(f"\n已输出到 {out_dir}/({len(shards)} 个分片,共 {total/1024:.0f} KB)")
    print(f"下一步:把该目录整个拷到 client/public/models/seq_student/")


if __name__ == "__main__":
    main()
