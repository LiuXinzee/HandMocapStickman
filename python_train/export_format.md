# 序列数据集导出格式（seq-1.0）

浏览器端 `/train-seq` 页面的「导出数据集给 Python」按钮产出两个文件，放进 `python_train/data/`：

```
python_train/data/
  dataset.bin    # 所有 TypedArray 顺序拼接的原始字节
  dataset.json   # 索引清单
```

## 为什么不是纯 JSON

一条 1.5s @ 50Hz 双手带视觉的序列约 64KB 二进制。600 条展开成 JSON 文本约 150MB，
浏览器端 `JSON.stringify` 会直接卡死，Python 侧解析也要几十秒。二进制 + 偏移表可以让
`np.frombuffer` 零拷贝切片。

## dataset.json 结构

```jsonc
{
  "version": "seq-1.0",
  "exportedAt": "2026-08-10T12:00:00.000Z",
  "totalSequences": 612,
  "sensorN": 137,          // 每帧每手的传感点数
  "imuN": 10,              // quat(4) + acc(3) + attitude(3)
  "landmarkN": 63,         // 21 关键点 × xyz
  "labels": ["_idle", "come", "go", "hello", ...],   // 已排序
  "arrayLayout": {
    "timestamps":     { "perFrame": 1,   "dtype": "float32" },
    "leftSensor":     { "perFrame": 137, "dtype": "uint8"   },
    "rightSensor":    { "perFrame": 137, "dtype": "uint8"   },
    "leftImu":        { "perFrame": 10,  "dtype": "float32" },
    "rightImu":       { "perFrame": 10,  "dtype": "float32" },
    "leftLandmarks":  { "perFrame": 63,  "dtype": "float32" },
    "rightLandmarks": { "perFrame": 63,  "dtype": "float32" }
  },
  "sequences": [
    {
      "segments": [{ "label": "hello", "startFrame": 0, "endFrame": 75 }],
      "primaryLabel": "hello",
      "frameCount": 75,
      "durationMs": 1500,
      "sourceFps": 50,
      "origin": "recorded",        // 或 "synthesized"
      "timestamp": 1770000000000,
      "arrays": {
        "timestamps":     { "offset": 0,     "length": 75,    "dtype": "float32" },
        "leftSensor":     { "offset": 300,   "length": 10275, "dtype": "uint8"   },
        "rightLandmarks": null,     // 该手/该模态缺失
        ...
      }
    }
  ]
}
```

## 关键约定

**`offset` 是 dataset.bin 里的字节偏移，`length` 是元素个数（不是字节数）。**
uint8 段字节数 = length，float32 段字节数 = length × 4。

**float32 段一定 4 字节对齐。** 编码时会在 uint8 段之后插 padding。numpy 在部分平台上
对未对齐的 buffer 调 `frombuffer` 会直接报错，所以这条不是可选项。

**`null` 表示该模态真的不存在**，不是全 0：
- 单手打的词 → 另一只手的 `*Sensor` / `*Imu` / `*Landmarks` 全为 null
- 没开摄像头录的 → `leftLandmarks` / `rightLandmarks` 为 null
- **不要把 null 补成 0 再喂进模型**：0 是合法的传感读数和关键点坐标，补 0 等于告诉模型
  "这只手贴在原点且完全没受力"，比缺失更糟。`load_dataset.py` 用 mask 表达缺失。

**视觉数组内部可能含 NaN**，表示那一帧 MediaPipe 丢手了（整条序列有视觉，但某几帧没有）。
`load_dataset.py` 会沿时间轴插值补齐，与浏览器端 `fillVisionGaps` 同一策略。

**`segments` 是词边界列表**。孤立词是长度为 1、覆盖全段的特例；句子级（连续手语）
用同一个 schema 装多个 segment，不需要改数据库或重新采集。CTC 训练直接读
`[seg.label for seg in segments]` 作为目标标签序列。

**`origin: "synthesized"`** 的样本是从旧的单帧静态样本合成出来的（低通相关噪声扩帧），
帧间统计与真实录制不完全一致。`load_dataset.py --recorded-only` 可以只要真实数据。

## 每帧特征维度（与浏览器端 sequenceFeatures.ts 严格一致）

每只手 147 维：

| 区间 | 内容 |
|---|---|
| `[0:137]` | 传感点 / 255 |
| `[137:141]` | 相对首帧的四元数 `q₀⁻¹ ⊗ qₜ`（w,x,y,z） |
| `[141:144]` | 重力方向在手系的投影 `R(q)⁻¹·[0,0,1]` |
| `[144:147]` | 加速度 / 16，截断到 [-1,1] |

- 学生（部署，纯触觉）：147 × 2 = **294 维/帧**
- 教师（训练，融合视觉）：294 + 63 × 2 = **420 维/帧**

用相对四元数而不是绝对朝向：IMU 有固定偏置（实测 yaw 偏 −53.7°）且佩戴姿态每次都不同，
绝对值不可信；动态词的信息本来就在朝向**变化**里。重力方向在手系的投影补上绝对倾角信息，
且它对绕重力轴的旋转不变，所以不受 yaw 偏置影响。

关键点归一化：手腕（点 0）保留绝对坐标承载轨迹，其余 20 点相对手腕平移并按手长（点 0→9 距离）
缩放承载手型。
