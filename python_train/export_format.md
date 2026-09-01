# 序列数据集导出格式（seq-1.1）

> **1.0 → 1.1**：每条序列多了 `trimSpan`（见下）。二进制布局一个字节都没变，
> 1.0 的数据集照样读得进来 —— 但**要求按 span 裁剪时会报错而不是静默不裁**。


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
  "version": "seq-1.1",
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
      },
      "trimSpan": {                  // seq-1.1 起。null = 导出时没算
        "startFrame": 12,            // 含
        "endFrame": 68,              // 不含
        "applied": true,             // false 时 start/end 就是整条
        "reason": "applied",         // applied|full_span|no_vision|no_run|too_short
        "keptRatio": 0.74,
        "tactileRan": true           // 第三层（触觉静止段）跑过没有
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

**`trimSpan` 是「动作真正开始/结束」的帧区间**，由浏览器端 `sequenceTrim.detectSignSpan`
三层判据给出：① 手在画面里的最长连续段；② 段内的速度谷底（抬手 transport 的终点，
带弯折成形钳位）；③ 触觉能量低于静止门限的头尾段。**只有第三层会裁尾。**

- 为什么不在 Python 里判：判据要视觉关键点 **和** 浏览器 `localStorage` 里的弯折两点标定。
  在这边重实现一遍就是本文件开头那条禁令说的双实现。一份实现、两处消费。
- `applied: false` 时 `reason` 仍有信息量：`full_span` 是"判过了、没什么可裁"，
  `no_vision` / `no_run` 是"判据压根没运行"，这两者混在一个数据集里 = 同一个词两种时间口径。
- `tactileRan: false` 意味着**收尾静止还在数据里**（导出时没做弯折标定）。合成句子时
  这一段会顶在句尾，而推理端 `sentenceEnvelope` 会把尾部 800ms 静止掐掉 —— 口径就差了。
- 整条为 `null`（1.0 的数据集也一样）时，要求裁剪的调用方**必须报错**，不能静默不裁：
  静默的唯一症状是 WER 差几个点，没有任何线索指回这里。

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
