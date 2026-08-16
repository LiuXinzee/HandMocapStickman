# -*- coding: utf-8 -*-
"""生成 deaf-kit 模型结构示意图 SVG。坐标全部由布局函数算出，避免手写 SVG 对不齐。"""
import io

W, H = 1720, 1210
BG = "#0a0e1a"
CYAN = "#00f0ff"
PURPLE = "#a855f7"
GREEN = "#00e5a0"
AMBER = "#f59e0b"
TXT = "#c8d4e0"
DIM = "#7a8899"
MONO = "JetBrains Mono, Consolas, monospace"

out = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, stroke, fill="none", rx=4, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def text(x, y, s, size=14, fill=TXT, anchor="start", weight="normal", op=1.0):
    out.append(
        f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" opacity="{op}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color=DIM, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#a-{color[1:]})"{d}/>'
    )


def marker_defs(colors):
    ms = []
    for c in colors:
        ms.append(
            f'<marker id="a-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
        )
    return "".join(ms)


def layer_stack(x, y, w, rows, color, row_h=40, gap=9):
    """rows: [(主文案, 右侧注释, 是否强调)]  返回底部 y"""
    cy = y
    for main, note, strong in rows:
        fill = f"{color}1f" if strong else "#111726"
        rect(x, cy, w, row_h, color if strong else "#2a3446", fill=fill, rx=3,
             sw=1.4 if strong else 1.0)
        text(x + 12, cy + row_h / 2 + 5, main, 14, TXT if strong else DIM,
             weight="bold" if strong else "normal")
        if note:
            text(x + w - 12, cy + row_h / 2 + 5, note, 12, DIM, anchor="end")
        cy += row_h + gap
    return cy - gap


# ============ 背景 / 标题 ============
out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
text(W / 2, 46, "deaf-kit 手语识别 · 模型与数据流", 24, CYAN, anchor="middle", weight="bold")
text(W / 2, 70, "TensorFlow.js 4.22 · 浏览器内训练与推理 · WebGL(GPU) 后端", 13, DIM, anchor="middle")

# ============ 采集期 ============
rect(60, 96, W - 120, 132, "#2a3446", fill="#0d1220", rx=6, dash="6 4")
text(78, 118, "采集期（只有这一段需要摄像头）", 13, AMBER)

rect(110, 132, 460, 76, GREEN, fill="#0d1a18")
text(130, 158, "手套 ×2  COM 921600", 15, GREEN, weight="bold")
text(130, 182, "137 传感点 + 四元数（+加速度/姿态角：仅 296B 固件）", 12, DIM)

rect(620, 132, 460, 76, AMBER, fill="#1a1508")
text(640, 158, "摄像头 → MediaPipe Hands", 15, AMBER, weight="bold")
text(640, 182, "21 关键点 ×3 = 63D / 手", 12, DIM)

rect(1130, 132, 480, 76, "#2a3446", fill="#111726")
text(1150, 158, "IndexedDB  datasetStore.ts", 15, TXT, weight="bold")
text(1150, 182, "samples(单帧) · sequences(序列) · models", 12, DIM)

arrow(570, 170, 615, 170, GREEN)
arrow(1080, 170, 1125, 170, AMBER)

# ============ 两条链路 ============
PANEL_Y = 262
PANEL_H = 700

# ---- 静态（左） ----
rect(60, PANEL_Y, 760, 500, CYAN, fill="#080d18", rx=8)
text(84, PANEL_Y + 30, "静态链路  /train", 18, CYAN, weight="bold")
text(84, PANEL_Y + 52, "signLanguageModel.ts · 单帧 MLP · 卷积层 0 层", 12, DIM)

st_teacher = [
    ("输入 408D", "2×(137+4) + 2×63", True),
    ("Dense 256 + ReLU", "BatchNorm · Drop 0.3", False),
    ("Dense 128 + ReLU", "BatchNorm · Drop 0.2", False),
    ("Dense 64 + ReLU", "Drop 0.1", False),
    ("Dense C + softmax", "", True),
]
st_student = [
    ("输入 282D", "2×(137+4)  无视觉", True),
    ("Dense 128 + ReLU", "BatchNorm · Drop 0.3", False),
    ("Dense 64 + ReLU", "BatchNorm · Drop 0.2", False),
    ("Dense 32 + ReLU", "Drop 0.1", False),
    ("Dense C + softmax", "", True),
]
text(96, PANEL_Y + 92, "教师（训练用，吃视觉）", 13, TXT, weight="bold")
text(456, PANEL_Y + 92, "学生（部署用，纯触觉）", 13, GREEN, weight="bold")
b1 = layer_stack(96, PANEL_Y + 106, 300, st_teacher, CYAN)
b2 = layer_stack(456, PANEL_Y + 106, 300, st_student, GREEN)

# ---- 时序（右） ----
rect(900, PANEL_Y, 760, PANEL_H, PURPLE, fill="#0d0818", rx=8)
text(924, PANEL_Y + 30, "时序链路  /train-seq", 18, PURPLE, weight="bold")
text(924, PANEL_Y + 52, "sequenceModel.ts · TCN · 卷积层 3 层（Conv1D k=5）", 12, DIM)

sq_teacher = [
    ("输入 [T=32, 420]", "294 触觉 + 126 视觉", True),
    ("Conv1D k5 ×128", "BN + ReLU     T=32", True),
    ("MaxPool1D /2", "T=16", False),
    ("Conv1D k5 ×128", "BN + ReLU     T=16", True),
    ("MaxPool1D /2", "T=8", False),
    ("Conv1D k5 ×256", "BN + ReLU     T=8", True),
    ("GlobalAvgPool1D", "[256]", False),
    ("Dense 128 + ReLU", "Drop 0.3", False),
    ("Dense C + softmax", "", True),
]
sq_student = [
    ("输入 [T=32, 294]", "只有触觉", True),
    ("Conv1D k5 ×64", "BN + ReLU     T=32", True),
    ("MaxPool1D /2", "T=16", False),
    ("Conv1D k5 ×64", "BN + ReLU     T=16", True),
    ("MaxPool1D /2", "T=8", False),
    ("Conv1D k5 ×128", "BN + ReLU     T=8", True),
    ("GlobalAvgPool1D", "[128]", False),
    ("Dense 64 + ReLU", "Drop 0.2", False),
    ("Dense C + softmax", "", True),
]
text(936, PANEL_Y + 92, "教师（训练用，吃视觉）", 13, TXT, weight="bold")
text(1296, PANEL_Y + 92, "学生（部署用，纯触觉）", 13, GREEN, weight="bold")
b3 = layer_stack(936, PANEL_Y + 106, 300, sq_teacher, PURPLE)
b4 = layer_stack(1296, PANEL_Y + 106, 300, sq_student, GREEN)

# 蒸馏箭头（两条链路各一条）
for x0, x1, ytop in ((396, 456, PANEL_Y + 300), (1236, 1296, PANEL_Y + 300)):
    arrow(x0 + 4, ytop, x1 - 4, ytop, AMBER, sw=2)
    text((x0 + x1) / 2, ytop - 14, "蒸馏", 12, AMBER, anchor="middle", weight="bold")

# 每条链路底部的蒸馏公式
for bx, by in ((96, PANEL_Y + 380), (936, PANEL_Y + 596)):
    rect(bx, by, 660, 92, AMBER, fill="#150f04", rx=5)
    text(bx + 14, by + 26, "知识蒸馏  soft = softmax( log p / T )   T=3", 13, AMBER)
    text(bx + 14, by + 48, "学生目标 = α·soft + (1−α)·hard        α=0.5", 13, AMBER)
    text(bx + 14, by + 74, "视觉覆盖 < 80% → 教师与蒸馏整个跳过，学生用 hard label 直接训", 12, DIM)

# 每帧特征布局（左下空位）
FB_Y = PANEL_Y + 520
rect(60, FB_Y, 560, 180, PURPLE, fill="#0d0818", rx=8)
text(84, FB_Y + 28, "每手每帧 147 维  sequenceFeatures.ts", 14, PURPLE, weight="bold")
for i, line in enumerate([
    "[  0..136]  137 个传感点 / 255",
    "[137..140]  相对首帧的四元数（绕开无绝对 yaw）",
    "[141..143]  重力方向在手系的投影  ← 由四元数算",
    "[144..146]  加速度 / 16   ← 旧固件(272B)无此字段，恒 0",
]):
    text(84, FB_Y + 56 + i * 24, line, 12, DIM)
text(84, FB_Y + 162, "147 ×2 手 = 294（学生）  + 归一化关键点 63×2 = 420（教师）", 12, TXT)

# 数据来源箭头
arrow(1290, 216, 640, PANEL_Y - 6, CYAN, dash="5 4")
arrow(1370, 216, 1280, PANEL_Y - 6, PURPLE, dash="5 4")

# ============ 推理 ============
INF_Y = 985
rect(60, INF_Y, 1600, 176, GREEN, fill="#08150f", rx=8)
text(84, INF_Y + 30, "推理  /translate（不需要摄像头 —— 这正是蒸馏的目的）", 17, GREEN, weight="bold")

rect(96, INF_Y + 48, 340, 100, GREEN, fill="#0d1a18", rx=4)
text(114, INF_Y + 86, "手套帧流 30Hz", 15, TXT)
text(114, INF_Y + 110, "sensorMapping / gloveProtocol", 11, DIM)

# 两条链路在推理端是**二选一**（MODE 开关），不是串联
rect(520, INF_Y + 44, 420, 48, CYAN, fill="#08131a", rx=4)
text(538, INF_Y + 74, "静态：单帧 282D · 每 100ms 一次", 14, TXT)

rect(520, INF_Y + 104, 420, 48, PURPLE, fill="#120818", rx=4)
text(538, INF_Y + 134, "时序：滑窗 [32,294] · 最近 1500ms", 14, TXT)

text(950, INF_Y + 102, "二选一", 12, DIM)

rect(1080, INF_Y + 48, 540, 100, GREEN, fill="#0d1a18", rx=4)
text(1098, INF_Y + 86, "词 + 置信度 → 平滑 → 输出", 15, GREEN)
text(1098, INF_Y + 110, "阈值 0.7 · 平滑窗口 5 帧 · 只有学生模型参与", 11, DIM)

arrow(440, INF_Y + 92, 516, INF_Y + 68, GREEN)
arrow(440, INF_Y + 104, 516, INF_Y + 128, GREEN)
arrow(944, INF_Y + 68, 1076, INF_Y + 92, GREEN)
arrow(944, INF_Y + 128, 1076, INF_Y + 104, GREEN)

# 学生 → 推理
arrow(700, PANEL_Y + 504, 700, INF_Y - 6, GREEN, dash="5 4")
arrow(1446, PANEL_Y + 704, 1446, INF_Y - 6, GREEN, dash="5 4")
text(712, PANEL_Y + 560, "学生权重", 12, GREEN)
text(1458, PANEL_Y + 740, "学生权重", 12, GREEN)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}"><defs>{marker_defs([DIM, GREEN, AMBER, CYAN, PURPLE])}</defs>'
    + "".join(out)
    + "</svg>"
)
io.open(r"D:\JOB\projects\deaf-kit\hand_mocap_stickman_2\模型结构图.svg", "w", encoding="utf-8").write(svg)
print("written", len(svg), "bytes")
