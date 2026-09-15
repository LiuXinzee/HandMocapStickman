/*
 * HandModel — 由「弯折通道 + IMU 四元数」直接驱动的 3D 手模（**单手一个 Canvas**）
 *
 * 移植自 glove_visual/cc_part2 的 HandScene.tsx（同一份硬件协议的分叉项目），
 * 按 deaf-kit 改成单手 + Cyberpunk HUD 配色 + 白手套材质。
 *
 * ⚠ **一手一个 Canvas，这是全项目唯一的手模视口**：`/mocap` 的四格自检、
 * 向导里的示意手模、`/translate` 底部的两格，用的都是这一个组件。
 *
 * 曾经还有一个 `SigningStage.tsx`：两只手 + 躯干剪影放进同一个场景，读起来像
 * 一个人在打手语。**已经删掉了**，理由不是观感 —— 手语的**位置**是语言学通道
 * （额头 / 下巴 / 胸前是不同的词），而手套只有弯折 + 压力 + IMU：IMU 给朝向不给
 * 位置，加速度二次积分在手上无 ZUPT 可重置，六轴还观测不到绝对 yaw。所以那个
 * 场景里手的位置只能是常量，看着像"在打手语"却仍然展示不出一个完整的手语词，
 * 只是把"位置是编的"藏得更深。要合回去，先得真有位置观测。
 *
 * 骨骼驱动那段（`SEGMENT_MAX_DEG` / rest 四元数 / slerp 收敛）**不要复制第二份**：
 * 那几个角度是离屏 FK 验算过的解剖学行程，复制出去两边就会各自漂。
 *
 * 为什么不用骨架回归模型的 21 关键点来驱动：
 *  1. 这条链路要能在**没有任何已训练模型**时就工作，否则新装机器上一片黑；
 *  2. skeletonModel 的输出层是 sigmoid，而训练目标里 MediaPipe 的 z 是可负的
 *     手腕相对深度 —— 深度通道实际上是废的（详见计划文档），拿它撑 3D 姿态不可靠；
 *  3. 与左侧 2D 火柴人构成**两条独立链路**，不一致时能立刻看出回归模型跑偏。
 *
 * 数据流全部走 ref：手套 100Hz，若用 props/state 传值会把整页每秒重渲染上百次。
 * 父页面在自己已有的 rAF 循环里填 driveRef，这里在 useFrame 里读，互不干扰。
 */

import { ContactShadows, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { memo, Suspense, useEffect, useMemo, useRef, type RefObject } from "react";
import {
  Bone,
  type BufferGeometry,
  Color,
  DoubleSide,
  Group,
  InstancedMesh,
  MathUtils,
  Matrix4,
  Mesh,
  Object3D,
  Quaternion,
  Vector3,
} from "three";
import { clone as cloneSkeleton } from "three/addons/utils/SkeletonUtils.js";

const MODEL_URL = "/assets/hand1.glb";

/*
 * ===== hand1.glb 的实测尺寸 =====
 *
 * 全部从 glb 里量出来的，不是估的：网格空间的 POSITION 包围盒 + skin 的
 * inverseBindMatrices（骨骼在网格空间的位置）。量法记在这里，换模型时能复现：
 *
 *   POSITION  x∈[-4.81, 2.14]  y∈[-2.08, 1.10]  z∈[-9.19, 7.06]
 *   Wrist 骨在 (0.18, 0.31, 0.00) —— **腕就在网格原点**，所以下面 AnimatedHand 里
 *   那个 `position={[0,-1.2,0]}` 放的确实是腕关节。
 *   Forearm_01（小臂中段）在 z=-4.09，Forearm_00（肘端）在 z=-8.12，
 *   Finger_23（中指尖）在 z=+6.78。
 *
 * 前臂末端那个**切面**（正对镜头时画面里那坨白饼）的形心量下来是
 * `(0.668, -0.092, -9.101)`，椭圆半径 x 1.54 / y 1.35；量法是取 z < zmin+0.25
 * 那一层的 29 个顶点求形心。它几乎正落在「腕 → 肘」的延长线上（以 Wrist 为 0、
 * Forearm_00 为 1，在 t = 1.122 处，偏离那条线只有 0.21 = 臂长的 2.3%），
 * 所以换模型时 `切面 ≈ 腕 + 1.12 × (肘 − 腕)` 够用，不必重量顶点。
 * 记在这儿只是免得再量一遍 —— 拿它当转动轴心的那一版已经回退，见下面 return 那段。
 *
 * ⚠ 这三个骨骼坐标是**重测过的**（节点层级 FK 与 skin 的 inverseBindMatrices
 *   两法一致）。原注释写的是 Wrist (0.28,0.48,0)、Forearm_00 z=-12.68、
 *   Finger_23 z=+10.59 —— 都不对。改模型时用上面这几个。
 *
 * ⚠ glb 的根节点 `LeapMotion_Basehand_Rig_Left` 自己带了 `rotation` = +90°X、
 *   `scale` = [-0.8, 0.8, 0.8]，**但这两样都不生效**，净缩放就是 0.78。
 *   曾经有一版注释断定净缩放是 0.78 × 0.8 = 0.624、这张表偏大 20% —— 那是错的，
 *   别再照着 0.624 推。两条独立的证据：
 *     1. 骨头（Wrist / Forearm_00 / Finger_23）全在 glb 的**顶层**，不是那个根节点的
 *        子节点，所以它们完全不受它的旋转缩放影响；而蒙皮网格按 glTF 规范由
 *        `骨骼世界矩阵 × inverseBindMatrix` 定位，跟着骨头走，不是跟着自己的节点走。
 *        这也正是内层 `rotation={[-π/2,0,0]}` 真的把指尖转朝上的原因 —— 它抵消的
 *        不是根节点，网格空间的 +z（指尖方向）本来就要转到 +y 才竖起来。
 *     2. 下面那三档取景是**上屏量出来的**，其中"手占多少帧高"只有 0.78 对得上：
 *        `[0,1.0,13.1]` 档记的是 70%，7.06 × 0.78 / 8.02 = 69% ✓（0.624 只有 55%）；
 *        现在这档记的是 ~59%，5.51 / 9.42 = 58.5% ✓。
 *
 * 内层 group 的 `rotation={[-π/2,0,0]}` 把网格的 z 转成竖直方向，外层 scale 0.78，
 * 于是相对腕关节、以世界单位算：
 *   腕 → 指尖（上）  7.06 × 0.78 = 5.51
 *   腕 → 前臂末端（下） 9.19 × 0.78 = 7.17  ← **比手掌本身还长**
 *   腕 → 拇指侧（横） 4.81 × 0.78 = 3.75
 *   腕 → 小指侧（横） 2.14 × 0.78 = 1.67
 *
 * 两条用得上的推论：
 *  1. **模型自带前臂**（`Forearm_00/01` 两根骨头）。别再往腕上加一根 capsule ——
 *     加过，结果是同一个腕点上长出两根前臂、还呈夹角岔开，看着就是两根柱子。
 *     自带那根跟着手刚性转，"前臂倾斜"由它提供；代价是手转 90° 时会横扫出画。
 *     骨架档画的小臂也是这两根骨头，不是另画的柱子 —— 见 FOREARM_COLOR 那段。
 *  2. 下面 Canvas 的取景：竖直 FOV 34° @ 距离 15.4 → 半高 15.4·tan17° = 4.71，
 *     画面中心抬到 y=0.34（相机和 OrbitControls 的 target 一起抬，否则轨道会绕着
 *     手底下转），于是取景是世界 y∈[-4.37, 5.05]。**取景由竖直方向定尺寸**：
 *     视口有多高就决定手有多大，把框加宽一点用都没有。调用方分配高度时按这个来。
 *
 *     ⚠ 这一档是**上屏量出来的**，而且是在两个方向上都撞过墙之后停在这里的：
 *       `[0, 0.15, 17]`   半高 5.20，手占 53% 帧高，上方 8% 空白 + 下方 39% 前臂
 *                          —— 那两块加起来比手还大，手太小
 *       `[0, 1.0, 13.1]`  半高 4.01，手占 70%，放大 1.3×
 *                          —— **前臂只剩腕下 1.8 个世界单位，看不见手臂了**
 *       `[0, 0.34, 15.4]` 半高 4.71，手占 ~59%，前臂留腕下 3.2 个世界单位 ← 现在这档
 *     前臂不能裁光：它是手腕朝向的视觉线索（见上一条），而且是唯一还能看出
 *     "手转到哪个朝向"的东西。所以这一档是**手的大小与手臂可见长度之间的平衡点**，
 *     不是"还没调到位"。要再放大手，先想清楚拿什么补手臂那条线索。
 *
 *     上边框那侧没有余量可挖：实心手模档的指尖比骨架档的关节球还高一截。
 *     **别照着数字继续调**，改了就上屏看手模档、手指伸直朝上那个姿势
 *     （那是最高的姿势，转起来只会变矮）。
 */

/*
 * ===== 前臂拉长：把末端切面推出取景框 =====
 *
 * 切面进画的几何账（数字全来自上面那段实测）：切面距腕 9.12 网格单位 =
 * 7.17 世界单位，而默认取景半高 4.71、宽视口下半宽 ~9.4，腕又钉在画面中央附近 ——
 * 前臂横扫时切面必然进画。轴心挪到切面那条路已经试过并回退（见 return 那段），
 * 剩下的办法就是把切面推远：**直接在几何顶点上沿手臂轴线做渐进拉伸**。
 *
 * 为什么改顶点而不是改骨头：切面附近的顶点全权重在 Forearm_00（肘端）上，
 * 而 Forearm_00 是整条骨链的**根**（Forearm_00 → Forearm_01 → Wrist → 手指），
 * 挪它会连腕带手一起挪走；要"只挪肘端"得同时给 Forearm_01 做反向补偿，
 * 还要过它们各自的局部旋转 —— 改绑定姿态的顶点则一步到位，骨骼驱动完全不动。
 *
 * 拉伸区间从 z=-4.5（Forearm_01 在 -4.09，再往下才开始拉）线性渐变到原切面
 * z=-9.19 处满量 8 个网格单位，切面被推到距腕 17.1 网格 = 13.4 世界单位 ——
 * 超过取景框对角的最远可达距离（宽视口下 ~12.1），又仍在默认相机距离（15.4）
 * 之内，手臂正对镜头时末端不会穿到相机背后被近平面剖开。腕以上的手掌手指
 * 一个顶点都不动，所以手的大小、取景三档的账全都不变。
 *
 * 光拉长挡不住所有姿态（斜对角、正对镜头时理论上多长都可能露端），所以末端
 * 最后 15%（t ∈ [0.85, 1]，世界 11.9 → 13.4）再做**径向收锥**：顶点朝手臂轴线
 * 收拢，开口的切面环几乎闭成一点。就算极端姿态下末端真进了画，看到的也是
 * 自然收细的臂端，不再是一个平的横截面圆盘。收锥只改径向偏移、不动法线 ——
 * 尖端荫影略糙，但它只在取景框最角上才可能露脸，不值得为它重算整手法线。
 *
 * ⚠ 几何被所有克隆实例**共享**（SkeletonUtils.clone 不克隆 geometry），
 * 所以拉伸做一次就够，用 geometry.userData 防止重复叠加。
 */
const FOREARM_STRETCH_START_Z = -4.5;
const FOREARM_STRETCH_END_Z = -9.19;
const FOREARM_STRETCH_EXTRA = 8;
/** 收锥起点（占拉伸渐变量 t 的比例）与末端保留的径向比例 */
const FOREARM_TAPER_START_T = 0.85;
const FOREARM_TAPER_MIN_SCALE = 0.03;
/** 腕(0.18,0.31,0) → 切面形心(0.668,-0.092,-9.101) 的单位向量，网格空间 */
const FOREARM_STRETCH_DIR = { x: 0.0535, y: -0.0441, z: -0.9976 } as const;
/** 手臂轴线按 z 参数化：axis(z) = 腕 + 该斜率 × z（由上面的方向向量折算） */
const FOREARM_AXIS = { x0: 0.18, dxdz: -0.0536, y0: 0.31, dydz: 0.0442 } as const;

function stretchForearmGeometry(geometry: BufferGeometry) {
  if (geometry.userData.forearmStretched) return;
  geometry.userData.forearmStretched = true;
  const pos = geometry.attributes.position;
  const span = FOREARM_STRETCH_START_Z - FOREARM_STRETCH_END_Z;
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i);
    if (z >= FOREARM_STRETCH_START_Z) continue;
    const t = Math.min(1, (FOREARM_STRETCH_START_Z - z) / span);
    // 径向收锥：先把顶点对手臂轴线的偏移量缩掉，再做轴向位移
    const axisX = FOREARM_AXIS.x0 + FOREARM_AXIS.dxdz * z;
    const axisY = FOREARM_AXIS.y0 + FOREARM_AXIS.dydz * z;
    let radial = 1;
    if (t > FOREARM_TAPER_START_T) {
      const s = (t - FOREARM_TAPER_START_T) / (1 - FOREARM_TAPER_START_T);
      const smooth = s * s * (3 - 2 * s);
      radial = 1 - smooth * (1 - FOREARM_TAPER_MIN_SCALE);
    }
    const d = FOREARM_STRETCH_EXTRA * t;
    pos.setXYZ(
      i,
      axisX + (pos.getX(i) - axisX) * radial + FOREARM_STRETCH_DIR.x * d,
      axisY + (pos.getY(i) - axisY) * radial + FOREARM_STRETCH_DIR.y * d,
      z + FOREARM_STRETCH_DIR.z * d
    );
  }
  pos.needsUpdate = true;
  // 包围体不更新的话，拉长的那截会在斜视角下被视锥剔除、整只手闪没
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}

/**
 * 实物手套是**白色**的（规格书 p12 写"黑色面料"与实物不符），
 * 所以用 cc_part2 的 white 预设：亮面料 + 低粗糙度 + 一点点金属感。
 */
const HAND_MATERIAL = {
  color: "#dce9f0",
  roughness: 0.4,
  metalness: 0.08,
} as const;

/**
 * 缺手那一侧的材质：灰、半透明。
 *
 * **不能什么都不画。** 一只手套没连上时画面里少一只手，看起来和"这个词是单手词"
 * 一模一样 —— 用户会以为系统在正常工作。画成灰影 + 画面内文字才分得出
 * "没戴/没连" 和 "戴着不动"。
 */
const HAND_MATERIAL_DIMMED = {
  color: "#b8c4d2",
  roughness: 0.9,
  metalness: 0,
  opacity: 0.38,
} as const;

/**
 * 关节光点色 / 强调色。
 *
 * ⚠ **这里不能用 `var(--hud-*)`**，和页面上那些 className 不一样：
 * 这几个值最终喂给 three.js 的 `Color.set()`，它自己解析字符串、不经过 CSS，
 * 拿到 `var(...)` 会当成未知颜色（结果是黑）。所以浅色皮肤下的对应值写死在这里。
 * 改配色时记得两边一起改：index.css 的 --hud-warn / --hud-accent 是同一对颜色。
 */
export const AMBER = "#d97706";
export const CYAN = "#1677ff";

/** 接触阴影的颜色。浅底上不能用强调色 —— 蓝色阴影会变成手底下一坨蓝斑 */
export const SHADOW_COLOR = "#64748b";

/**
 * 骨架档的配色：**骨头统一中性灰，颜色只落在 21 个关节球上。**
 *
 * ⚠ 这里**不再**跟 `HandCanvas.tsx`（首页 / 采集页那副**摄像头**骨架）一致。
 * 上一版是照抄它的 `FINGER_COLORS`，为的是跨页看到"同一只手"。放弃那条的理由：
 * 那套色其实是把应用的**状态色**直接当五指颜色用了 —— `--hud-accent` 蓝 /
 * `--hud-ok` 绿 / `--hud-warn` 橙 / `--hud-err` 玫红 / `--hud-violet` 紫。
 * 于是一只手上同时挂着"成功""警告""错误"三种语义色，加上五个色相跨度过大，
 * 在浅底上读起来像仪表盘报警而不是一只手。现在换成同一段冷色里的渐变，
 * 没有哪个色相还对应着状态。
 * **两页的骨架从此颜色不同 —— 这是知情的取舍，不是漏改。**
 *
 * 关节球仍然一指一色：手指在 3D 里互相遮挡时，这是唯一还能分出"哪根是哪根"的线索。
 *
 * 同样**不能写 `var(--hud-f1)`**：这些值喂给 three.js 的 `Color.set()`，
 * 它自己解析字符串、不过 CSS，拿到 `var(...)` 会当未知颜色处理（结果是黑）。
 * 和上面 AMBER / CYAN 那段是同一条约束。
 */
const JOINT_COLORS = [
  "#0e8a8f", // 拇指 · 青    3.60:1
  "#1178c4", // 食指 · 天蓝  4.03:1
  "#2563eb", // 中指 · 蓝    4.47:1
  "#4f46e5", // 无名 · 靛蓝  5.44:1
  "#7c3aed", // 小指 · 紫    4.93:1
] as const;

/** 腕关节球：比五指都深（7.55:1），当整只手的锚点 */
const WRIST_COLOR = "#1e40af";

/**
 * 所有手骨（含掌弓）统一这一个中性灰蓝 —— 分辨手指的活全交给关节球。
 *
 * 值是按**对比度**定的（都是对视口底色 `--hud-stage` #eaeff6 算的）：
 * 这个色 3.29:1。这几档都试过，别再从头调一遍：
 *   `#94a3b8` 2.22:1 —— 在"图形元素看得清"的 3:1 门限以下，整副骨架糊成一片
 *   `#74849b` 3.29:1 —— **现在这档**，刚过门限
 *   `#586880` 4.90:1 —— 连同关节色一起加深过一版，看着太重，退回来了
 *
 * 上限是硬的：**骨头必须比关节球浅。** 关节色是 3.6~5.4:1，骨头再深就会反超，
 * 层次翻过来、关节球不再是主角。所以"骨架不够明显"要靠 BONE_RADIUS 加粗
 * 和光晕（JOINT_GLOW_*）补，不是把这个色调深 —— 加深那一版就是这么翻车的。
 */
const BONE_COLOR = "#74849b";

/** 没连手套那只手的骨架色：灰。理由同 HAND_MATERIAL_DIMMED —— 不能不画 */
const BONE_COLOR_DIMMED = "#b8c4d2";

/**
 * 小臂那两段的颜色与半径。**故意比手骨更细、更淡。**
 *
 * 它是参照物不是主角：给出一条"手臂朝哪"的轴线就够了，画粗会把视线从手上抢走。
 * 变淡靠的是**贴近底色**而不是透明度 —— 骨头是一个 InstancedMesh，
 * 逐实例透明度得自己写 shader，而颜色本来就是逐实例的。
 *
 * 同样是按对比度定的。这一档走过三步，别再从头试：
 *   `#d0d8e2` 1.24:1 —— 等于没画
 *   `#aeb9c9` 1.72:1 —— 上一档。取景还很宽、小臂占 39% 帧高时够用
 *   `#94a3b8` 2.22:1 —— **现在这档**
 * 改深的触发是**取景收紧**（见文件头第 2 条）：相机推近之后腕以下只剩 1.8 个
 * 世界单位、约 22% 帧高，同样的淡色摊在这么短一截上就读不出来了 ——
 * 一条又短又淡的线看着像画面边上的一道划痕，不像手臂。
 * 上限仍然是硬的：**必须明显弱于手骨的 3.29:1**，小臂是参照物不是主角。
 * **手骨调深时这里要跟着调**，不然整副手变实、小臂却原地不动，又掉回"看不见"。
 *
 * 半径反过来比手骨**粗**（0.12 vs 0.075），这不是笔误：小臂**不参与光晕**
 * （见 JOINT_GLOW_* 那段），而手骨外面套着 2.6× 的半透明壳。所以几何半径相等
 * 时上屏是不等的 —— 原来两边都写 0.07 上下，画出来小臂细得像一道划痕。
 * 0.12 裸柱 ≈ 0.075 带壳的视觉粗细，"小臂比手骨退一档"这条仍然成立，
 * 只是现在靠颜色退（2.22:1 vs 3.29:1），不再靠粗细退。
 * 要给小臂加光晕请先重读 JOINT_GLOW_* 那段：它退到后面去是整个设定。
 *

 * 小臂的**朝向是真的**：整副骨架被 `handRef` 上的 IMU 四元数整体旋转，而朝向正是
 * 陀螺仪唯一真正观测到的量。所以"手臂朝这边倾"是测出来的，不是编的
 * —— 这一点和文件头那段被删掉的 SigningStage（位置只能是常量）性质完全不同。
 *
 * 但**腕相对小臂的角度是假的**：小臂在 rig 里是腕的刚性父级，那个夹角永远冻在
 * 静止姿态。单个 IMU 装在手背上，分不开"整条手臂转了"和"只有手腕折了"，它只给
 * 手的绝对朝向。结果就是：**不动小臂只折手腕，画面上的小臂也会跟着甩。**
 *
 * 实心手模档今天就是这个行为（自带前臂网格跟着手刚性转），所以骨架档画它
 * 不新增任何失真，只是把这条已有的性质摆到明面上。画得比手骨还淡也是在说
 * "这是结构参照，不是量到的关节" —— 它也确实不属于 MediaPipe 那 21 点。
 */
const FOREARM_COLOR = "#94a3b8";
const FOREARM_RADIUS = 0.12;

/**
 * 骨架的几个尺寸，模型单位（外层还有缩放，净 0.624，见文件头）。
 * 参照物：食指全长约 3.17、分 4 段，所以一段约 0.8。
 *
 * **这几个数是上屏调出来的**，想改就直接改、然后看屏幕，别照着比例推。
 * 骨头从最初的 0.055 加粗到 0.075，是因为"骨架不够明显"只能从这里补
 * —— 颜色那一路被 BONE_COLOR 上面那段的上下限夹死了。
 */
const JOINT_RADIUS = 0.135;
const BONE_RADIUS = 0.075;

/**
 * 「发光」怎么做的 —— **不是 bloom，也不能是 bloom。**
 *
 * 辉光后处理是**加光**的：它把亮的地方往更亮推。视口底色是 #eaeff6，已经接近白，
 * 往上加光加不出东西来，只会糊出一层雾。霓虹发光是**深色底**的语汇，
 * 这一页是浅底，照搬过来必然失败 —— 顺带还要装 `@react-three/postprocessing`、
 * 给两个 Canvas 各挂一条 EffectComposer，而这两个画面还在和 tfjs 抢 GPU。
 *
 * 浅底上等效的做法是**半透明同色外壳**：在本体外面套一个更大、低不透明度、
 * 同色的球/柱，边缘于是有一圈由浓到淡的晕。文件里 `JointBeacons` 用的就是这招
 * （0.12 的球 + scale 1.78 / opacity 0.14 的壳），这里只是把它搬到 InstancedMesh 上。
 *
 * 小臂不参与光晕：它的整个设定就是退到后面去（见 FOREARM_COLOR）。
 */
const JOINT_GLOW_SCALE = 2.05;
const JOINT_GLOW_OPACITY = 0.17;
/** 骨头光晕只放大半径、不动长度，否则骨头会从关节球里两头戳出来 */
const BONE_GLOW_SCALE = 2.6;
const BONE_GLOW_OPACITY = 0.1;

/**
 * 关节球画 21 个（MediaPipe 那 21 点），但位置要算 23 个 —— 多出来的
 * 21 / 22 是小臂的 `Forearm_01` / `Forearm_00`，只当骨头段的端点用。
 *
 * **小臂上不画球**：`Forearm_01` 是小臂中段、不是关节，画个球会凭空多出一个
 * 解剖学上不存在的关节；`Forearm_00`（肘端）在取景外，画了也看不见，而且
 * 单 IMU 观测不到肘的位置，画出一个肘关节点是在暗示一个不存在的观测量。
 */
const JOINT_COUNT = 21;
const POINT_COUNT = 23;

/**
 * 骨架拓扑：`[起点, 终点, 段类型]`，端点用上面那套 23 点编号
 * （0 = 腕，1~20 = `Finger_<(i-1)/4><(i-1)%4>`，前 21 个和 MediaPipe 一一对应）。
 *
 * **按结构生成，不抄表**：`useHandTracking.ts` 里的 `FINGER_CONNECTION_GROUPS`
 * 是同一套拓扑，但不 import 它 —— 那个常量挂在摄像头 hook 上，import 会把
 * gloveVision / handDetectionFusion / mediaPipeHandsRunner 整条链路拖进本组件，
 * 而 VirtualMocap 也在用本组件。生成的好处是没有表可以抄错。
 *
 * 掌部那 3 段横连（食指~小指的 MCP 之间）**必须有**：只从腕点朝 5 个方向放射的话，
 * 画出来读起来是一只海星，不是手掌。
 */
/**
 * 段类型：0~4 = 属于哪根手指，5 = 掌弓，6 = 小臂。
 *
 * 现在只有"是不是小臂"影响颜色和粗细（手骨全是 `BONE_COLOR`），前 6 档在
 * 渲染上是等价的。**留着不是冗余**：这是每一段的归属，想回到一指一色只要改
 * 取色那一行；把它压成布尔值就得重新推一遍拓扑。
 */
const PALM_LINK = 5;
const FOREARM_LINK = 6;

const BONE_LINKS: [number, number, number][] = [];
for (let f = 0; f < 5; f++) {
  const base = 1 + f * 4; // 这根手指的 MCP 在 21 点编号里的位置
  BONE_LINKS.push([0, base, f]); // 腕 → MCP
  for (let s = 0; s < 3; s++) BONE_LINKS.push([base + s, base + s + 1, f]);
}
// 掌弓：食指 MCP → 中指 MCP → 无名 MCP → 小指 MCP（拇指不参与，它从腕单独长出去）
for (let f = 1; f < 4; f++) BONE_LINKS.push([1 + f * 4, 1 + (f + 1) * 4, PALM_LINK]);
// 小臂：腕 → 小臂中段 → 肘端。肘端落在取景外，读起来就是"手臂从画面外伸进来"，
// 正好不必画一个假的肘关节点。用的是 glb 自带那两根骨头（见文件头推论 1）
BONE_LINKS.push([0, 21, FOREARM_LINK]);
BONE_LINKS.push([21, 22, FOREARM_LINK]);

/** 23 点编号 → glb 骨骼名 */
function jointBoneName(index: number): string {
  if (index === 0) return "Wrist";
  if (index === 21) return "Forearm_01";
  if (index === 22) return "Forearm_00";
  const i = index - 1;
  return `Finger_${Math.floor(i / 4)}${i % 4}`;
}

/** 指节绕这个局部轴弯曲；右手镜像组会把局部旋转一并镜像，故左右同符号 */
const AXIS = new Vector3(0, 0, 1);

/** hand1.glb 的骨骼命名：Finger_<手指0-4><指节0-3> */
const fingerBoneNames = Array.from({ length: 5 }, (_, finger) =>
  [0, 1, 2, 3].map((segment) => `Finger_${finger}${segment}`)
);

/**
 * 满量程（curl=1）时各指节转多少度，下标 = 指节序号：
 *   0 = MCP 掌指关节、1 = PIP 近端指间、2 = DIP 远端指间、3 = 指尖
 *
 * **第 3 节是 0**：hand1.glb 的 `Finger_x3` 是指尖骨、没有子节点，它不对应真实关节，
 * 转它只会把指尖掰断。
 *
 * 为什么不是原来那样每节都给满 90°：那样四节累计 **360°**，是绕一整圈而不是握拳
 * ——离屏 FK 验算过，指尖会卷穿手掌再从另一侧探出来。改成解剖学行程后累计 255°，
 * 指尖正好落在掌面上（食指 |指尖−MCP| = 1.04，指尖到掌心 1.46，模型单位下指长 3.17）。
 *
 * 一句提醒：如果实机握拳仍然显得不够弯，那是**弯折通道量程**的问题
 * （未标定的柔和预览只到 0.42，或两点标定时握得不够紧），不要回来加大这里的角度
 * —— 这几个数是手的解剖极限，改大只会得到一个反关节的手。
 */
const SEGMENT_MAX_DEG = [85, 100, 70, 0];
/** 拇指只有两个指间关节且行程短得多，单独一套 */
const THUMB_SEGMENT_MAX_DEG = [45, 55, 45, 0];

/** 父页面每帧填这个对象，本组件只读 */
export interface HandDrive {
  /** IMU 四元数 [w, x, y, z]（协议原序；已过朝向标定的话就是标定后的值） */
  quaternion: [number, number, number, number];
  /** 5 指弯曲度 0~1，canonical 拇指→小指。1 = 握拳，角度分配见 SEGMENT_MAX_DEG */
  curl: number[];
  /** 是否有活数据；false 时手模停在静止姿态 */
  hasData: boolean;
}

export function makeHandDrive(): HandDrive {
  return { quaternion: [1, 0, 0, 0], curl: [0, 0, 0, 0, 0], hasData: false };
}

function JointBeacons({
  model,
  rootRef,
}: {
  model: Object3D;
  rootRef: RefObject<Group | null>;
}) {
  const beaconRefs = useRef<(Group | null)[]>([]);
  useFrame(() => {
    fingerBoneNames.forEach((names, index) => {
      const target = model.getObjectByName(names[1]);
      const beacon = beaconRefs.current[index];
      const root = rootRef.current;
      if (!target || !beacon || !root) return;
      const localPosition = target.getWorldPosition(new Vector3());
      root.worldToLocal(localPosition);
      beacon.position.copy(localPosition);
    });
  });
  return (
    <>
      {fingerBoneNames.map((_, index) => (
        <group
          key={`beacon-${index}`}
          ref={(node) => {
            beaconRefs.current[index] = node;
          }}
        >
          <mesh>
            <sphereGeometry args={[0.12, 18, 18]} />
            <meshBasicMaterial color={AMBER} transparent opacity={0.92} />
          </mesh>
          <mesh scale={1.78}>
            <sphereGeometry args={[0.12, 18, 18]} />
            <meshBasicMaterial color={AMBER} transparent opacity={0.14} />
          </mesh>
        </group>
      ))}
    </>
  );
}

/**
 * 骨架档的本体：21 个关节球 + 25 根骨头圆柱（23 段手 + 2 段小臂），
 * 外加同数量的半透明光晕壳（见 JOINT_GLOW_SCALE 那段：浅底上"发光"只能这么做）。
 *
 * 数据源是 **glb 自己的骨骼** —— `AnimatedHand` 每帧已经在写这些骨头的四元数了，
 * 这里只是换一种画法读同一份结果。所以骨架和实心手模是**同一条链路**，
 * 不需要摄像头、也不需要任何已训练模型（`lib/skeletonModel.ts` 那条触觉→骨架回归
 * 要模型、新机器上会一片黑，而且它的深度通道实际是废的，见文件头）。
 *
 * 坐标换算沿用 `JointBeacons` 的写法：读骨头世界坐标 → `root.worldToLocal`。
 * 本组件挂在 `handRef` 群组内，换算回来的局部坐标再被那一层的 IMU 旋转 + ±0.78
 * 缩放重新作用一次，净效果正确。
 *
 * 用 4 个 `InstancedMesh`（本体 2 + 光晕壳 2）而不是上百个 `<mesh>`：
 * 一只手 4 个 draw call。
 * 材质是**不受光**的 basic —— 除了霓虹观感，它还顺带避开了镜像的坑：glb 根节点里
 * 藏着一个 `scale.x = -0.8`（rig 名字就叫 `..._Left`），和这里 `side` 那个 ±0.78
 * 相乘之后，**两侧之中恰好有一侧净行列式为负**、法线被翻转。basic 材质不看法线，
 * 所以两侧都不受影响 —— 也就不必去确认到底是哪一侧。
 */
function HandBones({
  model,
  rootRef,
  dimmed,
}: {
  model: Object3D;
  rootRef: RefObject<Group | null>;
  dimmed: boolean;
}) {
  const jointsRef = useRef<InstancedMesh>(null);
  const bonesRef = useRef<InstancedMesh>(null);
  /* 光晕壳，矩阵跟着本体走、只是放大（见 JOINT_GLOW_SCALE 那段） */
  const jointGlowRef = useRef<InstancedMesh>(null);
  const boneGlowRef = useRef<InstancedMesh>(null);

  /* 每帧要摆 92 个实例，临时对象全部复用 —— 在 useFrame 里 new 会给 GC 添活 */
  const tmp = useMemo(
    () => ({
      mid: new Vector3(),
      dir: new Vector3(),
      quat: new Quaternion(),
      scale: new Vector3(),
      matrix: new Matrix4(),
      /* 端点位置缓存（含小臂那 2 个）。骨头段直接从这里取两端，不重复查骨骼树 */
      points: Array.from({ length: POINT_COUNT }, () => new Vector3()),
      identity: new Quaternion(),
      one: new Vector3(1, 1, 1),
      up: new Vector3(0, 1, 0),
    }),
    []
  );

  /* 颜色只在挂载（和连接状态变化）时设一次，之后每帧只动矩阵 */
  useEffect(() => {
    const joints = jointsRef.current;
    const bones = bonesRef.current;
    const jointGlow = jointGlowRef.current;
    const boneGlow = boneGlowRef.current;
    if (!joints || !bones || !jointGlow || !boneGlow) return;
    const color = new Color();
    for (let i = 0; i < JOINT_COUNT; i++) {
      // 关节归它所在的手指；腕点单独一个更深的锚点色
      color.set(
        dimmed
          ? BONE_COLOR_DIMMED
          : i === 0
            ? WRIST_COLOR
            : JOINT_COLORS[Math.floor((i - 1) / 4)]
      );
      joints.setColorAt(i, color);
      // 光晕和本体同色 —— 换个色相就不是"发光"了，是套了个圈
      jointGlow.setColorAt(i, color);
    }
    BONE_LINKS.forEach(([, , linkType], i) => {
      // 小臂在 dimmed 档也保持自己那个更浅的灰：换成 BONE_COLOR_DIMMED 反而比
      // 手骨更深，断连时小臂会变得比手还显眼
      color.set(
        linkType === FOREARM_LINK
          ? FOREARM_COLOR
          : dimmed
            ? BONE_COLOR_DIMMED
            : BONE_COLOR
      );
      bones.setColorAt(i, color);
      boneGlow.setColorAt(i, color);
    });
    for (const m of [joints, bones, jointGlow, boneGlow]) {
      if (m.instanceColor) m.instanceColor.needsUpdate = true;
    }
  }, [dimmed]);

  useFrame(() => {
    const root = rootRef.current;
    const joints = jointsRef.current;
    const bones = bonesRef.current;
    const jointGlow = jointGlowRef.current;
    const boneGlow = boneGlowRef.current;
    if (!root || !joints || !bones || !jointGlow || !boneGlow) return;

    for (let i = 0; i < POINT_COUNT; i++) {
      const bone = model.getObjectByName(jointBoneName(i));
      // 查不到就**沿用上一帧**：归零会把这个关节甩到腕原点，凭空拉出一根长骨头
      if (!bone) continue;
      bone.getWorldPosition(tmp.points[i]);
      root.worldToLocal(tmp.points[i]);
    }

    for (let i = 0; i < JOINT_COUNT; i++) {
      tmp.matrix.compose(tmp.points[i], tmp.identity, tmp.one);
      joints.setMatrixAt(i, tmp.matrix);
      // 光晕壳：同一个位置，整体放大
      tmp.matrix.compose(
        tmp.points[i],
        tmp.identity,
        tmp.scale.setScalar(JOINT_GLOW_SCALE)
      );
      jointGlow.setMatrixAt(i, tmp.matrix);
    }
    joints.instanceMatrix.needsUpdate = true;
    jointGlow.instanceMatrix.needsUpdate = true;

    BONE_LINKS.forEach(([from, to, linkType], i) => {
      const a = tmp.points[from];
      const b = tmp.points[to];
      tmp.mid.addVectors(a, b).multiplyScalar(0.5);
      tmp.dir.subVectors(b, a);
      const length = tmp.dir.length();
      const forearm = linkType === FOREARM_LINK;
      if (length < 1e-6) {
        // 两端重合（还没拿到骨骼时会这样）：缩到 0，别留一根长度未定义的柱子
        tmp.matrix.compose(tmp.mid, tmp.identity, tmp.scale.set(0, 0, 0));
        bones.setMatrixAt(i, tmp.matrix);
        boneGlow.setMatrixAt(i, tmp.matrix);
        return;
      }
      tmp.dir.divideScalar(length);
      // 圆柱几何是沿 +Y 的单位长度，转到骨头方向、再把 Y 拉到实际长度。
      // XZ 那一路是粗细：手骨是 1（就是 BONE_RADIUS），小臂按比例改 ——
      // 用缩放而不是再开一个 InstancedMesh，省一个 draw call
      const radius = forearm ? FOREARM_RADIUS / BONE_RADIUS : 1;
      tmp.quat.setFromUnitVectors(tmp.up, tmp.dir);
      tmp.matrix.compose(
        tmp.mid,
        tmp.quat,
        tmp.scale.set(radius, length, radius)
      );
      bones.setMatrixAt(i, tmp.matrix);

      // 光晕壳：只放粗、长度照抄本体（放长会从关节球两头戳出来）。
      // 小臂缩到 0 —— 它不参与发光，见 FOREARM_COLOR 那段
      const glow = forearm ? 0 : radius * BONE_GLOW_SCALE;
      tmp.matrix.compose(
        tmp.mid,
        tmp.quat,
        tmp.scale.set(glow, glow === 0 ? 0 : length, glow)
      );
      boneGlow.setMatrixAt(i, tmp.matrix);
    });
    bones.instanceMatrix.needsUpdate = true;
    boneGlow.instanceMatrix.needsUpdate = true;
  });

  return (
    <>
      {/*
        光晕壳先画、本体后画。两个壳都 `depthWrite={false}`：它们把本体整个套在
        里面，一旦写深度，本体就会被自己的壳挡掉一部分、边上出现硬边。
        `depthTest` 保持开着 —— 手指前后交叠时光晕仍然要被挡住，否则远处那根手指
        的晕会浮到近处手指上面，深度关系就乱了。
      */}
      <instancedMesh
        ref={boneGlowRef}
        args={[undefined, undefined, BONE_LINKS.length]}
        frustumCulled={false}
      >
        <cylinderGeometry args={[BONE_RADIUS, BONE_RADIUS, 1, 8, 1, true]} />
        <meshBasicMaterial
          toneMapped={false}
          transparent
          depthWrite={false}
          opacity={dimmed ? 0 : BONE_GLOW_OPACITY}
        />
      </instancedMesh>
      <instancedMesh
        ref={jointGlowRef}
        args={[undefined, undefined, JOINT_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[JOINT_RADIUS, 12, 12]} />
        {/* 断连那只手不发光：灰骨架 + 会发光，读起来仍像在正常工作 */}
        <meshBasicMaterial
          toneMapped={false}
          transparent
          depthWrite={false}
          opacity={dimmed ? 0 : JOINT_GLOW_OPACITY}
        />
      </instancedMesh>
      {/* frustumCulled 关掉：InstancedMesh 的包围球不跟着实例矩阵走，
          手转到一定角度时整副骨架会被判成出画而整体消失 */}
      <instancedMesh
        ref={jointsRef}
        args={[undefined, undefined, JOINT_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[JOINT_RADIUS, 12, 12]} />
        {/* toneMapped=false：不经色调映射，屏幕上拿到的就是 JOINT_COLORS 那几个
            十六进制原值。这套冷色是拉平明度调出来的，过一遍色调映射就不平了 */}
        <meshBasicMaterial
          toneMapped={false}
          transparent
          opacity={dimmed ? 0.5 : 1}
        />
      </instancedMesh>
      <instancedMesh
        ref={bonesRef}
        args={[undefined, undefined, BONE_LINKS.length]}
        frustumCulled={false}
      >
        {/* openEnded：两头的盖子都埋在关节球里，画了也看不见 */}
        <cylinderGeometry args={[BONE_RADIUS, BONE_RADIUS, 1, 8, 1, true]} />
        {/* 连上时不透明：骨头本来就压在门限附近，再打个 0.95 的折就掉到 3:1 以下 */}
        <meshBasicMaterial
          toneMapped={false}
          transparent
          opacity={dimmed ? 0.45 : 1}
        />
      </instancedMesh>
    </>
  );
}

/**
 * 一只手的骨骼驱动。**全项目只有这一份** —— 见文件头那条警告。
 *
 * 它只负责"手本身"：位置固定在 `[0,-1.2,0]`（该 group 的原点就是腕关节的旋转支点，
 * 外层再包一层 `position` 做摆位是安全的）。前臂是 glb 自带的，不用也不该另画。
 */
export function AnimatedHand({
  driveRef,
  side,
  dimmed = false,
  mode = "mesh",
}: {
  driveRef: RefObject<HandDrive>;
  side: "left" | "right";
  /** 这只手套没连上：画成灰影、并且不画关节光点（光点会让它看着像在工作） */
  dimmed?: boolean;
  /**
   * 画实心手模还是骨架。**默认 mesh** —— `/mocap` 自检、向导、VirtualMocap
   * 都不传这个参数，加了骨架档也一行不用改、观感完全不变。
   * 只有 `/translate` 会传 `"skeleton"`。
   */
  mode?: "mesh" | "skeleton";
}) {
  const gltf = useGLTF(MODEL_URL);
  // 必须 clone：直接用 gltf.scene 会让所有实例共享同一套骨骼状态
  const model = useMemo(() => {
    const cloned = cloneSkeleton(gltf.scene);
    // 前臂拉长（幂等，见 stretchForearmGeometry 上那段）
    cloned.traverse((node: Object3D) => {
      if (node instanceof Mesh) stretchForearmGeometry(node.geometry);
    });
    return cloned;
  }, [gltf.scene]);
  const handRef = useRef<Group>(null);
  const restQuaternions = useRef(new Map<string, Quaternion>());
  const wrist = useRef<Object3D | null>(null);
  const targetQuat = useRef(new Quaternion());

  useEffect(() => {
    model.traverse((node: Object3D) => {
      if (node instanceof Bone && node.name.startsWith("Finger_")) {
        restQuaternions.current.set(node.uuid, node.quaternion.clone());
      }
      if (node.name === "Wrist") wrist.current = node;
      if (node instanceof Mesh) {
        /*
         * 骨架档把蒙皮网格藏掉。
         * `visible=false` 只影响渲染，**不影响 `updateMatrixWorld`** —— 骨头的
         * 世界矩阵照常每帧更新，所以 HandBones 读到的姿态不会停在藏起来那一刻。
         */
        node.visible = mode === "mesh";
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        // 克隆材质避免实例间共享；负缩放镜像会翻转三角形绕序，必须双面渲染，
        // 否则右手只剩内表面可见（看起来像"反面"）。
        const preset = dimmed ? HAND_MATERIAL_DIMMED : HAND_MATERIAL;
        const cloned = materials.map((material) => {
          const copy = material.clone();
          copy.side = DoubleSide;
          if ("color" in copy && copy.color) copy.color.set(preset.color);
          if ("roughness" in copy) copy.roughness = preset.roughness;
          if ("metalness" in copy) copy.metalness = preset.metalness;
          // 透明只在灰影档开：整个模型开 transparent 会走另一条排序路径，
          // 正常档没必要为此付代价、也会让指缝出现穿透感
          copy.transparent = dimmed;
          copy.opacity = dimmed ? HAND_MATERIAL_DIMMED.opacity : 1;
          copy.depthWrite = !dimmed;
          return copy;
        });
        node.material = Array.isArray(node.material) ? cloned : cloned[0];
      }
    });
    // dimmed 进依赖：连上/断开手套要立刻换材质。少了它断开后手仍是白的，
    // 而"白手 + 不动"看起来就是"戴着没动"。
    // mode 进依赖：切档要立刻藏/显网格
  }, [model, dimmed, mode]);

  useFrame((_, delta) => {
    const drive = driveRef.current;

    if (handRef.current && drive?.hasData) {
      // 协议的四元数是 [w,x,y,z]，three.js 构造函数是 (x,y,z,w) —— 顺序不能照抄
      const [w, x, y, z] = drive.quaternion;
      targetQuat.current.set(x, y, z, w);
      handRef.current.quaternion.slerp(targetQuat.current, 1 - Math.exp(-delta * 9));
    }

    fingerBoneNames.forEach((names, fingerIndex) => {
      const raw = drive?.curl[fingerIndex] ?? 0;
      const curl = raw <= 0 ? 0 : raw > 1 ? 1 : raw;
      const maxBySegment =
        fingerIndex === 0 ? THUMB_SEGMENT_MAX_DEG : SEGMENT_MAX_DEG;
      names.forEach((name, segmentIndex) => {
        const bone = model.getObjectByName(name);
        if (!(bone instanceof Bone)) return;
        const rest = restQuaternions.current.get(bone.uuid);
        if (!rest) return;
        // 每节按自己的解剖学行程折算；行程 0 的指尖骨会被 slerp 回静止姿态
        const signedAngle = MathUtils.degToRad(
          curl * (maxBySegment[segmentIndex] ?? 0) * -1
        );
        const target = rest
          .clone()
          .multiply(new Quaternion().setFromAxisAngle(AXIS, signedAngle));
        // 越靠近指根收敛越快，让整根手指看起来是一条连续的弧
        bone.quaternion.slerp(target, 1 - Math.exp(-delta * (12 - segmentIndex)));
      });
    });

    if (wrist.current) wrist.current.updateMatrixWorld();
  });

  /*
   * ===== 轴心就是腕关节 =====
   *
   * 手模是被 `handRef` 上的 IMU 四元数**整体**转的，所以"绕哪儿转"就是这个群组的
   * 原点在哪儿 —— 这里是腕，腕钉死在画面中央，整条前臂绕着腕扫。
   *
   * ⚠ 试过把原点挪到前臂末端那个切面上（外层平移到切面、内层反向平移回来，
   *   静止画面逐像素不变，只改转动中心），为的是让切面永远沉在取景底边之下、
   *   不再荡进画面。**上屏效果不行，已回退**，别再走这条路：轴心挪到肘端之后
   *   手离轴心 7～12.6 个世界单位，同样的转角画面里扫过的距离是绕腕的 2.3 倍，
   *   倾到 45° 上下手就整个甩出画了。切面偶尔露脸，比手到处乱飞好。
   */
  return (
    <group
      ref={handRef}
      position={[0, -1.2, 0]}
      scale={side === "right" ? [-0.78, 0.78, 0.78] : [0.78, 0.78, 0.78]}
    >
      {/* 基准姿态：把网格空间的 +z（指尖方向）转到 +y，指尖朝上；
          IMU 旋转在外层群组叠加 */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <primitive object={model} />
      </group>
      {/* 骨架档不画 JointBeacons：那 5 颗琥珀光点会和骨架自己的关节球撞在一起 */}
      {mode === "mesh" && !dimmed && <JointBeacons model={model} rootRef={handRef} />}
      {mode === "skeleton" && (
        <HandBones model={model} rootRef={handRef} dimmed={dimmed} />
      )}
    </group>
  );
}

/**
 * 场景灯光。**所有手模视口共用这一份** —— /mocap 自检那格和 /translate 那两格
 * 必须看着是同一只手，各写一份灯光的话两页的手会呈现出不同的材质感，
 * 很容易被当成"手模不一样"。
 */
export function HandSceneLights() {
  return (
    <>
      <ambientLight intensity={1.35} />
      <directionalLight position={[4, 7, 7]} intensity={2.3} color="#ffffff" />
      {/* 补光换成中性冷白：原来这两路是青 + 琥珀（深色底上给白手套镀边用的），
          浅底上会把白手染出一层青绿，看着像脏了 */}
      <directionalLight position={[-6, 1, 4]} intensity={1.05} color="#e8eef8" />
      <pointLight position={[0, -4, 4]} intensity={0.8} color="#f2f5fa" />
    </>
  );
}

export interface HandModelProps {
  driveRef: RefObject<HandDrive>;
  side: "left" | "right";
  /**
   * 实心手模还是骨架。**默认 mesh**，不传就是原样 —— 见 `AnimatedHand` 上那段。
   */
  mode?: "mesh" | "skeleton";
}

/**
 * memo：父页面为了刷新比例条会以约 12Hz 重渲染，而本组件的 props 全是稳定引用
 * （ref + 字符串），没必要跟着重建 r3f 场景树。
 */
export default memo(function HandModel({ driveRef, side, mode = "mesh" }: HandModelProps) {
  return (
    <Canvas
      camera={{ position: [0, 0.34, 15.4], fov: 34 }}
      dpr={[1, 1.6]}
      gl={{ antialias: true, alpha: true }}
    >
      <HandSceneLights />
      <Suspense fallback={null}>
        <AnimatedHand driveRef={driveRef} side={side} mode={mode} />
      </Suspense>
      {/* 骨架档不投接触阴影：没有实心网格，投下来的是一地碎斑。
          地面高度要跟着取景走（见文件头第 2 条）：相机是平视的，这个水平面在默认
          视角下**本来就看不见**（边看过去是一条线），只在用户拖着 OrbitControls
          俯视时才现身 —— 那时候它得还在取景里。现在取景底是 -4.37，-3.45 在里面 */}
      {mode === "mesh" && (
        <ContactShadows
          position={[0, -3.45, 0]}
          opacity={0.18}
          scale={11}
          blur={2.4}
          far={7}
          color={SHADOW_COLOR}
        />
      )}
      {/* target 必须跟着相机的 y 一起抬：`enablePan={false}` 之下轨道是绕 target 转的，
          留在原点的话画面会绕着手腕下面那截前臂打转，拖两下手就甩出画。
          minDistance 从 9 放到 6：默认距离已经是 13.1，9 那道下限只剩 1.45× 可推近 */}
      <OrbitControls
        target={[0, 0.34, 0]}
        enablePan={false}
        minDistance={6}
        maxDistance={22}
        minPolarAngle={0.75}
        maxPolarAngle={2.1}
      />
    </Canvas>
  );
});

useGLTF.preload(MODEL_URL);
