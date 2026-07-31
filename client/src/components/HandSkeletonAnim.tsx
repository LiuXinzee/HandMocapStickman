/*
 * HandSkeletonAnim — 骨架线条风格手语动画组件
 * DESIGN: Cyberpunk HUD 风格，霓虹色骨架线条 + 关节发光
 *
 * 使用 SVG 绘制手部骨架，通过 CSS 动画展示手势动作。
 * 每个手语词汇有对应的骨架姿态和动画。
 */
import { useEffect, useState } from "react";

interface HandSkeletonAnimProps {
  wordId: string;
  size?: number;
}

// 手部关节坐标定义 (归一化 0-100)
// 手腕 → 掌根 → 各手指关节
interface HandPose {
  wrist: [number, number];
  // 每根手指 4 个关节: [MCP, PIP, DIP, TIP]
  thumb: [[number, number], [number, number], [number, number], [number, number]];
  index: [[number, number], [number, number], [number, number], [number, number]];
  middle: [[number, number], [number, number], [number, number], [number, number]];
  ring: [[number, number], [number, number], [number, number], [number, number]];
  pinky: [[number, number], [number, number], [number, number], [number, number]];
}

// 预定义手势姿态
const POSES: Record<string, HandPose[]> = {
  // 你好: 食指和中指伸出，向前点头动作
  hello: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [30, 65], [25, 58], [22, 52]],
      index: [[44, 68], [42, 50], [41, 35], [40, 22]],
      middle: [[50, 67], [50, 48], [50, 33], [50, 18]],
      ring: [[56, 70], [58, 68], [60, 72], [61, 76]],
      pinky: [[62, 74], [65, 73], [67, 77], [68, 80]],
    },
    {
      wrist: [50, 90],
      thumb: [[38, 72], [30, 65], [25, 58], [22, 52]],
      index: [[44, 68], [40, 48], [37, 33], [35, 20]],
      middle: [[50, 67], [48, 46], [46, 31], [44, 16]],
      ring: [[56, 70], [58, 68], [60, 72], [61, 76]],
      pinky: [[62, 74], [65, 73], [67, 77], [68, 80]],
    },
  ],

  // 谢谢: 手心向下从嘴边向前伸出
  thank_you: [
    {
      wrist: [50, 85],
      thumb: [[38, 70], [30, 62], [28, 55], [26, 50]],
      index: [[45, 68], [40, 58], [37, 50], [35, 43]],
      middle: [[50, 66], [48, 55], [46, 47], [44, 40]],
      ring: [[55, 68], [55, 57], [54, 49], [53, 42]],
      pinky: [[60, 72], [62, 62], [62, 55], [62, 48]],
    },
    {
      wrist: [50, 85],
      thumb: [[35, 68], [25, 58], [22, 48], [20, 40]],
      index: [[42, 65], [34, 52], [28, 42], [24, 33]],
      middle: [[48, 63], [42, 49], [37, 38], [33, 29]],
      ring: [[54, 65], [50, 52], [46, 41], [43, 32]],
      pinky: [[60, 69], [58, 57], [56, 47], [54, 38]],
    },
  ],

  // 对不起: 握拳放在胸前
  sorry: [
    {
      wrist: [50, 88],
      thumb: [[40, 72], [35, 66], [38, 62], [42, 60]],
      index: [[46, 68], [48, 62], [52, 60], [50, 64]],
      middle: [[52, 67], [54, 60], [56, 58], [54, 62]],
      ring: [[57, 69], [59, 63], [60, 60], [58, 64]],
      pinky: [[62, 73], [63, 67], [63, 64], [62, 68]],
    },
    {
      wrist: [50, 82],
      thumb: [[40, 66], [35, 60], [38, 56], [42, 54]],
      index: [[46, 62], [48, 56], [52, 54], [50, 58]],
      middle: [[52, 61], [54, 54], [56, 52], [54, 56]],
      ring: [[57, 63], [59, 57], [60, 54], [58, 58]],
      pinky: [[62, 67], [63, 61], [63, 58], [62, 62]],
    },
  ],

  // 再见: 手掌左右摆动
  goodbye: [
    {
      wrist: [50, 88],
      thumb: [[36, 72], [28, 64], [24, 56], [22, 48]],
      index: [[42, 66], [36, 50], [33, 38], [30, 26]],
      middle: [[48, 64], [44, 47], [42, 34], [40, 22]],
      ring: [[54, 66], [52, 50], [51, 38], [50, 26]],
      pinky: [[60, 70], [60, 55], [60, 43], [60, 32]],
    },
    {
      wrist: [50, 88],
      thumb: [[64, 72], [72, 64], [76, 56], [78, 48]],
      index: [[58, 66], [64, 50], [67, 38], [70, 26]],
      middle: [[52, 64], [56, 47], [58, 34], [60, 22]],
      ring: [[46, 66], [48, 50], [49, 38], [50, 26]],
      pinky: [[40, 70], [40, 55], [40, 43], [40, 32]],
    },
  ],

  // 请: 手掌向上向前伸出
  please: [
    {
      wrist: [50, 88],
      thumb: [[36, 72], [28, 66], [24, 60], [22, 54]],
      index: [[44, 66], [38, 54], [34, 44], [30, 36]],
      middle: [[50, 64], [46, 51], [43, 40], [40, 30]],
      ring: [[56, 66], [54, 53], [52, 42], [50, 32]],
      pinky: [[62, 70], [62, 58], [61, 48], [60, 38]],
    },
    {
      wrist: [50, 85],
      thumb: [[34, 68], [24, 60], [20, 52], [18, 44]],
      index: [[42, 62], [34, 46], [28, 34], [24, 24]],
      middle: [[48, 60], [42, 43], [37, 30], [33, 18]],
      ring: [[54, 62], [50, 46], [47, 34], [44, 22]],
      pinky: [[60, 66], [58, 52], [56, 40], [54, 28]],
    },
  ],

  // 零: 拇指和食指圈成O
  num_0: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [30, 62], [28, 52], [34, 46]],
      index: [[46, 68], [42, 55], [38, 46], [36, 48]],
      middle: [[52, 67], [54, 55], [55, 48], [55, 55]],
      ring: [[57, 70], [60, 60], [61, 55], [60, 60]],
      pinky: [[62, 74], [65, 65], [66, 60], [65, 65]],
    },
  ],

  // 一: 伸出食指
  num_1: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [32, 67], [35, 63], [39, 62]],
      index: [[46, 68], [44, 50], [43, 35], [42, 20]],
      middle: [[52, 67], [54, 62], [56, 60], [54, 64]],
      ring: [[57, 70], [59, 65], [60, 63], [58, 67]],
      pinky: [[62, 74], [64, 70], [64, 68], [63, 72]],
    },
  ],

  // 二: 食指和中指伸出
  num_2: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [32, 67], [35, 63], [39, 62]],
      index: [[44, 68], [40, 50], [38, 35], [36, 20]],
      middle: [[52, 66], [52, 48], [52, 33], [52, 18]],
      ring: [[58, 70], [60, 66], [61, 64], [59, 68]],
      pinky: [[63, 74], [65, 71], [65, 69], [64, 73]],
    },
  ],

  // 三: 食指、中指、无名指伸出
  num_3: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [32, 67], [35, 63], [39, 62]],
      index: [[43, 68], [38, 50], [35, 35], [33, 20]],
      middle: [[50, 66], [50, 47], [50, 32], [50, 17]],
      ring: [[57, 68], [60, 50], [62, 35], [64, 20]],
      pinky: [[63, 74], [66, 72], [67, 70], [66, 74]],
    },
  ],

  // 四: 四指伸出，拇指弯曲
  num_4: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [33, 68], [36, 65], [40, 64]],
      index: [[42, 68], [36, 50], [33, 35], [30, 20]],
      middle: [[48, 66], [46, 47], [45, 32], [44, 17]],
      ring: [[55, 67], [56, 48], [57, 33], [58, 18]],
      pinky: [[62, 70], [64, 52], [66, 37], [68, 22]],
    },
  ],

  // 五: 五指张开
  num_5: [
    {
      wrist: [50, 90],
      thumb: [[36, 72], [26, 62], [20, 52], [16, 42]],
      index: [[42, 66], [35, 48], [30, 34], [26, 20]],
      middle: [[50, 64], [48, 44], [47, 28], [46, 14]],
      ring: [[58, 66], [62, 48], [64, 34], [66, 20]],
      pinky: [[64, 70], [70, 55], [74, 42], [78, 30]],
    },
  ],

  // 六: 拇指和小指伸出
  num_6: [
    {
      wrist: [50, 90],
      thumb: [[36, 72], [26, 62], [20, 52], [16, 42]],
      index: [[46, 68], [48, 63], [50, 61], [48, 65]],
      middle: [[52, 67], [54, 62], [55, 60], [53, 64]],
      ring: [[57, 70], [59, 65], [60, 63], [58, 67]],
      pinky: [[63, 73], [68, 58], [72, 45], [76, 32]],
    },
  ],

  // 七: 拇指食指中指捏在一起
  num_7: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [32, 62], [30, 52], [35, 44]],
      index: [[46, 68], [42, 55], [39, 46], [37, 44]],
      middle: [[52, 66], [50, 54], [47, 46], [45, 44]],
      ring: [[57, 70], [59, 65], [60, 63], [58, 67]],
      pinky: [[62, 74], [64, 70], [65, 68], [63, 72]],
    },
  ],

  // 八: 拇指和食指伸开成L
  num_8: [
    {
      wrist: [50, 90],
      thumb: [[36, 72], [26, 64], [20, 56], [16, 48]],
      index: [[46, 68], [44, 50], [43, 35], [42, 20]],
      middle: [[52, 67], [54, 63], [55, 61], [53, 65]],
      ring: [[57, 70], [59, 66], [60, 64], [58, 68]],
      pinky: [[62, 74], [64, 71], [65, 69], [63, 73]],
    },
  ],

  // 九: 食指弯曲成钩
  num_9: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [32, 67], [35, 63], [39, 62]],
      index: [[46, 68], [42, 52], [46, 46], [50, 52]],
      middle: [[52, 67], [54, 63], [55, 61], [53, 65]],
      ring: [[57, 70], [59, 66], [60, 64], [58, 68]],
      pinky: [[62, 74], [64, 71], [65, 69], [63, 73]],
    },
  ],

  // 十: 食指交叉
  num_10: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [32, 67], [35, 63], [39, 62]],
      index: [[46, 68], [44, 50], [43, 35], [42, 20]],
      middle: [[52, 67], [54, 63], [55, 61], [53, 65]],
      ring: [[57, 70], [59, 66], [60, 64], [58, 68]],
      pinky: [[62, 74], [64, 71], [65, 69], [63, 73]],
    },
  ],

  // 吃: 手指捏在一起送向嘴边
  eat: [
    {
      wrist: [50, 90],
      thumb: [[38, 72], [33, 62], [32, 52], [36, 46]],
      index: [[46, 68], [42, 56], [39, 48], [38, 46]],
      middle: [[52, 66], [49, 55], [47, 48], [46, 46]],
      ring: [[56, 69], [55, 58], [53, 52], [52, 48]],
      pinky: [[60, 73], [60, 64], [58, 58], [57, 52]],
    },
    {
      wrist: [50, 88],
      thumb: [[38, 68], [33, 56], [32, 44], [36, 36]],
      index: [[46, 64], [42, 48], [39, 38], [38, 36]],
      middle: [[52, 62], [49, 47], [47, 38], [46, 36]],
      ring: [[56, 65], [55, 52], [53, 44], [52, 38]],
      pinky: [[60, 69], [60, 58], [58, 50], [57, 44]],
    },
  ],

  // 喝: 拇指和小指伸出送向嘴边
  drink: [
    {
      wrist: [50, 90],
      thumb: [[36, 72], [26, 64], [20, 56], [16, 48]],
      index: [[46, 68], [48, 63], [50, 61], [48, 65]],
      middle: [[52, 67], [54, 62], [55, 60], [53, 64]],
      ring: [[57, 70], [59, 65], [60, 63], [58, 67]],
      pinky: [[63, 73], [68, 58], [72, 45], [76, 32]],
    },
    {
      wrist: [50, 82],
      thumb: [[36, 64], [26, 54], [20, 44], [16, 36]],
      index: [[46, 60], [48, 55], [50, 53], [48, 57]],
      middle: [[52, 59], [54, 54], [55, 52], [53, 56]],
      ring: [[57, 62], [59, 57], [60, 55], [58, 59]],
      pinky: [[63, 65], [68, 48], [72, 33], [76, 20]],
    },
  ],

  // 停: 手掌向前伸出
  stop: [
    {
      wrist: [50, 90],
      thumb: [[36, 72], [26, 64], [20, 56], [16, 48]],
      index: [[42, 66], [36, 50], [33, 38], [30, 26]],
      middle: [[48, 64], [46, 47], [45, 34], [44, 22]],
      ring: [[54, 66], [54, 49], [54, 36], [54, 24]],
      pinky: [[60, 70], [62, 54], [63, 42], [64, 30]],
    },
  ],
};

// 默认姿态（张开手掌）
const DEFAULT_POSE: HandPose = {
  wrist: [50, 90],
  thumb: [[36, 72], [26, 62], [20, 52], [16, 42]],
  index: [[42, 66], [35, 48], [30, 34], [26, 20]],
  middle: [[50, 64], [48, 44], [47, 28], [46, 14]],
  ring: [[58, 66], [62, 48], [64, 34], [66, 20]],
  pinky: [[64, 70], [70, 55], [74, 42], [78, 30]],
};

// 线性插值
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpPose(pose1: HandPose, pose2: HandPose, t: number): HandPose {
  const lerpPt = (p1: [number, number], p2: [number, number]): [number, number] => [
    lerp(p1[0], p2[0], t),
    lerp(p1[1], p2[1], t),
  ];
  const lerpFinger = (
    f1: [[number, number], [number, number], [number, number], [number, number]],
    f2: [[number, number], [number, number], [number, number], [number, number]]
  ): [[number, number], [number, number], [number, number], [number, number]] => [
    lerpPt(f1[0], f2[0]),
    lerpPt(f1[1], f2[1]),
    lerpPt(f1[2], f2[2]),
    lerpPt(f1[3], f2[3]),
  ];

  return {
    wrist: lerpPt(pose1.wrist, pose2.wrist),
    thumb: lerpFinger(pose1.thumb, pose2.thumb),
    index: lerpFinger(pose1.index, pose2.index),
    middle: lerpFinger(pose1.middle, pose2.middle),
    ring: lerpFinger(pose1.ring, pose2.ring),
    pinky: lerpFinger(pose1.pinky, pose2.pinky),
  };
}

// 手指颜色
const FINGER_COLORS = {
  thumb: "#4dabf7",
  index: "#69db7c",
  middle: "#ffd43b",
  ring: "#ffa94d",
  pinky: "#ff6b6b",
};

const JOINT_COLOR = "#00f0ff";
const WRIST_COLOR = "#da77f2";

export default function HandSkeletonAnim({ wordId, size = 180 }: HandSkeletonAnimProps) {
  const [currentPose, setCurrentPose] = useState<HandPose>(DEFAULT_POSE);
  const [animFrame, setAnimFrame] = useState(0);

  const poses = POSES[wordId];

  useEffect(() => {
    if (!poses || poses.length === 0) {
      setCurrentPose(DEFAULT_POSE);
      return;
    }

    if (poses.length === 1) {
      setCurrentPose(poses[0]);
      return;
    }

    // 多帧动画
    let frameId: number;
    let startTime = performance.now();
    const duration = 1200; // 每个过渡 1.2s

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const totalDuration = duration * poses.length;
      const loopTime = elapsed % totalDuration;
      const poseIdx = Math.floor(loopTime / duration);
      const t = (loopTime % duration) / duration;

      // 使用 easeInOut 缓动
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      const nextIdx = (poseIdx + 1) % poses.length;
      const interpolated = lerpPose(poses[poseIdx], poses[nextIdx], eased);
      setCurrentPose(interpolated);
      setAnimFrame(poseIdx);

      frameId = requestAnimationFrame(animate);
    };

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [wordId, poses]);

  // 渲染骨架
  const renderFinger = (
    joints: [[number, number], [number, number], [number, number], [number, number]],
    color: string,
    wrist: [number, number]
  ) => {
    const points = [wrist, ...joints];
    const lines = [];
    const circles = [];

    for (let i = 0; i < points.length - 1; i++) {
      lines.push(
        <line
          key={`l${i}`}
          x1={points[i][0]}
          y1={points[i][1]}
          x2={points[i + 1][0]}
          y2={points[i + 1][1]}
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.85"
          style={{ filter: `drop-shadow(0 0 2px ${color})` }}
        />
      );
    }

    for (let i = 0; i < points.length; i++) {
      const isJoint = i > 0;
      const isTip = i === points.length - 1;
      circles.push(
        <circle
          key={`c${i}`}
          cx={points[i][0]}
          cy={points[i][1]}
          r={isTip ? 2.5 : isJoint ? 2 : 3}
          fill={i === 0 ? WRIST_COLOR : JOINT_COLOR}
          opacity={isTip ? 1 : 0.9}
          style={{ filter: `drop-shadow(0 0 ${isTip ? 4 : 2}px ${i === 0 ? WRIST_COLOR : JOINT_COLOR})` }}
        />
      );
    }

    return [...lines, ...circles];
  };

  // 动画方向指示器
  const renderMotionArrow = () => {
    if (!poses || poses.length <= 1) return null;

    // 根据词汇类型显示不同的运动方向
    const motionTypes: Record<string, string> = {
      hello: "↕",
      thank_you: "→",
      sorry: "↕",
      goodbye: "↔",
      please: "→",
      eat: "↑",
      drink: "↑",
    };

    const arrow = motionTypes[wordId];
    if (!arrow) return null;

    return (
      <text
        x="88"
        y="15"
        fontSize="12"
        fill="#00f0ff"
        opacity="0.7"
        textAnchor="middle"
        style={{ filter: "drop-shadow(0 0 3px #00f0ff)" }}
      >
        {arrow}
      </text>
    );
  };

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {/* 背景网格 */}
      <svg
        viewBox="0 0 100 100"
        width={size}
        height={size}
        className="absolute inset-0"
      >
        {/* 网格线 */}
        <defs>
          <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path
              d="M 10 0 L 0 0 0 10"
              fill="none"
              stroke="#00f0ff"
              strokeWidth="0.1"
              opacity="0.2"
            />
          </pattern>
        </defs>
        <rect width="100" height="100" fill="url(#grid)" />

        {/* 扫描线动画 */}
        <line
          x1="0"
          y1="0"
          x2="100"
          y2="0"
          stroke="#00f0ff"
          strokeWidth="0.3"
          opacity="0.4"
          className="animate-scan-line"
        />
      </svg>

      {/* 手部骨架 */}
      <svg
        viewBox="0 0 100 100"
        width={size}
        height={size}
        className="absolute inset-0"
      >
        {/* 掌心连线 */}
        <polygon
          points={`${currentPose.thumb[0].join(",")
            } ${currentPose.index[0].join(",")
            } ${currentPose.middle[0].join(",")
            } ${currentPose.ring[0].join(",")
            } ${currentPose.pinky[0].join(",")}`}
          fill="rgba(0, 240, 255, 0.05)"
          stroke="#00f0ff"
          strokeWidth="0.5"
          opacity="0.4"
        />

        {/* 各手指骨架 */}
        {renderFinger(currentPose.thumb, FINGER_COLORS.thumb, currentPose.wrist)}
        {renderFinger(currentPose.index, FINGER_COLORS.index, currentPose.wrist)}
        {renderFinger(currentPose.middle, FINGER_COLORS.middle, currentPose.wrist)}
        {renderFinger(currentPose.ring, FINGER_COLORS.ring, currentPose.wrist)}
        {renderFinger(currentPose.pinky, FINGER_COLORS.pinky, currentPose.wrist)}

        {/* 手腕 */}
        <circle
          cx={currentPose.wrist[0]}
          cy={currentPose.wrist[1]}
          r="3.5"
          fill={WRIST_COLOR}
          opacity="0.9"
          style={{ filter: `drop-shadow(0 0 5px ${WRIST_COLOR})` }}
        />

        {/* 运动方向箭头 */}
        {renderMotionArrow()}
      </svg>

      {/* 角落装饰 */}
      <div className="absolute top-0 left-0 w-3 h-3 border-t border-l border-[#00f0ff]/40" />
      <div className="absolute top-0 right-0 w-3 h-3 border-t border-r border-[#00f0ff]/40" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b border-l border-[#00f0ff]/40" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b border-r border-[#00f0ff]/40" />

      {/* 帧指示器 */}
      {poses && poses.length > 1 && (
        <div className="absolute bottom-1 left-1/2 -translate-x-1/2 flex gap-1">
          {poses.map((_, i) => (
            <div
              key={i}
              className="w-1 h-1 rounded-full transition-all duration-200"
              style={{
                backgroundColor: i === animFrame ? "#00f0ff" : "#334455",
                boxShadow: i === animFrame ? "0 0 4px #00f0ff" : "none",
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
