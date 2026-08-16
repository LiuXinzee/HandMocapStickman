/*
 * HandModel — 由「弯折通道 + IMU 四元数」直接驱动的 3D 手模
 *
 * 移植自 glove_visual/cc_part2 的 HandScene.tsx（同一份硬件协议的分叉项目），
 * 按 deaf-kit 改成单手 + Cyberpunk HUD 配色 + 白手套材质。
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
  DoubleSide,
  Group,
  MathUtils,
  Mesh,
  Object3D,
  Quaternion,
  Vector3,
} from "three";
import { clone as cloneSkeleton } from "three/addons/utils/SkeletonUtils.js";

const MODEL_URL = "/assets/hand1.glb";

/**
 * 实物手套是**白色**的（规格书 p12 写"黑色面料"与实物不符），
 * 所以用 cc_part2 的 white 预设：亮面料 + 低粗糙度 + 一点点金属感。
 */
const HAND_MATERIAL = {
  color: "#dce9f0",
  roughness: 0.4,
  metalness: 0.08,
} as const;

/** 本页主色（琥珀）与全站强调色（青） */
const AMBER = "#f59e0b";
const CYAN = "#00f0ff";

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

function AnimatedHand({
  driveRef,
  side,
}: {
  driveRef: RefObject<HandDrive>;
  side: "left" | "right";
}) {
  const gltf = useGLTF(MODEL_URL);
  // 必须 clone：直接用 gltf.scene 会让所有实例共享同一套骨骼状态
  const model = useMemo(() => cloneSkeleton(gltf.scene), [gltf.scene]);
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
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        // 克隆材质避免实例间共享；负缩放镜像会翻转三角形绕序，必须双面渲染，
        // 否则右手只剩内表面可见（看起来像"反面"）。
        const cloned = materials.map((material) => {
          const copy = material.clone();
          copy.side = DoubleSide;
          if ("color" in copy && copy.color) copy.color.set(HAND_MATERIAL.color);
          if ("roughness" in copy) copy.roughness = HAND_MATERIAL.roughness;
          if ("metalness" in copy) copy.metalness = HAND_MATERIAL.metalness;
          return copy;
        });
        node.material = Array.isArray(node.material) ? cloned : cloned[0];
      }
    });
  }, [model]);

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

  return (
    <group
      ref={handRef}
      position={[0, -1.2, 0]}
      scale={side === "right" ? [-0.78, 0.78, 0.78] : [0.78, 0.78, 0.78]}
    >
      {/* 基准姿态：把模型从"指尖朝镜头"抬起 90°，指尖朝上；IMU 旋转在外层群组叠加 */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <primitive object={model} />
      </group>
      <JointBeacons model={model} rootRef={handRef} />
    </group>
  );
}

export interface HandModelProps {
  driveRef: RefObject<HandDrive>;
  side: "left" | "right";
}

/**
 * memo：父页面为了刷新比例条会以约 12Hz 重渲染，而本组件的 props 全是稳定引用
 * （ref + 字符串），没必要跟着重建 r3f 场景树。
 */
export default memo(function HandModel({ driveRef, side }: HandModelProps) {
  return (
    <Canvas
      camera={{ position: [0, 0.15, 17], fov: 34 }}
      dpr={[1, 1.6]}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={1.35} />
      <directionalLight position={[4, 7, 7]} intensity={2.3} color="#ffffff" />
      <directionalLight position={[-6, 1, 4]} intensity={1.15} color={CYAN} />
      <pointLight position={[0, -4, 4]} intensity={0.9} color={AMBER} />
      <Suspense fallback={null}>
        <AnimatedHand driveRef={driveRef} side={side} />
      </Suspense>
      <ContactShadows
        position={[0, -3.45, 0]}
        opacity={0.18}
        scale={11}
        blur={2.4}
        far={7}
        color={CYAN}
      />
      <OrbitControls
        enablePan={false}
        minDistance={9}
        maxDistance={22}
        minPolarAngle={0.75}
        maxPolarAngle={2.1}
      />
    </Canvas>
  );
});

useGLTF.preload(MODEL_URL);
