import { ContactShadows, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  Suspense,
  useEffect,
  useMemo,
  useRef,
  type CSSProperties,
  type RefObject,
} from "react";
import {
  Bone,
  DoubleSide,
  Group,
  MathUtils,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  Vector3,
} from "three";
import { clone as cloneSkeleton } from "three/addons/utils/SkeletonUtils.js";
import {
  expandGlbFingerBends,
  normalizeGlbQuaternion,
  type GlbQuaternion,
} from "@/lib/glbHandPose";

export const DEFAULT_GLB_HAND_MODEL_URL = "/assets/hand1.glb";

type HandSide = "left" | "right";
export type GlbHandTone = "black" | "white";

export interface GlbHandPose {
  /** Glove protocol quaternion in [w, x, y, z] order. */
  quaternion?: GlbQuaternion | null;
  /** Five per-finger angles or 20 finger-bone angles, in degrees. */
  fingerBends?: readonly number[] | null;
}

export interface GlbHandSceneProps {
  leftPose?: GlbHandPose | null;
  rightPose?: GlbHandPose | null;
  gloveTone?: GlbHandTone;
  modelUrl?: string;
  showJointBeacons?: boolean;
  enableOrbitControls?: boolean;
  className?: string;
  style?: CSSProperties;
}

const HAND_MATERIALS: Record<
  GlbHandTone,
  Record<HandSide, { color: string; roughness: number; metalness: number }>
> = {
  black: {
    left: { color: "#2e373e", roughness: 0.94, metalness: 0 },
    right: { color: "#39434b", roughness: 0.94, metalness: 0 },
  },
  white: {
    left: { color: "#d6e7ee", roughness: 0.4, metalness: 0.08 },
    right: { color: "#e1edf2", roughness: 0.4, metalness: 0.08 },
  },
};

const BEND_AXIS = new Vector3(0, 0, 1);
const FINGER_BONE_NAMES = Array.from({ length: 5 }, (_, fingerIndex) =>
  Array.from(
    { length: 4 },
    (_, segmentIndex) => `Finger_${fingerIndex}${segmentIndex}`
  )
);

function JointBeacons({
  model,
  rootRef,
  side,
}: {
  model: Object3D;
  rootRef: RefObject<Group | null>;
  side: HandSide;
}) {
  const beaconRefs = useRef<(Group | null)[]>([]);
  const worldPosition = useMemo(() => new Vector3(), []);

  useFrame(() => {
    FINGER_BONE_NAMES.forEach((names, fingerIndex) => {
      const target = model.getObjectByName(names[1]);
      const beacon = beaconRefs.current[fingerIndex];
      const root = rootRef.current;
      if (!target || !beacon || !root) return;

      target.getWorldPosition(worldPosition);
      root.worldToLocal(worldPosition);
      beacon.position.copy(worldPosition);
    });
  });

  return (
    <>
      {FINGER_BONE_NAMES.map((_, fingerIndex) => (
        <group
          key={`${side}-${fingerIndex}`}
          ref={node => {
            beaconRefs.current[fingerIndex] = node;
          }}
        >
          <mesh>
            <sphereGeometry args={[0.12, 18, 18]} />
            <meshBasicMaterial color="#037bbe" transparent opacity={0.92} />
          </mesh>
          <mesh scale={1.78}>
            <sphereGeometry args={[0.12, 18, 18]} />
            <meshBasicMaterial color="#037bbe" transparent opacity={0.14} />
          </mesh>
        </group>
      ))}
    </>
  );
}

interface AnimatedGlbHandProps {
  pose: GlbHandPose;
  side: HandSide;
  position: [number, number, number];
  gloveTone: GlbHandTone;
  modelUrl: string;
  showJointBeacons: boolean;
}

function AnimatedGlbHand({
  pose,
  side,
  position,
  gloveTone,
  modelUrl,
  showJointBeacons,
}: AnimatedGlbHandProps) {
  const gltf = useGLTF(modelUrl);
  const model = useMemo(() => {
    const cloned = cloneSkeleton(gltf.scene);
    cloned.traverse(node => {
      if (!(node instanceof Mesh)) return;
      const materials = Array.isArray(node.material)
        ? node.material
        : [node.material];
      const clonedMaterials = materials.map(material => {
        const copy = material.clone();
        copy.side = DoubleSide;
        return copy;
      });
      node.material = Array.isArray(node.material)
        ? clonedMaterials
        : clonedMaterials[0];
    });
    return cloned;
  }, [gltf.scene]);
  const handRef = useRef<Group>(null);

  const restQuaternions = useMemo(() => {
    const quaternions = new Map<string, Quaternion>();
    model.traverse(node => {
      if (node instanceof Bone && node.name.startsWith("Finger_")) {
        quaternions.set(node.name, node.quaternion.clone());
      }
    });
    return quaternions;
  }, [model]);

  useEffect(() => {
    const preset = HAND_MATERIALS[gloveTone][side];
    model.traverse(node => {
      if (!(node instanceof Mesh)) return;
      const materials = Array.isArray(node.material)
        ? node.material
        : [node.material];
      materials.forEach(material => {
        if (!(material instanceof MeshStandardMaterial)) return;
        material.color.set(preset.color);
        material.roughness = preset.roughness;
        material.metalness = preset.metalness;
        material.needsUpdate = true;
      });
    });
  }, [gloveTone, model, side]);

  const normalizedQuaternion = useMemo(
    () => normalizeGlbQuaternion(pose.quaternion),
    [pose.quaternion]
  );
  const targetHandQuaternion = useMemo(() => {
    if (!normalizedQuaternion) return null;
    const [w, x, y, z] = normalizedQuaternion;
    return new Quaternion(x, y, z, w);
  }, [normalizedQuaternion]);
  const expandedBends = useMemo(
    () => expandGlbFingerBends(pose.fingerBends),
    [pose.fingerBends]
  );
  const targetBoneQuaternions = useMemo(() => {
    const targets = new Map<string, Quaternion>();
    FINGER_BONE_NAMES.forEach((names, fingerIndex) => {
      names.forEach((name, segmentIndex) => {
        const rest = restQuaternions.get(name);
        if (!rest) return;
        const thumbFactor = fingerIndex === 0 ? 0.78 : 1;
        const angle = MathUtils.degToRad(
          (expandedBends[fingerIndex]?.[segmentIndex] ?? 0) * thumbFactor * -1
        );
        targets.set(
          name,
          rest
            .clone()
            .multiply(new Quaternion().setFromAxisAngle(BEND_AXIS, angle))
        );
      });
    });
    return targets;
  }, [expandedBends, restQuaternions]);

  useFrame((_, delta) => {
    if (handRef.current && targetHandQuaternion) {
      handRef.current.quaternion.slerp(
        targetHandQuaternion,
        1 - Math.exp(-delta * 9)
      );
    }

    targetBoneQuaternions.forEach((target, name) => {
      const bone = model.getObjectByName(name);
      if (!(bone instanceof Bone)) return;
      bone.quaternion.slerp(target, 1 - Math.exp(-delta * 11));
    });
  });

  return (
    <group
      ref={handRef}
      position={position}
      scale={side === "right" ? [-0.64, 0.64, 0.64] : [0.64, 0.64, 0.64]}
    >
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <primitive object={model} />
      </group>
      {showJointBeacons && (
        <JointBeacons model={model} rootRef={handRef} side={side} />
      )}
    </group>
  );
}

interface GlbHandModelsProps {
  leftPose: GlbHandPose | null;
  rightPose: GlbHandPose | null;
  gloveTone: GlbHandTone;
  modelUrl: string;
  showJointBeacons: boolean;
}

function GlbHandModels({
  leftPose,
  rightPose,
  gloveTone,
  modelUrl,
  showJointBeacons,
}: GlbHandModelsProps) {
  const { width, height } = useThree(state => state.size);
  const hasBothHands = Boolean(leftPose && rightPose);
  const aspect = height > 0 ? width / height : 1;
  const portraitFitAspect = hasBothHands ? 1.6 : 0.9;
  const sceneScale =
    aspect >= 1 ? 1 : Math.min(1, Math.max(0.24, aspect / portraitFitAspect));
  const leftPosition: [number, number, number] = hasBothHands
    ? [-2.9, -1.5, 0]
    : [0, -1.5, 0];
  const rightPosition: [number, number, number] = hasBothHands
    ? [2.9, -1.5, 0]
    : [0, -1.5, 0];

  return (
    <group scale={sceneScale}>
      {leftPose && (
        <AnimatedGlbHand
          pose={leftPose}
          side="left"
          position={leftPosition}
          gloveTone={gloveTone}
          modelUrl={modelUrl}
          showJointBeacons={showJointBeacons}
        />
      )}
      {rightPose && (
        <AnimatedGlbHand
          pose={rightPose}
          side="right"
          position={rightPosition}
          gloveTone={gloveTone}
          modelUrl={modelUrl}
          showJointBeacons={showJointBeacons}
        />
      )}
      {(leftPose || rightPose) && (
        <ContactShadows
          position={[0, -3.45, 0]}
          opacity={0.16}
          scale={11}
          blur={2.4}
          far={7}
          color="#0a7db8"
        />
      )}
    </group>
  );
}

/**
 * Optional GLB visualization ported from glove_app. Existing pages continue to
 * use HandCanvas unless they explicitly mount this component.
 */
export default function GlbHandScene({
  leftPose = null,
  rightPose = null,
  gloveTone = "black",
  modelUrl = DEFAULT_GLB_HAND_MODEL_URL,
  showJointBeacons = true,
  enableOrbitControls = true,
  className,
  style,
}: GlbHandSceneProps) {
  return (
    <div
      className={className}
      style={{ width: "100%", height: "100%", minHeight: 240, ...style }}
      role="img"
      aria-label="3D hand skeleton preview"
    >
      <Canvas
        camera={{ position: [0, 0.15, 19.5], fov: 34 }}
        dpr={[1, 1.6]}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={1.35} />
        <directionalLight
          position={[4, 7, 7]}
          intensity={2.3}
          color="#ffffff"
        />
        <directionalLight
          position={[-6, 1, 4]}
          intensity={1.25}
          color="#92c7f2"
        />
        <pointLight position={[0, -4, 4]} intensity={0.8} color="#0a7db8" />
        <Suspense fallback={null}>
          <GlbHandModels
            leftPose={leftPose}
            rightPose={rightPose}
            gloveTone={gloveTone}
            modelUrl={modelUrl}
            showJointBeacons={showJointBeacons}
          />
        </Suspense>
        {enableOrbitControls && (
          <OrbitControls
            enablePan={false}
            minDistance={10}
            maxDistance={20}
            minPolarAngle={0.75}
            maxPolarAngle={2.1}
          />
        )}
      </Canvas>
    </div>
  );
}

useGLTF.preload(DEFAULT_GLB_HAND_MODEL_URL);
