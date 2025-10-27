// OrientationCubeInCanvas.tsx
// Orientation cube rendered INSIDE the main Canvas using Hud for screen-space positioning

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Hud, OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";

interface OrientationCubeInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  onCubeClick?: (direction: THREE.Vector3) => void;
}

/**
 * Renders the orientation cube in the top-right corner using Hud.
 * Syncs rotation with main camera.
 * 
 * Benefits:
 * - No separate Canvas = no WebGL context issues
 * - Proper screen-space positioning via Hud
 * - Renders in same WebGL context as main scene
 * - Efficient and stable
 */
export function OrientationCubeInCanvas({ mainCameraRef, onCubeClick }: OrientationCubeInCanvasProps) {
  const cubeGroupRef = useRef<THREE.Group>(null);

  // Real-time rotation sync with main camera
  useFrame(() => {
    if (!mainCameraRef.current || !cubeGroupRef.current) return;
    cubeGroupRef.current.quaternion.copy(mainCameraRef.current.quaternion);
  });

  return (
    <Hud renderPriority={1}>
      {/* Orthographic camera for HUD rendering */}
      <OrthographicCamera
        makeDefault
        position={[0, 0, 10]}
        zoom={50}
        near={0.1}
        far={1000}
      />

      {/* Position cube in screen space (top-right corner) */}
      {/* Screen coordinates: (0,0) is center, positive X is right, positive Y is up */}
      <group position={[6.5, 4.5, 0]}>
        {/* Lighting for the cube */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-3, -3, -3]} intensity={0.3} />

        {/* Cube group that syncs with main camera rotation */}
        <group ref={cubeGroupRef}>
          <OrientationCubeMesh onFaceClick={onCubeClick} useSTL={false} />
        </group>

        {/* Subtle ground plane for depth */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.2, 0]}>
          <planeGeometry args={[3, 3]} />
          <shadowMaterial opacity={0.05} />
        </mesh>
      </group>
    </Hud>
  );
}
