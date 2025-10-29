// OrientationCubeInCanvas.tsx
// Orientation cube rendered INSIDE the main Canvas using Hud for screen-space positioning

import { useRef, Suspense, useCallback } from "react";
import { useFrame } from "@react-three/fiber";
import { Hud, OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";

interface OrientationCubeInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  controlsRef?: React.RefObject<any>;
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
export function OrientationCubeInCanvas({ mainCameraRef, controlsRef, onCubeClick }: OrientationCubeInCanvasProps) {
  const cubeGroupRef = useRef<THREE.Group>(null);

  // Real-time rotation sync with main camera
  useFrame(() => {
    if (!mainCameraRef.current || !cubeGroupRef.current) return;
    cubeGroupRef.current.quaternion.copy(mainCameraRef.current.quaternion);
  });

  // ✅ Drag-to-rotate handler with upside-down detection
  const handleDragRotate = useCallback(
    (deltaX: number, deltaY: number) => {
      if (!mainCameraRef.current || !controlsRef?.current) return;

      const camera = mainCameraRef.current;
      const controls = controlsRef.current;
      const target = controls.target.clone();

      // ✅ Detect if camera is upside down
      const isUpsideDown = camera.up.y < 0;

      const rotationSpeed = 0.005;

      // ✅ Invert BOTH horizontal and vertical when upside down
      const horizontalMultiplier = isUpsideDown ? 1 : -1;
      const verticalMultiplier = isUpsideDown ? 1 : -1;

      const deltaAzimuth = deltaX * rotationSpeed * horizontalMultiplier;
      const deltaPolar = deltaY * rotationSpeed * verticalMultiplier;

      const currentPosition = camera.position.clone();
      const currentUp = camera.up.clone();

      // Step 1: Horizontal rotation (around world Y-axis)
      const worldYAxis = new THREE.Vector3(0, 1, 0);
      let newPosition = currentPosition.clone().sub(target);
      newPosition.applyAxisAngle(worldYAxis, deltaAzimuth);
      newPosition.add(target);

      let newUp = currentUp.clone().applyAxisAngle(worldYAxis, deltaAzimuth);

      // Step 2: Vertical rotation (around camera's right vector)
      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);

      const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      newPosition = camera.position.clone().sub(target);
      newPosition.applyAxisAngle(cameraRight, deltaPolar);
      newPosition.add(target);

      newUp = camera.up.clone().applyAxisAngle(cameraRight, deltaPolar);

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();
    },
    [mainCameraRef, controlsRef],
  );

  return (
    <Hud renderPriority={1}>
      {/* Orthographic camera for HUD rendering */}
      <OrthographicCamera position={[0, 0, 10]} zoom={127} near={0.1} far={1000} />

      {/* Position cube in screen space (top-right corner) */}
      {/* Screen coordinates: (0,0) is center, positive X is right, positive Y is up */}
      <group position={[6.2, 4.2, 0]}>
        {/* Lighting for the cube */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-3, -3, -3]} intensity={0.3} />

        {/* Cube group that syncs with main camera rotation */}
        <group ref={cubeGroupRef} scale={1.2}>
          <Suspense
            fallback={
              <mesh>
                <boxGeometry args={[1.2, 1.2, 1.2]} />
                <meshStandardMaterial color="#64748b" opacity={0.8} transparent />
              </mesh>
            }
          >
            <OrientationCubeMesh groupRef={cubeGroupRef} onFaceClick={onCubeClick} onDragRotate={handleDragRotate} />
          </Suspense>
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
