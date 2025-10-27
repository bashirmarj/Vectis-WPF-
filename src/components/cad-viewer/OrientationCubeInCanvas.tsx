// OrientationCubeInCanvas.tsx
// Orientation cube rendered INSIDE the main Canvas to avoid WebGL context issues

import { useRef, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";

interface OrientationCubeInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  onCubeClick?: (direction: THREE.Vector3) => void;
}

/**
 * Renders the orientation cube in the corner of the main Canvas viewport.
 * Uses a separate OrthographicCamera positioned in screen space.
 * Syncs rotation with main camera.
 * 
 * Benefits:
 * - No separate Canvas = no WebGL context issues
 * - Renders in same WebGL context as main scene
 * - Efficient and stable
 */
export function OrientationCubeInCanvas({ mainCameraRef, onCubeClick }: OrientationCubeInCanvasProps) {
  const cubeCameraRef = useRef<THREE.OrthographicCamera>(null);
  const cubeGroupRef = useRef<THREE.Group>(null);
  const { size } = useThree();

  // Position the cube camera in the bottom-right corner
  const cubeViewportWidth = 160;
  const cubeViewportHeight = 160;

  // Real-time rotation sync with main camera
  useFrame(() => {
    if (!mainCameraRef.current || !cubeGroupRef.current) return;

    // Copy rotation from main camera to cube group
    cubeGroupRef.current.quaternion.copy(mainCameraRef.current.quaternion);
  });

  // Set up viewport rendering for the cube
  useEffect(() => {
    if (!cubeCameraRef.current) return;

    // Position orthographic camera for the cube viewport
    cubeCameraRef.current.position.set(0, 0, 5);
    cubeCameraRef.current.lookAt(0, 0, 0);
    cubeCameraRef.current.updateProjectionMatrix();
  }, []);

  return (
    <>
      {/* Orthographic camera for the cube (positioned in corner) */}
      <OrthographicCamera
        ref={cubeCameraRef}
        makeDefault={false}
        position={[0, 0, 5]}
        zoom={50}
        near={0.1}
        far={100}
      />

      {/* Cube group that syncs with main camera rotation */}
      <group ref={cubeGroupRef} position={[
        (size.width / 2) - (cubeViewportWidth / 2) - 20,
        -(size.height / 2) + (cubeViewportHeight / 2) + 20,
        0
      ]}>
        {/* Lighting for the cube */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-3, -3, -3]} intensity={0.3} />

        {/* The actual cube mesh */}
        <OrientationCubeMesh onFaceClick={onCubeClick} useSTL={false} />
      </group>
    </>
  );
}
