// src/components/cad-viewer/OrientationCubeInScene.tsx
import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface OrientationCubeInSceneProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  onCubeClick?: (direction: THREE.Vector3) => void;
}

// Create face textures once
const createFaceTextures = () => {
  const labels = ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"];
  return labels.map((label) => {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d")!;

    const gradient = ctx.createLinearGradient(0, 0, 256, 256);
    gradient.addColorStop(0, "#e8eaed");
    gradient.addColorStop(1, "#d0d3d7");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 256);

    ctx.strokeStyle = "#9aa0a6";
    ctx.lineWidth = 3;
    ctx.strokeRect(2, 2, 252, 252);

    ctx.fillStyle = "#3c4043";
    ctx.font = "bold 32px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, 128, 128);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  });
};

let cachedTextures: THREE.Texture[] | null = null;
const getTextures = () => {
  if (!cachedTextures) {
    cachedTextures = createFaceTextures();
  }
  return cachedTextures;
};

export function OrientationCubeInScene({ mainCameraRef, onCubeClick }: OrientationCubeInSceneProps) {
  const cubeGroupRef = useRef<THREE.Group>(null);
  const textures = getTextures();

  // ✅ CRITICAL: Sync cube rotation with main camera EVERY FRAME
  useFrame(() => {
    if (!cubeGroupRef.current || !mainCameraRef.current) return;

    const camera = mainCameraRef.current;

    // Position cube in front and to the right of the camera
    // This keeps it visible as the camera moves
    const distance = 25; // Distance from camera
    const offsetX = 8; // Offset to the right
    const offsetY = -6; // Offset down

    // Calculate position relative to camera
    const cameraForward = new THREE.Vector3(0, 0, -1);
    cameraForward.applyQuaternion(camera.quaternion);
    
    const cameraRight = new THREE.Vector3(1, 0, 0);
    cameraRight.applyQuaternion(camera.quaternion);
    
    const cameraUp = new THREE.Vector3(0, 1, 0);
    cameraUp.applyQuaternion(camera.quaternion);

    const position = camera.position.clone()
      .add(cameraForward.multiplyScalar(distance))
      .add(cameraRight.multiplyScalar(offsetX))
      .add(cameraUp.multiplyScalar(offsetY));

    cubeGroupRef.current.position.copy(position);

    // Invert main camera's quaternion and apply to cube
    const invertedQuat = camera.quaternion.clone().invert();
    cubeGroupRef.current.quaternion.copy(invertedQuat);
  });

  const handleClick = (event: any) => {
    event.stopPropagation();
    if (!onCubeClick || !cubeGroupRef.current) return;

    const point = event.point.clone();
    cubeGroupRef.current.worldToLocal(point);

    const absX = Math.abs(point.x);
    const absY = Math.abs(point.y);
    const absZ = Math.abs(point.z);

    let direction: THREE.Vector3;
    if (absX > absY && absX > absZ) {
      direction = new THREE.Vector3(Math.sign(point.x), 0, 0);
    } else if (absY > absX && absY > absZ) {
      direction = new THREE.Vector3(0, Math.sign(point.y), 0);
    } else {
      direction = new THREE.Vector3(0, 0, Math.sign(point.z));
    }

    console.log("🎯 Cube face clicked:", direction);
    onCubeClick(direction);
  };

  return (
    <group ref={cubeGroupRef} scale={[0.5, 0.5, 0.5]}>
      {/* Main cube body */}
      <mesh onClick={handleClick}>
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial
          color="#f5f5f5"
          metalness={0.2}
          roughness={0.4}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Edges */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(2, 2, 2), 15]} />
        <lineBasicMaterial color="#5f6368" transparent opacity={0.7} />
      </lineSegments>

      {/* Face labels */}
      {[
        { pos: [0, 0, 1.01], rot: [0, 0, 0], tex: 0 },
        { pos: [0, 0, -1.01], rot: [0, Math.PI, 0], tex: 1 },
        { pos: [1.01, 0, 0], rot: [0, Math.PI / 2, 0], tex: 2 },
        { pos: [-1.01, 0, 0], rot: [0, -Math.PI / 2, 0], tex: 3 },
        { pos: [0, 1.01, 0], rot: [-Math.PI / 2, 0, 0], tex: 4 },
        { pos: [0, -1.01, 0], rot: [Math.PI / 2, 0, 0], tex: 5 },
      ].map((face, i) => (
        <mesh
          key={i}
          position={face.pos as [number, number, number]}
          rotation={face.rot as [number, number, number]}
        >
          <planeGeometry args={[1.8, 1.8]} />
          <meshBasicMaterial map={textures[face.tex]} transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      ))}

      {/* Local lighting for the cube */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.4} />
    </group>
  );
}