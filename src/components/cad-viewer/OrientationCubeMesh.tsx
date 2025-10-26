import { useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import orientationCubeSTL from "@/assets/orientation-cube.stl?url";

interface OrientationCubeMeshProps {
  onCubeClick?: (direction: THREE.Vector3) => void;
  displayMode: "solid" | "wireframe" | "translucent";
}

// Helper function to classify click region (face, edge, or corner)
const classifyClickRegion = (localPoint: THREE.Vector3) => {
  const faceDistance = 1.0;
  const edgeThreshold = 0.25;

  const absX = Math.abs(localPoint.x);
  const absY = Math.abs(localPoint.y);
  const absZ = Math.abs(localPoint.z);

  const nearMaxX = absX > faceDistance - edgeThreshold;
  const nearMaxY = absY > faceDistance - edgeThreshold;
  const nearMaxZ = absZ > faceDistance - edgeThreshold;

  const edgeCount = [nearMaxX, nearMaxY, nearMaxZ].filter(Boolean).length;

  // CORNER: All 3 dimensions near maximum
  if (edgeCount === 3) {
    return new THREE.Vector3(
      Math.sign(localPoint.x),
      Math.sign(localPoint.y),
      Math.sign(localPoint.z),
    ).normalize();
  }

  // EDGE: Exactly 2 dimensions near maximum
  else if (edgeCount === 2) {
    return new THREE.Vector3(
      nearMaxX ? Math.sign(localPoint.x) : 0,
      nearMaxY ? Math.sign(localPoint.y) : 0,
      nearMaxZ ? Math.sign(localPoint.z) : 0,
    ).normalize();
  }

  // FACE: Only 1 dimension near maximum
  else {
    if (absX > absY && absX > absZ) {
      return new THREE.Vector3(Math.sign(localPoint.x), 0, 0);
    } else if (absY > absX && absY > absZ) {
      return new THREE.Vector3(0, Math.sign(localPoint.y), 0);
    } else {
      return new THREE.Vector3(0, 0, Math.sign(localPoint.z));
    }
  }
};

export function OrientationCubeMesh({ onCubeClick, displayMode }: OrientationCubeMeshProps) {
  const { gl } = useThree();
  const cubeRef = useRef<THREE.Mesh | null>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  // Load STL geometry
  useEffect(() => {
    const loader = new STLLoader();
    loader.load(orientationCubeSTL, (loadedGeometry) => {
      loadedGeometry.computeBoundingBox();
      loadedGeometry.center();
      setGeometry(loadedGeometry);
    });
  }, []);

  // Handle click
  const handleClick = (event: any) => {
    if (!cubeRef.current || !onCubeClick) return;
    event.stopPropagation();

    const localPoint = cubeRef.current.worldToLocal(event.point.clone());
    const direction = classifyClickRegion(localPoint);
    onCubeClick(direction);
  };

  // Handle hover
  const handlePointerMove = (event: any) => {
    event.stopPropagation();
    gl.domElement.style.cursor = "pointer";
  };

  const handlePointerOut = () => {
    gl.domElement.style.cursor = "default";
  };

  if (!geometry) return null;

  return (
    <mesh
      ref={cubeRef}
      geometry={geometry}
      onClick={handleClick}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    >
      <meshStandardMaterial
        color={0x4a5568}
        transparent={displayMode === "translucent"}
        opacity={displayMode === "translucent" ? 0.5 : 1}
        wireframe={displayMode === "wireframe"}
        roughness={0.7}
        metalness={0.3}
      />
      
      {/* Edge lines */}
      <lineSegments>
        <edgesGeometry args={[geometry, 30]} />
        <lineBasicMaterial color={0x1f2937} opacity={0.6} transparent />
      </lineSegments>

      {/* Face labels */}
      {[
        { text: "Front", position: [0, 0, 1], rotation: [0, 0, 0] },
        { text: "Back", position: [0, 0, -1], rotation: [0, Math.PI, 0] },
        { text: "Right", position: [1, 0, 0], rotation: [0, Math.PI / 2, 0] },
        { text: "Left", position: [-1, 0, 0], rotation: [0, -Math.PI / 2, 0] },
        { text: "Top", position: [0, 1, 0], rotation: [-Math.PI / 2, 0, 0] },
        { text: "Bottom", position: [0, -1, 0], rotation: [Math.PI / 2, 0, Math.PI] },
      ].map((face, index) => {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d")!;
        canvas.width = 512;
        canvas.height = 512;

        context.shadowColor = "rgba(255, 255, 255, 0.8)";
        context.shadowBlur = 8;
        context.shadowOffsetX = 2;
        context.shadowOffsetY = 2;

        context.fillStyle = "#000000";
        context.font = "bold 80px Arial";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(face.text, 256, 256);

        const texture = new THREE.CanvasTexture(canvas);

        return (
          <mesh
            key={index}
            position={[face.position[0] * 1.05, face.position[1] * 1.05, face.position[2] * 1.05]}
            rotation={[face.rotation[0], face.rotation[1], face.rotation[2]]}
          >
            <planeGeometry args={[0.8, 0.8]} />
            <meshBasicMaterial map={texture} transparent depthWrite={false} side={THREE.DoubleSide} />
          </mesh>
        );
      })}
    </mesh>
  );
}
