// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ FIXED: Loads STL with proper centering and bounding for raycasting

import { useRef, useMemo, useState } from "react";
import * as THREE from "three";
import { ThreeEvent, useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
}

export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredFace, setHoveredFace] = useState<string | null>(null);

  // ✅ Load STL and center it immediately
  const rawGeometry = useLoader(STLLoader, "/src/assets/orientation-cube.stl");

  const geometry = useMemo(() => {
    if (rawGeometry) {
      rawGeometry.center();
      rawGeometry.computeVertexNormals();
      rawGeometry.computeBoundingBox();
      rawGeometry.computeBoundingSphere();
      console.log("✅ STL centered and ready for raycasting");
    }
    return rawGeometry;
  }, [rawGeometry]);

  const baseMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: "#ffffff",
      metalness: 0.2,
      roughness: 0.7,
      transparent: true,
      opacity: 0.6,
      envMapIntensity: 1.2,
      flatShading: false,
    });
  }, []);

  const getFaceFromNormal = (normal: THREE.Vector3): string => {
    const absX = Math.abs(normal.x);
    const absY = Math.abs(normal.y);
    const absZ = Math.abs(normal.z);

    if (absX > absY && absX > absZ) {
      return normal.x > 0 ? "right" : "left";
    } else if (absY > absX && absY > absZ) {
      return normal.y > 0 ? "top" : "bottom";
    } else {
      return normal.z > 0 ? "front" : "back";
    }
  };

  const handlePointerEnter = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "pointer";

    if (event.face && meshRef.current) {
      const normal = event.face.normal.clone();
      normal.transformDirection(meshRef.current.matrixWorld);
      const face = getFaceFromNormal(normal);
      setHoveredFace(face);
    }
  };

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    if (event.face && meshRef.current) {
      const normal = event.face.normal.clone();
      normal.transformDirection(meshRef.current.matrixWorld);
      const face = getFaceFromNormal(normal);
      if (face !== hoveredFace) {
        setHoveredFace(face);
      }
    }
  };

  const handlePointerLeave = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "default";
    setHoveredFace(null);
  };

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    if (!onFaceClick || !event.face) return;

    const normal = event.face.normal.clone();

    if (meshRef.current) {
      normal.transformDirection(meshRef.current.matrixWorld);
    }

    const absX = Math.abs(normal.x);
    const absY = Math.abs(normal.y);
    const absZ = Math.abs(normal.z);

    let direction: THREE.Vector3;

    if (absX > absY && absX > absZ) {
      direction = new THREE.Vector3(Math.sign(normal.x), 0, 0);
    } else if (absY > absX && absY > absZ) {
      direction = new THREE.Vector3(0, Math.sign(normal.y), 0);
    } else {
      direction = new THREE.Vector3(0, 0, Math.sign(normal.z));
    }

    onFaceClick(direction);
  };

  const faceConfig: Record<string, { position: [number, number, number]; rotation: [number, number, number] }> =
    useMemo(
      () => ({
        right: { position: [0.91, 0, 0], rotation: [0, Math.PI / 2, 0] },
        left: { position: [-0.91, 0, 0], rotation: [0, -Math.PI / 2, 0] },
        top: { position: [0, 0.91, 0], rotation: [-Math.PI / 2, 0, 0] },
        bottom: { position: [0, -0.91, 0], rotation: [Math.PI / 2, 0, 0] },
        front: { position: [0, 0, 0.91], rotation: [0, 0, 0] },
        back: { position: [0, 0, -0.91], rotation: [0, Math.PI, 0] },
      }),
      [],
    );

  return (
    <group ref={groupRef} rotation={[0, 0, 0]} position={[0, 0, 0]}>
      <mesh
        ref={meshRef}
        geometry={geometry}
        material={baseMaterial}
        castShadow
        receiveShadow
        scale={0.95}
        onPointerEnter={handlePointerEnter}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onClick={handleClick}
      />

      {hoveredFace && faceConfig[hoveredFace] && (
        <mesh position={faceConfig[hoveredFace].position} rotation={faceConfig[hoveredFace].rotation}>
          <planeGeometry args={[1.7, 1.7]} />
          <meshBasicMaterial color="#60a5fa" transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}

      <lineSegments>
        <edgesGeometry args={[geometry, 25]} />
        <lineBasicMaterial color="#0f172a" linewidth={2} transparent opacity={0.7} />
      </lineSegments>
    </group>
  );
}
