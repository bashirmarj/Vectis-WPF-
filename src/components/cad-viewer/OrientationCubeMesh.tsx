// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ ISSUE #2 FIXED: Loading actual STL file with chamfer from assets
// Professional 3D Orientation Cube

import { useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeEvent, useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
}

/**
 * Professional Orientation Cube with:
 * - Loads actual STL file with chamfered edges
 * - Hover effects (blue glow)
 * - Click detection for camera rotation
 * - ✅ ISSUE #2 FIXED: Using real STL geometry instead of RoundedBoxGeometry
 */
export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredFace, setHoveredFace] = useState<string | null>(null);

  // ✅ ISSUE #2 FIXED: Load actual STL file with chamfers
  const geometry = useLoader(STLLoader, "/src/assets/orientation-cube.stl");

  // Center and scale the loaded geometry
  useEffect(() => {
    if (geometry) {
      geometry.center();
      geometry.computeVertexNormals();
      console.log("✅ STL loaded: orientation-cube.stl with chamfered edges");
    }
  }, [geometry]);

  // Simple single material - white/semi-transparent
  const baseMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: "#ffffff",
      metalness: 0.2,
      roughness: 0.7,
      transparent: true,
      opacity: 0.6,
      envMapIntensity: 1.2,
      flatShading: false, // Smooth shading for chamfer appearance
    });
  }, []);

  // Simplified face detection using world-space normal
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

  // Handle pointer events
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

  // Handle face clicks
  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    if (!onFaceClick || !event.face) return;

    // Get face normal (direction the face is pointing)
    const normal = event.face.normal.clone();

    // Transform normal by object's world matrix
    if (meshRef.current) {
      normal.transformDirection(meshRef.current.matrixWorld);
    }

    // Round to nearest axis-aligned direction
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

    console.log("🎯 Orientation Cube - Face clicked:", {
      normal: { x: normal.x.toFixed(3), y: normal.y.toFixed(3), z: normal.z.toFixed(3) },
      direction: { x: direction.x, y: direction.y, z: direction.z },
    });

    onFaceClick(direction);
  };

  // Face positions and rotations for highlight overlays
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
      {/* Main cube with STL geometry */}
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

      {/* Only ONE highlight plane, conditionally rendered */}
      {hoveredFace && faceConfig[hoveredFace] && (
        <mesh position={faceConfig[hoveredFace].position} rotation={faceConfig[hoveredFace].rotation}>
          <planeGeometry args={[1.7, 1.7]} />
          <meshBasicMaterial color="#60a5fa" transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}

      {/* Edge lines for visual definition */}
      <lineSegments>
        <edgesGeometry args={[geometry, 25]} />
        <lineBasicMaterial color="#0f172a" linewidth={2} transparent opacity={0.7} />
      </lineSegments>

      {/* Subtle outer glow for 3D effect */}
      <mesh geometry={geometry} scale={1.02}>
        <meshBasicMaterial color="#000000" transparent opacity={0.1} side={THREE.BackSide} />
      </mesh>
    </group>
  );
}
