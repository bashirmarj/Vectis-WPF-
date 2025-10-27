// src/components/cad-viewer/OrientationCubeMesh.tsx
// Professional 3D Orientation Cube with Chamfered Edges
// ✅ CORRECTED: Fixed highlight to show only ONE face at a time

import { useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeEvent } from "@react-three/fiber";
import { useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
}

/**
 * Professional Orientation Cube with:
 * - Clean single-material design
 * - Hover effects (blue glow)
 * - Click detection for camera rotation
 * - ✅ CORRECTED: Only ONE face highlights at a time
 */
export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredFace, setHoveredFace] = useState<string | null>(null);

  // Simple single material - always white/semi-transparent
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

  // Load the STL geometry with error handling
  let stlGeometry: THREE.BufferGeometry | null = null;
  try {
    stlGeometry = useLoader(STLLoader, "/orientation-cube.stl");
  } catch (error) {
    console.warn("⚠️ Failed to load STL, using fallback BoxGeometry:", error);
  }

  // Scale and center the STL geometry or use fallback
  // ✅ CORRECTED: Proper memoization to prevent reloading
  const cubeGeometry = useMemo(() => {
    if (!stlGeometry) {
      console.log("📦 Using fallback BoxGeometry (1.8x1.8x1.8)");
      return new THREE.BoxGeometry(1.8, 1.8, 1.8);
    }

    const geometry = stlGeometry.clone();
    geometry.computeBoundingBox();
    const boundingBox = geometry.boundingBox!;
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);

    // Center the geometry
    geometry.translate(-center.x, -center.y, -center.z);

    // Scale to fit in viewport - larger for better visibility
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 1.8 / maxDim;
    geometry.scale(scale, scale, scale);

    return geometry;
  }, [stlGeometry]);

  // ✅ CORRECTED: Simplified face detection using world-space normal
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
        console.log("🎯 Moved to face:", face);
      }
    }
  };

  const handlePointerLeave = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "default";
    setHoveredFace(null);
    console.log("👋 Left cube");
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
      // Click was on X face (RIGHT/LEFT)
      direction = new THREE.Vector3(Math.sign(normal.x), 0, 0);
    } else if (absY > absX && absY > absZ) {
      // Click was on Y face (TOP/BOTTOM)
      direction = new THREE.Vector3(0, Math.sign(normal.y), 0);
    } else {
      // Click was on Z face (FRONT/BACK)
      direction = new THREE.Vector3(0, 0, Math.sign(normal.z));
    }

    console.log("🎯 Orientation Cube - Face clicked:", {
      faceIndex: event.faceIndex !== undefined ? Math.floor(event.faceIndex / 2) : "unknown",
      normal: { x: normal.x.toFixed(3), y: normal.y.toFixed(3), z: normal.z.toFixed(3) },
      direction: { x: direction.x, y: direction.y, z: direction.z },
    });

    onFaceClick(direction);
  };

  // Log when cube is ready (only once)
  useEffect(() => {
    console.log("✅ OrientationCubeMesh: Rendered and ready");
  }, []); // Empty dependency array = run once

  // ✅ CORRECTED: Face positions and rotations for highlight overlays
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
      {/* Main cube with single material */}
      <mesh
        ref={meshRef}
        geometry={cubeGeometry}
        material={baseMaterial}
        castShadow
        receiveShadow
        scale={0.95}
        onPointerEnter={handlePointerEnter}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onClick={handleClick}
      />

      {/* ✅ CORRECTED: Only ONE highlight plane, conditionally rendered */}
      {hoveredFace && faceConfig[hoveredFace] && (
        <mesh position={faceConfig[hoveredFace].position} rotation={faceConfig[hoveredFace].rotation}>
          <planeGeometry args={[1.7, 1.7]} />
          <meshBasicMaterial color="#60a5fa" transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}

      {/* Edge lines for visual definition */}
      <lineSegments>
        <edgesGeometry args={[cubeGeometry, 25]} />
        <lineBasicMaterial color="#0f172a" linewidth={2} transparent opacity={0.7} />
      </lineSegments>

      {/* Subtle outer glow for 3D effect */}
      <mesh geometry={cubeGeometry} scale={1.02}>
        <meshBasicMaterial color="#000000" transparent opacity={0.1} side={THREE.BackSide} />
      </mesh>
    </group>
  );
}
