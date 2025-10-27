// src/components/cad-viewer/OrientationCubeMesh.tsx
// Professional 3D Orientation Cube with Chamfered Edges
// FULLY FIXED - Materials properly attached, cube renders and rotates correctly

import { useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeEvent } from "@react-three/fiber";
import { useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
}

/**
 * Simple Orientation Cube with:
 * - Clean single-material design
 * - Hover effects (blue glow)
 * - Click detection for camera rotation
 */
export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Simple single material - no face labels needed

  // Simple single material with hover effect
  const material = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: hoveredIndex !== null ? '#60a5fa' : '#ffffff', // Blue on hover, white for default
      metalness: 0.2,
      roughness: 0.7,
      transparent: true,
      opacity: hoveredIndex !== null ? 1.0 : 0.6, // 60% opacity when not hovered
      envMapIntensity: 1.2,
      emissive: hoveredIndex !== null ? new THREE.Color('#3b82f6') : new THREE.Color(0x000000),
      emissiveIntensity: hoveredIndex !== null ? 0.3 : 0,
      flatShading: false,
    });
  }, [hoveredIndex]);

  // Load the STL geometry with error handling
  let stlGeometry: THREE.BufferGeometry | null = null;
  try {
    stlGeometry = useLoader(STLLoader, "/orientation-cube.stl");
    console.log("✅ STL loaded successfully:", stlGeometry);
  } catch (error) {
    console.warn("⚠️ Failed to load STL, using fallback BoxGeometry:", error);
  }
  
  // Scale and center the STL geometry or use fallback
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
    
    console.log("✅ STL geometry processed, scale:", scale);
    return geometry;
  }, [stlGeometry]);

  // Handle pointer events
  const handlePointerEnter = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "pointer";

    // Get the face that was hovered
    if (event.face && event.faceIndex !== undefined) {
      const faceIndex = Math.floor(event.faceIndex / 2); // Each face has 2 triangles
      setHoveredIndex(faceIndex);
    }
  };

  const handlePointerLeave = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "default";
    setHoveredIndex(null);
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

  // Log when cube is ready
  useEffect(() => {
    console.log("✅ OrientationCubeMesh: Rendered and ready");
    console.log("   - Geometry:", cubeGeometry.type);
  }, [cubeGeometry]);

  return (
    <group ref={groupRef} rotation={[0, 0, 0]} position={[0, 0, 0]}>
      {/* Simple cube with single material */}
      <mesh
        ref={meshRef}
        geometry={cubeGeometry}
        material={material}
        onClick={handleClick}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        castShadow
        receiveShadow
        scale={0.95}
      />

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
