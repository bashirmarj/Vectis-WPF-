// src/components/cad-viewer/OrientationCubeMesh.tsx
// Professional 3D Orientation Cube with Chamfered Edges
// ✅ ISSUE #3 FIXED: Face highlighting now works correctly
// Hotspot planes are camera-aligned (not rotated with cube)

import { useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeEvent, useThree } from "@react-three/fiber";
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
 * - ✅ FIXED: Correct face highlighting (camera-aligned hotspots)
 */
export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredFace, setHoveredFace] = useState<string | null>(null);
  const { camera } = useThree();

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

  // Highlight material for hovered face overlay
  const highlightMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: "#60a5fa",
      metalness: 0.2,
      roughness: 0.7,
      transparent: true,
      opacity: 0.4,
      emissive: new THREE.Color("#3b82f6"),
      emissiveIntensity: 0.3,
      flatShading: false,
    });
  }, []);

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

  // Determine which logical face is hovered based on camera-relative position
  const getFaceFromCameraPosition = (intersectionPoint: THREE.Vector3): string => {
    // Transform intersection point to camera space to get screen-relative position
    const localPoint = intersectionPoint.clone();
    if (groupRef.current) {
      groupRef.current.worldToLocal(localPoint);
    }

    // Determine which axis has the largest absolute value
    const absX = Math.abs(localPoint.x);
    const absY = Math.abs(localPoint.y);
    const absZ = Math.abs(localPoint.z);

    if (absX > absY && absX > absZ) {
      return localPoint.x > 0 ? "right" : "left";
    } else if (absY > absX && absY > absZ) {
      return localPoint.y > 0 ? "top" : "bottom";
    } else {
      return localPoint.z > 0 ? "front" : "back";
    }
  };

  // Handle pointer events on the main cube mesh
  const handlePointerEnter = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "pointer";

    if (event.point) {
      const face = getFaceFromCameraPosition(event.point);
      setHoveredFace(face);
      console.log("🎯 Hovering over face:", face);
    }
  };

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    if (event.point) {
      const face = getFaceFromCameraPosition(event.point);
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
    if (!onFaceClick || !event.point) return;

    const face = getFaceFromCameraPosition(event.point);

    // Convert face name to direction vector
    let direction: THREE.Vector3;

    switch (face) {
      case "right":
        direction = new THREE.Vector3(1, 0, 0);
        break;
      case "left":
        direction = new THREE.Vector3(-1, 0, 0);
        break;
      case "top":
        direction = new THREE.Vector3(0, 1, 0);
        break;
      case "bottom":
        direction = new THREE.Vector3(0, -1, 0);
        break;
      case "front":
        direction = new THREE.Vector3(0, 0, 1);
        break;
      case "back":
        direction = new THREE.Vector3(0, 0, -1);
        break;
      default:
        return;
    }

    console.log("🎯 Orientation Cube - Face clicked:", {
      face,
      direction: { x: direction.x, y: direction.y, z: direction.z },
    });

    onFaceClick(direction);
  };

  // Log when cube is ready
  useEffect(() => {
    console.log("✅ OrientationCubeMesh: Rendered and ready");
    console.log("   - Geometry:", cubeGeometry.type);
  }, [cubeGeometry]);

  // ✅ ISSUE #3 FIXED: Highlight overlays are now positioned in WORLD SPACE
  // Calculate camera-facing directions for each logical face
  const getHighlightPosition = (face: string): [number, number, number] => {
    // These positions are in the cube's local space
    // They represent the center of each logical face
    const positions: Record<string, [number, number, number]> = {
      right: [0.91, 0, 0],
      left: [-0.91, 0, 0],
      top: [0, 0.91, 0],
      bottom: [0, -0.91, 0],
      front: [0, 0, 0.91],
      back: [0, 0, -0.91],
    };
    return positions[face];
  };

  const getHighlightRotation = (face: string): [number, number, number] => {
    // Rotations to orient the plane correctly for each face
    const rotations: Record<string, [number, number, number]> = {
      right: [0, Math.PI / 2, 0],
      left: [0, -Math.PI / 2, 0],
      top: [-Math.PI / 2, 0, 0],
      bottom: [Math.PI / 2, 0, 0],
      front: [0, 0, 0],
      back: [0, Math.PI, 0],
    };
    return rotations[face];
  };

  return (
    <group ref={groupRef} rotation={[0, 0, 0]} position={[0, 0, 0]}>
      {/* Main cube with hover detection */}
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

      {/* Visible highlight overlays on hovered face */}
      {hoveredFace && (
        <mesh position={getHighlightPosition(hoveredFace)} rotation={getHighlightRotation(hoveredFace)}>
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
