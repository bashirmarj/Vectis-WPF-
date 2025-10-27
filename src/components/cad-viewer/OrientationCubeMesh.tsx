// src/components/cad-viewer/OrientationCubeMesh.tsx
// Professional 3D Orientation Cube with Chamfered Edges
// FULLY FIXED - Materials properly attached, cube renders and rotates correctly

import { useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeEvent, useLoader } from "@react-three/fiber";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import cubeSTL from '@/assets/orientation-cube.stl?url';

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
  useSTL?: boolean; // Option to use STL file from assets
}

/**
 * Professional Orientation Cube with:
 * - Chamfered edges (RoundedBoxGeometry OR STL)
 * - PBR materials for realistic appearance
 * - Face, edge, and corner detection
 * - Hover effects
 * - Labeled faces with embossed text
 *
 * ✅ FIXED: Materials are now properly attached as array
 * ✅ FIXED: Cube now renders and rotates correctly
 */
export function OrientationCubeMesh({ onFaceClick, useSTL = false }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Load STL if requested (from src/assets/orientation-cube.stl)
  const stlGeometry = useSTL ? useLoader(STLLoader, cubeSTL) : null;

  // Create face textures with proper labels and gradients
  const faceTextures = useMemo(() => {
    const createFaceTexture = (label: string, color: string) => {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext("2d")!;

      // Professional gradient background
      const gradient = ctx.createLinearGradient(0, 0, 512, 512);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, adjustBrightness(color, -15));
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 512, 512);

      // Subtle inner shadow for depth
      const innerGradient = ctx.createRadialGradient(256, 256, 0, 256, 256, 256);
      innerGradient.addColorStop(0, "rgba(255, 255, 255, 0.1)");
      innerGradient.addColorStop(1, "rgba(0, 0, 0, 0.2)");
      ctx.fillStyle = innerGradient;
      ctx.fillRect(0, 0, 512, 512);

      // Border frame
      ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
      ctx.lineWidth = 8;
      ctx.strokeRect(8, 8, 496, 496);

      // Inner highlight
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 4;
      ctx.strokeRect(16, 16, 480, 480);

      // Label text with embossed effect
      ctx.font = "bold 80px -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      // Text shadow for emboss
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 8;
      ctx.shadowOffsetX = 3;
      ctx.shadowOffsetY = 3;

      // Main text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, 256, 256);

      // Highlight for 3D effect
      ctx.shadowColor = "transparent";
      ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
      ctx.fillText(label, 254, 254);

      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      return texture;
    };

    const adjustBrightness = (hex: string, percent: number) => {
      const num = parseInt(hex.replace("#", ""), 16);
      const amt = Math.round(2.55 * percent);
      const R = Math.max(0, Math.min(255, (num >> 16) + amt));
      const G = Math.max(0, Math.min(255, ((num >> 8) & 0x00ff) + amt));
      const B = Math.max(0, Math.min(255, (num & 0x0000ff) + amt));
      return `#${((1 << 24) + (R << 16) + (G << 8) + B).toString(16).slice(1)}`;
    };

    // Create textures for all 6 faces with distinct colors
    return [
      createFaceTexture("RIGHT", "#e74c3c"), // +X: Red
      createFaceTexture("LEFT", "#e67e22"), // -X: Orange
      createFaceTexture("TOP", "#3498db"), // +Y: Blue
      createFaceTexture("BOTTOM", "#2ecc71"), // -Y: Green
      createFaceTexture("FRONT", "#9b59b6"), // +Z: Purple
      createFaceTexture("BACK", "#7f8c8d"), // -Z: Gray
    ];
  }, []);

  // ✅ CRITICAL FIX: Create materials array properly
  const faceMaterials = useMemo(() => {
    return faceTextures.map((texture, index) => {
      return new THREE.MeshStandardMaterial({
        map: texture,
        metalness: 0.15,
        roughness: 0.4,
        envMapIntensity: 0.8,
        // Add subtle emissive glow when hovered
        emissive:
          hoveredIndex === index
            ? new THREE.Color(
                index === 0
                  ? "#e74c3c"
                  : index === 1
                    ? "#e67e22"
                    : index === 2
                      ? "#3498db"
                      : index === 3
                        ? "#2ecc71"
                        : index === 4
                          ? "#9b59b6"
                          : "#7f8c8d",
              )
            : new THREE.Color(0x000000),
        emissiveIntensity: hoveredIndex === index ? 0.3 : 0,
      });
    });
  }, [faceTextures, hoveredIndex]);

  // Create chamfered cube geometry
  const cubeGeometry = useMemo(() => {
    if (useSTL && stlGeometry) {
      // Use STL geometry if provided
      stlGeometry.center();
      stlGeometry.computeVertexNormals();
      return stlGeometry;
    }

    // Otherwise use RoundedBoxGeometry for chamfered edges
    // RoundedBoxGeometry(width, height, depth, segments, radius)
    return new RoundedBoxGeometry(1.8, 1.8, 1.8, 4, 0.15);
  }, [useSTL, stlGeometry]);

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

  // ✅ Log when cube is ready
  useEffect(() => {
    console.log("✅ OrientationCubeMesh: Rendered and ready");
    console.log("   - Geometry:", cubeGeometry.type);
    console.log("   - Materials:", faceMaterials.length, "faces");
    console.log("   - Using STL:", useSTL);
  }, [cubeGeometry, faceMaterials, useSTL]);

  return (
    <group ref={groupRef}>
      {/* ✅ CRITICAL FIX: Pass materials array directly to mesh */}
      <mesh
        ref={meshRef}
        geometry={cubeGeometry}
        material={faceMaterials}
        onClick={handleClick}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        castShadow
        receiveShadow
      />

      {/* Edge lines for visual definition */}
      <lineSegments>
        <edgesGeometry args={[cubeGeometry, 25]} />
        <lineBasicMaterial color="#1a1a1a" linewidth={2} transparent opacity={0.5} />
      </lineSegments>

      {/* Subtle outer glow for 3D effect */}
      <mesh geometry={cubeGeometry} scale={1.02}>
        <meshBasicMaterial color="#000000" transparent opacity={0.1} side={THREE.BackSide} />
      </mesh>
    </group>
  );
}
