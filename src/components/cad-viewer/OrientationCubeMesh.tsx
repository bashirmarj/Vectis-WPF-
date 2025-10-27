// src/components/cad-viewer/OrientationCubeMesh.tsx
// Professional 3D Orientation Cube with Chamfered Edges
// Industry-standard implementation matching SolidWorks, Fusion 360, Onshape

import { useRef, useMemo, useState } from "react";
import * as THREE from "three";
import { ThreeEvent } from "@react-three/fiber";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
}

/**
 * Professional Orientation Cube with:
 * - Chamfered edges (RoundedBoxGeometry)
 * - PBR materials for realistic appearance
 * - Face, edge, and corner detection
 * - Hover effects
 * - Labeled faces with embossed text
 */
export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const groupRef = useRef<THREE.Group>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Create face materials with proper PBR properties
  const faceMaterials = useMemo(() => {
    const createFaceMaterial = (label: string, color: string, faceIndex: number) => {
      // Create canvas texture for face label
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

      // Professional PBR material
      return new THREE.MeshStandardMaterial({
        map: texture,
        metalness: 0.15,
        roughness: 0.4,
        envMapIntensity: 0.8,
        // Add subtle emissive glow when hovered
        emissive: hoveredIndex === faceIndex ? new THREE.Color(color) : new THREE.Color(0x000000),
        emissiveIntensity: hoveredIndex === faceIndex ? 0.3 : 0,
      });
    };

    const adjustBrightness = (hex: string, percent: number) => {
      const num = parseInt(hex.replace("#", ""), 16);
      const amt = Math.round(2.55 * percent);
      const R = Math.max(0, Math.min(255, (num >> 16) + amt));
      const G = Math.max(0, Math.min(255, ((num >> 8) & 0x00ff) + amt));
      const B = Math.max(0, Math.min(255, (num & 0x0000ff) + amt));
      return `#${((1 << 24) + (R << 16) + (G << 8) + B).toString(16).slice(1)}`;
    };

    // Create materials for all 6 faces with distinct colors
    return [
      createFaceMaterial("RIGHT", "#e74c3c", 0), // +X: Red
      createFaceMaterial("LEFT", "#e67e22", 1), // -X: Orange  
      createFaceMaterial("TOP", "#3498db", 2), // +Y: Blue
      createFaceMaterial("BOTTOM", "#2ecc71", 3), // -Y: Green
      createFaceMaterial("FRONT", "#9b59b6", 4), // +Z: Purple
      createFaceMaterial("BACK", "#7f8c8d", 5), // -Z: Gray
    ];
  }, [hoveredIndex]);

  // Create chamfered cube geometry
  const cubeGeometry = useMemo(() => {
    // RoundedBoxGeometry(width, height, depth, segments, radius)
    // segments: number of segments per edge (higher = smoother)
    // radius: chamfer radius (edge rounding)
    return new RoundedBoxGeometry(1.8, 1.8, 1.8, 4, 0.15);
  }, []);

  // Handle pointer events
  const handlePointerEnter = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    document.body.style.cursor = "pointer";
    
    // Get the face that was hovered
    if (event.face) {
      const faceIndex = Math.floor(event.faceIndex! / 2); // Each face has 2 triangles
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
    if (groupRef.current) {
      normal.transformDirection(groupRef.current.matrixWorld);
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
      faceIndex: Math.floor(event.faceIndex! / 2),
      normal: { x: normal.x.toFixed(3), y: normal.y.toFixed(3), z: normal.z.toFixed(3) },
      direction: { x: direction.x, y: direction.y, z: direction.z },
    });

    onFaceClick(direction);
  };

  return (
    <group ref={groupRef}>
      {/* Main chamfered cube with labeled faces */}
      <mesh
        geometry={cubeGeometry}
        onClick={handleClick}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        castShadow
        receiveShadow
      >
        {faceMaterials.map((material, index) => (
          <primitive key={index} object={material} attach={`material-${index}`} />
        ))}
      </mesh>

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
