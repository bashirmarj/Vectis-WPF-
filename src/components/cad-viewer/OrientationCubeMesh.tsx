// src/components/cad-viewer/OrientationCubeMesh.tsx
// The actual 3D cube geometry with face labels and click handling

import { useRef, useMemo } from "react";
import * as THREE from "three";
import { ThreeEvent } from "@react-three/fiber";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
}

/**
 * OrientationCubeMesh - The 3D cube with labeled faces
 *
 * Face Layout (standard CAD convention):
 * - +Z: FRONT (blue face)
 * - -Z: BACK
 * - +X: RIGHT
 * - -X: LEFT
 * - +Y: TOP
 * - -Y: BOTTOM
 */
export function OrientationCubeMesh({ onFaceClick }: OrientationCubeMeshProps) {
  const cubeRef = useRef<THREE.Group>(null);

  // Create face textures with labels
  const faceTextures = useMemo(() => {
    const createFaceTexture = (label: string, color: string) => {
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext("2d")!;

      // Background gradient
      const gradient = ctx.createLinearGradient(0, 0, 256, 256);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, adjustBrightness(color, -20));
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 256, 256);

      // Border
      ctx.strokeStyle = "#555";
      ctx.lineWidth = 4;
      ctx.strokeRect(4, 4, 248, 248);

      // Label
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 40px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 4;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
      ctx.fillText(label, 128, 128);

      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      return texture;
    };

    const adjustBrightness = (hex: string, percent: number) => {
      const num = parseInt(hex.replace("#", ""), 16);
      const amt = Math.round(2.55 * percent);
      const R = (num >> 16) + amt;
      const G = ((num >> 8) & 0x00ff) + amt;
      const B = (num & 0x0000ff) + amt;
      return (
        "#" +
        (
          0x1000000 +
          (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x10000 +
          (G < 255 ? (G < 1 ? 0 : G) : 255) * 0x100 +
          (B < 255 ? (B < 1 ? 0 : B) : 255)
        )
          .toString(16)
          .slice(1)
      );
    };

    // Create textures for each face with distinct colors
    return [
      createFaceTexture("RIGHT", "#e74c3c"), // +X: Red
      createFaceTexture("LEFT", "#e67e22"), // -X: Orange
      createFaceTexture("TOP", "#3498db"), // +Y: Blue
      createFaceTexture("BOTTOM", "#2ecc71"), // -Y: Green
      createFaceTexture("FRONT", "#9b59b6"), // +Z: Purple
      createFaceTexture("BACK", "#95a5a6"), // -Z: Gray
    ];
  }, []);

  // Create materials array for the cube
  const materials = useMemo(() => {
    return faceTextures.map(
      (texture) =>
        new THREE.MeshStandardMaterial({
          map: texture,
          metalness: 0.1,
          roughness: 0.6,
        }),
    );
  }, [faceTextures]);

  // Handle face clicks
  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    if (!onFaceClick) return;

    // Get the face that was clicked
    const face = event.face;
    if (!face) return;

    // Get face normal (direction the face is pointing)
    const normal = face.normal.clone();

    // Round to nearest axis-aligned direction
    const absX = Math.abs(normal.x);
    const absY = Math.abs(normal.y);
    const absZ = Math.abs(normal.z);

    let direction: THREE.Vector3;

    if (absX > absY && absX > absZ) {
      // Click was on X face
      direction = new THREE.Vector3(Math.sign(normal.x), 0, 0);
    } else if (absY > absX && absY > absZ) {
      // Click was on Y face
      direction = new THREE.Vector3(0, Math.sign(normal.y), 0);
    } else {
      // Click was on Z face
      direction = new THREE.Vector3(0, 0, Math.sign(normal.z));
    }

    console.log("🎯 Cube face clicked:", {
      face: event.faceIndex,
      normal: { x: normal.x.toFixed(2), y: normal.y.toFixed(2), z: normal.z.toFixed(2) },
      direction: { x: direction.x, y: direction.y, z: direction.z },
    });

    onFaceClick(direction);
  };

  return (
    <group ref={cubeRef}>
      {/* Main cube with labeled faces */}
      <mesh onClick={handleClick}>
        <boxGeometry args={[1.5, 1.5, 1.5]} />
        {materials.map((material, index) => (
          <primitive key={index} object={material} attach={`material-${index}`} />
        ))}
      </mesh>

      {/* Edge lines for clarity */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(1.5, 1.5, 1.5), 15]} />
        <lineBasicMaterial color="#000000" linewidth={2} />
      </lineSegments>
    </group>
  );
}
