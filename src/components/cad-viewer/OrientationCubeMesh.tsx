// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ FIXED VERSION - October 28, 2025
// ✅ Scale: 1.0 (not 0.95) for visible chamfers
// ✅ Material: Enhanced properties for better light reflection
// ✅ 6 invisible planes for reliable face detection

import { useRef, useMemo, useState } from "react";
import * as THREE from "three";
import { ThreeEvent, useThree } from "@react-three/fiber";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
  onDragRotate?: (deltaX: number, deltaY: number) => void;
  groupRef: React.RefObject<THREE.Group>;
}

export function OrientationCubeMesh({ onFaceClick, onDragRotate, groupRef }: OrientationCubeMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoveredFace, setHoveredFace] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);
  const { gl } = useThree();

  // ✅ Use RoundedBoxGeometry for chamfered edges
  const geometry = useMemo(() => {
    const geo = new RoundedBoxGeometry(1.8, 1.8, 1.8, 4, 0.15);
    geo.center();
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    console.log("✅ RoundedBoxGeometry created with chamfered edges");
    return geo;
  }, []);

  // ✅ FIXED: Enhanced material properties for better visibility
  const baseMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: "#ffffff",
      metalness: 0.3, // ✅ Increased from 0.2
      roughness: 0.6, // ✅ Decreased from 0.7
      transparent: true,
      opacity: 0.7, // ✅ Increased from 0.6
      envMapIntensity: 1.5, // ✅ Increased from 1.2
      flatShading: false,
      side: THREE.FrontSide, // ✅ NEW: Cleaner rendering
    });
  }, []);

  // ✅ Define 6 clickable face planes with their directions
  const faceDefinitions = useMemo(
    () => [
      {
        name: "right",
        direction: new THREE.Vector3(1, 0, 0),
        position: [0.91, 0, 0] as [number, number, number],
        rotation: [0, Math.PI / 2, 0] as [number, number, number],
      },
      {
        name: "left",
        direction: new THREE.Vector3(-1, 0, 0),
        position: [-0.91, 0, 0] as [number, number, number],
        rotation: [0, -Math.PI / 2, 0] as [number, number, number],
      },
      {
        name: "top",
        direction: new THREE.Vector3(0, 1, 0),
        position: [0, 0.91, 0] as [number, number, number],
        rotation: [-Math.PI / 2, 0, 0] as [number, number, number],
      },
      {
        name: "bottom",
        direction: new THREE.Vector3(0, -1, 0),
        position: [0, -0.91, 0] as [number, number, number],
        rotation: [Math.PI / 2, 0, 0] as [number, number, number],
      },
      {
        name: "front",
        direction: new THREE.Vector3(0, 0, 1),
        position: [0, 0, 0.91] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
      },
      {
        name: "back",
        direction: new THREE.Vector3(0, 0, -1),
        position: [0, 0, -0.91] as [number, number, number],
        rotation: [0, Math.PI, 0] as [number, number, number],
      },
    ],
    [],
  );

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    setIsDragging(true);
    dragStartPos.current = { x: event.clientX, y: event.clientY };
    gl.domElement.style.cursor = "grabbing";

    (event.target as any).setPointerCapture?.(event.pointerId);
  };

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation(); // ✅ CRITICAL: Stop hover events from propagating

    // Only rotate if actually dragging (mouse button held down)
    if (isDragging && dragStartPos.current && onDragRotate) {
      const deltaX = event.clientX - dragStartPos.current.x;
      const deltaY = event.clientY - dragStartPos.current.y;

      if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
        onDragRotate(deltaX, deltaY);
        dragStartPos.current = { x: event.clientX, y: event.clientY };
      }
    }
  };

  const handlePointerUp = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();

    const wasClick =
      dragStartPos.current &&
      Math.abs(event.clientX - dragStartPos.current.x) < 5 &&
      Math.abs(event.clientY - dragStartPos.current.y) < 5;

    if (wasClick) {
      // Click was on main cube, not a specific face
      console.log("🖱️ Cube body clicked (no face selected)");
    }

    setIsDragging(false);
    dragStartPos.current = null;
    gl.domElement.style.cursor = "grab";

    (event.target as any).releasePointerCapture?.(event.pointerId);
  };

  const handleCubeEnter = () => {
    if (!isDragging) {
      gl.domElement.style.cursor = "grab";
    }
  };

  const handleCubeLeave = () => {
    if (!isDragging) {
      gl.domElement.style.cursor = "default";
      setHoveredFace(null);
    }
  };

  // ✅ Individual face click handlers
  const handleFaceClick = (faceName: string, direction: THREE.Vector3) => (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();

    const wasClick =
      dragStartPos.current &&
      Math.abs(event.clientX - dragStartPos.current.x) < 5 &&
      Math.abs(event.clientY - dragStartPos.current.y) < 5;

    if (wasClick && onFaceClick) {
      console.log(`🖱️ Face clicked: ${faceName.toUpperCase()} → direction:`, direction);
      onFaceClick(direction);
    }
  };

  const handleFaceEnter = (faceName: string) => () => {
    if (!isDragging) {
      setHoveredFace(faceName);
      console.log("🎯 Hovering face:", faceName.toUpperCase());
    }
  };

  const handleFaceLeave = () => {
    if (!isDragging) {
      setHoveredFace(null);
    }
  };

  return (
    <group ref={groupRef}>
      {/* Main cube mesh */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        material={baseMaterial}
        castShadow
        receiveShadow
        scale={1.0}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerEnter={handleCubeEnter}
        onPointerLeave={handleCubeLeave}
      />

      {/* ✅ 6 invisible clickable face planes */}
      {faceDefinitions.map((face) => (
        <mesh
          key={face.name}
          position={face.position}
          rotation={face.rotation}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handleFaceClick(face.name, face.direction)}
          onPointerEnter={handleFaceEnter(face.name)}
          onPointerLeave={handleFaceLeave}
        >
          <planeGeometry args={[1.7, 1.7]} />
          <meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ))}

      {/* Highlight overlay when hovering */}
      {hoveredFace && faceDefinitions.find((f) => f.name === hoveredFace) && !isDragging && (
        <mesh
          position={faceDefinitions.find((f) => f.name === hoveredFace)!.position}
          rotation={faceDefinitions.find((f) => f.name === hoveredFace)!.rotation}
        >
          <planeGeometry args={[1.7, 1.7]} />
          <meshBasicMaterial color="#60a5fa" transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}

      {/* Edge lines */}
      <lineSegments>
        <edgesGeometry args={[geometry, 25]} />
        <lineBasicMaterial color="#0f172a" linewidth={2} transparent opacity={0.7} />
      </lineSegments>
    </group>
  );
}
