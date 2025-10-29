// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ UPDATED VERSION - Using STL file for realistic appearance
// ✅ Loads orientation-cube.stl for proper chamfered edges
// ✅ All interaction logic preserved

import { useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeEvent, useThree, useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

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

  // ✅ Load STL geometry
  const geometry = useLoader(STLLoader, "/orientation-cube.stl");

  useEffect(() => {
    if (geometry) {
      geometry.center();
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      console.log("✅ Orientation cube STL loaded successfully");
    }
  }, [geometry]);

  // ✅ CRITICAL: Track mouse movement at window level during drag
  useEffect(() => {
    if (!isDragging) return;

    const handleWindowMouseMove = (e: MouseEvent) => {
      if (dragStartPos.current && onDragRotate) {
        const deltaX = e.clientX - dragStartPos.current.x;
        const deltaY = e.clientY - dragStartPos.current.y;

        if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
          onDragRotate(deltaX, deltaY);
          dragStartPos.current = { x: e.clientX, y: e.clientY };
        }
      }
    };

    const handleWindowMouseUp = (e: MouseEvent) => {
      console.log("🖱️ Window mouse UP - dragging stopped");
      setIsDragging(false);
      dragStartPos.current = null;
      gl.domElement.style.cursor = "default";
    };

    window.addEventListener("mousemove", handleWindowMouseMove);
    window.addEventListener("mouseup", handleWindowMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
    };
  }, [isDragging, onDragRotate, gl]);

  // ✅ Define 6 clickable face planes with their directions
  const faceDefinitions = [
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
  ];

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();

    if (isDragging) {
      console.warn("⚠️ Pointer DOWN while already dragging - ignoring");
      return;
    }

    setIsDragging(true);
    dragStartPos.current = { x: event.clientX, y: event.clientY };
    gl.domElement.style.cursor = "grabbing";
    console.log("🖱️ Pointer DOWN - dragging started (window tracking enabled)");
  };

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    // Window-level listener handles drag tracking, this just prevents event propagation
  };

  const handlePointerUp = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();

    // Window listener handles drag end, this is just for clicking the cube body
    if (!isDragging) {
      return;
    }

    const wasClick =
      dragStartPos.current &&
      Math.abs(event.clientX - dragStartPos.current.x) < 3 &&
      Math.abs(event.clientY - dragStartPos.current.y) < 3;

    if (wasClick) {
      console.log("🖱️ Cube body clicked (no specific face)");
    }

    // Window listener will handle state reset
  };

  const handleCubeEnter = () => {
    if (!isDragging) {
      gl.domElement.style.cursor = "grab";
    }
  };

  // ✅ Individual face click handlers
  const handleFaceClick = (faceName: string, direction: THREE.Vector3) => (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();

    const wasClick =
      dragStartPos.current &&
      Math.abs(event.clientX - dragStartPos.current.x) < 3 &&
      Math.abs(event.clientY - dragStartPos.current.y) < 3;

    if (wasClick && onFaceClick) {
      console.log(`🖱️ Face clicked: ${faceName.toUpperCase()} → direction:`, direction);
      onFaceClick(direction);
    }

    // Always reset drag state after click
    setIsDragging(false);
    dragStartPos.current = null;
  };

  const handleFaceEnter = (faceName: string) => () => {
    if (!isDragging) {
      setHoveredFace(faceName);
    }
  };

  const handleFaceLeave = () => {
    if (!isDragging) {
      setHoveredFace(null);
    }
  };

  return (
    <group ref={groupRef}>
      {/* Main cube mesh from STL */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        castShadow
        receiveShadow
        scale={0.018} // Scale STL to appropriate size
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerEnter={handleCubeEnter}
      >
        <meshStandardMaterial
          color="#ffffff"
          metalness={0.3}
          roughness={0.6}
          transparent={true}
          opacity={0.8}
          envMapIntensity={1.5}
          side={THREE.FrontSide}
        />
      </mesh>

      {/* ✅ 6 invisible clickable face planes - always rendered for face selection */}
      {faceDefinitions.map((face) => (
        <mesh
          key={face.name}
          position={face.position}
          rotation={face.rotation}
          onPointerDown={handlePointerDown}
          onPointerUp={handleFaceClick(face.name, face.direction)}
          onPointerEnter={handleFaceEnter(face.name)}
          onPointerLeave={handleFaceLeave}
          visible={!isDragging} // Hide during drag to prevent interference
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
