// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ COMPLETE VERSION - Face/Edge/Corner Detection
// ✅ STL geometry properly centered for rotation
// ✅ 6 faces + 12 edges + 8 corners = 26 clickable zones

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
  const [hoveredZone, setHoveredZone] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);
  const [centeredGeometry, setCenteredGeometry] = useState<THREE.BufferGeometry | null>(null);
  const { gl } = useThree();

  // ✅ Load STL geometry
  const loadedGeometry = useLoader(STLLoader, "/orientation-cube.stl");

  // ✅ Center the geometry at origin for correct rotation
  useEffect(() => {
    if (loadedGeometry) {
      // Clone the geometry so we don't modify the cached loader data
      const cloned = loadedGeometry.clone();

      // Compute bounding box
      cloned.computeBoundingBox();
      const bbox = cloned.boundingBox!;
      const center = new THREE.Vector3();
      bbox.getCenter(center);

      // Translate all vertices to center
      const positions = cloned.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        positions.setX(i, positions.getX(i) - center.x);
        positions.setY(i, positions.getY(i) - center.y);
        positions.setZ(i, positions.getZ(i) - center.z);
      }

      positions.needsUpdate = true;
      cloned.computeVertexNormals();
      cloned.computeBoundingBox();
      cloned.computeBoundingSphere();

      setCenteredGeometry(cloned);
    }
  }, [loadedGeometry]);

  // ✅ Window-level drag tracking
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

    const handleWindowMouseUp = () => {
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

  // ✅ Define 6 FACE click zones
  const faceDefinitions = [
    {
      name: "face-right",
      label: "RIGHT",
      direction: new THREE.Vector3(1, 0, 0),
      position: [0.91, 0, 0] as [number, number, number],
      rotation: [0, Math.PI / 2, 0] as [number, number, number],
    },
    {
      name: "face-left",
      label: "LEFT",
      direction: new THREE.Vector3(-1, 0, 0),
      position: [-0.91, 0, 0] as [number, number, number],
      rotation: [0, -Math.PI / 2, 0] as [number, number, number],
    },
    {
      name: "face-top",
      label: "TOP",
      direction: new THREE.Vector3(0, 1, 0),
      position: [0, 0.91, 0] as [number, number, number],
      rotation: [-Math.PI / 2, 0, 0] as [number, number, number],
    },
    {
      name: "face-bottom",
      label: "BOTTOM",
      direction: new THREE.Vector3(0, -1, 0),
      position: [0, -0.91, 0] as [number, number, number],
      rotation: [Math.PI / 2, 0, 0] as [number, number, number],
    },
    {
      name: "face-front",
      label: "FRONT",
      direction: new THREE.Vector3(0, 0, 1),
      position: [0, 0, 0.91] as [number, number, number],
      rotation: [0, 0, 0] as [number, number, number],
    },
    {
      name: "face-back",
      label: "BACK",
      direction: new THREE.Vector3(0, 0, -1),
      position: [0, 0, -0.91] as [number, number, number],
      rotation: [0, Math.PI, 0] as [number, number, number],
    },
  ];

  // ✅ Define 12 EDGE click zones (for aligned views)
  const edgeDefinitions = [
    // Top edges
    {
      name: "edge-top-front",
      direction: new THREE.Vector3(0, 1, 1).normalize(),
      position: [0, 0.65, 0.65] as [number, number, number],
    },
    {
      name: "edge-top-back",
      direction: new THREE.Vector3(0, 1, -1).normalize(),
      position: [0, 0.65, -0.65] as [number, number, number],
    },
    {
      name: "edge-top-left",
      direction: new THREE.Vector3(-1, 1, 0).normalize(),
      position: [-0.65, 0.65, 0] as [number, number, number],
    },
    {
      name: "edge-top-right",
      direction: new THREE.Vector3(1, 1, 0).normalize(),
      position: [0.65, 0.65, 0] as [number, number, number],
    },

    // Middle edges (vertical)
    {
      name: "edge-front-left",
      direction: new THREE.Vector3(-1, 0, 1).normalize(),
      position: [-0.65, 0, 0.65] as [number, number, number],
    },
    {
      name: "edge-front-right",
      direction: new THREE.Vector3(1, 0, 1).normalize(),
      position: [0.65, 0, 0.65] as [number, number, number],
    },
    {
      name: "edge-back-left",
      direction: new THREE.Vector3(-1, 0, -1).normalize(),
      position: [-0.65, 0, -0.65] as [number, number, number],
    },
    {
      name: "edge-back-right",
      direction: new THREE.Vector3(1, 0, -1).normalize(),
      position: [0.65, 0, -0.65] as [number, number, number],
    },

    // Bottom edges
    {
      name: "edge-bottom-front",
      direction: new THREE.Vector3(0, -1, 1).normalize(),
      position: [0, -0.65, 0.65] as [number, number, number],
    },
    {
      name: "edge-bottom-back",
      direction: new THREE.Vector3(0, -1, -1).normalize(),
      position: [0, -0.65, -0.65] as [number, number, number],
    },
    {
      name: "edge-bottom-left",
      direction: new THREE.Vector3(-1, -1, 0).normalize(),
      position: [-0.65, -0.65, 0] as [number, number, number],
    },
    {
      name: "edge-bottom-right",
      direction: new THREE.Vector3(1, -1, 0).normalize(),
      position: [0.65, -0.65, 0] as [number, number, number],
    },
  ];

  // ✅ Define 8 CORNER click zones (for isometric views)
  const cornerDefinitions = [
    {
      name: "corner-top-front-right",
      direction: new THREE.Vector3(1, 1, 1).normalize(),
      position: [0.65, 0.65, 0.65] as [number, number, number],
    },
    {
      name: "corner-top-front-left",
      direction: new THREE.Vector3(-1, 1, 1).normalize(),
      position: [-0.65, 0.65, 0.65] as [number, number, number],
    },
    {
      name: "corner-top-back-right",
      direction: new THREE.Vector3(1, 1, -1).normalize(),
      position: [0.65, 0.65, -0.65] as [number, number, number],
    },
    {
      name: "corner-top-back-left",
      direction: new THREE.Vector3(-1, 1, -1).normalize(),
      position: [-0.65, 0.65, -0.65] as [number, number, number],
    },
    {
      name: "corner-bottom-front-right",
      direction: new THREE.Vector3(1, -1, 1).normalize(),
      position: [0.65, -0.65, 0.65] as [number, number, number],
    },
    {
      name: "corner-bottom-front-left",
      direction: new THREE.Vector3(-1, -1, 1).normalize(),
      position: [-0.65, -0.65, 0.65] as [number, number, number],
    },
    {
      name: "corner-bottom-back-right",
      direction: new THREE.Vector3(1, -1, -1).normalize(),
      position: [0.65, -0.65, -0.65] as [number, number, number],
    },
    {
      name: "corner-bottom-back-left",
      direction: new THREE.Vector3(-1, -1, -1).normalize(),
      position: [-0.65, -0.65, -0.65] as [number, number, number],
    },
  ];

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (isDragging) return;

    setIsDragging(true);
    dragStartPos.current = { x: event.clientX, y: event.clientY };
    gl.domElement.style.cursor = "grabbing";
  };

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
  };

  const handlePointerUp = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
  };

  const handleCubeEnter = () => {
    if (!isDragging) {
      gl.domElement.style.cursor = "grab";
    }
  };

  // ✅ Generic zone click handler
  const handleZoneClick =
    (zoneName: string, zoneLabel: string, direction: THREE.Vector3) => (event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation();

      const wasClick =
        dragStartPos.current &&
        Math.abs(event.clientX - dragStartPos.current.x) < 3 &&
        Math.abs(event.clientY - dragStartPos.current.y) < 3;

      if (wasClick && onFaceClick) {
        console.log(`🖱️ ${zoneLabel} clicked →`, direction);
        onFaceClick(direction);
      }

      setIsDragging(false);
      dragStartPos.current = null;
    };

  const handleZoneEnter = (zoneName: string) => () => {
    if (!isDragging) {
      setHoveredZone(zoneName);
    }
  };

  const handleZoneLeave = () => {
    if (!isDragging) {
      setHoveredZone(null);
    }
  };

  // ✅ Detect zone from 3D point on interaction cube
  const detectZoneFromPoint = (point: THREE.Vector3) => {
    const abs = { x: Math.abs(point.x), y: Math.abs(point.y), z: Math.abs(point.z) };
    const max = Math.max(abs.x, abs.y, abs.z);
    const tolerance = 0.15;

    // Count how many axes are at maximum
    const atMax = [
      abs.x > max - tolerance ? "x" : null,
      abs.y > max - tolerance ? "y" : null,
      abs.z > max - tolerance ? "z" : null,
    ].filter(Boolean);

    if (atMax.length === 1) {
      // FACE: Only one axis at max
      if (point.x > 0.4) return { name: "face-right", direction: new THREE.Vector3(1, 0, 0) };
      if (point.x < -0.4) return { name: "face-left", direction: new THREE.Vector3(-1, 0, 0) };
      if (point.y > 0.4) return { name: "face-top", direction: new THREE.Vector3(0, 1, 0) };
      if (point.y < -0.4) return { name: "face-bottom", direction: new THREE.Vector3(0, -1, 0) };
      if (point.z > 0.4) return { name: "face-front", direction: new THREE.Vector3(0, 0, 1) };
      if (point.z < -0.4) return { name: "face-back", direction: new THREE.Vector3(0, 0, -1) };
    } else if (atMax.length === 2) {
      // EDGE: Two axes at max
      const dir = new THREE.Vector3(
        abs.x > max - tolerance ? Math.sign(point.x) : 0,
        abs.y > max - tolerance ? Math.sign(point.y) : 0,
        abs.z > max - tolerance ? Math.sign(point.z) : 0,
      );
      return {
        name: `edge-${Math.sign(point.x)}-${Math.sign(point.y)}-${Math.sign(point.z)}`,
        direction: dir.normalize(),
      };
    } else if (atMax.length === 3) {
      // CORNER: All three axes at max
      const dir = new THREE.Vector3(Math.sign(point.x), Math.sign(point.y), Math.sign(point.z));
      return {
        name: `corner-${Math.sign(point.x)}-${Math.sign(point.y)}-${Math.sign(point.z)}`,
        direction: dir.normalize(),
      };
    }

    return null;
  };

  // ✅ Click handler for interaction cube
  const handleInteractionCubeClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();

    const wasClick =
      dragStartPos.current &&
      Math.abs(event.clientX - dragStartPos.current.x) < 3 &&
      Math.abs(event.clientY - dragStartPos.current.y) < 3;

    if (wasClick && onFaceClick && event.point) {
      const zone = detectZoneFromPoint(event.point);
      if (zone) {
        console.log(`🖱️ ${zone.name} clicked →`, zone.direction);
        onFaceClick(zone.direction);
      }
    }

    setIsDragging(false);
    dragStartPos.current = null;
  };

  // ✅ Move handler for interaction cube
  const handleInteractionCubeMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (isDragging || !event.point) return;

    const zone = detectZoneFromPoint(event.point);
    setHoveredZone(zone?.name || null);
  };

  return (
    <group ref={groupRef}>
      {/* Main cube mesh from STL - properly centered */}
      {centeredGeometry && (
        <mesh
          ref={meshRef}
          geometry={centeredGeometry}
          castShadow
          receiveShadow
          scale={1.1}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerEnter={handleCubeEnter}
        >
          <meshStandardMaterial
            color="#FFAB00"
            metalness={0.3}
            roughness={0.5}
            transparent={false}
            opacity={1}
            envMapIntensity={1.5}
            side={THREE.FrontSide}
          />
        </mesh>
      )}

      {/* ✅ Third invisible interaction cube - handles all 26 zones */}
      {centeredGeometry && (
        <mesh
          position={[0, 0, 0]}
          scale={1.1}
          onClick={handleInteractionCubeClick}
          onPointerMove={handleInteractionCubeMove}
          onPointerEnter={() => (gl.domElement.style.cursor = "pointer")}
          onPointerLeave={() => {
            gl.domElement.style.cursor = "auto";
            setHoveredZone(null);
          }}
          visible={!isDragging}
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial transparent opacity={0} side={THREE.BackSide} depthWrite={false} />
        </mesh>
      )}

      {/* Highlight overlay when hovering ANY zone */}
      {hoveredZone && (
        <mesh position={[0, 0, 0]} raycast={() => null}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshBasicMaterial color="#2563eb" transparent={true} opacity={0.3} depthTest={false} depthWrite={false} />
        </mesh>
      )}

      {/* Edge lines - Dual-layer: Simple box for clean 12 outer edges */}
      {centeredGeometry && (
        <lineSegments scale={1.1}>
          <edgesGeometry args={[centeredGeometry]} />
          <lineBasicMaterial
            color="#0f172a"
            linewidth={2}
            transparent={true}
            opacity={0.9}
            depthTest={true}
            depthWrite={false}
          />
        </lineSegments>
      )}
    </group>
  );
}
