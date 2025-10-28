// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ FIXED: Face click detection + drag-to-rotate functionality

import { useRef, useMemo, useState, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";
import { ThreeEvent, useLoader, useThree } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
  onDragRotate?: (deltaX: number, deltaY: number) => void;
}

export const OrientationCubeMesh = forwardRef<THREE.Group, OrientationCubeMeshProps>(
  ({ onFaceClick, onDragRotate }, ref) => {
    const groupRef = useRef<THREE.Group>(null);
    const meshRef = useRef<THREE.Mesh>(null);
    const [hoveredFace, setHoveredFace] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const dragStartPos = useRef<{ x: number; y: number } | null>(null);
    const { gl } = useThree();

    // Expose group ref to parent
    useImperativeHandle(ref, () => groupRef.current!);

    // ✅ Load STL and center it immediately
    const rawGeometry = useLoader(STLLoader, "/src/assets/orientation-cube.stl");

    const geometry = useMemo(() => {
      if (rawGeometry) {
        rawGeometry.center();
        rawGeometry.computeVertexNormals();
        rawGeometry.computeBoundingBox();
        rawGeometry.computeBoundingSphere();
        console.log("✅ STL centered and ready for raycasting");
      }
      return rawGeometry;
    }, [rawGeometry]);

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

    const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      setIsDragging(true);
      dragStartPos.current = { x: event.clientX, y: event.clientY };
      gl.domElement.style.cursor = "grabbing";

      // Capture pointer to continue receiving events even outside canvas
      (event.target as any).setPointerCapture(event.pointerId);
    };

    const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
      if (isDragging && dragStartPos.current && onDragRotate) {
        const deltaX = event.clientX - dragStartPos.current.x;
        const deltaY = event.clientY - dragStartPos.current.y;

        // Only call onDragRotate if there's meaningful movement
        if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
          onDragRotate(deltaX, deltaY);
          dragStartPos.current = { x: event.clientX, y: event.clientY };
        }
      } else if (!isDragging && event.face && groupRef.current) {
        // Hover highlighting (only when not dragging)
        const normal = event.face.normal.clone();
        groupRef.current.updateMatrixWorld(true);
        normal.transformDirection(groupRef.current.matrixWorld);
        const face = getFaceFromNormal(normal);
        if (face !== hoveredFace) {
          setHoveredFace(face);
        }
      }
    };

    const handlePointerUp = (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();

      // Check if this was a click (minimal movement) or a drag
      const wasClick =
        dragStartPos.current &&
        Math.abs(event.clientX - dragStartPos.current.x) < 5 &&
        Math.abs(event.clientY - dragStartPos.current.y) < 5;

      if (wasClick && onFaceClick && event.face && groupRef.current) {
        // Handle face click
        const normal = event.face.normal.clone();
        groupRef.current.updateMatrixWorld(true);
        normal.transformDirection(groupRef.current.matrixWorld);

        console.log("🖱️ Cube clicked - transformed normal:", normal);

        const absX = Math.abs(normal.x);
        const absY = Math.abs(normal.y);
        const absZ = Math.abs(normal.z);

        let direction: THREE.Vector3;

        if (absX > absY && absX > absZ) {
          direction = new THREE.Vector3(Math.sign(normal.x), 0, 0);
          console.log("   → Detected X-axis face:", Math.sign(normal.x) > 0 ? "RIGHT" : "LEFT");
        } else if (absY > absX && absY > absZ) {
          direction = new THREE.Vector3(0, Math.sign(normal.y), 0);
          console.log("   → Detected Y-axis face:", Math.sign(normal.y) > 0 ? "TOP" : "BOTTOM");
        } else {
          direction = new THREE.Vector3(0, 0, Math.sign(normal.z));
          console.log("   → Detected Z-axis face:", Math.sign(normal.z) > 0 ? "FRONT" : "BACK");
        }

        onFaceClick(direction);
      }

      setIsDragging(false);
      dragStartPos.current = null;
      gl.domElement.style.cursor = "grab";

      // Release pointer capture
      (event.target as any).releasePointerCapture(event.pointerId);
    };

    const handlePointerEnter = (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      gl.domElement.style.cursor = "grab";

      if (event.face && groupRef.current && !isDragging) {
        const normal = event.face.normal.clone();
        groupRef.current.updateMatrixWorld(true);
        normal.transformDirection(groupRef.current.matrixWorld);
        const face = getFaceFromNormal(normal);
        setHoveredFace(face);
      }
    };

    const handlePointerLeave = (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      if (!isDragging) {
        gl.domElement.style.cursor = "default";
        setHoveredFace(null);
      }
    };

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
        <mesh
          ref={meshRef}
          geometry={geometry}
          material={baseMaterial}
          castShadow
          receiveShadow
          scale={0.95}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerEnter={handlePointerEnter}
          onPointerLeave={handlePointerLeave}
        />

        {hoveredFace && faceConfig[hoveredFace] && !isDragging && (
          <mesh position={faceConfig[hoveredFace].position} rotation={faceConfig[hoveredFace].rotation}>
            <planeGeometry args={[1.7, 1.7]} />
            <meshBasicMaterial color="#60a5fa" transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
          </mesh>
        )}

        <lineSegments>
          <edgesGeometry args={[geometry, 25]} />
          <lineBasicMaterial color="#0f172a" linewidth={2} transparent opacity={0.7} />
        </lineSegments>
      </group>
    );
  },
);

OrientationCubeMesh.displayName = "OrientationCubeMesh";
