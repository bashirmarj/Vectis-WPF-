// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ Reference-Accurate 54-Mesh PlaneGeometry Architecture
// All faces, edges, and corners are individual PlaneGeometry meshes
// Highlighting via material.color changes (orange -> blue)

import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import * as THREE from "three";
import { ThreeEvent } from "@react-three/fiber";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
  onDragRotate?: (deltaX: number, deltaY: number) => void;
  groupRef: React.RefObject<THREE.Group>;
}

// Constants matching reference code
const CUBE_SIZE = 1.5; // Increased from 1.0 for better visibility
const EDGE_SIZE = 0.1;
const FACE_SIZE = CUBE_SIZE - (EDGE_SIZE * 2); // 0.8
const FACE_OFFSET = CUBE_SIZE / 2; // 0.75
const BORDER_OFFSET = FACE_OFFSET - (EDGE_SIZE / 2); // 0.70

/**
 * Helper function to create text sprite for face labels
 */
function createTextSprite(text: string): THREE.Texture {
  const canvas = document.createElement('canvas');
  canvas.width = 200;
  canvas.height = 200;
  const context = canvas.getContext('2d')!;
  
  // Background
  context.fillStyle = 'rgba(255, 171, 0, 1.0)';
  context.fillRect(0, 0, 200, 200);
  
  // Text
  context.font = 'bold 30px Arial Narrow, sans-serif';
  context.fillStyle = 'rgba(15, 23, 42, 1.0)';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, 100, 100);
  
  const texture = new THREE.Texture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

// Helper to create a single PlaneGeometry face
function createFace(
  size: [number, number],
  position: [number, number, number],
  rotation: [number, number, number],
  name: string,
  textMap?: THREE.Texture
): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(size[0], size[1]);
  const material = new THREE.MeshStandardMaterial({
    color: 0xFFAB00,
    metalness: 0.3,
    roughness: 0.5,
    side: THREE.DoubleSide,
    map: textMap, // Apply texture if provided
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.rotation.set(rotation[0], rotation[1], rotation[2]);
  mesh.position.set(position[0], position[1], position[2]);
  return mesh;
}

export function OrientationCubeMesh({
  onFaceClick,
  onDragRotate,
  groupRef,
}: OrientationCubeMeshProps) {
  const [hoveredZoneName, setHoveredZoneName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);

  // Build all 54 PlaneGeometry meshes
  const cubeMeshes = useMemo(() => {
    const meshes: THREE.Mesh[] = [];
    
    // === 6 MAIN FACES (1 mesh each) with text labels ===
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, 0, FACE_OFFSET], [0, 0, 0], 'face-front', createTextSprite('FRONT')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [FACE_OFFSET, 0, 0], [0, Math.PI/2, 0], 'face-right', createTextSprite('RIGHT')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, 0, -FACE_OFFSET], [0, Math.PI, 0], 'face-back', createTextSprite('BACK')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [-FACE_OFFSET, 0, 0], [0, -Math.PI/2, 0], 'face-left', createTextSprite('LEFT')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, FACE_OFFSET, 0], [-Math.PI/2, 0, 0], 'face-top', createTextSprite('TOP')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, -FACE_OFFSET, 0], [Math.PI/2, 0, 0], 'face-bottom', createTextSprite('BOTTOM')));
    
    // === 12 EDGES (2 meshes each = 24 total) ===
    
    // Top 4 edges (horizontal)
    const e1 = 'edge-top-front';
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], e1));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, FACE_OFFSET, BORDER_OFFSET], [-Math.PI/2, 0, 0], e1));
    
    const e2 = 'edge-top-right';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [BORDER_OFFSET, FACE_OFFSET, 0], [Math.PI/2, Math.PI/2, 0], e2));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [FACE_OFFSET, BORDER_OFFSET, 0], [0, Math.PI/2, 0], e2));
    
    const e3 = 'edge-top-back';
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], e3));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, FACE_OFFSET, -BORDER_OFFSET], [-Math.PI/2, 0, 0], e3));
    
    const e4 = 'edge-top-left';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-BORDER_OFFSET, FACE_OFFSET, 0], [Math.PI/2, -Math.PI/2, 0], e4));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [-FACE_OFFSET, BORDER_OFFSET, 0], [0, -Math.PI/2, 0], e4));
    
    // Middle 4 edges (vertical)
    const e5 = 'edge-front-right';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [BORDER_OFFSET, 0, FACE_OFFSET], [0, 0, 0], e5));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [FACE_OFFSET, 0, BORDER_OFFSET], [0, Math.PI/2, 0], e5));
    
    const e6 = 'edge-back-right';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [BORDER_OFFSET, 0, -FACE_OFFSET], [0, Math.PI, 0], e6));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [FACE_OFFSET, 0, -BORDER_OFFSET], [0, Math.PI/2, 0], e6));
    
    const e7 = 'edge-back-left';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-BORDER_OFFSET, 0, -FACE_OFFSET], [0, Math.PI, 0], e7));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-FACE_OFFSET, 0, -BORDER_OFFSET], [0, -Math.PI/2, 0], e7));
    
    const e8 = 'edge-front-left';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-BORDER_OFFSET, 0, FACE_OFFSET], [0, 0, 0], e8));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-FACE_OFFSET, 0, BORDER_OFFSET], [0, -Math.PI/2, 0], e8));
    
    // Bottom 4 edges (horizontal)
    const e9 = 'edge-bottom-front';
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, -BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], e9));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, -FACE_OFFSET, BORDER_OFFSET], [Math.PI/2, 0, 0], e9));
    
    const e10 = 'edge-bottom-right';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [BORDER_OFFSET, -FACE_OFFSET, 0], [-Math.PI/2, Math.PI/2, 0], e10));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [FACE_OFFSET, -BORDER_OFFSET, 0], [0, Math.PI/2, 0], e10));
    
    const e11 = 'edge-bottom-back';
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, -BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], e11));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, -FACE_OFFSET, -BORDER_OFFSET], [Math.PI/2, 0, 0], e11));
    
    const e12 = 'edge-bottom-left';
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-BORDER_OFFSET, -FACE_OFFSET, 0], [-Math.PI/2, -Math.PI/2, 0], e12));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [-FACE_OFFSET, -BORDER_OFFSET, 0], [0, -Math.PI/2, 0], e12));
    
    // === 8 CORNERS (3 meshes each = 24 total) ===
    
    const c1 = 'corner-top-front-right';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c1));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, BORDER_OFFSET, BORDER_OFFSET], [0, Math.PI/2, 0], c1));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, FACE_OFFSET, BORDER_OFFSET], [-Math.PI/2, 0, 0], c1));
    
    const c2 = 'corner-top-back-right';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c2));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, BORDER_OFFSET, -BORDER_OFFSET], [0, Math.PI/2, 0], c2));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, FACE_OFFSET, -BORDER_OFFSET], [-Math.PI/2, 0, 0], c2));
    
    const c3 = 'corner-top-back-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c3));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, BORDER_OFFSET, -BORDER_OFFSET], [0, -Math.PI/2, 0], c3));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, FACE_OFFSET, -BORDER_OFFSET], [-Math.PI/2, 0, 0], c3));
    
    const c4 = 'corner-top-front-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c4));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, BORDER_OFFSET, BORDER_OFFSET], [0, -Math.PI/2, 0], c4));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, FACE_OFFSET, BORDER_OFFSET], [-Math.PI/2, 0, 0], c4));
    
    const c5 = 'corner-bottom-front-right';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c5));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, -BORDER_OFFSET, BORDER_OFFSET], [0, Math.PI/2, 0], c5));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -FACE_OFFSET, BORDER_OFFSET], [Math.PI/2, 0, 0], c5));
    
    const c6 = 'corner-bottom-back-right';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c6));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, -BORDER_OFFSET, -BORDER_OFFSET], [0, Math.PI/2, 0], c6));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -FACE_OFFSET, -BORDER_OFFSET], [Math.PI/2, 0, 0], c6));
    
    const c7 = 'corner-bottom-back-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c7));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, -BORDER_OFFSET, -BORDER_OFFSET], [0, -Math.PI/2, 0], c7));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -FACE_OFFSET, -BORDER_OFFSET], [Math.PI/2, 0, 0], c7));
    
    const c8 = 'corner-bottom-front-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c8));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, -BORDER_OFFSET, BORDER_OFFSET], [0, -Math.PI/2, 0], c8));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -FACE_OFFSET, BORDER_OFFSET], [Math.PI/2, 0, 0], c8));
    
    return meshes;
  }, []);

  // Window-level drag tracking
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging && dragStartPos.current && onDragRotate) {
        const deltaX = e.clientX - dragStartPos.current.x;
        const deltaY = e.clientY - dragStartPos.current.y;
        onDragRotate(deltaX, deltaY);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      dragStartPos.current = null;
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, onDragRotate]);

  // Event handlers
  const handlePointerMove = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    
    // Reset all meshes to orange
    cubeMeshes.forEach(mesh => {
      (mesh.material as THREE.MeshStandardMaterial).color.setHex(0xFFAB00);
    });
    
    // Highlight all meshes with matching name
    if (event.intersections.length > 0) {
      const hoveredName = event.intersections[0].object.name;
      if (hoveredName) {
        cubeMeshes.forEach(mesh => {
          if (mesh.name === hoveredName) {
            (mesh.material as THREE.MeshStandardMaterial).color.setHex(0x3b82f6);
          }
        });
        setHoveredZoneName(hoveredName);
        document.body.style.cursor = 'pointer';
      }
    }
  }, [cubeMeshes]);

  const handlePointerLeave = useCallback(() => {
    cubeMeshes.forEach(mesh => {
      (mesh.material as THREE.MeshStandardMaterial).color.setHex(0xFFAB00);
    });
    setHoveredZoneName(null);
    document.body.style.cursor = 'default';
  }, [cubeMeshes]);

  const handlePointerDown = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    setIsDragging(true);
    dragStartPos.current = { x: event.clientX, y: event.clientY };
  }, []);

  const handlePointerUp = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const wasDragging = isDragging;
    setIsDragging(false);
    dragStartPos.current = null;
    
    if (!wasDragging && event.intersections.length > 0) {
      const clickedName = event.intersections[0].object.name;
      
      // Extract direction from clicked zone
      if (clickedName.startsWith('face-')) {
        const directionMap: { [key: string]: THREE.Vector3 } = {
          'face-front': new THREE.Vector3(0, 0, 1),
          'face-back': new THREE.Vector3(0, 0, -1),
          'face-right': new THREE.Vector3(1, 0, 0),
          'face-left': new THREE.Vector3(-1, 0, 0),
          'face-top': new THREE.Vector3(0, 1, 0),
          'face-bottom': new THREE.Vector3(0, -1, 0),
        };
        const direction = directionMap[clickedName];
        if (direction && onFaceClick) {
          onFaceClick(direction);
        }
      } else if (clickedName.startsWith('edge-')) {
        // Edge directions (diagonal)
        const edgeDirectionMap: { [key: string]: THREE.Vector3 } = {
          'edge-top-front': new THREE.Vector3(0, 1, 1).normalize(),
          'edge-top-right': new THREE.Vector3(1, 1, 0).normalize(),
          'edge-top-back': new THREE.Vector3(0, 1, -1).normalize(),
          'edge-top-left': new THREE.Vector3(-1, 1, 0).normalize(),
          'edge-front-right': new THREE.Vector3(1, 0, 1).normalize(),
          'edge-back-right': new THREE.Vector3(1, 0, -1).normalize(),
          'edge-back-left': new THREE.Vector3(-1, 0, -1).normalize(),
          'edge-front-left': new THREE.Vector3(-1, 0, 1).normalize(),
          'edge-bottom-front': new THREE.Vector3(0, -1, 1).normalize(),
          'edge-bottom-right': new THREE.Vector3(1, -1, 0).normalize(),
          'edge-bottom-back': new THREE.Vector3(0, -1, -1).normalize(),
          'edge-bottom-left': new THREE.Vector3(-1, -1, 0).normalize(),
        };
        const direction = edgeDirectionMap[clickedName];
        if (direction && onFaceClick) {
          onFaceClick(direction);
        }
      } else if (clickedName.startsWith('corner-')) {
        // Corner directions (3D diagonal)
        const cornerDirectionMap: { [key: string]: THREE.Vector3 } = {
          'corner-top-front-right': new THREE.Vector3(1, 1, 1).normalize(),
          'corner-top-back-right': new THREE.Vector3(1, 1, -1).normalize(),
          'corner-top-back-left': new THREE.Vector3(-1, 1, -1).normalize(),
          'corner-top-front-left': new THREE.Vector3(-1, 1, 1).normalize(),
          'corner-bottom-front-right': new THREE.Vector3(1, -1, 1).normalize(),
          'corner-bottom-back-right': new THREE.Vector3(1, -1, -1).normalize(),
          'corner-bottom-back-left': new THREE.Vector3(-1, -1, -1).normalize(),
          'corner-bottom-front-left': new THREE.Vector3(-1, -1, 1).normalize(),
        };
        const direction = cornerDirectionMap[clickedName];
        if (direction && onFaceClick) {
          onFaceClick(direction);
        }
      }
    }
  }, [isDragging, onFaceClick]);

  return (
    <group ref={groupRef}>
      {/* Single interactive layer: 54 visible PlaneGeometry meshes */}
      <group
        onPointerMove={handlePointerMove}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
      >
        {cubeMeshes.map((mesh, index) => (
          <primitive key={index} object={mesh} />
        ))}
      </group>
      
      {/* Wireframe */}
      <lineSegments>
        <edgesGeometry>
          <boxGeometry args={[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]} />
        </edgesGeometry>
        <lineBasicMaterial color="#0f172a" linewidth={2} />
      </lineSegments>
    </group>
  );
}
