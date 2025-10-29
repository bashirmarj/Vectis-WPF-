// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ Reference-Accurate Implementation
// Matches ViewCube architecture from reference code

import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import * as THREE from "three";
import { ThreeEvent } from "@react-three/fiber";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
  onDragRotate?: (deltaX: number, deltaY: number) => void;
  groupRef: React.RefObject<THREE.Group>;
}

// Constants matching reference code
const CUBE_SIZE = 1.5;
const EDGE_SIZE = 0.2;
const FACE_SIZE = CUBE_SIZE - (EDGE_SIZE * 2);
const FACE_OFFSET = CUBE_SIZE / 2;
const BORDER_OFFSET = FACE_OFFSET - (EDGE_SIZE / 2);

// Color scheme from reference
const MAINCOLOR = 0xDDDDDD;
const ACCENTCOLOR = 0xF2F5CE;
const OUTLINECOLOR = 0xCCCCCC;

/**
 * Helper function to create text texture for face labels
 */
function createTextSprite(text: string): THREE.Texture {
  const fontface = 'Arial Narrow, sans-serif';
  const fontsize = 60;
  const width = 200;
  const height = 200;
  
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d')!;
  
  // Solid background matching face color
  context.fillStyle = 'rgb(221, 221, 221)'; // MAINCOLOR
  context.fillRect(0, 0, width, height);
  
  // Bold black text
  context.font = `bold ${fontsize}px ${fontface}`;
  context.fillStyle = 'rgb(0, 0, 0)'; // Black text
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, width / 2, height / 2);
  
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
  const material = new THREE.MeshBasicMaterial({
    color: MAINCOLOR,
    side: THREE.DoubleSide,
    map: textMap,
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
  const { meshes: cubeMeshes } = useMemo(() => {
    const meshes: THREE.Mesh[] = [];
    
    // === 6 MAIN FACES with text labels ===
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, 0, FACE_OFFSET], [0, 0, 0], 'face-front', createTextSprite('FRONT')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [FACE_OFFSET, 0, 0], [0, Math.PI/2, 0], 'face-right', createTextSprite('RIGHT')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, 0, -FACE_OFFSET], [0, Math.PI, 0], 'face-back', createTextSprite('BACK')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [-FACE_OFFSET, 0, 0], [0, -Math.PI/2, 0], 'face-left', createTextSprite('LEFT')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, FACE_OFFSET, 0], [-Math.PI/2, 0, 0], 'face-top', createTextSprite('TOP')));
    meshes.push(createFace([FACE_SIZE, FACE_SIZE], [0, -FACE_OFFSET, 0], [Math.PI/2, 0, 0], 'face-bottom', createTextSprite('BOTTOM')));
    
    // === 12 EDGES (2 meshes each = 24 total) ===
    // Top edges
    const e1 = 'edge-top-front', e2 = 'edge-top-right', e3 = 'edge-top-back', e4 = 'edge-top-left';
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], e1));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [0, BORDER_OFFSET, BORDER_OFFSET], [-Math.PI/2, 0, 0], e1));
    
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, 0], [-Math.PI/2, 0, 0], e2));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [FACE_OFFSET, BORDER_OFFSET, 0], [0, Math.PI/2, 0], e2));
    
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], e3));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [0, BORDER_OFFSET, -BORDER_OFFSET], [-Math.PI/2, 0, 0], e3));
    
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, 0], [-Math.PI/2, 0, 0], e4));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [-FACE_OFFSET, BORDER_OFFSET, 0], [0, -Math.PI/2, 0], e4));
    
    // Bottom edges
    const e5 = 'edge-bottom-front', e6 = 'edge-bottom-right', e7 = 'edge-bottom-back', e8 = 'edge-bottom-left';
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, -BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], e5));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [0, -BORDER_OFFSET, BORDER_OFFSET], [Math.PI/2, 0, 0], e5));
    
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, 0], [Math.PI/2, 0, 0], e6));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [FACE_OFFSET, -BORDER_OFFSET, 0], [0, Math.PI/2, 0], e6));
    
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [0, -BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], e7));
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [0, -BORDER_OFFSET, -BORDER_OFFSET], [Math.PI/2, 0, 0], e7));
    
    meshes.push(createFace([EDGE_SIZE, FACE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, 0], [Math.PI/2, 0, 0], e8));
    meshes.push(createFace([FACE_SIZE, EDGE_SIZE], [-FACE_OFFSET, -BORDER_OFFSET, 0], [0, -Math.PI/2, 0], e8));
    
    // Vertical edges
    const e9 = 'edge-front-right', e10 = 'edge-back-right', e11 = 'edge-back-left', e12 = 'edge-front-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, 0, FACE_OFFSET], [0, 0, 0], e9));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, 0, BORDER_OFFSET], [0, Math.PI/2, 0], e9));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, 0, -FACE_OFFSET], [0, Math.PI, 0], e10));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, 0, -BORDER_OFFSET], [0, Math.PI/2, 0], e10));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, 0, -FACE_OFFSET], [0, Math.PI, 0], e11));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, 0, -BORDER_OFFSET], [0, -Math.PI/2, 0], e11));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, 0, FACE_OFFSET], [0, 0, 0], e12));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, 0, BORDER_OFFSET], [0, -Math.PI/2, 0], e12));
    
    // === 8 CORNERS (3 meshes each = 24 total) ===
    // Top corners
    const c1 = 'corner-top-front-right', c2 = 'corner-top-back-right', c3 = 'corner-top-back-left', c4 = 'corner-top-front-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c1));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, BORDER_OFFSET, BORDER_OFFSET], [0, Math.PI/2, 0], c1));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, BORDER_OFFSET], [-Math.PI/2, 0, 0], c1));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c2));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, BORDER_OFFSET, -BORDER_OFFSET], [0, Math.PI/2, 0], c2));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, BORDER_OFFSET, -BORDER_OFFSET], [-Math.PI/2, 0, 0], c2));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c3));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, BORDER_OFFSET, -BORDER_OFFSET], [0, -Math.PI/2, 0], c3));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, -BORDER_OFFSET], [-Math.PI/2, 0, 0], c3));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c4));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, BORDER_OFFSET, BORDER_OFFSET], [0, -Math.PI/2, 0], c4));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, BORDER_OFFSET, BORDER_OFFSET], [-Math.PI/2, 0, 0], c4));
    
    // Bottom corners
    const c5 = 'corner-bottom-front-right', c6 = 'corner-bottom-back-right', c7 = 'corner-bottom-back-left', c8 = 'corner-bottom-front-left';
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c5));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, -BORDER_OFFSET, BORDER_OFFSET], [0, Math.PI/2, 0], c5));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, BORDER_OFFSET], [Math.PI/2, 0, 0], c5));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c6));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [FACE_OFFSET, -BORDER_OFFSET, -BORDER_OFFSET], [0, Math.PI/2, 0], c6));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [BORDER_OFFSET, -BORDER_OFFSET, -BORDER_OFFSET], [Math.PI/2, 0, 0], c6));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, -FACE_OFFSET], [0, Math.PI, 0], c7));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, -BORDER_OFFSET, -BORDER_OFFSET], [0, -Math.PI/2, 0], c7));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, -BORDER_OFFSET], [Math.PI/2, 0, 0], c7));
    
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, FACE_OFFSET], [0, 0, 0], c8));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-FACE_OFFSET, -BORDER_OFFSET, BORDER_OFFSET], [0, -Math.PI/2, 0], c8));
    meshes.push(createFace([EDGE_SIZE, EDGE_SIZE], [-BORDER_OFFSET, -BORDER_OFFSET, BORDER_OFFSET], [Math.PI/2, 0, 0], c8));
    
    return { meshes };
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
    
    // Reset all meshes to main color
    cubeMeshes.forEach(mesh => {
      if (mesh.name && mesh.material && 'color' in mesh.material) {
        (mesh.material as THREE.MeshBasicMaterial).color.setHex(MAINCOLOR);
      }
    });
    
    // Highlight all meshes with matching name
    if (event.intersections.length > 0) {
      const hoveredName = event.intersections[0].object.name;
      if (hoveredName) {
        cubeMeshes.forEach(mesh => {
          if (mesh.name === hoveredName) {
            (mesh.material as THREE.MeshBasicMaterial).color.setHex(ACCENTCOLOR);
          }
        });
        setHoveredZoneName(hoveredName);
        document.body.style.cursor = 'pointer';
      }
    }
  }, [cubeMeshes]);

  const handlePointerLeave = useCallback(() => {
    cubeMeshes.forEach((mesh) => {
      if (mesh.name && mesh.material && 'color' in mesh.material) {
        (mesh.material as THREE.MeshBasicMaterial).color.setHex(MAINCOLOR);
      }
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
      {/* Interactive meshes */}
      <group
        onPointerMove={handlePointerMove}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
      >
        {cubeMeshes.map((mesh, index) => (
          <primitive key={`mesh-${index}`} object={mesh} />
        ))}
      </group>
      
      {/* Cube outline wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)]} />
        <lineBasicMaterial color={OUTLINECOLOR} linewidth={1} />
      </lineSegments>
    </group>
  );
}
