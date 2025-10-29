// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ Group-Based Architecture Implementation
// Matches reference ViewCube structure using THREE.Group rotation

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

/**
 * Creates a corner group with 3 faces (like reference _createCornerFaces)
 */
function createCornerGroup(borderSize: number, offset: number, name: string): THREE.Group {
  const corner = new THREE.Group();
  const borderOffset = offset - borderSize / 2;
  
  // Face 1: Front-facing
  const geo1 = new THREE.PlaneGeometry(borderSize, borderSize);
  const mat1 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh1 = new THREE.Mesh(geo1, mat1);
  mesh1.name = name;
  mesh1.position.set(borderOffset, borderOffset, offset);
  corner.add(mesh1);
  
  // Face 2: Right-facing
  const geo2 = new THREE.PlaneGeometry(borderSize, borderSize);
  const mat2 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh2 = new THREE.Mesh(geo2, mat2);
  mesh2.name = name;
  mesh2.position.set(offset, borderOffset, borderOffset);
  mesh2.rotateOnAxis(new THREE.Vector3(0, 1, 0), Math.PI / 2);
  corner.add(mesh2);
  
  // Face 3: Top-facing
  const geo3 = new THREE.PlaneGeometry(borderSize, borderSize);
  const mat3 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh3 = new THREE.Mesh(geo3, mat3);
  mesh3.name = name;
  mesh3.position.set(borderOffset, offset, borderOffset);
  mesh3.rotateOnAxis(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  corner.add(mesh3);
  
  return corner;
}

/**
 * Creates horizontal edge group with 2 faces
 */
function createHorzEdgeGroup(width: number, height: number, offset: number, name: string): THREE.Group {
  const edge = new THREE.Group();
  const borderOffset = offset - height / 2;
  
  // Face 1: Front
  const geo1 = new THREE.PlaneGeometry(width, height);
  const mat1 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh1 = new THREE.Mesh(geo1, mat1);
  mesh1.name = name;
  mesh1.position.set(0, borderOffset, offset);
  edge.add(mesh1);
  
  // Face 2: Top
  const geo2 = new THREE.PlaneGeometry(width, height);
  const mat2 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh2 = new THREE.Mesh(geo2, mat2);
  mesh2.name = name;
  mesh2.position.set(0, offset, borderOffset);
  mesh2.rotateOnAxis(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  edge.add(mesh2);
  
  return edge;
}

/**
 * Creates vertical edge group with 2 faces
 */
function createVertEdgeGroup(width: number, height: number, offset: number, name: string): THREE.Group {
  const edge = new THREE.Group();
  const borderOffset = offset - width / 2;
  
  // Face 1: Front
  const geo1 = new THREE.PlaneGeometry(width, height);
  const mat1 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh1 = new THREE.Mesh(geo1, mat1);
  mesh1.name = name;
  mesh1.position.set(borderOffset, 0, offset);
  edge.add(mesh1);
  
  // Face 2: Right
  const geo2 = new THREE.PlaneGeometry(width, height);
  const mat2 = new THREE.MeshBasicMaterial({ color: MAINCOLOR });
  const mesh2 = new THREE.Mesh(geo2, mat2);
  mesh2.name = name;
  mesh2.position.set(offset, 0, borderOffset);
  mesh2.rotateOnAxis(new THREE.Vector3(0, 1, 0), Math.PI / 2);
  edge.add(mesh2);
  
  return edge;
}

export function OrientationCubeMesh({
  onFaceClick,
  onDragRotate,
  groupRef,
}: OrientationCubeMeshProps) {
  const [hoveredZoneName, setHoveredZoneName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);

  // Build cube using group rotation pattern (like reference _build())
  const cubeGroup = useMemo(() => {
    const mainGroup = new THREE.Group();
    
    // === 6 MAIN FACES ===
    const frontMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(FACE_SIZE, FACE_SIZE),
      new THREE.MeshBasicMaterial({ color: MAINCOLOR, map: createTextSprite('FRONT') })
    );
    frontMesh.name = 'face-front';
    frontMesh.position.set(0, 0, FACE_OFFSET);
    mainGroup.add(frontMesh);
    
    const rightMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(FACE_SIZE, FACE_SIZE),
      new THREE.MeshBasicMaterial({ color: MAINCOLOR, map: createTextSprite('RIGHT') })
    );
    rightMesh.name = 'face-right';
    rightMesh.position.set(FACE_OFFSET, 0, 0);
    rightMesh.rotateOnAxis(new THREE.Vector3(0, 1, 0), Math.PI / 2);
    mainGroup.add(rightMesh);
    
    const backMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(FACE_SIZE, FACE_SIZE),
      new THREE.MeshBasicMaterial({ color: MAINCOLOR, map: createTextSprite('BACK') })
    );
    backMesh.name = 'face-back';
    backMesh.position.set(0, 0, -FACE_OFFSET);
    backMesh.rotateOnAxis(new THREE.Vector3(0, 1, 0), Math.PI);
    mainGroup.add(backMesh);
    
    const leftMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(FACE_SIZE, FACE_SIZE),
      new THREE.MeshBasicMaterial({ color: MAINCOLOR, map: createTextSprite('LEFT') })
    );
    leftMesh.name = 'face-left';
    leftMesh.position.set(-FACE_OFFSET, 0, 0);
    leftMesh.rotateOnAxis(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
    mainGroup.add(leftMesh);
    
    const topMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(FACE_SIZE, FACE_SIZE),
      new THREE.MeshBasicMaterial({ color: MAINCOLOR, map: createTextSprite('TOP') })
    );
    topMesh.name = 'face-top';
    topMesh.position.set(0, FACE_OFFSET, 0);
    topMesh.rotateOnAxis(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    mainGroup.add(topMesh);
    
    const bottomMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(FACE_SIZE, FACE_SIZE),
      new THREE.MeshBasicMaterial({ color: MAINCOLOR, map: createTextSprite('BOTTOM') })
    );
    bottomMesh.name = 'face-bottom';
    bottomMesh.position.set(0, -FACE_OFFSET, 0);
    bottomMesh.rotateOnAxis(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    mainGroup.add(bottomMesh);
    
    // === 8 CORNERS (using rotation like reference) ===
    // Top 4 corners
    const topCornerNames = [
      'corner-top-front-right',
      'corner-top-back-right', 
      'corner-top-back-left',
      'corner-top-front-left'
    ];
    
    for (let i = 0; i < 4; i++) {
      const corner = createCornerGroup(EDGE_SIZE, FACE_OFFSET, topCornerNames[i]);
      corner.rotateOnAxis(new THREE.Vector3(0, 1, 0), (i * 90) * Math.PI / 180);
      mainGroup.add(corner);
    }
    
    // Bottom 4 corners
    const bottomCornersGroup = new THREE.Group();
    const bottomCornerNames = [
      'corner-bottom-back-right',
      'corner-bottom-front-right',
      'corner-bottom-front-left',
      'corner-bottom-back-left'
    ];
    
    for (let i = 0; i < 4; i++) {
      const corner = createCornerGroup(EDGE_SIZE, FACE_OFFSET, bottomCornerNames[i]);
      corner.rotateOnAxis(new THREE.Vector3(0, 1, 0), (i * 90) * Math.PI / 180);
      bottomCornersGroup.add(corner);
    }
    bottomCornersGroup.rotateOnAxis(new THREE.Vector3(1, 0, 0), Math.PI);
    mainGroup.add(bottomCornersGroup);
    
    // === 12 EDGES ===
    // Top 4 horizontal edges
    const topEdgeNames = [
      'edge-top-front',
      'edge-top-right',
      'edge-top-back',
      'edge-top-left'
    ];
    
    for (let i = 0; i < 4; i++) {
      const edge = createHorzEdgeGroup(FACE_SIZE, EDGE_SIZE, FACE_OFFSET, topEdgeNames[i]);
      edge.rotateOnAxis(new THREE.Vector3(0, 1, 0), (i * 90) * Math.PI / 180);
      mainGroup.add(edge);
    }
    
    // Bottom 4 horizontal edges
    const bottomEdgesGroup = new THREE.Group();
    const bottomEdgeNames = [
      'edge-bottom-back',
      'edge-bottom-right',
      'edge-bottom-front',
      'edge-bottom-left'
    ];
    
    for (let i = 0; i < 4; i++) {
      const edge = createHorzEdgeGroup(FACE_SIZE, EDGE_SIZE, FACE_OFFSET, bottomEdgeNames[i]);
      edge.rotateOnAxis(new THREE.Vector3(0, 1, 0), (i * 90) * Math.PI / 180);
      bottomEdgesGroup.add(edge);
    }
    bottomEdgesGroup.rotateOnAxis(new THREE.Vector3(1, 0, 0), Math.PI);
    mainGroup.add(bottomEdgesGroup);
    
    // 4 vertical side edges
    const verticalEdgeNames = [
      'edge-front-right',
      'edge-back-right',
      'edge-back-left',
      'edge-front-left'
    ];
    
    for (let i = 0; i < 4; i++) {
      const edge = createVertEdgeGroup(EDGE_SIZE, FACE_SIZE, FACE_OFFSET, verticalEdgeNames[i]);
      edge.rotateOnAxis(new THREE.Vector3(0, 1, 0), (i * 90) * Math.PI / 180);
      mainGroup.add(edge);
    }
    
    return mainGroup;
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
    
    // Reset all colors by traversing the group
    cubeGroup.traverse((obj) => {
      if (obj instanceof THREE.Mesh && obj.name) {
        (obj.material as THREE.MeshBasicMaterial).color.setHex(MAINCOLOR);
      }
    });
    
    // Highlight hovered
    if (event.intersections.length > 0) {
      const hoveredName = event.intersections[0].object.name;
      if (hoveredName) {
        cubeGroup.traverse((obj) => {
          if (obj instanceof THREE.Mesh && obj.name === hoveredName) {
            (obj.material as THREE.MeshBasicMaterial).color.setHex(ACCENTCOLOR);
          }
        });
        setHoveredZoneName(hoveredName);
        document.body.style.cursor = 'pointer';
      }
    }
  }, [cubeGroup]);

  const handlePointerLeave = useCallback(() => {
    cubeGroup.traverse((obj) => {
      if (obj instanceof THREE.Mesh && obj.name) {
        (obj.material as THREE.MeshBasicMaterial).color.setHex(MAINCOLOR);
      }
    });
    setHoveredZoneName(null);
    document.body.style.cursor = 'default';
  }, [cubeGroup]);

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
      <primitive 
        object={cubeGroup}
        onPointerMove={handlePointerMove}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
      />
      
      {/* Cube outline wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)]} />
        <lineBasicMaterial color={OUTLINECOLOR} linewidth={1} />
      </lineSegments>
    </group>
  );
}
