// src/components/cad-viewer/OrientationCubeMesh.tsx
// ✅ Layered Geometry Architecture - Industry Standard Pattern
// Layer 1: Visual STL cube (chamfered, beautiful)
// Layer 2: Interaction primitives (26 invisible meshes)
// Layer 3: Dynamic highlight mesh
// Layer 4: Wireframe edges

import { useRef, useState, useEffect, useMemo } from "react";
import * as THREE from "three";
import { ThreeEvent, useThree, useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

interface OrientationCubeMeshProps {
  onFaceClick?: (direction: THREE.Vector3) => void;
  onDragRotate?: (deltaX: number, deltaY: number) => void;
  groupRef: React.RefObject<THREE.Group>;
}

interface ZoneData {
  type: 'face' | 'edge' | 'corner';
  name: string;
  position: [number, number, number];
  direction: THREE.Vector3;
  size?: [number, number, number]; // For edges
  radius?: number; // For corners
  rotation?: [number, number, number]; // For faces
}

export function OrientationCubeMesh({ onFaceClick, onDragRotate, groupRef }: OrientationCubeMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const interactionGroupRef = useRef<THREE.Group>(null);
  const [hoveredZoneData, setHoveredZoneData] = useState<ZoneData | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);
  const [centeredGeometry, setCenteredGeometry] = useState<THREE.BufferGeometry | null>(null);
  const { gl, camera } = useThree();

  // ✅ Load STL geometry
  const loadedGeometry = useLoader(STLLoader, "/orientation-cube.stl");

  // ✅ Center the geometry at origin for correct rotation
  useEffect(() => {
    if (loadedGeometry) {
      const cloned = loadedGeometry.clone();
      cloned.computeBoundingBox();
      const bbox = cloned.boundingBox!;
      const center = new THREE.Vector3();
      bbox.getCenter(center);

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
          
          if (groupRef.current) {
            const rotationSpeed = 0.01;
            groupRef.current.rotation.y += deltaX * rotationSpeed;
            groupRef.current.rotation.x += deltaY * rotationSpeed;
          }
          
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
  }, [isDragging, onDragRotate, gl, groupRef]);

  // ✅ Define 6 FACE zones
  const faceDefinitions: ZoneData[] = useMemo(() => [
    {
      type: 'face',
      name: "face-right",
      direction: new THREE.Vector3(1, 0, 0),
      position: [0.575, 0, 0],
      rotation: [0, Math.PI / 2, 0],
    },
    {
      type: 'face',
      name: "face-left",
      direction: new THREE.Vector3(-1, 0, 0),
      position: [-0.575, 0, 0],
      rotation: [0, -Math.PI / 2, 0],
    },
    {
      type: 'face',
      name: "face-top",
      direction: new THREE.Vector3(0, 1, 0),
      position: [0, 0.575, 0],
      rotation: [-Math.PI / 2, 0, 0],
    },
    {
      type: 'face',
      name: "face-bottom",
      direction: new THREE.Vector3(0, -1, 0),
      position: [0, -0.575, 0],
      rotation: [Math.PI / 2, 0, 0],
    },
    {
      type: 'face',
      name: "face-front",
      direction: new THREE.Vector3(0, 0, 1),
      position: [0, 0, 0.575],
      rotation: [0, 0, 0],
    },
    {
      type: 'face',
      name: "face-back",
      direction: new THREE.Vector3(0, 0, -1),
      position: [0, 0, -0.575],
      rotation: [0, Math.PI, 0],
    },
  ], []);

  // ✅ Define 12 EDGE zones
  const edgeDefinitions: ZoneData[] = useMemo(() => [
    // Top edges
    {
      type: 'edge',
      name: "edge-top-front",
      direction: new THREE.Vector3(0, 1, 1).normalize(),
      position: [0, 0.575, 0.575],
      size: [1.15, 0.05, 0.05],
    },
    {
      type: 'edge',
      name: "edge-top-back",
      direction: new THREE.Vector3(0, 1, -1).normalize(),
      position: [0, 0.575, -0.575],
      size: [1.15, 0.05, 0.05],
    },
    {
      type: 'edge',
      name: "edge-top-left",
      direction: new THREE.Vector3(-1, 1, 0).normalize(),
      position: [-0.575, 0.575, 0],
      size: [0.05, 0.05, 1.15],
    },
    {
      type: 'edge',
      name: "edge-top-right",
      direction: new THREE.Vector3(1, 1, 0).normalize(),
      position: [0.575, 0.575, 0],
      size: [0.05, 0.05, 1.15],
    },
    // Middle vertical edges
    {
      type: 'edge',
      name: "edge-front-left",
      direction: new THREE.Vector3(-1, 0, 1).normalize(),
      position: [-0.575, 0, 0.575],
      size: [0.05, 1.15, 0.05],
    },
    {
      type: 'edge',
      name: "edge-front-right",
      direction: new THREE.Vector3(1, 0, 1).normalize(),
      position: [0.575, 0, 0.575],
      size: [0.05, 1.15, 0.05],
    },
    {
      type: 'edge',
      name: "edge-back-left",
      direction: new THREE.Vector3(-1, 0, -1).normalize(),
      position: [-0.575, 0, -0.575],
      size: [0.05, 1.15, 0.05],
    },
    {
      type: 'edge',
      name: "edge-back-right",
      direction: new THREE.Vector3(1, 0, -1).normalize(),
      position: [0.575, 0, -0.575],
      size: [0.05, 1.15, 0.05],
    },
    // Bottom edges
    {
      type: 'edge',
      name: "edge-bottom-front",
      direction: new THREE.Vector3(0, -1, 1).normalize(),
      position: [0, -0.575, 0.575],
      size: [1.15, 0.05, 0.05],
    },
    {
      type: 'edge',
      name: "edge-bottom-back",
      direction: new THREE.Vector3(0, -1, -1).normalize(),
      position: [0, -0.575, -0.575],
      size: [1.15, 0.05, 0.05],
    },
    {
      type: 'edge',
      name: "edge-bottom-left",
      direction: new THREE.Vector3(-1, -1, 0).normalize(),
      position: [-0.575, -0.575, 0],
      size: [0.05, 0.05, 1.15],
    },
    {
      type: 'edge',
      name: "edge-bottom-right",
      direction: new THREE.Vector3(1, -1, 0).normalize(),
      position: [0.575, -0.575, 0],
      size: [0.05, 0.05, 1.15],
    },
  ], []);

  // ✅ Define 8 CORNER zones
  const cornerDefinitions: ZoneData[] = useMemo(() => [
    {
      type: 'corner',
      name: "corner-top-front-right",
      direction: new THREE.Vector3(1, 1, 1).normalize(),
      position: [0.575, 0.575, 0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-top-front-left",
      direction: new THREE.Vector3(-1, 1, 1).normalize(),
      position: [-0.575, 0.575, 0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-top-back-right",
      direction: new THREE.Vector3(1, 1, -1).normalize(),
      position: [0.575, 0.575, -0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-top-back-left",
      direction: new THREE.Vector3(-1, 1, -1).normalize(),
      position: [-0.575, 0.575, -0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-bottom-front-right",
      direction: new THREE.Vector3(1, -1, 1).normalize(),
      position: [0.575, -0.575, 0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-bottom-front-left",
      direction: new THREE.Vector3(-1, -1, 1).normalize(),
      position: [-0.575, -0.575, 0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-bottom-back-right",
      direction: new THREE.Vector3(1, -1, -1).normalize(),
      position: [0.575, -0.575, -0.575],
      radius: 0.08,
    },
    {
      type: 'corner',
      name: "corner-bottom-back-left",
      direction: new THREE.Vector3(-1, -1, -1).normalize(),
      position: [-0.575, -0.575, -0.575],
      radius: 0.08,
    },
  ], []);

  // ✅ Create interaction layer (26 invisible primitive meshes)
  const interactionGroup = useMemo(() => {
    const group = new THREE.Group();
    
    // Create face interaction planes
    faceDefinitions.forEach(face => {
      const geometry = new THREE.PlaneGeometry(1.15, 1.15);
      const material = new THREE.MeshBasicMaterial({ 
        visible: false,
        side: THREE.DoubleSide 
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...face.position);
      mesh.rotation.set(...(face.rotation || [0, 0, 0]));
      mesh.name = face.name;
      mesh.userData = face;
      group.add(mesh);
    });
    
    // Create edge interaction boxes
    edgeDefinitions.forEach(edge => {
      const geometry = new THREE.BoxGeometry(...(edge.size || [0.1, 0.1, 0.1]));
      const material = new THREE.MeshBasicMaterial({ visible: false });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...edge.position);
      mesh.name = edge.name;
      mesh.userData = edge;
      group.add(mesh);
    });
    
    // Create corner interaction spheres
    cornerDefinitions.forEach(corner => {
      const geometry = new THREE.SphereGeometry(corner.radius || 0.08, 8, 8);
      const material = new THREE.MeshBasicMaterial({ visible: false });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...corner.position);
      mesh.name = corner.name;
      mesh.userData = corner;
      group.add(mesh);
    });
    
    return group;
  }, [faceDefinitions, edgeDefinitions, cornerDefinitions]);

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (isDragging) return;

    setIsDragging(true);
    dragStartPos.current = { x: event.clientX, y: event.clientY };
    gl.domElement.style.cursor = "default";
  };

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    
    if (isDragging) return;
    
    // R3F already did the raycasting! Just check the intersections
    if (event.intersections.length > 0) {
      // Find first object with userData.type
      for (const intersection of event.intersections) {
        if (intersection.object.userData.type) {
          setHoveredZoneData(intersection.object.userData as ZoneData);
          gl.domElement.style.cursor = "default";
          return;
        }
      }
    }
    
    setHoveredZoneData(null);
    gl.domElement.style.cursor = "default";
  };

  const handlePointerUp = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    
    const wasClick =
      dragStartPos.current &&
      Math.abs(event.clientX - dragStartPos.current.x) < 3 &&
      Math.abs(event.clientY - dragStartPos.current.y) < 3;

    if (wasClick && hoveredZoneData && onFaceClick) {
      console.log(`🖱️ ${hoveredZoneData.name} clicked →`, hoveredZoneData.direction);
      onFaceClick(hoveredZoneData.direction);
    }

    setIsDragging(false);
    dragStartPos.current = null;
  };

  const handleCubeEnter = () => {
    if (!isDragging) {
      gl.domElement.style.cursor = "default";
    }
  };

  const handleCubeLeave = () => {
    if (!isDragging) {
      setHoveredZoneData(null);
      gl.domElement.style.cursor = "default";
    }
  };

  return (
    <>
    <group ref={groupRef}>
      {/* Layer 1: Visual STL cube (orange, chamfered, beautiful) - NO INTERACTION */}
      {centeredGeometry && (
        <mesh
          ref={meshRef}
          geometry={centeredGeometry}
          castShadow
          receiveShadow
          scale={1.1}
          raycast={() => null}
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

      {/* Layer 2: Interaction primitives (26 invisible meshes) - ALL EVENTS HERE */}
      <group
        scale={1.16}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerEnter={handleCubeEnter}
        onPointerLeave={handleCubeLeave}
      >
        <primitive object={interactionGroup} ref={interactionGroupRef} />
      </group>

      {/* Layer 4: Wireframe edges */}
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

    {/* Layer 3: Dynamic highlight mesh - OUTSIDE rotating group, in world space */}
    {hoveredZoneData && groupRef.current && (
      <mesh 
        position={(() => {
          const localPos = new THREE.Vector3(...hoveredZoneData.position);
          return groupRef.current!.localToWorld(localPos.clone());
        })()}
        rotation={(() => {
          const worldQuat = new THREE.Quaternion();
          groupRef.current!.getWorldQuaternion(worldQuat);
          
          if (hoveredZoneData.rotation) {
            const zoneEuler = new THREE.Euler(...hoveredZoneData.rotation);
            const zoneQuat = new THREE.Quaternion().setFromEuler(zoneEuler);
            worldQuat.multiply(zoneQuat);
          }
          
          const euler = new THREE.Euler().setFromQuaternion(worldQuat);
          return euler.toArray().slice(0, 3) as [number, number, number];
        })()}
        scale={1.16}
        renderOrder={999}
      >
        {hoveredZoneData.type === 'face' && (
          <planeGeometry args={[1.15, 1.15]} />
        )}
        {hoveredZoneData.type === 'edge' && (
          <boxGeometry args={hoveredZoneData.size} />
        )}
        {hoveredZoneData.type === 'corner' && (
          <sphereGeometry args={[hoveredZoneData.radius, 16, 16]} />
        )}
        <meshBasicMaterial 
          color="#3b82f6" 
          transparent 
          opacity={0.5}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>
    )}
    </>
  );
}
