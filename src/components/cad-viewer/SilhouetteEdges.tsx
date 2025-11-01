import { useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

interface EdgeData {
  vertices: [THREE.Vector3, THREE.Vector3];
  triangles: Array<{
    normal: THREE.Vector3;
    centroid: THREE.Vector3;
  }>;
  angle?: number; // Angle between faces (in degrees)
}

interface SilhouetteEdgesProps {
  geometry: THREE.BufferGeometry;
  mesh?: THREE.Mesh | null;
  updateThreshold?: number;
  staticFeatureEdges: THREE.BufferGeometry;
}

export function SilhouetteEdges({ 
  geometry,
  mesh,
  updateThreshold = 0.1,
  staticFeatureEdges
}: SilhouetteEdgesProps) {
  const { camera } = useThree();
  const lastCameraPos = useRef<THREE.Vector3>(new THREE.Vector3());
  const lastCameraQuat = useRef<THREE.Quaternion>(new THREE.Quaternion());
  const [silhouettePositions, setSilhouettePositions] = useState<Float32Array>(
    new Float32Array(0)
  );

  // Build edge-to-triangle adjacency map (computed once)
  const edgeMap = useMemo(() => {
    return buildEdgeMap(geometry);
  }, [geometry]);

  // Update silhouette edges when camera moves
  useFrame(() => {
    const currentPos = camera.position;
    const currentQuat = camera.quaternion;

    // Check if camera moved or rotated significantly
    const posChanged = currentPos.distanceTo(lastCameraPos.current) > updateThreshold;
    const rotChanged = currentQuat.angleTo(lastCameraQuat.current) > 0.05; // ~3 degrees

    if (posChanged || rotChanged) {
      // Transform camera position to local space for accurate view direction
      let localCameraPos = currentPos;
      if (mesh) {
        const worldToLocal = mesh.matrixWorld.clone().invert();
        localCameraPos = currentPos.clone().applyMatrix4(worldToLocal);
      }
      
      const silhouettes = computeSilhouetteEdges(edgeMap, localCameraPos);
      console.log("🔄 Silhouette update:", silhouettes.length / 6, "segments");
      
      setSilhouettePositions(silhouettes);
      
      lastCameraPos.current.copy(currentPos);
      lastCameraQuat.current.copy(currentQuat);
    }
  });

  return (
    <group>
      {/* Static feature edges (sharp angles, circles, boundaries) - ALWAYS visible */}
      <lineSegments geometry={staticFeatureEdges}>
        <lineBasicMaterial 
          color="#000000" 
          toneMapped={false}
          polygonOffset={true}
          polygonOffsetFactor={-1}
          polygonOffsetUnits={-1}
        />
      </lineSegments>
      
      {/* Dynamic silhouette edges (smooth surfaces only) - VIEW DEPENDENT */}
      {silhouettePositions.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={silhouettePositions.length / 3}
              array={silhouettePositions}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial 
            color="#000000" 
            toneMapped={false}
            polygonOffset={true}
            polygonOffsetFactor={-1}
            polygonOffsetUnits={-1}
          />
        </lineSegments>
      )}
    </group>
  );
}

// ============= Helper Functions =============

/**
 * Build a map of edges to their adjacent triangles
 * Key: "x1,y1,z1|x2,y2,z2" (sorted vertices)
 * Value: Array of triangles sharing this edge
 */
function buildEdgeMap(geometry: THREE.BufferGeometry): Map<string, EdgeData> {
  const edgeMap = new Map<string, EdgeData>();
  const positions = geometry.attributes.position.array as Float32Array;
  const indices = geometry.index?.array;

  if (!indices) {
    console.warn("SilhouetteEdges: No indices found in geometry");
    return edgeMap;
  }

  // Process each triangle
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i] * 3;
    const i1 = indices[i + 1] * 3;
    const i2 = indices[i + 2] * 3;

    const v0 = new THREE.Vector3(positions[i0], positions[i0 + 1], positions[i0 + 2]);
    const v1 = new THREE.Vector3(positions[i1], positions[i1 + 1], positions[i1 + 2]);
    const v2 = new THREE.Vector3(positions[i2], positions[i2 + 1], positions[i2 + 2]);

    // Compute triangle normal
    const edge1 = new THREE.Vector3().subVectors(v1, v0);
    const edge2 = new THREE.Vector3().subVectors(v2, v0);
    const normal = new THREE.Vector3().crossVectors(edge1, edge2).normalize();

    // Compute centroid
    const centroid = new THREE.Vector3()
      .add(v0)
      .add(v1)
      .add(v2)
      .multiplyScalar(1 / 3);

    const triangleData = { normal, centroid };

    // Add all three edges of this triangle
    addEdgeToMap(edgeMap, v0, v1, triangleData);
    addEdgeToMap(edgeMap, v1, v2, triangleData);
    addEdgeToMap(edgeMap, v2, v0, triangleData);
  }

  // Pre-compute angles for edges with 2 triangles (feature edge detection)
  edgeMap.forEach((edgeData) => {
    if (edgeData.triangles.length === 2) {
      const tri1 = edgeData.triangles[0];
      const tri2 = edgeData.triangles[1];
      const dotProduct = Math.max(-1, Math.min(1, tri1.normal.dot(tri2.normal)));
      const angle = Math.acos(dotProduct);
      edgeData.angle = angle * (180 / Math.PI);
    }
  });

  return edgeMap;
}

/**
 * Add an edge to the edge map
 */
function addEdgeToMap(
  edgeMap: Map<string, EdgeData>,
  v1: THREE.Vector3,
  v2: THREE.Vector3,
  triangleData: { normal: THREE.Vector3; centroid: THREE.Vector3 }
) {
  const key = makeEdgeKey(v1, v2);
  
  if (!edgeMap.has(key)) {
    edgeMap.set(key, {
      vertices: [v1.clone(), v2.clone()],
      triangles: [],
    });
  }
  
  edgeMap.get(key)!.triangles.push(triangleData);
}

/**
 * Create a unique key for an edge (order-independent)
 */
function makeEdgeKey(v1: THREE.Vector3, v2: THREE.Vector3): string {
  // Sort vertices to ensure consistent key regardless of order
  const vertices = [v1, v2].sort((a, b) => {
    if (a.x !== b.x) return a.x - b.x;
    if (a.y !== b.y) return a.y - b.y;
    return a.z - b.z;
  });

  return `${vertices[0].x.toFixed(6)},${vertices[0].y.toFixed(6)},${vertices[0].z.toFixed(6)}|${vertices[1].x.toFixed(6)},${vertices[1].y.toFixed(6)},${vertices[1].z.toFixed(6)}`;
}

/**
 * Compute silhouette edges based on camera position
 * A silhouette edge is one where one adjacent face is front-facing
 * and the other is back-facing (or it's a boundary edge)
 */
function computeSilhouetteEdges(
  edgeMap: Map<string, EdgeData>,
  cameraPos: THREE.Vector3
): Float32Array {
  const silhouetteEdges: number[] = [];

  edgeMap.forEach((edgeData) => {
    const { vertices, triangles } = edgeData;

    // Boundary edges (only 1 triangle) - all boundaries in staticFeatureEdges
    if (triangles.length === 1) {
      // Skip all boundary edges - they're already rendered by staticFeatureEdges
      // Note: We cannot distinguish sharp vs smooth boundaries without angle data
      // (angle is only computed for edges with 2 adjacent triangles)
      return;
    }

    // For interior edges (2 triangles), check silhouette only on SMOOTH surfaces
    if (triangles.length === 2) {
      // Skip feature edges (sharp angles >= 30°) - already in staticFeatureEdges
      if (edgeData.angle !== undefined && edgeData.angle >= 30) {
        return;
      }
      
      const tri1 = triangles[0];
      const tri2 = triangles[1];

      // Use edge midpoint for accurate silhouette detection
      const edgeMidpoint = new THREE.Vector3()
        .addVectors(vertices[0], vertices[1])
        .multiplyScalar(0.5);
      
      const viewDir = new THREE.Vector3()
        .subVectors(cameraPos, edgeMidpoint)
        .normalize();

      // Check if faces are front or back facing with smooth threshold
      const dot1 = tri1.normal.dot(viewDir);
      const dot2 = tri2.normal.dot(viewDir);
      
      const threshold = 0.01; // Small threshold to avoid flickering
      const isFrontFacing1 = dot1 > threshold;
      const isFrontFacing2 = dot2 > threshold;

      // Silhouette edge: one front, one back (view-dependent only)
      const isSilhouetteEdge = isFrontFacing1 !== isFrontFacing2;

      // ONLY show true silhouette edges (no feature edges)
      if (isSilhouetteEdge) {
        silhouetteEdges.push(
          vertices[0].x, vertices[0].y, vertices[0].z,
          vertices[1].x, vertices[1].y, vertices[1].z
        );
      }
    }

    // Skip edges with more than 2 triangles (non-manifold geometry)
  });

  return new Float32Array(silhouetteEdges);
}
