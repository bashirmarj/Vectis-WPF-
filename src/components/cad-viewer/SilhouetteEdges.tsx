import { useMemo, useRef, useState, useEffect } from "react";
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
  showHiddenEdges?: boolean;
  controlsRef?: React.RefObject<any>;
  displayMode?: "solid" | "wireframe" | "translucent";
}

export function SilhouetteEdges({
  geometry,
  mesh,
  updateThreshold = 0.5,
  staticFeatureEdges,
  showHiddenEdges = false,
  controlsRef,
  displayMode = "solid"
}: SilhouetteEdgesProps) {
  const { camera } = useThree();
  const lastCameraPos = useRef<THREE.Vector3>(new THREE.Vector3());
  const lastCameraQuat = useRef<THREE.Quaternion>(new THREE.Quaternion());
  const [silhouettePositions, setSilhouettePositions] = useState<Float32Array>(
    new Float32Array(0)
  );

  // Performance optimizations
  const frameCount = useRef(0);
  const isDragging = useRef(false);

  // Mode-aware performance configuration
  const performanceConfig = useMemo(() => {
    if (displayMode === "wireframe") {
      return {
        frameSkip: 2,           // Update every 2 frames - 50% reduction
        positionThreshold: 0.3, // Slightly less sensitive
        rotationThreshold: 0.05, // ~3 degrees
        pauseOnDrag: false      // Keep updating
      };
    }

    // Solid mode (default)
    return {
      frameSkip: 3,            // Update every 3 frames
      positionThreshold: 0.5,  // Less sensitive
      rotationThreshold: 0.1,  // ~6 degrees
      pauseOnDrag: true        // Pause for smooth dragging
    };
  }, [displayMode]);

  // Build edge-to-triangle adjacency map (computed once)
  // Add null safety check for geometry
  const edgeMap = useMemo(() => {
    if (!geometry?.attributes?.position) {
      return new Map();
    }
    return buildEdgeMap(geometry);
  }, [geometry]);

  // Build a Set of all static feature edge keys for fast duplicate detection
  const staticEdgeKeys = useMemo(() => {
    const keys = new Set<string>();
    // Add null safety check for staticFeatureEdges
    const positions = staticFeatureEdges?.attributes?.position?.array as Float32Array;

    if (positions) {
      // staticFeatureEdges is a LineSegments geometry (pairs of vertices)
      for (let i = 0; i < positions.length; i += 6) {
        const v1 = new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]);
        const v2 = new THREE.Vector3(positions[i + 3], positions[i + 4], positions[i + 5]);
        keys.add(makeEdgeKey(v1, v2));
      }
    }

    return keys;
  }, [staticFeatureEdges]);

  // Listen to controls drag events
  useEffect(() => {
    if (controlsRef?.current) {
      const controls = controlsRef.current;

      const handleStart = () => {
        isDragging.current = true;
      };

      const handleEnd = () => {
        isDragging.current = false;
        frameCount.current = 0; // Force immediate update
      };

      controls.addEventListener('start', handleStart);
      controls.addEventListener('end', handleEnd);

      return () => {
        controls.removeEventListener('start', handleStart);
        controls.removeEventListener('end', handleEnd);
      };
    }
  }, [controlsRef]);

  // Update silhouette edges when camera moves (optimized)
  useFrame(() => {
    // Skip if actively dragging (only in solid mode)
    if (isDragging.current && performanceConfig.pauseOnDrag) {
      return;
    }

    // Frame skipping: based on mode
    frameCount.current++;
    if (frameCount.current % performanceConfig.frameSkip !== 0) {
      return;
    }

    const currentPos = camera.position;
    const currentQuat = camera.quaternion;

    // Check if camera moved or rotated (mode-aware thresholds)
    const posChanged = currentPos.distanceTo(lastCameraPos.current) > performanceConfig.positionThreshold;
    const rotChanged = currentQuat.angleTo(lastCameraQuat.current) > performanceConfig.rotationThreshold;

    if (posChanged || rotChanged) {
      // Transform camera position to local space for accurate view direction
      let localCameraPos = currentPos;
      if (mesh) {
        const worldToLocal = mesh.matrixWorld.clone().invert();
        localCameraPos = currentPos.clone().applyMatrix4(worldToLocal);
      }

      const silhouettes = computeSilhouetteEdges(edgeMap, localCameraPos, staticEdgeKeys);
      setSilhouettePositions(silhouettes);

      lastCameraPos.current.copy(currentPos);
      lastCameraQuat.current.copy(currentQuat);
    }
  });

  // Create proper geometry with bounding sphere for dynamic silhouettes
  const dynamicSilhouetteGeometry = useMemo(() => {
    if (silhouettePositions.length === 0) return null;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(silhouettePositions, 3));
    geo.computeBoundingSphere(); // CRITICAL: Needed for proper rendering

    return geo;
  }, [silhouettePositions]);

  // Cleanup effect for geometry disposal
  useEffect(() => {
    return () => {
      if (dynamicSilhouetteGeometry) {
        dynamicSilhouetteGeometry.dispose();
      }
    };
  }, [dynamicSilhouetteGeometry]);

  // Null safety check - don't render if no valid data
  if (!geometry?.attributes?.position) {
    console.warn("⚠️ SilhouetteEdges: No valid geometry data available");
    return null;
  }

  return (
    <group>
      {/* Static feature edges - ONLY in wireframe mode (solid mode uses MeshModel edges) */}
      {displayMode === "wireframe" && staticFeatureEdges?.attributes?.position && (
        <lineSegments
          geometry={staticFeatureEdges}
          frustumCulled={true}
          key={`static-edges-${showHiddenEdges}`}
        >
          <lineBasicMaterial
            color="#000000"
            toneMapped={false}
            polygonOffset={true}
            polygonOffsetFactor={-10}
            polygonOffsetUnits={-10}
            depthTest={!showHiddenEdges}
            depthWrite={false}
          />
        </lineSegments>
      )}

      {/* Dynamic silhouette edges (smooth surfaces only) - VIEW DEPENDENT */}
      {dynamicSilhouetteGeometry && (
        <lineSegments
          geometry={dynamicSilhouetteGeometry}
          frustumCulled={true}
          key={`dynamic-edges-${showHiddenEdges}`}
        >
          <lineBasicMaterial
            color="#000000"
            toneMapped={false}
            polygonOffset={true}
            polygonOffsetFactor={-10}
            polygonOffsetUnits={-10}
            depthTest={!showHiddenEdges}
            depthWrite={false}
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

  if (!geometry?.attributes?.position) {
    console.warn("SilhouetteEdges: No position attribute found in geometry");
    return edgeMap;
  }

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
  cameraPos: THREE.Vector3,
  staticEdgeKeys: Set<string>
): Float32Array {
  // Pre-allocate array (30% of edges are typically silhouettes)
  const estimatedSize = Math.floor(edgeMap.size * 0.3) * 6;
  const positions = new Float32Array(estimatedSize);
  let posIndex = 0;

  // Reusable vectors (avoid allocations)
  const edgeMidpoint = new THREE.Vector3();
  const viewDir = new THREE.Vector3();

  edgeMap.forEach((edgeData, edgeKey) => {
    const { vertices, triangles } = edgeData;

    // Skip if already in static edges (fast Set lookup)
    if (staticEdgeKeys.has(edgeKey)) {
      return;
    }

    // Skip boundary edges (in static features)
    if (triangles.length !== 2) {
      return;
    }

    // Skip feature edges (already in static)
    if (edgeData.angle !== undefined && edgeData.angle >= 30) {
      return;
    }

    const tri1 = triangles[0];
    const tri2 = triangles[1];

    // Inline midpoint calculation (reuse vector)
    edgeMidpoint.addVectors(vertices[0], vertices[1]).multiplyScalar(0.5);

    // Inline view direction (reuse vector)
    viewDir.subVectors(cameraPos, edgeMidpoint).normalize();

    // Dot products
    const dot1 = tri1.normal.dot(viewDir);
    const dot2 = tri2.normal.dot(viewDir);

    const threshold = 0.01;
    const isFrontFacing1 = dot1 > threshold;
    const isFrontFacing2 = dot2 > threshold;

    // Silhouette edge: one front, one back
    if (isFrontFacing1 !== isFrontFacing2) {
      // Direct array assignment (faster than push)
      if (posIndex + 5 < positions.length) {
        positions[posIndex++] = vertices[0].x;
        positions[posIndex++] = vertices[0].y;
        positions[posIndex++] = vertices[0].z;
        positions[posIndex++] = vertices[1].x;
        positions[posIndex++] = vertices[1].y;
        positions[posIndex++] = vertices[1].z;
      }
    }
  });

  // Trim to actual size
  return positions.slice(0, posIndex);
}
