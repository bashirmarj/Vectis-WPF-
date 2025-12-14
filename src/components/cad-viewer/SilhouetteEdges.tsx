import { useMemo, useRef, useState, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

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

  // NEW: State for visible feature edges (computed per-frame based on face normals)
  const [visibleFeaturePositions, setVisibleFeaturePositions] = useState<Float32Array>(
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

  // Build edge map from mesh geometry (once, memoized)
  const edgeMap = useMemo(() => {
    if (!geometry?.attributes?.position) return null;
    return buildEdgeMap(geometry);
  }, [geometry]);

  // State for silhouette edges (view-dependent outlines on smooth surfaces)
  const [silhouettePositions, setSilhouettePositions] = useState<Float32Array>(
    new Float32Array(0)
  );

  // Update edges when camera moves
  useFrame(() => {
    // Skip if actively dragging (only in solid mode)
    if (isDragging.current && performanceConfig.pauseOnDrag) {
      return;
    }

    // Frame skipping
    frameCount.current++;
    if (frameCount.current % performanceConfig.frameSkip !== 0) {
      return;
    }

    const currentPos = camera.position;
    const currentQuat = camera.quaternion;

    // Check if camera moved or rotated
    const posChanged = currentPos.distanceTo(lastCameraPos.current) > performanceConfig.positionThreshold;
    const rotChanged = currentQuat.angleTo(lastCameraQuat.current) > performanceConfig.rotationThreshold;

    if (posChanged || rotChanged) {
      // Transform camera position to local space for accurate view direction
      let localCameraPos = currentPos;
      if (mesh) {
        const worldToLocal = mesh.matrixWorld.clone().invert();
        localCameraPos = currentPos.clone().applyMatrix4(worldToLocal);
      }

      // Compute visible feature edges based on backend normals
      if (!showHiddenEdges && staticFeatureEdges?.attributes?.position) {
        const visibleEdges = computeVisibleFeatureEdges(
          staticFeatureEdges,
          localCameraPos
        );
        setVisibleFeaturePositions(visibleEdges);
      }

      // Compute silhouette edges from mesh geometry (outline on smooth surfaces)
      if (!showHiddenEdges && edgeMap) {
        const silhouettes = computeSilhouetteEdges(edgeMap, localCameraPos);
        setSilhouettePositions(silhouettes);
      }

      lastCameraPos.current.copy(currentPos);
      lastCameraQuat.current.copy(currentQuat);
    }
  });

  // Create geometry for visible feature edges
  const visibleFeatureGeometry = useMemo(() => {
    if (visibleFeaturePositions.length === 0) return null;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(visibleFeaturePositions, 3));
    geo.computeBoundingSphere();

    return geo;
  }, [visibleFeaturePositions]);

  // Create geometry for silhouette edges
  const silhouetteGeometry = useMemo(() => {
    if (silhouettePositions.length === 0) return null;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(silhouettePositions, 3));
    geo.computeBoundingSphere();

    return geo;
  }, [silhouettePositions]);

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (visibleFeatureGeometry) {
        visibleFeatureGeometry.dispose();
      }
      if (silhouetteGeometry) {
        silhouetteGeometry.dispose();
      }
    };
  }, [visibleFeatureGeometry, silhouetteGeometry]);

  // Null safety check
  if (!geometry?.attributes?.position) {
    return null;
  }

  return (
    <group>
      {/* Static feature edges - show ALL when showHiddenEdges=true (any display mode) */}
      {showHiddenEdges && staticFeatureEdges?.attributes?.position && (
        <lineSegments
          geometry={staticFeatureEdges}
          frustumCulled={true}
          key="static-edges-all"
        >
          <lineBasicMaterial
            color="#000000"
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </lineSegments>
      )}

      {/* Visible feature edges - computed per-frame based on backend data (any display mode) */}
      {!showHiddenEdges && visibleFeatureGeometry && (
        <lineSegments
          geometry={visibleFeatureGeometry}
          frustumCulled={true}
          key="visible-feature-edges"
        >
          <lineBasicMaterial
            color="#000000"
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </lineSegments>
      )}

      {/* Silhouette edges - outline on smooth curved surfaces (cylinders, etc.) */}
      {!showHiddenEdges && silhouetteGeometry && (
        <lineSegments
          geometry={silhouetteGeometry}
          frustumCulled={true}
          key="silhouette-edges"
        >
          <lineBasicMaterial
            color="#000000"
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </lineSegments>
      )}

      {/* Fallback: Show all edges if no normals computed yet */}
      {!showHiddenEdges && !visibleFeatureGeometry && staticFeatureEdges?.attributes?.position && (
        <lineSegments
          geometry={staticFeatureEdges}
          frustumCulled={true}
          key="fallback-edges"
        >
          <lineBasicMaterial
            color="#000000"
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </lineSegments>
      )}
    </group>
  );
}

// ============= Helper Functions =============

/**
 * Compute which feature edges are visible based on BACKEND PROVIDED normals.
 * An edge is visible if AT LEAST ONE of its adjacent faces is front-facing.
 * This checks the "faceNormal1" and "faceNormal2" attributes directly.
 */
function computeVisibleFeatureEdges(
  staticFeatureEdges: THREE.BufferGeometry,
  cameraPos: THREE.Vector3
): Float32Array {
  const positions = staticFeatureEdges.attributes.position.array as Float32Array;

  // Backend provided normals (set in MeshModel)
  const n1Attr = staticFeatureEdges.attributes.faceNormal1;
  const n2Attr = staticFeatureEdges.attributes.faceNormal2;

  // FALLBACK: If no backend normals, return ALL edges as visible
  // This ensures edges always render even if normal data isn't available
  if (!n1Attr || !n2Attr) {
    return positions.slice(); // Return copy of all positions
  }

  const n1Array = n1Attr.array as Float32Array;
  const n2Array = n2Attr.array as Float32Array;
  const numSegments = positions.length / 6;

  // Pre-allocate (most edges will be visible)
  const visiblePositions = new Float32Array(positions.length);
  let posIndex = 0;

  // Reusable vectors
  const edgeMidpoint = new THREE.Vector3();
  const viewDir = new THREE.Vector3();
  const normal1 = new THREE.Vector3();
  const normal2 = new THREE.Vector3();

  for (let i = 0; i < numSegments; i++) {
    const offset = i * 6;

    // Vertex 1
    const v1x = positions[offset];
    const v1y = positions[offset + 1];
    const v1z = positions[offset + 2];

    // Vertex 2
    const v2x = positions[offset + 3];
    const v2y = positions[offset + 4];
    const v2z = positions[offset + 5];

    // Compute view direction
    edgeMidpoint.set(v1x + v2x, v1y + v2y, v1z + v2z).multiplyScalar(0.5);
    viewDir.subVectors(cameraPos, edgeMidpoint).normalize();

    // Get Backend Normals for this segment
    // Note: We stored them per-vertex (MeshModel.tsx), so we can just grab from the first vertex of the segment
    normal1.set(n1Array[offset], n1Array[offset + 1], n1Array[offset + 2]);
    normal2.set(n2Array[offset], n2Array[offset + 1], n2Array[offset + 2]);

    // Check visibility
    // Visible if ANY adjacent face is front-facing (dot > 0)
    // Using a small epsilon 0.01 to avoid flickering on exactly 90-degree edges
    const isFace1Visible = normal1.dot(viewDir) > 0.01;
    const isFace2Visible = normal2.dot(viewDir) > 0.01;

    // Check if this is a "boundary" edge (only 1 face)
    // If n2 is zero-length, it implies only 1 adjacent face provided
    const isBoundary = normal2.lengthSq() < 0.001;

    if (isFace1Visible || isFace2Visible || (isBoundary && isFace1Visible)) {
      visiblePositions[posIndex++] = v1x;
      visiblePositions[posIndex++] = v1y;
      visiblePositions[posIndex++] = v1z;
      visiblePositions[posIndex++] = v2x;
      visiblePositions[posIndex++] = v2y;
      visiblePositions[posIndex++] = v2z;
    }
  }

  // Trim to actual size
  return visiblePositions.slice(0, posIndex);
}

/**
 * Build an edge map from mesh geometry for silhouette edge computation.
 * Maps each edge (by vertex pair key) to its adjacent triangle normals.
 */
function buildEdgeMap(geometry: THREE.BufferGeometry): Map<string, { normal1: THREE.Vector3; normal2: THREE.Vector3 | null; v1: THREE.Vector3; v2: THREE.Vector3 }> {
  const positions = geometry.attributes.position.array as Float32Array;
  const indices = geometry.index?.array;
  const edgeMap = new Map<string, { normal1: THREE.Vector3; normal2: THREE.Vector3 | null; v1: THREE.Vector3; v2: THREE.Vector3 }>();

  const getVertex = (index: number) => {
    const i = index * 3;
    return new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]);
  };

  const makeEdgeKey = (i1: number, i2: number) => {
    return i1 < i2 ? `${i1}_${i2}` : `${i2}_${i1}`;
  };

  const computeTriangleNormal = (v0: THREE.Vector3, v1: THREE.Vector3, v2: THREE.Vector3) => {
    const edge1 = new THREE.Vector3().subVectors(v1, v0);
    const edge2 = new THREE.Vector3().subVectors(v2, v0);
    return new THREE.Vector3().crossVectors(edge1, edge2).normalize();
  };

  const processTriangle = (i0: number, i1: number, i2: number) => {
    const v0 = getVertex(i0);
    const v1 = getVertex(i1);
    const v2 = getVertex(i2);
    const normal = computeTriangleNormal(v0, v1, v2);

    // Process each edge of the triangle
    const edges = [[i0, i1, v0, v1], [i1, i2, v1, v2], [i2, i0, v2, v0]] as const;
    for (const [idx1, idx2, vert1, vert2] of edges) {
      const key = makeEdgeKey(idx1, idx2);
      const existing = edgeMap.get(key);
      if (existing) {
        existing.normal2 = normal.clone();
      } else {
        edgeMap.set(key, { normal1: normal.clone(), normal2: null, v1: vert1.clone(), v2: vert2.clone() });
      }
    }
  };

  if (indices) {
    for (let i = 0; i < indices.length; i += 3) {
      processTriangle(indices[i], indices[i + 1], indices[i + 2]);
    }
  } else {
    const numVertices = positions.length / 3;
    for (let i = 0; i < numVertices; i += 3) {
      processTriangle(i, i + 1, i + 2);
    }
  }

  return edgeMap;
}

/**
 * Compute silhouette edges from mesh geometry.
 * Silhouette edges are where one adjacent triangle is front-facing and the other is back-facing.
 * These are the "outline" edges visible on smooth curved surfaces like cylinders.
 */
function computeSilhouetteEdges(
  edgeMap: Map<string, { normal1: THREE.Vector3; normal2: THREE.Vector3 | null; v1: THREE.Vector3; v2: THREE.Vector3 }>,
  cameraPos: THREE.Vector3
): Float32Array {
  const silhouetteEdges: number[] = [];
  const viewDir = new THREE.Vector3();

  edgeMap.forEach((edge) => {
    // Skip boundary edges (only 1 adjacent face) - these are already feature edges
    if (!edge.normal2) return;

    // Compute view direction from edge midpoint
    const midpoint = new THREE.Vector3().addVectors(edge.v1, edge.v2).multiplyScalar(0.5);
    viewDir.subVectors(cameraPos, midpoint).normalize();

    // Check if the two adjacent faces have opposite facing
    const dot1 = edge.normal1.dot(viewDir);
    const dot2 = edge.normal2.dot(viewDir);

    // Silhouette edge: one face front-facing, one back-facing
    // Using small threshold to avoid flickering
    const isSilhouette = (dot1 > 0.001 && dot2 < -0.001) || (dot1 < -0.001 && dot2 > 0.001);

    if (isSilhouette) {
      silhouetteEdges.push(
        edge.v1.x, edge.v1.y, edge.v1.z,
        edge.v2.x, edge.v2.y, edge.v2.z
      );
    }
  });

  return new Float32Array(silhouetteEdges);
}

