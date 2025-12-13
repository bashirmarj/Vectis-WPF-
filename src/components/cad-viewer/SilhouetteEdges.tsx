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

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (visibleFeatureGeometry) {
        visibleFeatureGeometry.dispose();
      }
    };
  }, [visibleFeatureGeometry]);

  // Null safety check
  if (!geometry?.attributes?.position) {
    return null;
  }

  return (
    <group>
      {/* Static feature edges - show ALL when showHiddenEdges=true */}
      {displayMode === "wireframe" && showHiddenEdges && staticFeatureEdges?.attributes?.position && (
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

      {/* Visible feature edges - computed per-frame based on backend normals */}
      {displayMode === "wireframe" && !showHiddenEdges && visibleFeatureGeometry && (
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

  // STRICT PRODUCTION REQUIREMENT: 
  // If no backend normals are provided, we return empty.
  // We do NOT guess or compute connectivity on the frontend.
  if (!n1Attr || !n2Attr) {
    return new Float32Array(0);
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
