import * as THREE from "three";

interface BackendFaceClassification {
  face_id: number;
  type: "internal" | "external" | "through" | "planar";
  center: [number, number, number];
  normal: [number, number, number];
  area: number;
  surface_type: "cylinder" | "plane" | "other";
  radius?: number;
}

interface MeshData {
  vertex_face_ids?: number[];
  face_classifications?: BackendFaceClassification[];
}

/**
 * Get face classification from raycast intersection using geometric matching
 * This is more robust than triangle index matching
 */
export function getFaceFromIntersection(
  intersection: THREE.Intersection,
  meshData: MeshData | null
): BackendFaceClassification | null {
  if (!meshData?.face_classifications) {
    console.warn("❌ No face classifications available");
    return null;
  }

  if (!intersection.face || !intersection.point) {
    console.warn("❌ Invalid intersection data");
    return null;
  }

  // Get intersection point and normal (in world space)
  const intersectionPoint = intersection.point.clone();
  const intersectionNormal = intersection.face.normal.clone();
  
  // If the mesh has a transform, apply it to the normal
  if (intersection.object) {
    intersectionNormal.transformDirection(intersection.object.matrixWorld);
  }
  intersectionNormal.normalize();

  console.log("🔍 Intersection data:", {
    point: intersectionPoint.toArray(),
    normal: intersectionNormal.toArray(),
    faceIndex: intersection.faceIndex,
  });

  // Find the best matching face by comparing geometry
  let bestMatch: BackendFaceClassification | null = null;
  let bestScore = Infinity;

  for (const face of meshData.face_classifications) {
    const faceCenter = new THREE.Vector3(...face.center);
    const faceNormal = new THREE.Vector3(...face.normal);

    // Calculate distance from intersection point to face center
    const distanceToCenter = intersectionPoint.distanceTo(faceCenter);

    // Calculate normal alignment (dot product)
    // 1.0 = perfect alignment, 0.0 = perpendicular, -1.0 = opposite
    const normalAlignment = Math.abs(intersectionNormal.dot(faceNormal));

    // Combined score: prioritize normal alignment, then distance
    // Lower score is better
    const score = distanceToCenter * (2.0 - normalAlignment);

    console.log(`  Face ${face.face_id} (${face.surface_type}):`, {
      distanceToCenter: distanceToCenter.toFixed(3),
      normalAlignment: normalAlignment.toFixed(3),
      score: score.toFixed(3),
    });

    if (score < bestScore) {
      bestScore = score;
      bestMatch = face;
    }
  }

  if (bestMatch) {
    console.log("✅ Best match:", {
      face_id: bestMatch.face_id,
      surface_type: bestMatch.surface_type,
      score: bestScore.toFixed(3),
    });
  } else {
    console.warn("❌ No matching face found");
  }

  return bestMatch;
}

/**
 * Calculate distance between two faces based on their surface types
 */
export function calculateFaceToFaceDistance(
  face1: BackendFaceClassification,
  face2: BackendFaceClassification
): { distance: number; type: string; description: string } {
  const center1 = new THREE.Vector3(...face1.center);
  const center2 = new THREE.Vector3(...face2.center);
  const normal1 = new THREE.Vector3(...face1.normal);
  const normal2 = new THREE.Vector3(...face2.normal);

  // Case 1: Plane to Plane (Parallel)
  if (face1.surface_type === "plane" && face2.surface_type === "plane") {
    // Check if parallel (normals aligned or opposite)
    const dot = Math.abs(normal1.dot(normal2));
    
    if (dot > 0.99) { // Nearly parallel (< 8 degrees difference)
      // Distance = projection of center-to-center vector onto normal
      const centerVector = new THREE.Vector3().subVectors(center2, center1);
      const distance = Math.abs(centerVector.dot(normal1));
      
      return {
        distance,
        type: "plane-to-plane",
        description: "Parallel Planes"
      };
    } else {
      // Non-parallel planes - use center-to-center distance
      return {
        distance: center1.distanceTo(center2),
        type: "plane-to-plane",
        description: "Non-Parallel Planes"
      };
    }
  }

  // Case 2: Cylinder to Cylinder (Coaxial)
  if (face1.surface_type === "cylinder" && face2.surface_type === "cylinder") {
    if (face1.radius && face2.radius) {
      return {
        distance: Math.abs(face1.radius - face2.radius),
        type: "cylinder-to-cylinder",
        description: `Radial Difference`
      };
    }
  }

  // Case 3: Plane to Cylinder
  if (
    (face1.surface_type === "plane" && face2.surface_type === "cylinder") ||
    (face1.surface_type === "cylinder" && face2.surface_type === "plane")
  ) {
    const planeFace = face1.surface_type === "plane" ? face1 : face2;
    const cylinderFace = face1.surface_type === "cylinder" ? face1 : face2;
    const planeCenter = new THREE.Vector3(...planeFace.center);
    const planeNormal = new THREE.Vector3(...planeFace.normal);
    const cylinderCenter = new THREE.Vector3(...cylinderFace.center);

    // Distance from cylinder center to plane
    const centerVector = new THREE.Vector3().subVectors(cylinderCenter, planeCenter);
    const distance = Math.abs(centerVector.dot(planeNormal));

    return {
      distance,
      type: "plane-to-cylinder",
      description: "Plane to Cylinder"
    };
  }

  // Default: Center-to-center distance
  return {
    distance: center1.distanceTo(center2),
    type: "point-to-point",
    description: "Center-to-Center"
  };
}
