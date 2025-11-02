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
 * Get face classification from raycast intersection
 */
export function getFaceFromIntersection(
  intersection: THREE.Intersection,
  meshData: MeshData | null
): BackendFaceClassification | null {
  if (!meshData?.vertex_face_ids || !meshData.face_classifications) {
    return null;
  }

  const faceIndex = intersection.faceIndex;
  if (faceIndex === undefined) return null;

  // Each face has 3 vertices (triangle), get face_id from first vertex
  const vertexIndex = faceIndex * 3;
  const faceId = meshData.vertex_face_ids[vertexIndex];

  return meshData.face_classifications.find(f => f.face_id === faceId) || null;
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
