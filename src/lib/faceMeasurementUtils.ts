import * as THREE from 'three';

// Constants from reference (geometry.js) - EXACT VALUES
export const BigEps = 0.0001;
export const RadDeg = 57.29577951308232;

// Helper function for epsilon comparison
export function isEqualEps(a: number, b: number, eps: number = BigEps): boolean {
  return Math.abs(a - b) < eps;
}

// Port from reference: GetFaceWorldNormal
export function getFaceWorldNormal(intersection: THREE.Intersection): THREE.Vector3 {
  const normalMatrix = new THREE.Matrix4();
  intersection.object.updateWorldMatrix(true, false);
  normalMatrix.extractRotation(intersection.object.matrixWorld);
  
  const faceNormal = intersection.face!.normal.clone();
  faceNormal.applyMatrix4(normalMatrix);
  
  return faceNormal;
}

// Port from reference: CalculateMarkerValues
export interface MarkerValues {
  pointsDistance: number;
  parallelFacesDistance: number | null;
  facesAngle: number;
}

export function calculateMarkerValues(
  aIntersection: THREE.Intersection,
  bIntersection: THREE.Intersection
): MarkerValues {
  const aNormal = getFaceWorldNormal(aIntersection);
  const bNormal = getFaceWorldNormal(bIntersection);
  
  const pointsDistance = aIntersection.point.distanceTo(bIntersection.point);
  const facesAngle = aNormal.angleTo(bNormal);
  
  let parallelFacesDistance: number | null = null;
  
  // Check if faces are parallel (0° or 180°)
  if (
    Math.abs(facesAngle) < BigEps || 
    Math.abs(facesAngle - Math.PI) < BigEps
  ) {
    const aPlane = new THREE.Plane().setFromNormalAndCoplanarPoint(
      aNormal,
      aIntersection.point
    );
    parallelFacesDistance = Math.abs(aPlane.distanceToPoint(bIntersection.point));
  }
  
  return {
    pointsDistance,
    parallelFacesDistance,
    facesAngle,
  };
}
