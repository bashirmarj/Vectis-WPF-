import * as THREE from 'three';

export interface MarkerData {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  intersection: THREE.Intersection;
}

/**
 * Extract the world-space normal of the face at the intersection point
 */
export function getFaceWorldNormal(intersection: THREE.Intersection): THREE.Vector3 {
  const normalMatrix = new THREE.Matrix4();
  intersection.object.updateWorldMatrix(true, false);
  normalMatrix.extractRotation(intersection.object.matrixWorld);
  
  const faceNormal = intersection.face!.normal.clone();
  faceNormal.applyMatrix4(normalMatrix);
  faceNormal.normalize();
  
  return faceNormal;
}

/**
 * Calculate measurement values between two markers
 */
export function calculateMarkerValues(
  markerA: MarkerData,
  markerB: MarkerData
): {
  pointsDistance: number;
  parallelFacesDistance: number | null;
  facesAngle: number;
} {
  // Calculate point-to-point distance
  const pointsDistance = markerA.position.distanceTo(markerB.position);
  
  // Calculate angle between face normals
  const facesAngle = markerA.normal.angleTo(markerB.normal);
  
  // Convert to degrees
  const facesAngleDegrees = (facesAngle * 180) / Math.PI;
  
  // Check if faces are parallel (angle ≈ 0° or ≈ 180°)
  let parallelFacesDistance: number | null = null;
  const parallelThreshold = 0.01; // ~0.57 degrees
  const isParallel = 
    Math.abs(facesAngle) < parallelThreshold || 
    Math.abs(facesAngle - Math.PI) < parallelThreshold;
  
  if (isParallel) {
    // Calculate perpendicular distance between parallel faces
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
      markerA.normal,
      markerA.position
    );
    parallelFacesDistance = Math.abs(plane.distanceToPoint(markerB.position));
  }
  
  return {
    pointsDistance,
    parallelFacesDistance,
    facesAngle: facesAngleDegrees
  };
}
