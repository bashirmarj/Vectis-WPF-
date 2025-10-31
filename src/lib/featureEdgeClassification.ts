import * as THREE from "three";

export interface EdgeClassificationResult {
  type: 'line' | 'circle' | 'arc';
  length?: number;
  diameter?: number;
  radius?: number;
  center?: THREE.Vector3;
  arcLength?: number;
  arcAngle?: number;
  connectedEdges?: THREE.Line3[];
}

/**
 * Find all edges connected to the given edge (within tolerance)
 */
export const findConnectedEdges = (
  edge: THREE.Line3,
  allEdges: THREE.Line3[],
  tolerance: number = 0.01
): THREE.Line3[] => {
  const connected: THREE.Line3[] = [];
  const edgeStart = edge.start;
  const edgeEnd = edge.end;

  for (const otherEdge of allEdges) {
    if (otherEdge === edge) continue;

    const otherStart = otherEdge.start;
    const otherEnd = otherEdge.end;

    // Check if endpoints are connected
    const startToOtherStart = edgeStart.distanceTo(otherStart);
    const startToOtherEnd = edgeStart.distanceTo(otherEnd);
    const endToOtherStart = edgeEnd.distanceTo(otherStart);
    const endToOtherEnd = edgeEnd.distanceTo(otherEnd);

    if (
      startToOtherStart < tolerance ||
      startToOtherEnd < tolerance ||
      endToOtherStart < tolerance ||
      endToOtherEnd < tolerance
    ) {
      connected.push(otherEdge);
    }
  }

  return connected;
};

/**
 * Check if edges form a closed loop
 */
const isClosedLoop = (edges: THREE.Line3[], tolerance: number = 0.01): boolean => {
  if (edges.length < 3) return false;

  // Walk through edges to see if they form a continuous chain back to start
  const visited = new Set<THREE.Line3>();
  let current = edges[0];
  visited.add(current);
  let currentEnd = current.end.clone();

  for (let i = 1; i < edges.length; i++) {
    let foundNext = false;
    
    for (const edge of edges) {
      if (visited.has(edge)) continue;
      
      if (currentEnd.distanceTo(edge.start) < tolerance) {
        visited.add(edge);
        currentEnd = edge.end.clone();
        foundNext = true;
        break;
      } else if (currentEnd.distanceTo(edge.end) < tolerance) {
        visited.add(edge);
        currentEnd = edge.start.clone();
        foundNext = true;
        break;
      }
    }
    
    if (!foundNext) return false;
  }

  // Check if end connects back to start
  return currentEnd.distanceTo(edges[0].start) < tolerance;
};

/**
 * Calculate center point from a set of edges forming a circle
 */
const calculateCircleCenter = (edges: THREE.Line3[]): THREE.Vector3 | null => {
  if (edges.length < 3) return null;

  // Sample points from edges
  const points: THREE.Vector3[] = [];
  edges.forEach(edge => {
    points.push(edge.start.clone());
    const center = new THREE.Vector3();
    edge.getCenter(center);
    points.push(center);
  });

  // Use first 3 non-collinear points to find center
  const p1 = points[0];
  const p2 = points[Math.floor(points.length / 3)];
  const p3 = points[Math.floor(2 * points.length / 3)];

  // Calculate circle center using perpendicular bisectors
  const mid1 = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
  const mid2 = new THREE.Vector3().addVectors(p2, p3).multiplyScalar(0.5);

  const dir1 = new THREE.Vector3().subVectors(p2, p1).normalize();
  const dir2 = new THREE.Vector3().subVectors(p3, p2).normalize();

  // Get perpendiculars in the plane
  const normal = new THREE.Vector3().crossVectors(dir1, dir2);
  if (normal.length() < 0.01) return null; // Collinear points

  const perp1 = new THREE.Vector3().crossVectors(normal, dir1).normalize();
  const perp2 = new THREE.Vector3().crossVectors(normal, dir2).normalize();

  // Find intersection of perpendicular bisectors
  // This is a simplified approach - using average for robustness
  const center = new THREE.Vector3();
  points.forEach(p => center.add(p));
  center.divideScalar(points.length);

  return center;
};

/**
 * Detect if edges form a circle and return properties
 */
const detectCircleFromEdges = (
  edge: THREE.Line3,
  connectedEdges: THREE.Line3[]
): EdgeClassificationResult | null => {
  const allEdges = [edge, ...connectedEdges];
  
  if (allEdges.length < 4) return null; // Need at least 4 edges for a circle
  if (!isClosedLoop(allEdges)) return null;

  const center = calculateCircleCenter(allEdges);
  if (!center) return null;

  // Calculate radius from all edge points
  const radii: number[] = [];
  allEdges.forEach(e => {
    radii.push(center.distanceTo(e.start));
    radii.push(center.distanceTo(e.end));
    const edgeCenter = new THREE.Vector3();
    e.getCenter(edgeCenter);
    radii.push(center.distanceTo(edgeCenter));
  });

  const avgRadius = radii.reduce((a, b) => a + b, 0) / radii.length;
  
  // Check if all points are within tolerance of average radius (circle test)
  const maxDeviation = Math.max(...radii.map(r => Math.abs(r - avgRadius)));
  if (maxDeviation > avgRadius * 0.05) return null; // 5% tolerance

  return {
    type: 'circle',
    diameter: avgRadius * 2,
    radius: avgRadius,
    center,
    connectedEdges: allEdges,
  };
};

/**
 * Detect if edges form an arc and return properties
 */
const detectArcFromEdges = (
  edge: THREE.Line3,
  connectedEdges: THREE.Line3[]
): EdgeClassificationResult | null => {
  const allEdges = [edge, ...connectedEdges];
  
  if (allEdges.length < 2) return null;
  if (isClosedLoop(allEdges)) return null; // It's a circle, not an arc

  const center = calculateCircleCenter(allEdges);
  if (!center) return null;

  // Calculate radius
  const radii: number[] = [];
  allEdges.forEach(e => {
    radii.push(center.distanceTo(e.start));
    radii.push(center.distanceTo(e.end));
  });

  const avgRadius = radii.reduce((a, b) => a + b, 0) / radii.length;
  
  // Check if all points are within tolerance (arc test)
  const maxDeviation = Math.max(...radii.map(r => Math.abs(r - avgRadius)));
  if (maxDeviation > avgRadius * 0.1) return null; // 10% tolerance for arcs

  // Calculate arc length and angle
  let arcLength = 0;
  allEdges.forEach(e => {
    arcLength += e.distance();
  });

  const arcAngle = (arcLength / avgRadius) * (180 / Math.PI); // degrees

  return {
    type: 'arc',
    radius: avgRadius,
    center,
    arcLength,
    arcAngle,
    connectedEdges: allEdges,
  };
};

/**
 * Classify a feature edge as line, circle, or arc
 */
export const classifyFeatureEdge = (
  edge: THREE.Line3,
  allEdges: THREE.Line3[]
): EdgeClassificationResult => {
  // Find connected edges
  const connectedEdges = findConnectedEdges(edge, allEdges, 0.01);

  // Try to detect circle first
  if (connectedEdges.length >= 3) {
    const circleResult = detectCircleFromEdges(edge, connectedEdges);
    if (circleResult) return circleResult;
  }

  // Try to detect arc
  if (connectedEdges.length >= 1) {
    const arcResult = detectArcFromEdges(edge, connectedEdges);
    if (arcResult) return arcResult;
  }

  // Default to straight line
  return {
    type: 'line',
    length: edge.distance(),
  };
};
