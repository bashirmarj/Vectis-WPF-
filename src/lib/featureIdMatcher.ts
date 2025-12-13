import * as THREE from "three";

/**
 * Edge type from backend - supports all OpenCascade curve types
 */
export type EdgeType =
  | "line"
  | "circle"
  | "arc"
  | "ellipse"
  | "hyperbola"
  | "parabola"
  | "bezier"
  | "bspline"
  | "offset"
  | "other";

/**
 * Tagged edge from backend with analytical geometry data (SolidWorks-level precision)
 */
export interface TaggedFeatureEdge {
  feature_id: number;
  type: EdgeType;
  length?: number;
  is_closed?: boolean;
  start: [number, number, number];
  end: [number, number, number];

  // Analytical data for circles/arcs
  center?: [number, number, number];
  radius?: number;
  diameter?: number;
  normal?: [number, number, number];
  start_angle?: number;
  end_angle?: number;
  sweep_angle?: number;
  is_full_circle?: boolean;
  arc_length?: number;

  // Analytical data for ellipses
  major_radius?: number;
  minor_radius?: number;

  // Analytical data for lines
  direction?: [number, number, number];

  // Sparse snap points for mouse interaction (max 12 points per edge)
  snap_points?: [number, number, number][];

  // Topology
  adjacent_face_count?: number;

  // Legacy compatibility
  iso_type?: string;
}

/**
 * Find the closest tagged edge to a given 3D point.
 * Uses snap_points for efficient and accurate matching.
 *
 * @param point - The 3D point to find the closest edge to
 * @param taggedEdges - Array of tagged edges from backend
 * @param maxDistance - Maximum distance threshold in METERS (default 0.002 = 2mm)
 * @returns Tagged edge if found within threshold, undefined otherwise
 */
export function findClosestTaggedEdge(
  point: THREE.Vector3,
  taggedEdges?: TaggedFeatureEdge[],
  maxDistance: number = 0.002, // 2mm in meters - coordinates from backend are in meters
): TaggedFeatureEdge | undefined {
  if (!taggedEdges || taggedEdges.length === 0) {
    return undefined;
  }

  let closestEdge: TaggedFeatureEdge | undefined = undefined;
  let minDistance = maxDistance;

  for (const tagged of taggedEdges) {
    // Use snap_points if available (preferred - more accurate for curves)
    if (tagged.snap_points && tagged.snap_points.length >= 2) {
      // Check distance to each segment in the polyline
      for (let i = 0; i < tagged.snap_points.length - 1; i++) {
        const p1 = tagged.snap_points[i];
        const p2 = tagged.snap_points[i + 1];
        const start = new THREE.Vector3(p1[0], p1[1], p1[2]);
        const end = new THREE.Vector3(p2[0], p2[1], p2[2]);
        const line = new THREE.Line3(start, end);

        const closestOnSegment = new THREE.Vector3();
        line.closestPointToPoint(point, true, closestOnSegment);

        const distance = point.distanceTo(closestOnSegment);
        if (distance < minDistance) {
          minDistance = distance;
          closestEdge = tagged;
        }
      }

      // For closed edges (full circles), also check the closing segment
      if (tagged.is_closed && tagged.snap_points.length > 2) {
        const firstPt = tagged.snap_points[0];
        const lastPt = tagged.snap_points[tagged.snap_points.length - 1];
        const start = new THREE.Vector3(firstPt[0], firstPt[1], firstPt[2]);
        const end = new THREE.Vector3(lastPt[0], lastPt[1], lastPt[2]);
        const line = new THREE.Line3(start, end);

        const closestOnSegment = new THREE.Vector3();
        line.closestPointToPoint(point, true, closestOnSegment);

        const distance = point.distanceTo(closestOnSegment);
        if (distance < minDistance) {
          minDistance = distance;
          closestEdge = tagged;
        }
      }
    } else {
      // Fallback: use start/end points only
      const start = new THREE.Vector3(tagged.start[0], tagged.start[1], tagged.start[2]);
      const end = new THREE.Vector3(tagged.end[0], tagged.end[1], tagged.end[2]);
      const line = new THREE.Line3(start, end);

      const closestOnEdge = new THREE.Vector3();
      line.closestPointToPoint(point, true, closestOnEdge);

      const distance = point.distanceTo(closestOnEdge);
      if (distance < minDistance) {
        minDistance = distance;
        closestEdge = tagged;
      }
    }
  }

  return closestEdge;
}

/**
 * Find edge by direct feature_id lookup using clicked line segment.
 * Enhanced to use snap_points for better matching.
 *
 * @param clickedEdge - The line segment that was clicked
 * @param taggedEdges - Array of tagged edge segments from backend
 * @returns Tagged edge if found, undefined otherwise
 */
export function findEdgeByFeatureId(
  clickedEdge: THREE.Line3,
  taggedEdges?: TaggedFeatureEdge[],
): TaggedFeatureEdge | undefined {
  if (!taggedEdges || taggedEdges.length === 0) {
    console.warn("⚠️ No tagged_edges available for matching");
    return undefined;
  }

  // Use the midpoint of the clicked edge for proximity matching
  const clickedMidpoint = new THREE.Vector3()
    .addVectors(clickedEdge.start, clickedEdge.end)
    .multiplyScalar(0.5);

  // Use the new findClosestTaggedEdge function (0.002m = 2mm threshold)
  return findClosestTaggedEdge(clickedMidpoint, taggedEdges, 0.002);
}

/**
 * Get all segments belonging to the same feature_id
 *
 * @param featureId - The feature ID to search for
 * @param taggedEdges - Array of tagged edge segments
 * @returns Array of all segments with matching feature_id
 */
export function getFeatureSegments(featureId: number, taggedEdges?: TaggedFeatureEdge[]): TaggedFeatureEdge[] {
  if (!taggedEdges) return [];
  return taggedEdges.filter((edge) => edge.feature_id === featureId);
}
