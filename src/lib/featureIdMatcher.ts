import * as THREE from "three";

/**
 * Tagged edge segment from backend with feature_id
 */
export interface TaggedFeatureEdge {
  feature_id: number;
  start: [number, number, number];
  end: [number, number, number];
  type: "line" | "circle" | "arc" | string; // Allow other types for defensive handling
  diameter?: number;
  radius?: number;
  length?: number;
  iso_type?: string; // Backend adds this for UIso/VIso curves
}

/**
 * Find edge by direct feature_id lookup using clicked line segment.
 * Enhanced to handle circular edges with many segments.
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

  // ✅ INCREASED: More tolerant matching for tessellated circular edges
  const exactThreshold = 0.5; // For direct segment matches
  const midpointThreshold = 2.0; // For circular edge segment detection

  // STRATEGY 1: Direct endpoint matching (for line edges and exact segment matches)
  for (const tagged of taggedEdges) {
    const taggedStart = new THREE.Vector3(...tagged.start);
    const taggedEnd = new THREE.Vector3(...tagged.end);

    const startMatch = clickedEdge.start.distanceTo(taggedStart) < exactThreshold;
    const endMatch = clickedEdge.end.distanceTo(taggedEnd) < exactThreshold;
    const startMatchReverse = clickedEdge.start.distanceTo(taggedEnd) < exactThreshold;
    const endMatchReverse = clickedEdge.end.distanceTo(taggedStart) < exactThreshold;

    if ((startMatch && endMatch) || (startMatchReverse && endMatchReverse)) {
      return tagged;
    }
  }

  // STRATEGY 2: Midpoint proximity matching (for circular/arc edges with many segments)
  // This handles cases where we click on a circular edge but the exact segment doesn't match
  const clickedMidpoint = new THREE.Vector3().addVectors(clickedEdge.start, clickedEdge.end).multiplyScalar(0.5);

  let closestFeature: TaggedFeatureEdge | undefined = undefined;
  let closestDist = midpointThreshold;

  for (const tagged of taggedEdges) {
    // Skip line edges for midpoint matching (they should use exact matching)
    if (tagged.type === "line") continue;

    const taggedStart = new THREE.Vector3(...tagged.start);
    const taggedEnd = new THREE.Vector3(...tagged.end);
    const taggedMidpoint = new THREE.Vector3().addVectors(taggedStart, taggedEnd).multiplyScalar(0.5);

    const dist = clickedMidpoint.distanceTo(taggedMidpoint);

    if (dist < closestDist) {
      closestDist = dist;
      closestFeature = tagged;
    }
  }

  if (closestFeature) {
    return closestFeature;
  }

  return undefined;
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
