import * as THREE from "three";

/**
 * Tagged edge segment from backend with feature_id
 */
export interface TaggedFeatureEdge {
  feature_id: number;
  start: [number, number, number];
  end: [number, number, number];
  type: 'line' | 'circle' | 'arc';
  diameter?: number;
  radius?: number;
  length?: number;
}

/**
 * Find edge by direct feature_id lookup using clicked line segment.
 * This replaces unreliable distance-based matching with exact segment matching.
 * 
 * @param clickedEdge - The line segment that was clicked
 * @param taggedEdges - Array of tagged edge segments from backend
 * @returns Tagged edge if found, undefined otherwise
 */
export function findEdgeByFeatureId(
  clickedEdge: THREE.Line3,
  taggedEdges?: TaggedFeatureEdge[]
): TaggedFeatureEdge | undefined {
  if (!taggedEdges || taggedEdges.length === 0) return undefined;

  const threshold = 0.5; // Increased from 0.01 to 0.5mm for Line2 geometry tolerance

  // Find tagged edge segment that matches clicked edge
  for (const tagged of taggedEdges) {
    const taggedStart = new THREE.Vector3(...tagged.start);
    const taggedEnd = new THREE.Vector3(...tagged.end);

    // Check if segments match (handle both directions)
    const startMatch = clickedEdge.start.distanceTo(taggedStart) < threshold;
    const endMatch = clickedEdge.end.distanceTo(taggedEnd) < threshold;
    const startMatchReverse = clickedEdge.start.distanceTo(taggedEnd) < threshold;
    const endMatchReverse = clickedEdge.end.distanceTo(taggedStart) < threshold;

    if ((startMatch && endMatch) || (startMatchReverse && endMatchReverse)) {
      return tagged;
    }
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
export function getFeatureSegments(
  featureId: number,
  taggedEdges?: TaggedFeatureEdge[]
): TaggedFeatureEdge[] {
  if (!taggedEdges) return [];
  return taggedEdges.filter(edge => edge.feature_id === featureId);
}
