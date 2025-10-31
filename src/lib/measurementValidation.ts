import * as THREE from "three";
import { BackendEdgeClassification, MeasurementValidation } from "./measurementUtils";

/**
 * Validate a frontend measurement against backend ground truth
 * 
 * @param frontendValue - Measurement calculated by frontend from tessellation
 * @param segmentCount - Number of segments used in frontend calculation
 * @param backendClassification - Ground truth from backend CAD analysis
 * @returns Validation result with confidence score and warnings
 */
export function validateMeasurement(
  frontendValue: number,
  segmentCount: number,
  measurementType: "length" | "diameter" | "radius",
  backendClassification?: BackendEdgeClassification
): MeasurementValidation {
  const warnings: string[] = [];
  
  // No backend data available - can't validate
  if (!backendClassification) {
    return {
      isValid: false,
      confidence: 0.5,
      frontendValue,
      warnings: ["No backend validation data available"],
    };
  }

  // Extract backend value based on measurement type
  let backendValue: number | undefined;
  if (measurementType === "diameter") {
    backendValue = backendClassification.diameter;
  } else if (measurementType === "radius") {
    backendValue = backendClassification.radius;
  } else {
    backendValue = backendClassification.length;
  }

  // Backend doesn't have this measurement type
  if (backendValue === undefined) {
    return {
      isValid: false,
      confidence: 0.3,
      frontendValue,
      warnings: [`Backend has no ${measurementType} data for this feature`],
      backendType: backendClassification.type,
    };
  }

  // Calculate deviation
  const deviationAbs = Math.abs(frontendValue - backendValue);
  const deviation = (deviationAbs / backendValue) * 100;

  // Check segment count matches expected tessellation
  const segmentCountMatch = segmentCount === backendClassification.segment_count;
  if (!segmentCountMatch) {
    warnings.push(
      `Segment count mismatch: expected ${backendClassification.segment_count}, got ${segmentCount}`
    );
  }

  // Determine validation status based on deviation
  let isValid = true;
  let confidence = 1.0;

  if (deviation > 10) {
    // >10% deviation - measurement is invalid
    isValid = false;
    confidence = 0.1;
    warnings.push(`Large deviation: ${deviation.toFixed(1)}% from CAD geometry`);
  } else if (deviation > 2) {
    // 2-10% deviation - measurement is questionable
    isValid = false;
    confidence = 0.5;
    warnings.push(`Moderate deviation: ${deviation.toFixed(1)}% from CAD geometry`);
  } else if (deviation > 0.5) {
    // 0.5-2% deviation - acceptable approximation
    isValid = true;
    confidence = 0.8;
    warnings.push(`Minor deviation: ${deviation.toFixed(2)}% (acceptable)`);
  } else {
    // <0.5% deviation - excellent match
    isValid = true;
    confidence = 1.0;
  }

  return {
    isValid,
    confidence,
    backendValue,
    frontendValue,
    deviation,
    deviationAbs,
    warnings,
    backendType: backendClassification.type,
    segmentCountMatch,
  };
}

/**
 * Find the closest backend edge classification to a given point
 * 
 * @param point - 3D point to match
 * @param edgeClassifications - Array of backend classifications
 * @returns Closest edge classification or undefined
 */
export function findClosestBackendEdge(
  point: THREE.Vector3,
  edgeClassifications?: BackendEdgeClassification[]
): BackendEdgeClassification | undefined {
  if (!edgeClassifications || edgeClassifications.length === 0) {
    return undefined;
  }

  let closestEdge: BackendEdgeClassification | undefined;
  let minDistance = Infinity;

  for (const edge of edgeClassifications) {
    // Calculate distance to edge midpoint
    const startPoint = new THREE.Vector3(...edge.start_point);
    const endPoint = new THREE.Vector3(...edge.end_point);
    const midpoint = new THREE.Vector3().addVectors(startPoint, endPoint).multiplyScalar(0.5);
    
    const distance = point.distanceTo(midpoint);
    
    if (distance < minDistance) {
      minDistance = distance;
      closestEdge = edge;
    }
  }

  return closestEdge;
}

/**
 * Get validation badge color based on confidence
 */
export function getValidationBadgeColor(confidence: number): string {
  if (confidence >= 0.9) return "success";    // Green
  if (confidence >= 0.7) return "warning";    // Yellow
  return "destructive";                       // Red
}

/**
 * Get validation badge text
 */
export function getValidationBadgeText(validation: MeasurementValidation): string {
  if (validation.confidence >= 0.9) {
    return `✓ Validated (±${validation.deviation?.toFixed(2) || 0}%)`;
  }
  if (validation.confidence >= 0.7) {
    return `⚠ Approximation (±${validation.deviation?.toFixed(1) || 0}%)`;
  }
  return `✗ Uncertain (±${validation.deviation?.toFixed(1) || 0}%)`;
}
