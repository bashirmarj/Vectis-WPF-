import * as THREE from "three";
import { SnapType } from "@/stores/measurementStore";

export interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  vertex_face_ids?: number[];  // Map vertex index to face_id
  face_classifications?: BackendFaceClassification[];  // Detailed face data
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: BackendEdgeClassification[];  // Backend ground truth
  tagged_edges?: BackendTaggedEdge[];  // Exact BREP edges from CAD
}

export interface SnapResult {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  surfaceType: SnapType;
  confidence: number;
  edgeIndex?: number;
  faceIndex?: number;
  metadata?: {
    edgeType?: "line" | "arc" | "circle";
    radius?: number;
    diameter?: number;
    center?: THREE.Vector3;
    startPoint?: THREE.Vector3;
    endPoint?: THREE.Vector3;
    backendFeatureId?: number;
    backendLength?: number;
    backendDiameter?: number;
    backendMatch?: boolean;
    isFullCircle?: boolean;
    sweepAngle?: number;
    arcLength?: number;
  };
}

export interface EdgeClassification {
  type: "line" | "arc" | "circle";
  start: THREE.Vector3;
  end: THREE.Vector3;
  confidence: number;
  length?: number;
  radius?: number;
  center?: THREE.Vector3;
  angle?: number;
}

// Backend classification with ground truth from CAD geometry
export interface BackendEdgeClassification {
  id: number;
  feature_id?: number;  // Unique ID for this geometric feature (optional for backwards compatibility)
  type: "line" | "arc" | "circle";
  start_point: [number, number, number];
  end_point: [number, number, number];
  length?: number;
  radius?: number;
  diameter?: number;
  center?: [number, number, number];
  segment_count: number;
}

export type EdgeType =
  | "line"
  | "circle"
  | "ellipse"
  | "hyperbola"
  | "parabola"
  | "bezier"
  | "bspline"
  | "offset"
  | "other";

export interface BackendTaggedEdge {
  feature_id: number;
  type: EdgeType;
  length: number;
  is_closed: boolean;
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

  // Sparse snap points for mouse interaction
  snap_points?: [number, number, number][];

  // Topology
  adjacent_face_count?: number;

  // Legacy compatibility
  iso_type?: string;
}

// Backend face classification with ground truth  
export interface BackendFaceClassification {
  face_id: number;
  type: "internal" | "external" | "through" | "planar";
  center: [number, number, number];
  normal: [number, number, number];
  area: number;
  surface_type: "cylinder" | "plane" | "other";
  radius?: number;
}

export interface MeasurementValidation {
  isValid: boolean;
  confidence: number;  // 0-1 score
  backendValue?: number;
  frontendValue: number;
  deviation?: number;  // Percentage difference
  deviationAbs?: number;  // Absolute difference
  warnings: string[];
  backendType?: string;
  segmentCountMatch?: boolean;
}

/**
 * Calculate distance between two points with high precision
 * @param p1 First point
 * @param p2 Second point
 * @returns Distance in world units (mm)
 */
export function calculateDistance(p1: THREE.Vector3, p2: THREE.Vector3): number {
  return p1.distanceTo(p2);
}

/**
 * Calculate angle between three points (p1-p2-p3) in degrees
 * @param p1 First point
 * @param p2 Vertex point (angle measured here)
 * @param p3 Third point
 * @returns Angle in degrees
 */
export function calculateAngle(p1: THREE.Vector3, p2: THREE.Vector3, p3: THREE.Vector3): number {
  const v1 = new THREE.Vector3().subVectors(p1, p2);
  const v2 = new THREE.Vector3().subVectors(p3, p2);

  const angle = v1.angleTo(v2);
  return THREE.MathUtils.radToDeg(angle);
}

/**
 * Calculate radius of a circle from 3 points
 * @param p1 First point on circle
 * @param p2 Second point on circle
 * @param p3 Third point on circle
 * @returns Radius in world units (mm)
 */
export function calculateRadius(p1: THREE.Vector3, p2: THREE.Vector3, p3: THREE.Vector3): number {
  // Calculate side lengths of triangle
  const a = p1.distanceTo(p2);
  const b = p2.distanceTo(p3);
  const c = p3.distanceTo(p1);

  // Heron's formula for area
  const s = (a + b + c) / 2;
  const area = Math.sqrt(s * (s - a) * (s - b) * (s - c));

  // Radius from area and sides: R = abc / (4 * area)
  const radius = (a * b * c) / (4 * area);

  return radius;
}

/**
 * Format measurement value with appropriate precision
 * @param value Measurement value
 * @param unit Unit type ('mm' or 'deg')
 * @returns Formatted string
 */
export function formatMeasurement(value: number, unit: string): string {
  if (unit === "deg") {
    return `${value.toFixed(2)}°`;
  }

  if (unit === "coord") {
    return `${value.toFixed(3)}`;
  }

  // For mm measurements - adaptive precision
  if (value < 1) {
    return `${value.toFixed(3)} ${unit}`; // Sub-millimeter: 3 decimals
  } else if (value < 10) {
    return `${value.toFixed(2)} ${unit}`; // Small: 2 decimals
  } else if (value < 100) {
    return `${value.toFixed(1)} ${unit}`; // Medium: 1 decimal
  } else {
    return `${value.toFixed(0)} ${unit}`; // Large: whole numbers
  }
}

/**
 * Snap point to nearest vertex with configurable threshold
 * @param point Point to snap
 * @param meshData Mesh data containing vertices
 * @param snapDistance Maximum snap distance
 * @returns Snap result or null if no vertex within threshold
 */
export function snapToVertex(point: THREE.Vector3, meshData: MeshData, snapDistance: number = 2): SnapResult | null {
  let closestVertex: THREE.Vector3 | null = null;
  let closestNormal: THREE.Vector3 = new THREE.Vector3(0, 1, 0);
  let minDistance = snapDistance;

  // Check all vertices
  for (let i = 0; i < meshData.vertices.length; i += 3) {
    const vertex = new THREE.Vector3(meshData.vertices[i], meshData.vertices[i + 1], meshData.vertices[i + 2]);

    const distance = point.distanceTo(vertex);
    if (distance < minDistance) {
      minDistance = distance;
      closestVertex = vertex;

      // Get normal for this vertex
      if (meshData.normals && i < meshData.normals.length) {
        closestNormal = new THREE.Vector3(meshData.normals[i], meshData.normals[i + 1], meshData.normals[i + 2]);
      }
    }
  }

  if (closestVertex) {
    return {
      position: closestVertex,
      normal: closestNormal,
      surfaceType: "vertex",
      confidence: 1 - minDistance / snapDistance,
    };
  }

  return null;
}

export function extractEdges(meshData: MeshData): THREE.Line3[] {
  const edges: THREE.Line3[] = [];
  const edgeSet = new Set<string>();

  for (let i = 0; i < meshData.indices.length; i += 3) {
    const i0 = meshData.indices[i] * 3;
    const i1 = meshData.indices[i + 1] * 3;
    const i2 = meshData.indices[i + 2] * 3;

    const v0 = new THREE.Vector3(meshData.vertices[i0], meshData.vertices[i0 + 1], meshData.vertices[i0 + 2]);
    const v1 = new THREE.Vector3(meshData.vertices[i1], meshData.vertices[i1 + 1], meshData.vertices[i1 + 2]);
    const v2 = new THREE.Vector3(meshData.vertices[i2], meshData.vertices[i2 + 1], meshData.vertices[i2 + 2]);

    const edgePairs = [
      [v0, v1],
      [v1, v2],
      [v2, v0],
    ];

    edgePairs.forEach(([start, end]) => {
      const key = `${start.x.toFixed(6)},${start.y.toFixed(6)},${start.z.toFixed(6)}-${end.x.toFixed(6)},${end.y.toFixed(6)},${end.z.toFixed(6)}`;
      const reverseKey = `${end.x.toFixed(6)},${end.y.toFixed(6)},${end.z.toFixed(6)}-${start.x.toFixed(6)},${start.y.toFixed(6)},${start.z.toFixed(6)}`;

      if (!edgeSet.has(key) && !edgeSet.has(reverseKey)) {
        edgeSet.add(key);
        edges.push(new THREE.Line3(start, end));
      }
    });
  }

  return edges;
}

function checkCollinearity(points: THREE.Vector3[], tolerance: number): boolean {
  if (points.length < 3) return true;

  const firstDir = new THREE.Vector3().subVectors(points[1], points[0]).normalize();

  for (let i = 2; i < points.length; i++) {
    const currentDir = new THREE.Vector3().subVectors(points[i], points[i - 1]).normalize();
    const cross = new THREE.Vector3().crossVectors(firstDir, currentDir);

    if (cross.length() > tolerance) {
      return false;
    }
  }

  return true;
}

function fitCircleToPoints(points: THREE.Vector3[]): { center: THREE.Vector3; radius: number } | null {
  if (points.length < 3) return null;

  // ✅ ENHANCED: Filter outliers before fitting
  const distances: number[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    distances.push(points[i].distanceTo(points[i + 1]));
  }
  distances.sort((a, b) => a - b);
  const medianDist = distances[Math.floor(distances.length / 2)];

  const filteredPoints = points.filter((p, i) => {
    if (i === 0 || i === points.length - 1) return true;
    const dist = p.distanceTo(points[i - 1]);
    return Math.abs(dist - medianDist) < medianDist * 2; // Allow 2x deviation
  });

  if (filteredPoints.length < 3) return null;

  const p1 = filteredPoints[0];
  const p2 = filteredPoints[Math.floor(filteredPoints.length / 2)];
  const p3 = filteredPoints[filteredPoints.length - 1];

  const mid1 = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
  const mid2 = new THREE.Vector3().addVectors(p2, p3).multiplyScalar(0.5);

  const dir1 = new THREE.Vector3().subVectors(p2, p1);
  const dir2 = new THREE.Vector3().subVectors(p3, p2);

  const normal = new THREE.Vector3().crossVectors(dir1, dir2).normalize();

  if (normal.length() < 0.001) {
    return null;
  }

  const perp1 = new THREE.Vector3().crossVectors(dir1, normal).normalize();
  const perp2 = new THREE.Vector3().crossVectors(dir2, normal).normalize();

  const radius1 = p1.distanceTo(mid1);
  const radius2 = p2.distanceTo(mid1);
  const radius3 = p3.distanceTo(mid2);

  const avgRadius = (radius1 + radius2 + radius3) / 3;

  const center = new THREE.Vector3().addVectors(p1, p2).add(p3).divideScalar(3);

  let maxError = 0;
  for (const point of filteredPoints) {
    const dist = point.distanceTo(center);
    const error = Math.abs(dist - avgRadius);
    maxError = Math.max(maxError, error);
  }

  if (maxError > avgRadius * 0.2) {
    return null;
  }

  return { center, radius: avgRadius };
}

export function classifyEdge(edge: THREE.Line3, meshData: MeshData): EdgeClassification {
  const tolerance = 0.01;

  // ✅ ENHANCED: Increased samples for better circle/arc detection
  const samples = 40; // Increased from 20 for better detection
  const edgePoints: THREE.Vector3[] = [];

  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const point = new THREE.Vector3();
    edge.at(t, point);

    let nearestPoint = point.clone();
    let minDist = Infinity;

    for (let j = 0; j < meshData.vertices.length; j += 3) {
      const meshPoint = new THREE.Vector3(meshData.vertices[j], meshData.vertices[j + 1], meshData.vertices[j + 2]);

      const dist = point.distanceTo(meshPoint);
      // ✅ ENHANCED: Increased search radius for better mesh point matching
      if (dist < minDist && dist < 2.0) {
        // Increased from 1.0
        minDist = dist;
        nearestPoint = meshPoint;
      }
    }

    edgePoints.push(nearestPoint);
  }

  const isCollinear = checkCollinearity(edgePoints, tolerance);

  if (isCollinear) {
    return {
      type: "line",
      start: edge.start,
      end: edge.end,
      length: edge.distance(),
      confidence: 95,
    };
  }

  const circleFit = fitCircleToPoints(edgePoints);

  if (circleFit) {
    const { center, radius } = circleFit;

    const startVec = new THREE.Vector3().subVectors(edgePoints[0], center);
    const endVec = new THREE.Vector3().subVectors(edgePoints[edgePoints.length - 1], center);
    const angle = startVec.angleTo(endVec);

    // ✅ ENHANCED: Better circle detection with relaxed threshold
    const startToEndDist = edgePoints[0].distanceTo(edgePoints[edgePoints.length - 1]);
    const radiusTolerance = radius * 0.05; // 5% of radius (more lenient)

    // Check if arc angle covers nearly full circle (>351 degrees = >6.13 radians)
    if (startToEndDist < radiusTolerance || angle > Math.PI * 1.95) {
      return {
        type: "circle",
        start: edge.start,
        end: edge.end,
        radius: radius,
        center: center,
        angle: Math.PI * 2,
        confidence: 90,
      };
    } else {
      return {
        type: "arc",
        start: edge.start,
        end: edge.end,
        radius: radius,
        center: center,
        angle: angle,
        confidence: 85,
      };
    }
  }

  return {
    type: "line",
    start: edge.start,
    end: edge.end,
    length: edge.distance(),
    confidence: 80,
  };
}

/**
 * Snap to backend-provided tagged edges (exact CAD geometry)
 * This uses BREP edges from backend with analytical data (SolidWorks-level precision)
 *
 * Priority:
 * 1. Use snap_points for efficient distance calculation
 * 2. Use analytical center/radius for circles/arcs (exact, not reconstructed)
 * 3. Fall back to start/end for simple lines
 */
export function snapToBackendEdge(
  point: THREE.Vector3,
  taggedEdges: BackendTaggedEdge[],
  snapDistance: number = 2
): SnapResult | null {
  let closestPoint: THREE.Vector3 | null = null;
  let minDistance = snapDistance;
  let closestEdge: BackendTaggedEdge | null = null;

  // Find closest tagged edge using snap_points (sparse polyline for efficiency)
  for (const edge of taggedEdges) {
    // Use snap_points if available (preferred - optimized for interaction)
    if (edge.snap_points && edge.snap_points.length >= 2) {
      // Check distance to each segment in the polyline
      for (let i = 0; i < edge.snap_points.length - 1; i++) {
        const p1 = edge.snap_points[i];
        const p2 = edge.snap_points[i + 1];
        const start = new THREE.Vector3(p1[0], p1[1], p1[2]);
        const end = new THREE.Vector3(p2[0], p2[1], p2[2]);
        const line = new THREE.Line3(start, end);

        const closestOnSegment = new THREE.Vector3();
        line.closestPointToPoint(point, true, closestOnSegment);

        const distance = point.distanceTo(closestOnSegment);
        if (distance < minDistance) {
          minDistance = distance;
          closestPoint = closestOnSegment;
          closestEdge = edge;
        }
      }

      // For closed edges (full circles), also check the closing segment
      if (edge.is_closed && edge.snap_points.length > 2) {
        const firstPt = edge.snap_points[0];
        const lastPt = edge.snap_points[edge.snap_points.length - 1];
        const start = new THREE.Vector3(firstPt[0], firstPt[1], firstPt[2]);
        const end = new THREE.Vector3(lastPt[0], lastPt[1], lastPt[2]);
        const line = new THREE.Line3(start, end);

        const closestOnSegment = new THREE.Vector3();
        line.closestPointToPoint(point, true, closestOnSegment);

        const distance = point.distanceTo(closestOnSegment);
        if (distance < minDistance) {
          minDistance = distance;
          closestPoint = closestOnSegment;
          closestEdge = edge;
        }
      }
    } else {
      // Fallback: use start/end points only
      const start = new THREE.Vector3(edge.start[0], edge.start[1], edge.start[2]);
      const end = new THREE.Vector3(edge.end[0], edge.end[1], edge.end[2]);
      const line = new THREE.Line3(start, end);

      const closestOnEdge = new THREE.Vector3();
      line.closestPointToPoint(point, true, closestOnEdge);

      const distance = point.distanceTo(closestOnEdge);
      if (distance < minDistance) {
        minDistance = distance;
        closestPoint = closestOnEdge;
        closestEdge = edge;
      }
    }
  }

  if (closestPoint && closestEdge) {
    const start = new THREE.Vector3(closestEdge.start[0], closestEdge.start[1], closestEdge.start[2]);
    const end = new THREE.Vector3(closestEdge.end[0], closestEdge.end[1], closestEdge.end[2]);

    // Use analytical center from backend (exact, not reconstructed)
    let center: THREE.Vector3 | undefined;
    if (closestEdge.center) {
      center = new THREE.Vector3(
        closestEdge.center[0],
        closestEdge.center[1],
        closestEdge.center[2]
      );
    }

    // Determine edge type for display
    let displayType: "line" | "arc" | "circle" = "line";
    if (closestEdge.type === "circle") {
      displayType = closestEdge.is_full_circle ? "circle" : "arc";
    }

    return {
      position: closestPoint,
      normal: closestEdge.normal
        ? new THREE.Vector3(closestEdge.normal[0], closestEdge.normal[1], closestEdge.normal[2])
        : new THREE.Vector3(0, 1, 0),
      surfaceType: "edge",
      confidence: 1 - minDistance / snapDistance,
      metadata: {
        edgeType: displayType,
        radius: closestEdge.radius,
        diameter: closestEdge.diameter,
        center: center,
        startPoint: start,
        endPoint: end,
        backendFeatureId: closestEdge.feature_id,
        backendLength: closestEdge.length,
        backendDiameter: closestEdge.diameter,
        backendMatch: true,
        isFullCircle: closestEdge.is_full_circle,
        sweepAngle: closestEdge.sweep_angle,
        arcLength: closestEdge.arc_length,
      },
    };
  }

  return null;
}

export function snapToEdge(point: THREE.Vector3, meshData: MeshData, snapDistance: number = 2): SnapResult | null {
  // ✅ PRIORITY: Use backend tagged edges if available (exact CAD geometry)
  if (meshData.tagged_edges && meshData.tagged_edges.length > 0) {
    const backendResult = snapToBackendEdge(point, meshData.tagged_edges, snapDistance);
    if (backendResult) {
      return backendResult;
    }
  }

  // ❌ FALLBACK: Extract edges from mesh triangles (less accurate)
  const edges = extractEdges(meshData);
  let closestPoint: THREE.Vector3 | null = null;
  let minDistance = snapDistance;
  let closestEdgeIndex = -1;
  let closestEdge: THREE.Line3 | null = null;

  edges.forEach((edge, index) => {
    const closestOnEdge = new THREE.Vector3();
    edge.closestPointToPoint(point, true, closestOnEdge);

    const distance = point.distanceTo(closestOnEdge);
    if (distance < minDistance) {
      minDistance = distance;
      closestPoint = closestOnEdge;
      closestEdgeIndex = index;
      closestEdge = edge;
    }
  });

  if (closestPoint && closestEdge) {
    const classification = classifyEdge(closestEdge, meshData);

    return {
      position: closestPoint,
      normal: new THREE.Vector3(0, 1, 0),
      surfaceType: "edge",
      confidence: 1 - minDistance / snapDistance,
      edgeIndex: closestEdgeIndex,
      metadata: {
        edgeType: classification.type,
        radius: classification.radius,
        center: classification.center,
        startPoint: classification.start,
        endPoint: classification.end,
        backendMatch: false,
      },
    };
  }

  return null;
}

export function snapToMidpoint(point: THREE.Vector3, meshData: MeshData, snapDistance: number = 2): SnapResult | null {
  const edges = extractEdges(meshData);
  let closestMidpoint: THREE.Vector3 | null = null;
  let minDistance = snapDistance;
  let closestEdgeIndex = -1;

  edges.forEach((edge, index) => {
    const midpoint = new THREE.Vector3().addVectors(edge.start, edge.end).multiplyScalar(0.5);
    const distance = point.distanceTo(midpoint);

    if (distance < minDistance) {
      minDistance = distance;
      closestMidpoint = midpoint;
      closestEdgeIndex = index;
    }
  });

  if (closestMidpoint) {
    return {
      position: closestMidpoint,
      normal: new THREE.Vector3(0, 1, 0),
      surfaceType: "midpoint",
      confidence: 1 - minDistance / snapDistance,
      edgeIndex: closestEdgeIndex,
    };
  }

  return null;
}

export function snapToFaceCenter(
  point: THREE.Vector3,
  meshData: MeshData,
  snapDistance: number = 2,
): SnapResult | null {
  let closestCenter: THREE.Vector3 | null = null;
  let closestNormal: THREE.Vector3 | null = null;
  let minDistance = snapDistance;
  let closestFaceIndex = -1;

  for (let i = 0; i < meshData.indices.length; i += 3) {
    const i0 = meshData.indices[i] * 3;
    const i1 = meshData.indices[i + 1] * 3;
    const i2 = meshData.indices[i + 2] * 3;

    const v0 = new THREE.Vector3(meshData.vertices[i0], meshData.vertices[i0 + 1], meshData.vertices[i0 + 2]);
    const v1 = new THREE.Vector3(meshData.vertices[i1], meshData.vertices[i1 + 1], meshData.vertices[i1 + 2]);
    const v2 = new THREE.Vector3(meshData.vertices[i2], meshData.vertices[i2 + 1], meshData.vertices[i2 + 2]);

    const center = new THREE.Vector3().addVectors(v0, v1).add(v2).divideScalar(3);

    const distance = point.distanceTo(center);
    if (distance < minDistance) {
      minDistance = distance;
      closestCenter = center;
      closestFaceIndex = i / 3;

      const e1 = new THREE.Vector3().subVectors(v1, v0);
      const e2 = new THREE.Vector3().subVectors(v2, v0);
      closestNormal = new THREE.Vector3().crossVectors(e1, e2).normalize();
    }
  }

  if (closestCenter && closestNormal) {
    return {
      position: closestCenter,
      normal: closestNormal,
      surfaceType: "center",
      confidence: 1 - minDistance / snapDistance,
      faceIndex: closestFaceIndex,
    };
  }

  return null;
}

export function performMultiSnap(
  point: THREE.Vector3,
  meshData: MeshData,
  activeSnapTypes: SnapType[],
  snapDistance: number = 2,
): SnapResult | null {
  const snapFunctions = {
    vertex: () => snapToVertex(point, meshData, snapDistance),
    midpoint: () => snapToMidpoint(point, meshData, snapDistance),
    center: () => snapToFaceCenter(point, meshData, snapDistance),
    edge: () => snapToEdge(point, meshData, snapDistance),
    face: () => snapToFaceCenter(point, meshData, snapDistance * 1.5),
    intersection: () => null,
  };

  for (const snapType of activeSnapTypes) {
    const snapFunc = snapFunctions[snapType];
    if (snapFunc) {
      const result = snapFunc();
      if (result && result.confidence > 0.5) {
        return result;
      }
    }
  }

  return null;
}

/**
 * Find the midpoint between two points
 * @param p1 First point
 * @param p2 Second point
 * @returns Midpoint
 */
export function getMidpoint(p1: THREE.Vector3, p2: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
}

/**
 * Generate label for measurement
 * @param type Measurement type
 * @param value Measurement value
 * @param unit Unit string
 * @returns Formatted label string
 */
export function generateMeasurementLabel(type: string, value: number, unit: string): string {
  const formattedValue = formatMeasurement(value, unit);

  switch (type) {
    case "distance":
      return `Distance: ${formattedValue}`;
    case "angle":
      return `Angle: ${formattedValue}`;
    case "radius":
      return `Radius: ${formattedValue}`;
    case "diameter":
      return `Diameter: ${formattedValue}`;
    case "edge-to-edge":
      return `Edge Distance: ${formattedValue}`;
    case "face-to-face":
      return `Face Distance: ${formattedValue}`;
    case "coordinate":
      return `Coordinate: ${formattedValue}`;
    case "edge-select":
      return formattedValue;
    default:
      return formattedValue;
  }
}

/**
 * Format coordinate with precision
 * @param coord Coordinate value
 * @returns Formatted coordinate string
 */
export function formatCoordinate(coord: number): string {
  return coord.toFixed(2);
}

/**
 * Generate unique ID for measurement
 * @returns Unique ID string
 */
export function generateMeasurementId(): string {
  return `measurement-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function calculateEdgeToEdgeDistance(
  edge1Start: THREE.Vector3,
  edge1End: THREE.Vector3,
  edge2Start: THREE.Vector3,
  edge2End: THREE.Vector3,
): { distance: number; point1: THREE.Vector3; point2: THREE.Vector3 } {
  const line1 = new THREE.Line3(edge1Start, edge1End);
  const line2 = new THREE.Line3(edge2Start, edge2End);

  let minDist = Infinity;
  let closestP1 = edge1Start.clone();
  let closestP2 = edge2Start.clone();

  const samples = 20;
  for (let i = 0; i <= samples; i++) {
    const t1 = i / samples;
    const p1 = new THREE.Vector3();
    line1.at(t1, p1);

    for (let j = 0; j <= samples; j++) {
      const t2 = j / samples;
      const p2 = new THREE.Vector3();
      line2.at(t2, p2);

      const dist = p1.distanceTo(p2);
      if (dist < minDist) {
        minDist = dist;
        closestP1 = p1.clone();
        closestP2 = p2.clone();
      }
    }
  }

  return {
    distance: minDist,
    point1: closestP1,
    point2: closestP2,
  };
}

export function calculateFaceToFaceDistance(
  face1Center: THREE.Vector3,
  face1Normal: THREE.Vector3,
  face2Center: THREE.Vector3,
  face2Normal: THREE.Vector3,
): number | null {
  const dot = Math.abs(face1Normal.dot(face2Normal));

  if (dot > 0.99) {
    const vector = new THREE.Vector3().subVectors(face2Center, face1Center);
    const distance = Math.abs(vector.dot(face1Normal));
    return distance;
  }

  return null;
}

export function calculateDeltas(
  p1: THREE.Vector3,
  p2: THREE.Vector3,
): {
  deltaX: number;
  deltaY: number;
  deltaZ: number;
} {
  return {
    deltaX: Math.abs(p2.x - p1.x),
    deltaY: Math.abs(p2.y - p1.y),
    deltaZ: Math.abs(p2.z - p1.z),
  };
}

/**
 * Calculate XYZ deltas for SolidWorks-style measurements
 * Returns SIGNED deltas (positive or negative) in millimeters
 * @param point1 First point (from)
 * @param point2 Second point (to)
 * @returns Object with dX, d

Y, dZ in mm
 */
export function calculateXYZDeltas(
  point1: THREE.Vector3,
  point2: THREE.Vector3
): {
  dX: number;
  dY: number;
  dZ: number;
  distance: number;
} {
  // Points are already in meters, convert to mm
  const dX = (point2.x - point1.x) * 1000;
  const dY = (point2.y - point1.y) * 1000;
  const dZ = (point2.z - point1.z) * 1000;
  const distance = point1.distanceTo(point2) * 1000;

  return {
    dX,
    dY,
    dZ,
    distance
  };
}


/**
 * Detect if two points lie on a cylindrical surface
 * @param p1 First point
 * @param p2 Second point
 * @param meshData Mesh data for surface analysis
 * @returns Cylinder info or null
 */
export function detectCylindricalSurface(
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  meshData: MeshData,
): { axis: THREE.Vector3; radius: number; center: THREE.Vector3; arcLength: number } | null {
  // Sample points along straight line between p1 and p2
  const samples = 10;
  const points: THREE.Vector3[] = [];

  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const point = new THREE.Vector3().lerpVectors(p1, p2, t);

    // Find closest mesh point
    let closestDist = Infinity;
    let closestMeshPoint: THREE.Vector3 | null = null;

    for (let j = 0; j < meshData.vertices.length; j += 3) {
      const meshPoint = new THREE.Vector3(meshData.vertices[j], meshData.vertices[j + 1], meshData.vertices[j + 2]);
      const dist = point.distanceTo(meshPoint);
      if (dist < closestDist) {
        closestDist = dist;
        closestMeshPoint = meshPoint;
      }
    }

    if (closestMeshPoint && closestDist < 5) {
      points.push(closestMeshPoint);
    }
  }

  if (points.length < 5) return null;

  // Try to fit cylinder
  // Calculate potential axis (perpendicular to p1-p2 vector)
  const direction = new THREE.Vector3().subVectors(p2, p1);

  // Check if points deviate from straight line (indicating curvature)
  let maxDeviation = 0;
  for (const point of points) {
    const projected = new THREE.Vector3();
    const line = new THREE.Line3(p1, p2);
    line.closestPointToPoint(point, false, projected);
    const deviation = point.distanceTo(projected);
    maxDeviation = Math.max(maxDeviation, deviation);
  }

  // If deviation is significant, likely a curved surface
  if (maxDeviation > 2 && maxDeviation < direction.length() * 0.5) {
    // Estimate cylinder parameters
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

    // Project points to estimate radius
    const radii: number[] = [];
    for (const point of points) {
      const projected = new THREE.Vector3();
      const line = new THREE.Line3(p1, p2);
      line.closestPointToPoint(point, false, projected);
      const radius = point.distanceTo(projected);
      if (radius > 1) radii.push(radius);
    }

    const avgRadius = radii.reduce((a, b) => a + b, 0) / radii.length;

    // Calculate arc length
    const straightDist = p1.distanceTo(p2);
    const theta = straightDist / avgRadius; // Angle in radians
    const arcLength = avgRadius * theta;

    return {
      axis: direction.normalize(),
      radius: avgRadius,
      center: midpoint,
      arcLength,
    };
  }

  return null;
}

/**
 * Calculate face perimeter from boundary edges
 * @param faceId Face ID to calculate perimeter for
 * @param meshData Mesh data containing vertices and face mapping
 * @returns Perimeter in mm, or null if cannot calculate
 */
export function calculateFacePerimeter(
  faceId: number,
  meshData: MeshData
): number | null {
  if (!meshData.vertex_face_ids || !meshData.vertices || !meshData.indices) {
    return null;
  }

  // Find all triangles belonging to this face
  const faceTriangles: number[] = [];
  for (let i = 0; i < meshData.indices.length; i += 3) {
    const idx0 = meshData.indices[i];
    const idx1 = meshData.indices[i + 1];
    const idx2 = meshData.indices[i + 2];

    // Check if all vertices belong to this face
    if (
      meshData.vertex_face_ids[idx0] === faceId ||
      meshData.vertex_face_ids[idx1] === faceId ||
      meshData.vertex_face_ids[idx2] === faceId
    ) {
      faceTriangles.push(i / 3);
    }
  }

  if (faceTriangles.length === 0) return null;

  // Extract boundary edges (edges that appear only once)
  const edgeMap = new Map<string, number>();

  for (const triIdx of faceTriangles) {
    const i = triIdx * 3;
    const idx0 = meshData.indices[i];
    const idx1 = meshData.indices[i + 1];
    const idx2 = meshData.indices[i + 2];

    // Three edges per triangle
    const edges = [
      [idx0, idx1],
      [idx1, idx2],
      [idx2, idx0],
    ];

    for (const [a, b] of edges) {
      const key = a < b ? `${a}-${b}` : `${b}-${a}`;
      edgeMap.set(key, (edgeMap.get(key) || 0) + 1);
    }
  }

  // Boundary edges appear only once
  let perimeter = 0;
  for (const [key, count] of edgeMap.entries()) {
    if (count === 1) {
      const [a, b] = key.split('-').map(Number);
      const v1 = new THREE.Vector3(
        meshData.vertices[a * 3],
        meshData.vertices[a * 3 + 1],
        meshData.vertices[a * 3 + 2]
      );
      const v2 = new THREE.Vector3(
        meshData.vertices[b * 3],
        meshData.vertices[b * 3 + 1],
        meshData.vertices[b * 3 + 2]
      );
      perimeter += v1.distanceTo(v2);
    }
  }

  return perimeter * 1000; // Convert to mm
}

/**
 * Format vertex coordinates for display
 * @param position Vertex position
 * @param precision Decimal places (0-5)
 * @returns Formatted coordinate string
 */
export function formatVertexCoordinates(
  position: THREE.Vector3,
  precision: number = 2
): string {
  // Convert from meters to mm
  const x = (position.x * 1000).toFixed(precision);
  const y = (position.y * 1000).toFixed(precision);
  const z = (position.z * 1000).toFixed(precision);

  return `(${x}, ${y}, ${z}) mm`;
}

/**
 * Calculate edge-to-edge distance with min, max, and center-to-center options
 * @param edge1Start First edge start point
 * @param edge1End First edge end point
 * @param edge2Start Second edge start point
 * @param edge2End Second edge end point
 * @returns Object with min, max, centerToCenter distances and closest points
 */
export function calculateEdgeToEdgeDistances(
  edge1Start: THREE.Vector3,
  edge1End: THREE.Vector3,
  edge2Start: THREE.Vector3,
  edge2End: THREE.Vector3
): {
  min: number;
  max: number;
  centerToCenter: number;
  minPoint1: THREE.Vector3;
  minPoint2: THREE.Vector3;
} {
  const line1 = new THREE.Line3(edge1Start, edge1End);
  const line2 = new THREE.Line3(edge2Start, edge2End);

  // Find minimum distance
  let minDist = Infinity;
  let minP1 = edge1Start.clone();
  let minP2 = edge2Start.clone();

  const samples = 20;
  for (let i = 0; i <= samples; i++) {
    const t1 = i / samples;
    const p1 = new THREE.Vector3();
    line1.at(t1, p1);

    for (let j = 0; j <= samples; j++) {
      const t2 = j / samples;
      const p2 = new THREE.Vector3();
      line2.at(t2, p2);

      const dist = p1.distanceTo(p2);
      if (dist < minDist) {
        minDist = dist;
        minP1 = p1.clone();
        minP2 = p2.clone();
      }
    }
  }

  // Calculate maximum distance (between endpoints)
  const distances = [
    edge1Start.distanceTo(edge2Start),
    edge1Start.distanceTo(edge2End),
    edge1End.distanceTo(edge2Start),
    edge1End.distanceTo(edge2End),
  ];
  const maxDist = Math.max(...distances);

  // Center-to-center distance
  const center1 = new THREE.Vector3().addVectors(edge1Start, edge1End).multiplyScalar(0.5);
  const center2 = new THREE.Vector3().addVectors(edge2Start, edge2End).multiplyScalar(0.5);
  const centerDist = center1.distanceTo(center2);

  return {
    min: minDist * 1000, // Convert to mm
    max: maxDist * 1000,
    centerToCenter: centerDist * 1000,
    minPoint1: minP1,
    minPoint2: minP2,
  };
}
