/**
 * measurementApiService.ts
 *
 * API service layer for communicating with the standalone measurement backend.
 * This is a STANDALONE service that integrates with the existing application
 * WITHOUT modifying any existing code.
 *
 * Features:
 * - RESTful API communication with measurement service
 * - Type-safe request/response handling
 * - Error handling and retry logic
 * - Caching for performance optimization
 */

import * as THREE from 'three';

// === Configuration ===

const MEASUREMENT_SERVICE_URL = import.meta.env.VITE_MEASUREMENT_SERVICE_URL || 'http://localhost:8081';
const REQUEST_TIMEOUT = 10000; // 10 seconds
const MAX_RETRIES = 2;

// === Type Definitions ===

export type UnitSystemApi = 'metric' | 'imperial';

export interface Point3DApi {
  x: number;
  y: number;
  z: number;
}

export interface EdgeInfoApi {
  edge_id: number;
  edge_type: 'line' | 'circle' | 'arc' | 'ellipse' | 'spline';
  start_point: [number, number, number];
  end_point: [number, number, number];
  length?: number;
  radius?: number;
  diameter?: number;
  center?: [number, number, number];
  axis?: [number, number, number];
  angle?: number;
}

export interface FaceInfoApi {
  face_id: number;
  surface_type: 'plane' | 'cylinder' | 'cone' | 'sphere' | 'torus' | 'other';
  center: [number, number, number];
  normal: [number, number, number];
  area: number;
  radius?: number;
  cylinder_axis?: [number, number, number];
}

export interface MeasurementResultApi {
  success: boolean;
  measurement_type: string;
  value: number;
  unit: string;
  label: string;
  display_value: string;
  delta_x?: number;
  delta_y?: number;
  delta_z?: number;
  edge_type?: string;
  diameter?: number;
  radius?: number;
  center?: [number, number, number];
  face_id?: number;
  face_type?: string;
  surface_type?: string;
  area?: number;
  normal?: [number, number, number];
  face1_id?: number;
  face2_id?: number;
  angle?: number;
  is_parallel?: boolean;
  perpendicular_distance?: number;
  point1?: [number, number, number];
  point2?: [number, number, number];
  confidence: number;
  backend_match: boolean;
  processing_time_ms: number;
  error?: string;
  correlation_id?: string;
}

export interface GeometryAnalysisResult {
  success: boolean;
  file_hash: string;
  face_count: number;
  edge_count: number;
  faces: FaceInfoApi[];
  edges: EdgeInfoApi[];
  processing_time_ms: number;
  error?: string;
  correlation_id?: string;
}

export interface ConversionResult {
  success: boolean;
  original_value: number;
  original_unit: string;
  converted_value: number;
  converted_unit: string;
  display_value: string;
  error?: string;
}

// === API Error Class ===

export class MeasurementApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public correlationId?: string
  ) {
    super(message);
    this.name = 'MeasurementApiError';
  }
}

// === Helper Functions ===

const generateCorrelationId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

const point3DToArray = (point: THREE.Vector3 | Point3DApi): [number, number, number] => {
  if (point instanceof THREE.Vector3) {
    return [point.x, point.y, point.z];
  }
  return [point.x, point.y, point.z];
};

const arrayToVector3 = (arr: [number, number, number] | undefined): THREE.Vector3 | undefined => {
  if (!arr) return undefined;
  return new THREE.Vector3(arr[0], arr[1], arr[2]);
};

// === Fetch with Retry ===

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  retries: number = MAX_RETRIES
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new MeasurementApiError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      return response;
    } catch (error) {
      lastError = error as Error;

      if (attempt < retries && !(error instanceof MeasurementApiError && error.statusCode === 400)) {
        // Wait before retry (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 500));
        continue;
      }
    }
  }

  throw lastError || new Error('Unknown error');
}

// === API Service Class ===

class MeasurementApiService {
  private baseUrl: string;
  private geometryCache: Map<string, GeometryAnalysisResult>;

  constructor(baseUrl: string = MEASUREMENT_SERVICE_URL) {
    this.baseUrl = baseUrl;
    this.geometryCache = new Map();
  }

  /**
   * Check if measurement service is healthy
   */
  async healthCheck(): Promise<{
    status: string;
    service: string;
    version: string;
    capabilities: string[];
  }> {
    const response = await fetchWithRetry(`${this.baseUrl}/health`, {
      method: 'GET',
    });

    return response.json();
  }

  /**
   * Measure point-to-point distance with XYZ deltas
   */
  async measurePointToPoint(
    point1: THREE.Vector3 | [number, number, number],
    point2: THREE.Vector3 | [number, number, number],
    unitSystem: UnitSystemApi = 'metric'
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const p1 = point1 instanceof THREE.Vector3 ? [point1.x, point1.y, point1.z] : point1;
    const p2 = point2 instanceof THREE.Vector3 ? [point2.x, point2.y, point2.z] : point2;

    const response = await fetchWithRetry(`${this.baseUrl}/measure/point-to-point`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        point1: p1,
        point2: p2,
        unit_system: unitSystem,
      }),
    });

    return response.json();
  }

  /**
   * Measure an edge (length, diameter, or radius based on type)
   */
  async measureEdge(
    edgeInfo: {
      edge_id?: number;
      edge_type: 'line' | 'circle' | 'arc';
      start: [number, number, number];
      end: [number, number, number];
      length?: number;
      radius?: number;
      diameter?: number;
      center?: [number, number, number];
    },
    unitSystem: UnitSystemApi = 'metric'
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/measure/edge`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        edge_info: {
          edge_id: edgeInfo.edge_id || 0,
          edge_type: edgeInfo.edge_type,
          start_point: edgeInfo.start,
          end_point: edgeInfo.end,
          length: edgeInfo.length,
          radius: edgeInfo.radius,
          diameter: edgeInfo.diameter,
          center: edgeInfo.center,
        },
        unit_system: unitSystem,
      }),
    });

    return response.json();
  }

  /**
   * Measure face area
   */
  async measureFaceArea(
    faceInfo: {
      face_id?: number;
      surface_type: string;
      center: [number, number, number];
      normal: [number, number, number];
      area: number;
    },
    unitSystem: UnitSystemApi = 'metric'
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/measure/face-area`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        face_info: faceInfo,
        unit_system: unitSystem,
      }),
    });

    return response.json();
  }

  /**
   * Measure distance and angle between two faces
   */
  async measureFaceToFace(
    face1: { center: [number, number, number]; normal: [number, number, number] },
    face2: { center: [number, number, number]; normal: [number, number, number] },
    unitSystem: UnitSystemApi = 'metric'
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/measure/face-to-face`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        face1,
        face2,
        unit_system: unitSystem,
      }),
    });

    return response.json();
  }

  /**
   * Measure angle between two vectors or three points
   */
  async measureAngle(
    params:
      | { vector1: [number, number, number]; vector2: [number, number, number] }
      | { point1: [number, number, number]; vertex: [number, number, number]; point2: [number, number, number] }
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/measure/angle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify(params),
    });

    return response.json();
  }

  /**
   * Get coordinate information for a point
   */
  async measureCoordinate(
    point: [number, number, number],
    unitSystem: UnitSystemApi = 'metric'
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/measure/coordinate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        point,
        unit_system: unitSystem,
      }),
    });

    return response.json();
  }

  /**
   * Analyze STEP file geometry for measurements
   */
  async analyzeGeometry(file: File): Promise<GeometryAnalysisResult> {
    const correlationId = generateCorrelationId();

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetchWithRetry(`${this.baseUrl}/analyze-geometry`, {
      method: 'POST',
      headers: {
        'X-Correlation-ID': correlationId,
      },
      body: formData,
    });

    const result = await response.json();

    // Cache the result
    if (result.success && result.file_hash) {
      this.geometryCache.set(result.file_hash, result);
    }

    return result;
  }

  /**
   * Get cached geometry analysis
   */
  getCachedGeometry(fileHash: string): GeometryAnalysisResult | undefined {
    return this.geometryCache.get(fileHash);
  }

  /**
   * Convert units
   */
  async convertUnits(
    value: number,
    fromUnit: string,
    toUnit: string,
    decimals: number = 3
  ): Promise<ConversionResult> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/convert-units`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        value,
        from_unit: fromUnit,
        to_unit: toUnit,
        decimals,
      }),
    });

    return response.json();
  }

  /**
   * Unified measurement from mesh data (frontend integration)
   */
  async measureFromMesh(
    measurementType: 'point_to_point' | 'edge' | 'face_area' | 'face_to_face',
    data: any,
    unitSystem: UnitSystemApi = 'metric'
  ): Promise<MeasurementResultApi> {
    const correlationId = generateCorrelationId();

    const response = await fetchWithRetry(`${this.baseUrl}/measure/from-mesh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': correlationId,
      },
      body: JSON.stringify({
        measurement_type: measurementType,
        data,
        unit_system: unitSystem,
      }),
    });

    return response.json();
  }

  /**
   * Clear geometry cache
   */
  clearCache(): void {
    this.geometryCache.clear();
  }
}

// === Singleton Instance ===

export const measurementApi = new MeasurementApiService();

// === Local Calculation Fallbacks ===
// These functions provide local calculations when the backend is unavailable

export const localCalculations = {
  /**
   * Calculate point-to-point distance locally
   */
  pointToPoint(
    point1: THREE.Vector3,
    point2: THREE.Vector3,
    unitSystem: UnitSystemApi = 'metric'
  ): MeasurementResultApi {
    const startTime = performance.now();

    const distance = point1.distanceTo(point2);
    const deltaX = Math.abs(point2.x - point1.x);
    const deltaY = Math.abs(point2.y - point1.y);
    const deltaZ = Math.abs(point2.z - point1.z);

    const displayValue = unitSystem === 'imperial'
      ? `${(distance * 0.0393701).toFixed(3)} in`
      : `${distance.toFixed(3)} mm`;

    return {
      success: true,
      measurement_type: 'point_to_point',
      value: distance,
      unit: 'mm',
      label: `Distance: ${displayValue}`,
      display_value: displayValue,
      delta_x: deltaX,
      delta_y: deltaY,
      delta_z: deltaZ,
      point1: [point1.x, point1.y, point1.z],
      point2: [point2.x, point2.y, point2.z],
      confidence: 1.0,
      backend_match: false,
      processing_time_ms: Math.round(performance.now() - startTime),
    };
  },

  /**
   * Calculate edge measurement locally
   */
  edgeMeasurement(
    edgeType: 'line' | 'circle' | 'arc',
    options: {
      start?: THREE.Vector3;
      end?: THREE.Vector3;
      length?: number;
      radius?: number;
      diameter?: number;
    },
    unitSystem: UnitSystemApi = 'metric'
  ): MeasurementResultApi {
    const startTime = performance.now();

    let value: number;
    let measurementType: string;
    let label: string;

    if (edgeType === 'circle' && options.diameter) {
      value = options.diameter;
      measurementType = 'edge_diameter';
      label = 'Diameter';
    } else if (edgeType === 'arc' && options.radius) {
      value = options.radius;
      measurementType = 'edge_radius';
      label = 'Radius';
    } else if (options.length) {
      value = options.length;
      measurementType = 'edge_length';
      label = 'Length';
    } else if (options.start && options.end) {
      value = options.start.distanceTo(options.end);
      measurementType = 'edge_length';
      label = 'Length';
    } else {
      value = 0;
      measurementType = 'edge_length';
      label = 'Length';
    }

    const displayValue = unitSystem === 'imperial'
      ? `${(value * 0.0393701).toFixed(3)} in`
      : `${value.toFixed(3)} mm`;

    const labelPrefix = edgeType === 'circle' ? 'Ø ' : edgeType === 'arc' ? 'R ' : '';

    return {
      success: true,
      measurement_type: measurementType,
      value,
      unit: 'mm',
      label: `${label}: ${labelPrefix}${displayValue}`,
      display_value: `${labelPrefix}${displayValue}`,
      edge_type: edgeType,
      diameter: options.diameter,
      radius: options.radius,
      confidence: 1.0,
      backend_match: false,
      processing_time_ms: Math.round(performance.now() - startTime),
    };
  },

  /**
   * Calculate face-to-face distance and angle locally
   */
  faceToFace(
    face1: { center: THREE.Vector3; normal: THREE.Vector3 },
    face2: { center: THREE.Vector3; normal: THREE.Vector3 },
    unitSystem: UnitSystemApi = 'metric'
  ): MeasurementResultApi {
    const startTime = performance.now();

    const distance = face1.center.distanceTo(face2.center);
    const dot = Math.abs(face1.normal.dot(face2.normal));
    const angle = Math.acos(Math.min(1.0, Math.max(-1.0, dot))) * (180 / Math.PI);
    const isParallel = angle < 5 || angle > 175;

    let perpDistance: number | undefined;
    if (isParallel) {
      const diff = new THREE.Vector3().subVectors(face2.center, face1.center);
      perpDistance = Math.abs(diff.dot(face1.normal));
    }

    const displayValue = unitSystem === 'imperial'
      ? `${(distance * 0.0393701).toFixed(3)} in`
      : `${distance.toFixed(3)} mm`;

    return {
      success: true,
      measurement_type: 'face_to_face',
      value: distance,
      unit: 'mm',
      label: `Distance: ${displayValue}`,
      display_value: `${displayValue} (${angle.toFixed(1)}°)`,
      angle,
      is_parallel: isParallel,
      perpendicular_distance: perpDistance,
      point1: [face1.center.x, face1.center.y, face1.center.z],
      point2: [face2.center.x, face2.center.y, face2.center.z],
      confidence: 1.0,
      backend_match: false,
      processing_time_ms: Math.round(performance.now() - startTime),
    };
  },

  /**
   * Calculate angle between two vectors locally
   */
  angleBetweenVectors(
    v1: THREE.Vector3,
    v2: THREE.Vector3
  ): MeasurementResultApi {
    const startTime = performance.now();

    const n1 = v1.clone().normalize();
    const n2 = v2.clone().normalize();
    const dot = n1.dot(n2);
    const angle = Math.acos(Math.min(1.0, Math.max(-1.0, dot))) * (180 / Math.PI);

    return {
      success: true,
      measurement_type: 'angle',
      value: angle,
      unit: 'deg',
      label: `Angle: ${angle.toFixed(2)}°`,
      display_value: `${angle.toFixed(2)}°`,
      angle,
      confidence: 1.0,
      backend_match: false,
      processing_time_ms: Math.round(performance.now() - startTime),
    };
  },
};

export default measurementApi;
