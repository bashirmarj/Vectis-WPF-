/**
 * useMeasureAPI - Standalone hook for SolidWorks-style measurement API
 * 
 * This hook communicates with /api/measure endpoint.
 * All values are in MILLIMETERS.
 */

import { useState, useCallback } from 'react';
import * as THREE from 'three';

// Types for measurement results
export interface MeasureResult {
    success: boolean;
    measurement_type: 'edge' | 'point_to_point' | 'face_to_face';
    edge_type?: 'line' | 'circle' | 'ellipse' | 'arc' | 'other';
    value_mm: number;
    label: string;
    metadata: {
        delta_x_mm?: number;
        delta_y_mm?: number;
        delta_z_mm?: number;
        radius_mm?: number;
        diameter_mm?: number;
        arc_length_mm?: number;
        is_full_circle?: boolean;
        is_parallel?: boolean;
        center_mm?: [number, number, number];
        point1_mm?: [number, number, number];
        point2_mm?: [number, number, number];
        major_radius_mm?: number;
        minor_radius_mm?: number;
        angle_deg?: number;
    };
    error?: string;
}

export interface TaggedEdge {
    feature_id: number;
    type: string;
    length?: number;
    diameter?: number;
    radius?: number;
    arc_length?: number;
    is_full_circle?: boolean;
    is_closed?: boolean;
    center?: [number, number, number];
    start: [number, number, number];
    end: [number, number, number];
    normal?: [number, number, number];
    snap_points?: [number, number, number][];
    major_radius?: number;
    minor_radius?: number;
}

export interface FaceData {
    face_id: number;
    center: [number, number, number];
    normal: [number, number, number];
    surface_type?: string;
}

/**
 * Hook for performing measurements via API
 * For now, performs calculations client-side using backend edge data
 * Future: Can call actual /api/measure endpoint
 */
export function useMeasureAPI() {
    const [isLoading, setIsLoading] = useState(false);
    const [lastResult, setLastResult] = useState<MeasureResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    /**
     * Measure an edge using backend tagged_edge data
     * All values already in MM from backend
     */
    const measureEdge = useCallback((taggedEdge: TaggedEdge): MeasureResult => {
        const edgeType = taggedEdge.type;

        const result: MeasureResult = {
            success: true,
            measurement_type: 'edge',
            edge_type: edgeType as any,
            value_mm: 0,
            label: '',
            metadata: {}
        };

        if (edgeType === 'circle') {
            if (taggedEdge.is_full_circle && taggedEdge.diameter) {
                // Full circle - return diameter
                result.value_mm = taggedEdge.diameter;
                result.label = `Ø ${taggedEdge.diameter.toFixed(2)} mm`;
                result.metadata.is_full_circle = true;
                result.metadata.diameter_mm = taggedEdge.diameter;
                result.metadata.radius_mm = taggedEdge.radius || taggedEdge.diameter / 2;
            } else if (taggedEdge.radius) {
                // Arc
                result.value_mm = taggedEdge.radius;
                result.label = `R ${taggedEdge.radius.toFixed(2)} mm`;
                result.metadata.is_full_circle = false;
                result.metadata.radius_mm = taggedEdge.radius;
                if (taggedEdge.arc_length) {
                    result.label += ` | Arc: ${taggedEdge.arc_length.toFixed(2)} mm`;
                    result.metadata.arc_length_mm = taggedEdge.arc_length;
                }
            }
        } else if (edgeType === 'line' && taggedEdge.length) {
            result.value_mm = taggedEdge.length;
            result.label = `${taggedEdge.length.toFixed(2)} mm`;

            // Calculate deltas
            const start = taggedEdge.start;
            const end = taggedEdge.end;
            result.metadata.delta_x_mm = Math.abs(end[0] - start[0]);
            result.metadata.delta_y_mm = Math.abs(end[1] - start[1]);
            result.metadata.delta_z_mm = Math.abs(end[2] - start[2]);
        } else if (edgeType === 'ellipse' && taggedEdge.major_radius && taggedEdge.minor_radius) {
            result.value_mm = taggedEdge.major_radius;
            result.label = `Ellipse: ${taggedEdge.major_radius.toFixed(2)} × ${taggedEdge.minor_radius.toFixed(2)} mm`;
            result.metadata.major_radius_mm = taggedEdge.major_radius;
            result.metadata.minor_radius_mm = taggedEdge.minor_radius;
        } else if (taggedEdge.length) {
            // Generic edge with length
            result.value_mm = taggedEdge.length;
            result.label = `${taggedEdge.length.toFixed(2)} mm`;
        } else {
            result.success = false;
            result.label = 'No measurement available';
        }

        // Add center if available
        if (taggedEdge.center) {
            result.metadata.center_mm = taggedEdge.center;
        }

        setLastResult(result);
        return result;
    }, []);

    /**
     * Measure distance between two points
     * Points should be in MM
     */
    const measurePointToPoint = useCallback((
        point1: THREE.Vector3 | [number, number, number],
        point2: THREE.Vector3 | [number, number, number]
    ): MeasureResult => {
        const p1 = Array.isArray(point1) ? point1 : [point1.x, point1.y, point1.z];
        const p2 = Array.isArray(point2) ? point2 : [point2.x, point2.y, point2.z];

        const dx = p2[0] - p1[0];
        const dy = p2[1] - p1[1];
        const dz = p2[2] - p1[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        const result: MeasureResult = {
            success: true,
            measurement_type: 'point_to_point',
            value_mm: distance,
            label: `${distance.toFixed(2)} mm`,
            metadata: {
                delta_x_mm: Math.abs(dx),
                delta_y_mm: Math.abs(dy),
                delta_z_mm: Math.abs(dz),
                point1_mm: p1 as [number, number, number],
                point2_mm: p2 as [number, number, number]
            }
        };

        setLastResult(result);
        return result;
    }, []);

    /**
     * Measure distance between two faces
     */
    const measureFaceToFace = useCallback((
        face1: FaceData,
        face2: FaceData
    ): MeasureResult => {
        const n1 = face1.normal;
        const n2 = face2.normal;
        const c1 = face1.center;
        const c2 = face2.center;

        // Check if parallel
        const dot = Math.abs(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]);
        const isParallel = dot > 0.99;

        let distance: number;
        let label: string;

        if (isParallel) {
            // Perpendicular distance for parallel faces
            const dx = c2[0] - c1[0];
            const dy = c2[1] - c1[1];
            const dz = c2[2] - c1[2];
            distance = Math.abs(dx * n1[0] + dy * n1[1] + dz * n1[2]);
            label = `${distance.toFixed(2)} mm (parallel)`;
        } else {
            // Center-to-center distance
            const dx = c2[0] - c1[0];
            const dy = c2[1] - c1[1];
            const dz = c2[2] - c1[2];
            distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            const angleDeg = Math.acos(Math.min(1, Math.max(-1, dot))) * (180 / Math.PI);
            label = `${distance.toFixed(2)} mm (${angleDeg.toFixed(1)}°)`;
        }

        const result: MeasureResult = {
            success: true,
            measurement_type: 'face_to_face',
            value_mm: distance,
            label: label,
            metadata: {
                is_parallel: isParallel
            }
        };

        setLastResult(result);
        return result;
    }, []);

    /**
     * Clear last result and error
     */
    const reset = useCallback(() => {
        setLastResult(null);
        setError(null);
    }, []);

    return {
        measureEdge,
        measurePointToPoint,
        measureFaceToFace,
        isLoading,
        lastResult,
        error,
        reset
    };
}
