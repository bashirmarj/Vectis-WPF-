import React from "react";
import { useMeasurementStore, Measurement } from "@/stores/measurementStore";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { LinearDimension } from "./LinearDimension";
import { RadialDimension } from "./RadialDimension";
import { AngularDimension } from "./AngularDimension";

/**
 * DimensionOverlay - Professional SVG overlay for SolidWorks-style dimension rendering
 * Projects 3D measurements to 2D screen space for crisp dimension lines and callouts
 */
export function DimensionOverlay() {
    const { measurements } = useMeasurementStore();
    const { camera, size } = useThree();

    // Project 3D point to 2D screen coordinates
    const project3DTo2D = (point: THREE.Vector3): { x: number; y: number } => {
        const vector = point.clone().project(camera);
        return {
            x: ((vector.x + 1) / 2) * size.width,
            y: ((-vector.y + 1) / 2) * size.height,
        };
    };

    // Check if measurement is visible (in camera frustum)
    const isVisible = (measurement: Measurement): boolean => {
        if (!measurement.visible) return false;

        // Check if any point is in view
        for (const point of measurement.points) {
            const projected = point.position.clone().project(camera);
            if (Math.abs(projected.x) <= 1 && Math.abs(projected.y) <= 1 && projected.z <= 1) {
                return true;
            }
        }
        return false;
    };

    // Render appropriate dimension component based on measurement type
    const renderDimension = (measurement: Measurement) => {
        if (!isVisible(measurement)) return null;

        const metadata = measurement.metadata;
        const edgeType = metadata?.edgeType;

        // Linear dimensions (lines, edges)
        if (edgeType === "line" && measurement.points.length >= 2) {
            const point1 = project3DTo2D(measurement.points[0].position);
            const point2 = project3DTo2D(measurement.points[1].position);

            return (
                <LinearDimension
                    key={measurement.id}
                    point1={point1}
                    point2={point2}
                    value={measurement.value}
                    unit={measurement.unit}
                    label={measurement.label}
                    deltaX={metadata?.deltaX}
                    deltaY={metadata?.deltaY}
                    deltaZ={metadata?.deltaZ}
                />
            );
        }

        // Radial dimensions (circles, arcs)
        if ((edgeType === "circle" || edgeType === "arc") && metadata?.arcCenter && metadata?.arcRadius) {
            const center = project3DTo2D(
                new THREE.Vector3(
                    metadata.arcCenter.x,
                    metadata.arcCenter.y,
                    metadata.arcCenter.z
                )
            );
            const radiusPoint = measurement.points[0]
                ? project3DTo2D(measurement.points[0].position)
                : { x: center.x + 50, y: center.y };

            return (
                <RadialDimension
                    key={measurement.id}
                    center={center}
                    radiusPoint={radiusPoint}
                    radius={metadata.arcRadius}
                    isCircle={metadata.isFullCircle ?? false}
                    isDiameter={false}
                    label={measurement.label}
                />
            );
        }

        // Angular dimensions (face angles)
        if (metadata?.facesAngle !== undefined && measurement.points.length >= 2) {
            const point1 = project3DTo2D(measurement.points[0].position);
            const point2 = project3DTo2D(measurement.points[1].position);

            return (
                <AngularDimension
                    key={measurement.id}
                    point1={point1}
                    point2={point2}
                    angle={metadata.facesAngle}
                    label={measurement.label}
                />
            );
        }

        // Default: Simple callout for other measurement types
        if (measurement.points.length > 0) {
            const midpoint = measurement.points.length === 1
                ? project3DTo2D(measurement.points[0].position)
                : (() => {
                    const p1 = project3DTo2D(measurement.points[0].position);
                    const p2 = project3DTo2D(measurement.points[1].position);
                    return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
                })();

            return (
                <g key={measurement.id}>
                    {/* Simple callout box */}
                    <rect
                        x={midpoint.x - 40}
                        y={midpoint.y - 15}
                        width={80}
                        height={24}
                        fill="rgba(255, 255, 255, 0.95)"
                        stroke="#0066cc"
                        strokeWidth={1}
                        rx={2}
                    />
                    <text
                        x={midpoint.x}
                        y={midpoint.y}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize={12}
                        fontWeight="bold"
                        fill="#0066cc"
                    >
                        {measurement.label}
                    </text>
                </g>
            );
        }

        return null;
    };

    return (
        <svg
            style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                zIndex: 10,
            }}
        >
            {measurements.map(renderDimension)}
        </svg>
    );
}
