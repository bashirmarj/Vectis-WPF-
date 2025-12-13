import React from "react";

interface AngularDimensionProps {
    point1: { x: number; y: number };
    point2: { x: number; y: number };
    angle: number;
    label: string;
}

/**
 * AngularDimension - Professional angular dimension with arc and angle marking
 * Renders SolidWorks-style dimension annotation for angle measurements
 */
export function AngularDimension({
    point1,
    point2,
    angle,
    label,
}: AngularDimensionProps) {
    // Calculate midpoint (vertex of angle)
    const midpoint = {
        x: (point1.x + point2.x) / 2,
        y: (point1.y + point2.y) / 2,
    };

    // Vectors from midpoint to points
    const v1x = point1.x - midpoint.x;
    const v1y = point1.y - midpoint.y;
    const v2x = point2.x - midpoint.x;
    const v2y = point2.y - midpoint.y;

    const length1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const length2 = Math.sqrt(v2x * v2x + v2y * v2y);

    if (length1 < 10 || length2 < 10) return null;

    // Arc radius (shorter of the two arms)
    const arcRadius = Math.min(length1, length2, 50);

    // Calculate angles in radians
    const angle1 = Math.atan2(v1y, v1x);
    const angle2 = Math.atan2(v2y, v2x);

    // Arc endpoints
    const arcStart = {
        x: midpoint.x + Math.cos(angle1) * arcRadius,
        y: midpoint.y + Math.sin(angle1) * arcRadius,
    };
    const arcEnd = {
        x: midpoint.x + Math.cos(angle2) * arcRadius,
        y: midpoint.y + Math.sin(angle2) * arcRadius,
    };

    // Determine sweep direction
    let angleDiff = angle2 - angle1;
    if (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    if (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
    const sweepFlag = angleDiff > 0 ? 1 : 0;

    // Arc path
    const arcPath = `
    M ${arcStart.x} ${arcStart.y}
    A ${arcRadius} ${arcRadius} 0 0 ${sweepFlag} ${arcEnd.x} ${arcEnd.y}
  `;

    // Callout position (midway along arc)
    const midAngle = (angle1 + angle2) / 2;
    const calloutDistance = arcRadius + 20;
    const calloutX = midpoint.x + Math.cos(midAngle) * calloutDistance;
    const calloutY = midpoint.y + Math.sin(midAngle) * calloutDistance;

    return (
        <g>
            {/* Reference lines (dashed) */}
            <line
                x1={midpoint.x}
                y1={midpoint.y}
                x2={point1.x}
                y2={point1.y}
                stroke="#666666"
                strokeWidth={0.5}
                strokeDasharray="2,2"
                opacity={0.5}
            />
            <line
                x1={midpoint.x}
                y1={midpoint.y}
                x2={point2.x}
                y2={point2.y}
                stroke="#666666"
                strokeWidth={0.5}
                strokeDasharray="2,2"
                opacity={0.5}
            />

            {/* Angle arc */}
            <path
                d={arcPath}
                fill="none"
                stroke="#999999"
                strokeWidth={1.5}
            />

            {/* Vertex point */}
            <circle
                cx={midpoint.x}
                cy={midpoint.y}
                r={2}
                fill="#666666"
            />

            {/* Callout box */}
            <rect
                x={calloutX - 30}
                y={calloutY - 12}
                width={60}
                height={24}
                fill="rgba(0, 102, 204, 0.95)"
                stroke="#005bb5"
                strokeWidth={1}
                rx={2}
            />

            {/* Angle value */}
            <text
                x={calloutX}
                y={calloutY}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={13}
                fontWeight="bold"
                fill="#ffffff"
            >
                {angle.toFixed(1)}°
            </text>
        </g>
    );
}
