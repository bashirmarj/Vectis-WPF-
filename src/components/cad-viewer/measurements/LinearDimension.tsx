import React from "react";

interface LinearDimensionProps {
    point1: { x: number; y: number };
    point2: { x: number; y: number };
    value: number;
    unit: string;
    label: string;
    deltaX?: number;
    deltaY?: number;
    deltaZ?: number;
}

/**
 * LinearDimension - Professional linear dimension with arrows and extension lines
 * Renders SolidWorks-style dimension annotation for linear measurements
 */
export function LinearDimension({
    point1,
    point2,
    value,
    unit,
    label,
    deltaX,
    deltaY,
    deltaZ,
}: LinearDimensionProps) {
    // Calculate dimension line offset (perpendicular to measurement)
    const dx = point2.x - point1.x;
    const dy = point2.y - point1.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    if (length < 10) return null; // Too small to render

    // Perpendicular vector for offset
    const perpX = -dy / length;
    const perpY = dx / length;

    // Offset distance from points
    const offset = 30;

    // Dimension line points (offset from actual points)
    const dimStart = {
        x: point1.x + perpX * offset,
        y: point1.y + perpY * offset,
    };
    const dimEnd = {
        x: point2.x + perpX * offset,
        y: point2.y + perpY * offset,
    };

    // Midpoint for label
    const midpoint = {
        x: (dimStart.x + dimEnd.x) / 2,
        y: (dimStart.y + dimEnd.y) / 2,
    };

    // Extension lines (from points to dimension line)
    const extLength = offset + 10;
    const ext1End = {
        x: point1.x + perpX * extLength,
        y: point1.y + perpY * extLength,
    };
    const ext2End = {
        x: point2.x + perpX * extLength,
        y: point2.y + perpY * extLength,
    };

    return (
        <g>
            {/* Arrow marker definition */}
            <defs>
                <marker
                    id={`arrow-${label.replace(/\s/g, "")}`}
                    markerWidth="6"
                    markerHeight="6"
                    refX="3"
                    refY="3"
                    orient="auto"
                    markerUnits="strokeWidth"
                >
                    <path d="M0,0 L0,6 L6,3 z" fill="#000000" />
                </marker>
            </defs>

            {/* Extension lines */}
            <line
                x1={point1.x}
                y1={point1.y}
                x2={ext1End.x}
                y2={ext1End.y}
                stroke="#666666"
                strokeWidth={0.5}
                strokeDasharray="2,2"
            />
            <line
                x1={point2.x}
                y1={point2.y}
                x2={ext2End.x}
                y2={ext2End.y}
                stroke="#666666"
                strokeWidth={0.5}
                strokeDasharray="2,2"
            />

            {/* Dimension line with arrows */}
            <line
                x1={dimStart.x}
                y1={dimStart.y}
                x2={dimEnd.x}
                y2={dimEnd.y}
                stroke="#000000"
                strokeWidth={1}
                markerStart={`url(#arrow-${label.replace(/\s/g, "")})`}
                markerEnd={`url(#arrow-${label.replace(/\s/g, "")})`}
            />

            {/* Callout box for value */}
            <rect
                x={midpoint.x - 45}
                y={midpoint.y - 28}
                width={90}
                height={deltaX !== undefined ? 44 : 24}
                fill="rgba(0, 102, 204, 0.95)"
                stroke="#005bb5"
                strokeWidth={1}
                rx={2}
            />

            {/* Primary value */}
            <text
                x={midpoint.x}
                y={midpoint.y - (deltaX !== undefined ? 10 : 0)}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={13}
                fontWeight="bold"
                fill="#ffffff"
            >
                {label}
            </text>

            {/* XYZ deltas (if available) */}
            {deltaX !== undefined && deltaY !== undefined && deltaZ !== undefined && (
                <text
                    x={midpoint.x}
                    y={midpoint.y + 12}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={9}
                    fill="#ffffff"
                    fontFamily="monospace"
                >
                    dX:{deltaX.toFixed(1)} dY:{deltaY.toFixed(1)} dZ:{deltaZ.toFixed(1)}
                </text>
            )}
        </g>
    );
}
