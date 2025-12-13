import React from "react";

interface RadialDimensionProps {
    center: { x: number; y: number };
    radiusPoint: { x: number; y: number };
    radius: number;
    isCircle: boolean;
    isDiameter: boolean;
    label: string;
}

/**
 * RadialDimension - Professional radial/diameter dimension with leader line
 * Renders SolidWorks-style dimension annotation for circles and arcs
 */
export function RadialDimension({
    center,
    radiusPoint,
    radius,
    isCircle,
    isDiameter,
    label,
}: RadialDimensionProps) {
    // Calculate leader line direction
    const dx = radiusPoint.x - center.x;
    const dy = radiusPoint.y - center.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    if (length < 10) return null;

    const dirX = dx / length;
    const dirY = dy / length;

    // Leader line points
    const leaderStart = center;
    const leaderEnd = {
        x: center.x + dirX * (length + 40),
        y: center.y + dirY * (length + 40),
    };

    // Callout position (at end of leader)
    const calloutX = leaderEnd.x;
    const calloutY = leaderEnd.y - 12;

    // Symbol (R for radius, Ø for diameter)
    const symbol = isDiameter ? "Ø" : "R";

    // Calculate screen radius for circle rendering
    const screenRadius = length;

    return (
        <g>
            {/* Circle/Arc outline (dashed for arcs, solid for circles) */}
            <circle
                cx={center.x}
                cy={center.y}
                r={screenRadius}
                fill="none"
                stroke="#999999"
                strokeWidth={0.5}
                strokeDasharray={isCircle ? "none" : "4,2"}
                opacity={0.5}
            />

            {/* Center point */}
            <circle
                cx={center.x}
                cy={center.y}
                r={2}
                fill="#666666"
            />

            {/* Leader line */}
            <line
                x1={leaderStart.x}
                y1={leaderStart.y}
                x2={leaderEnd.x}
                y2={leaderEnd.y}
                stroke="#666666"
                strokeWidth={1}
            />

            {/* Callout box */}
            <rect
                x={calloutX - 40}
                y={calloutY - 12}
                width={80}
                height={24}
                fill="rgba(0, 102, 204, 0.95)"
                stroke="#005bb5"
                strokeWidth={1}
                rx={2}
            />

            {/* Value with symbol */}
            <text
                x={calloutX}
                y={calloutY}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={13}
                fontWeight="bold"
                fill="#ffffff"
            >
                {symbol} {label}
            </text>
        </g>
    );
}
