/**
 * SolidWorksMeasurementCallout.tsx
 *
 * SolidWorks-style measurement callout component for 3D visualization.
 * Renders professional-looking measurement labels in the 3D viewer.
 *
 * Features:
 * - Multiple callout styles (distance, diameter, radius, angle, area)
 * - Dimension lines with arrows
 * - Leader lines for offset labels
 * - Unit display with precision control
 * - Interactive hover states
 */

import React, { useMemo } from 'react';
import { Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { type SWMeasurement, type UnitSystem } from '@/stores/solidworksMeasureStore';

// === Type Definitions ===

interface CalloutProps {
  measurement: SWMeasurement;
  unitSystem: UnitSystem;
  precision?: number;
  scale?: number;
  interactive?: boolean;
  onHover?: (isHovered: boolean) => void;
  onClick?: () => void;
}

interface DimensionLineProps {
  start: THREE.Vector3;
  end: THREE.Vector3;
  offset?: number;
  color?: string;
  arrowSize?: number;
}

// === Utility Functions ===

const formatDisplayValue = (
  value: number,
  unit: string,
  unitSystem: UnitSystem,
  precision: number
): string => {
  if (unit === 'deg') {
    return `${value.toFixed(precision)}°`;
  }

  if (unit === 'mm²' || unit === 'in²') {
    if (unitSystem === 'imperial') {
      return `${(value * 0.00155).toFixed(precision)} in²`;
    }
    return `${value.toFixed(precision)} mm²`;
  }

  if (unitSystem === 'imperial') {
    return `${(value * 0.0393701).toFixed(precision)} in`;
  }
  return `${value.toFixed(precision)} mm`;
};

const getMeasurementPrefix = (type: SWMeasurement['type']): string => {
  switch (type) {
    case 'edge_diameter':
      return 'Ø';
    case 'edge_radius':
      return 'R';
    default:
      return '';
  }
};

const getMeasurementColor = (type: SWMeasurement['type']): string => {
  switch (type) {
    case 'edge_diameter':
    case 'edge_radius':
      return '#0066CC';
    case 'face_area':
      return '#00AA55';
    case 'face_to_face':
      return '#AA5500';
    case 'angle':
      return '#AA00AA';
    default:
      return '#1a5fb4';
  }
};

// === Dimension Line Component ===

const DimensionLine: React.FC<DimensionLineProps> = ({
  start,
  end,
  offset = 0,
  color = '#1a5fb4',
  arrowSize = 2,
}) => {
  const geometry = useMemo(() => {
    const dir = new THREE.Vector3().subVectors(end, start).normalize();
    const length = start.distanceTo(end);

    // Calculate perpendicular offset direction
    const up = new THREE.Vector3(0, 1, 0);
    let perpendicular = new THREE.Vector3().crossVectors(dir, up).normalize();
    if (perpendicular.length() < 0.01) {
      perpendicular = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(1, 0, 0)).normalize();
    }

    // Offset points
    const offsetVec = perpendicular.multiplyScalar(offset);
    const offsetStart = start.clone().add(offsetVec);
    const offsetEnd = end.clone().add(offsetVec);

    // Arrow geometry
    const arrowDir = new THREE.Vector3().subVectors(offsetEnd, offsetStart).normalize();
    const arrowPerp = perpendicular.clone().multiplyScalar(arrowSize * 0.4);

    // Arrow at end
    const arrow1Start = offsetEnd.clone().sub(arrowDir.clone().multiplyScalar(arrowSize));
    const arrow1End = offsetEnd;
    const arrow1Left = arrow1Start.clone().add(arrowPerp);
    const arrow1Right = arrow1Start.clone().sub(arrowPerp);

    // Arrow at start (reversed)
    const arrow2End = offsetStart;
    const arrow2Start = offsetStart.clone().add(arrowDir.clone().multiplyScalar(arrowSize));
    const arrow2Left = arrow2Start.clone().add(arrowPerp);
    const arrow2Right = arrow2Start.clone().sub(arrowPerp);

    return {
      mainLine: [offsetStart, offsetEnd],
      extensionLine1: offset !== 0 ? [start, offsetStart] : null,
      extensionLine2: offset !== 0 ? [end, offsetEnd] : null,
      arrow1: [arrow1Left, arrow1End, arrow1Right],
      arrow2: [arrow2Left, arrow2End, arrow2Right],
    };
  }, [start, end, offset, arrowSize]);

  return (
    <group>
      {/* Main dimension line */}
      <Line points={geometry.mainLine} color={color} lineWidth={1.5} />

      {/* Extension lines */}
      {geometry.extensionLine1 && (
        <Line points={geometry.extensionLine1} color={color} lineWidth={1} dashed dashSize={1} gapSize={0.5} />
      )}
      {geometry.extensionLine2 && (
        <Line points={geometry.extensionLine2} color={color} lineWidth={1} dashed dashSize={1} gapSize={0.5} />
      )}

      {/* Arrow heads */}
      <Line points={geometry.arrow1} color={color} lineWidth={1.5} />
      <Line points={geometry.arrow2} color={color} lineWidth={1.5} />
    </group>
  );
};

// === Main Callout Component ===

export const SolidWorksMeasurementCallout: React.FC<CalloutProps> = ({
  measurement,
  unitSystem,
  precision = 3,
  scale = 1,
  interactive = true,
  onHover,
  onClick,
}) => {
  const [isHovered, setIsHovered] = React.useState(false);

  const handlePointerEnter = () => {
    setIsHovered(true);
    onHover?.(true);
  };

  const handlePointerLeave = () => {
    setIsHovered(false);
    onHover?.(false);
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClick?.();
  };

  const color = getMeasurementColor(measurement.type);
  const prefix = getMeasurementPrefix(measurement.type);
  const displayValue = formatDisplayValue(measurement.value, measurement.unit, unitSystem, precision);

  // Calculate position based on measurement type
  const labelPosition = useMemo(() => {
    if (measurement.point1 && measurement.point2) {
      // Midpoint for distance measurements
      return new THREE.Vector3()
        .addVectors(measurement.point1, measurement.point2)
        .multiplyScalar(0.5);
    }
    if (measurement.point1) {
      return measurement.point1;
    }
    return new THREE.Vector3(0, 0, 0);
  }, [measurement]);

  // Style based on hover state
  const labelStyle = useMemo(() => ({
    background: isHovered ? 'rgba(26, 95, 180, 0.98)' : 'rgba(255, 255, 255, 0.98)',
    color: isHovered ? 'white' : '#1a1a1a',
    padding: `${8 * scale}px ${14 * scale}px`,
    borderRadius: `${4 * scale}px`,
    fontSize: `${14 * scale}px`,
    fontWeight: '600' as const,
    border: `${2 * scale}px solid ${color}`,
    boxShadow: isHovered
      ? '0 4px 16px rgba(26, 95, 180, 0.4)'
      : '0 2px 8px rgba(0,0,0,0.15)',
    cursor: interactive ? 'pointer' : 'default',
    transition: 'all 0.15s ease',
    fontFamily: '"Segoe UI", Arial, sans-serif',
    userSelect: 'none' as const,
    minWidth: `${80 * scale}px`,
    textAlign: 'center' as const,
  }), [isHovered, color, scale, interactive]);

  return (
    <group>
      {/* Dimension line for distance measurements */}
      {measurement.type === 'point_to_point' && measurement.point1 && measurement.point2 && (
        <DimensionLine
          start={measurement.point1}
          end={measurement.point2}
          color={color}
          arrowSize={3 * scale}
        />
      )}

      {/* End point markers */}
      {measurement.point1 && (
        <mesh position={measurement.point1}>
          <sphereGeometry args={[1.2 * scale, 16, 16]} />
          <meshBasicMaterial color={color} />
        </mesh>
      )}
      {measurement.point2 && (
        <mesh position={measurement.point2}>
          <sphereGeometry args={[1.2 * scale, 16, 16]} />
          <meshBasicMaterial color={color} />
        </mesh>
      )}

      {/* Label */}
      <Html
        position={labelPosition}
        center
        distanceFactor={1.5}
        style={{ pointerEvents: interactive ? 'auto' : 'none' }}
      >
        <div
          style={labelStyle}
          onPointerEnter={handlePointerEnter}
          onPointerLeave={handlePointerLeave}
          onClick={handleClick}
        >
          {/* Main value */}
          <div style={{
            fontSize: `${20 * scale}px`,
            fontWeight: 'bold',
            color: isHovered ? 'white' : color,
            marginBottom: measurement.deltaX !== undefined ? `${4 * scale}px` : 0,
          }}>
            {prefix}{displayValue}
          </div>

          {/* Delta values for distance measurements */}
          {measurement.deltaX !== undefined && (
            <div style={{
              fontSize: `${10 * scale}px`,
              color: isHovered ? 'rgba(255,255,255,0.8)' : '#666',
              display: 'flex',
              gap: `${6 * scale}px`,
              justifyContent: 'center',
            }}>
              <span>
                <span style={{ color: isHovered ? '#ff8888' : '#dc2626' }}>ΔX:</span>{' '}
                {formatDisplayValue(measurement.deltaX, 'mm', unitSystem, precision)}
              </span>
              <span>
                <span style={{ color: isHovered ? '#88ff88' : '#16a34a' }}>ΔY:</span>{' '}
                {formatDisplayValue(measurement.deltaY || 0, 'mm', unitSystem, precision)}
              </span>
              <span>
                <span style={{ color: isHovered ? '#8888ff' : '#2563eb' }}>ΔZ:</span>{' '}
                {formatDisplayValue(measurement.deltaZ || 0, 'mm', unitSystem, precision)}
              </span>
            </div>
          )}

          {/* Angle for face-to-face measurements */}
          {measurement.type === 'face_to_face' && measurement.angle !== undefined && (
            <div style={{
              fontSize: `${10 * scale}px`,
              color: isHovered ? 'rgba(255,255,255,0.8)' : '#666',
              marginTop: `${4 * scale}px`,
            }}>
              Angle: {measurement.angle.toFixed(1)}°
              {measurement.isParallel && (
                <span style={{
                  marginLeft: `${6 * scale}px`,
                  background: isHovered ? 'rgba(255,255,255,0.2)' : 'rgba(26, 95, 180, 0.1)',
                  padding: `${2 * scale}px ${6 * scale}px`,
                  borderRadius: `${3 * scale}px`,
                  fontSize: `${9 * scale}px`,
                }}>
                  Parallel
                </span>
              )}
            </div>
          )}

          {/* Edge type indicator */}
          {(measurement.type === 'edge_diameter' || measurement.type === 'edge_radius' || measurement.type === 'edge_length') && measurement.edgeType && (
            <div style={{
              fontSize: `${9 * scale}px`,
              color: isHovered ? 'rgba(255,255,255,0.6)' : '#999',
              marginTop: `${2 * scale}px`,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}>
              {measurement.edgeType}
            </div>
          )}
        </div>
      </Html>
    </group>
  );
};

// === Alternative Compact Callout ===

export const CompactCallout: React.FC<{
  position: THREE.Vector3;
  value: string;
  color?: string;
  scale?: number;
}> = ({ position, value, color = '#1a5fb4', scale = 1 }) => {
  return (
    <Html position={position} center>
      <div
        style={{
          background: color,
          color: 'white',
          padding: `${4 * scale}px ${8 * scale}px`,
          borderRadius: `${3 * scale}px`,
          fontSize: `${12 * scale}px`,
          fontWeight: '600',
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          boxShadow: '0 2px 6px rgba(0,0,0,0.25)',
          fontFamily: '"Segoe UI", Arial, sans-serif',
        }}
      >
        {value}
      </div>
    </Html>
  );
};

// === Hover Callout (for edge/face preview) ===

export const HoverCallout: React.FC<{
  position: THREE.Vector3;
  label: string;
  type?: 'edge' | 'face' | 'point';
}> = ({ position, label, type = 'edge' }) => {
  const bgColor = type === 'edge' ? 'rgba(26, 95, 180, 0.95)' :
                  type === 'face' ? 'rgba(0, 170, 85, 0.95)' :
                  'rgba(170, 85, 0, 0.95)';

  return (
    <Html position={position} center>
      <div
        style={{
          background: bgColor,
          color: 'white',
          padding: '6px 12px',
          borderRadius: '4px',
          fontSize: '14px',
          fontWeight: '600',
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          fontFamily: '"Segoe UI", Arial, sans-serif',
          backdropFilter: 'blur(4px)',
        }}
      >
        {label}
      </div>
    </Html>
  );
};

export default SolidWorksMeasurementCallout;
