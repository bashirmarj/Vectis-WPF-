import React, { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import type { Measurement } from '@/stores/measurementStore';

interface DimensionAnnotationProps {
  measurement: Measurement;
}

export const DimensionAnnotation: React.FC<DimensionAnnotationProps> = ({ measurement }) => {
  const { geometries, labelPosition } = useMemo(() => {
    if (measurement.type === 'edge-select' && measurement.metadata?.edgeStart && measurement.metadata?.edgeEnd) {
      const start = new THREE.Vector3(
        measurement.metadata.edgeStart[0],
        measurement.metadata.edgeStart[1],
        measurement.metadata.edgeStart[2]
      );
      const end = new THREE.Vector3(
        measurement.metadata.edgeEnd[0],
        measurement.metadata.edgeEnd[1],
        measurement.metadata.edgeEnd[2]
      );
      
      // Calculate extension offset (perpendicular to edge)
      const edgeDir = new THREE.Vector3().subVectors(end, start).normalize();
      const edgeLength = start.distanceTo(end);
      
      // Find perpendicular direction (use camera up as reference)
      const arbitraryUp = new THREE.Vector3(0, 1, 0);
      const perpendicular = new THREE.Vector3().crossVectors(edgeDir, arbitraryUp).normalize();
      if (perpendicular.length() < 0.1) {
        perpendicular.crossVectors(edgeDir, new THREE.Vector3(1, 0, 0)).normalize();
      }
      
      const extensionOffset = edgeLength * 0.15; // 15% of edge length
      
      // Extension lines endpoints
      const ext1Start = start.clone();
      const ext1End = start.clone().add(perpendicular.clone().multiplyScalar(extensionOffset));
      const ext2Start = end.clone();
      const ext2End = end.clone().add(perpendicular.clone().multiplyScalar(extensionOffset));
      
      // Dimension line (parallel to edge, offset by extensionOffset)
      const dimLineStart = ext1End.clone();
      const dimLineEnd = ext2End.clone();
      const dimLineMidpoint = new THREE.Vector3().addVectors(dimLineStart, dimLineEnd).multiplyScalar(0.5);
      
      // Arrow geometry
      const arrowSize = edgeLength * 0.02;
      const arrowDir1 = new THREE.Vector3().subVectors(dimLineEnd, dimLineStart).normalize();
      const arrowDir2 = arrowDir1.clone().negate();
      
      return {
        geometries: {
          extension1: { start: ext1Start, end: ext1End },
          extension2: { start: ext2Start, end: ext2End },
          dimensionLine: { start: dimLineStart, end: dimLineEnd },
          arrow1: { pos: dimLineStart, dir: arrowDir1, size: arrowSize },
          arrow2: { pos: dimLineEnd, dir: arrowDir2, size: arrowSize }
        },
        labelPosition: dimLineMidpoint
      };
    }
    
    return { geometries: null, labelPosition: new THREE.Vector3() };
  }, [measurement]);

  if (!geometries) {
    // Fallback for non-edge-select measurements
    const position = measurement.points[0]?.position || new THREE.Vector3();
    return (
      <Html position={position} center distanceFactor={10}>
        <div
          style={{
            background: 'rgba(0, 0, 0, 0.85)',
            color: '#ffffff',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}
        >
          {measurement.label}
        </div>
      </Html>
    );
  }

  const createLineGeometry = (start: THREE.Vector3, end: THREE.Vector3) => {
    const points = [start, end];
    return new THREE.BufferGeometry().setFromPoints(points);
  };

  const createArrowGeometry = (position: THREE.Vector3, direction: THREE.Vector3, size: number) => {
    const perp1 = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0)).normalize();
    if (perp1.length() < 0.1) {
      perp1.crossVectors(direction, new THREE.Vector3(1, 0, 0)).normalize();
    }
    const perp2 = new THREE.Vector3().crossVectors(direction, perp1).normalize();
    
    const tip = position.clone();
    const base1 = position.clone().add(perp1.clone().multiplyScalar(size * 0.3)).add(direction.clone().multiplyScalar(-size));
    const base2 = position.clone().add(perp2.clone().multiplyScalar(size * 0.3)).add(direction.clone().multiplyScalar(-size));
    
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
      tip.x, tip.y, tip.z,
      base1.x, base1.y, base1.z,
      base2.x, base2.y, base2.z
    ]);
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    return geometry;
  };

  return (
    <>
      {/* Extension lines (thin, dashed) */}
      <primitive object={new THREE.Line(createLineGeometry(geometries.extension1.start, geometries.extension1.end), new THREE.LineBasicMaterial({ color: "#888888", transparent: true, opacity: 0.6 }))} />
      <primitive object={new THREE.Line(createLineGeometry(geometries.extension2.start, geometries.extension2.end), new THREE.LineBasicMaterial({ color: "#888888", transparent: true, opacity: 0.6 }))} />
      
      {/* Dimension line (thicker, solid) */}
      <primitive object={new THREE.Line(createLineGeometry(geometries.dimensionLine.start, geometries.dimensionLine.end), new THREE.LineBasicMaterial({ color: "#0066CC" }))} />
      
      {/* Arrow 1 */}
      <mesh geometry={createArrowGeometry(geometries.arrow1.pos, geometries.arrow1.dir, geometries.arrow1.size)}>
        <meshBasicMaterial color="#0066CC" side={THREE.DoubleSide} />
      </mesh>
      
      {/* Arrow 2 */}
      <mesh geometry={createArrowGeometry(geometries.arrow2.pos, geometries.arrow2.dir, geometries.arrow2.size)}>
        <meshBasicMaterial color="#0066CC" side={THREE.DoubleSide} />
      </mesh>
      
      {/* Dimension text label - SolidWorks style */}
      <Html position={labelPosition} center distanceFactor={10}>
        <div
          style={{
            background: 'rgba(255, 255, 255, 0.95)',
            color: '#000000',
            padding: '6px 12px',
            borderRadius: '2px',
            fontSize: '14px',
            fontWeight: '600',
            fontFamily: 'Arial, Helvetica, sans-serif',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            border: '1px solid rgba(0, 0, 0, 0.25)',
            boxShadow: '0 2px 6px rgba(0, 0, 0, 0.15)',
            letterSpacing: '0.3px'
          }}
        >
          {measurement.label}
        </div>
      </Html>
    </>
  );
};
