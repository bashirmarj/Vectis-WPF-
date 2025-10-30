import React from 'react';
import * as THREE from 'three';
import { useMeasurementStore } from '@/stores/measurementStore';
import { DimensionAnnotation } from './DimensionAnnotation';

const EdgeHighlight = ({ start, end, color }: { start: THREE.Vector3, end: THREE.Vector3, color: string }) => {
  const startVec = start;
  const endVec = end;
  const direction = new THREE.Vector3().subVectors(endVec, startVec);
  const length = direction.length();
  const midpoint = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5);
  
  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());

  return (
    <mesh position={midpoint} quaternion={quaternion}>
      <cylinderGeometry args={[1.5, 1.5, length, 8]} />
      <meshStandardMaterial 
        color={color} 
        emissive={color} 
        emissiveIntensity={0.4}
        transparent
        opacity={0.8}
      />
    </mesh>
  );
};

const FaceHighlight = ({ vertices, color }: { vertices: number[][], color: string }) => {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(vertices.flat());
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial 
        color={color} 
        transparent 
        opacity={0.2} 
        side={THREE.DoubleSide} 
      />
    </mesh>
  );
};

export function MeasurementRenderer() {
  const measurements = useMeasurementStore((state) => state.measurements);

  return (
    <>
      {measurements.map((measurement) => {
        if (!measurement.visible) return null;

        return (
          <React.Fragment key={measurement.id}>
            {/* Render persistent edge highlight */}
            {measurement.type === 'edge-select' && measurement.metadata?.edgeStart && measurement.metadata?.edgeEnd && (
              <EdgeHighlight 
                start={measurement.metadata.edgeStart}
                end={measurement.metadata.edgeEnd}
                color="#0066CC"
              />
            )}
            
            
            {/* Render dimension annotation */}
            <DimensionAnnotation measurement={measurement} />
          </React.Fragment>
        );
      })}
    </>
  );
}
