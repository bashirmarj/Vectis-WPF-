import { useMeasurementStore } from '@/stores/measurementStore';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

export function MeasurementRenderer() {
  const measurements = useMeasurementStore((state) => state.measurements);

  return (
    <>
      {measurements.map((measurement) => {
        if (!measurement.visible) return null;

        // Render edge-select measurements as thick lines
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
          const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
          
          // Create tube geometry for thick line
          const direction = new THREE.Vector3().subVectors(end, start);
          const length = direction.length();
          const cylinder = new THREE.CylinderGeometry(1, 1, length, 8);
          
          // Rotate cylinder to align with edge direction
          const quaternion = new THREE.Quaternion();
          quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize()
          );

          return (
            <group key={measurement.id}>
              {/* Thick edge line */}
              <mesh position={midpoint} quaternion={quaternion}>
                <primitive object={cylinder} />
                <meshBasicMaterial color={measurement.color || '#00ff00'} />
              </mesh>
              
              {/* Label at midpoint */}
              <Html position={midpoint} center>
                <div className="bg-background/90 text-foreground px-2 py-1 rounded text-xs border border-border whitespace-nowrap pointer-events-none">
                  {measurement.label}
                </div>
              </Html>
            </group>
          );
        }

        // Render other measurement types (distance, angle, etc.) - can be extended later
        return null;
      })}
    </>
  );
}
