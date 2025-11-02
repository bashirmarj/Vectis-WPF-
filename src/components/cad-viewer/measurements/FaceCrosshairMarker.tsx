import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { getFaceWorldNormal } from '@/lib/faceMeasurementUtils';

interface FaceCrosshairMarkerProps {
  intersection: THREE.Intersection;
  radius: number;
  visible?: boolean;
}

export function FaceCrosshairMarker({ 
  intersection, 
  radius, 
  visible = true 
}: FaceCrosshairMarkerProps) {
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    if (!groupRef.current) return;

    // Get face normal
    const faceNormal = getFaceWorldNormal(intersection);

    // Update marker position and orientation
    groupRef.current.position.set(0, 0, 0);
    groupRef.current.lookAt(faceNormal);
    groupRef.current.position.copy(intersection.point);
  }, [intersection, radius]);

  return (
    <group ref={groupRef} visible={visible}>
      {/* Circle */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={51}
            array={new Float32Array(
              new THREE.EllipseCurve(
                0, 0,           // center x, y
                radius, radius, // x radius, y radius
                0, 2 * Math.PI, // start angle, end angle
                false,          // clockwise
                0               // rotation
              )
                .getPoints(50)
                .flatMap(p => [p.x, p.y, 0])
            )}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial 
          color="#263238" 
          depthTest={false} 
          depthWrite={false}
        />
      </line>

      {/* Horizontal crosshair */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-radius, 0, 0, radius, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial 
          color="#263238" 
          depthTest={false} 
          depthWrite={false}
        />
      </line>

      {/* Vertical crosshair */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, -radius, 0, 0, radius, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial 
          color="#263238" 
          depthTest={false} 
          depthWrite={false}
        />
      </line>
    </group>
  );
}
