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

    // Get face normal in world space
    const faceNormal = getFaceWorldNormal(intersection);

    // Position marker at intersection point first
    groupRef.current.position.copy(intersection.point);
    
    // Align marker to face normal using lookAt
    // Create a target point in the direction of the normal
    const targetPoint = intersection.point.clone().add(faceNormal);
    groupRef.current.lookAt(targetPoint);
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
          transparent={false}
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
          transparent={false}
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
          transparent={false}
        />
      </line>
    </group>
  );
}
