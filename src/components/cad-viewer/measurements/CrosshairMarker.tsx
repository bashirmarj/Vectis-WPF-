import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

interface CrosshairMarkerProps {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  radius: number;
  isTemp?: boolean;
}

export const CrosshairMarker = ({ position, normal, radius, isTemp }: CrosshairMarkerProps) => {
  // Create circle curve points
  const circlePoints = useMemo(() => {
    const curve = new THREE.EllipseCurve(
      0, 0,           // center
      radius, radius, // x radius, y radius
      0, 2 * Math.PI, // start angle, end angle
      false, 0        // clockwise, rotation
    );
    return curve.getPoints(50).map(p => new THREE.Vector3(p.x, p.y, 0));
  }, [radius]);

  // Calculate rotation to align with normal
  const rotation = useMemo(() => {
    const quaternion = new THREE.Quaternion();
    const defaultNormal = new THREE.Vector3(0, 0, 1);
    quaternion.setFromUnitVectors(defaultNormal, normal.clone().normalize());
    return quaternion;
  }, [normal]);

  return (
    <group position={position} quaternion={rotation}>
      {/* Circle */}
      <Line
        points={circlePoints}
        color="#263238"
        lineWidth={2}
        depthTest={false}
        depthWrite={false}
        renderOrder={999}
      />
      
      {/* Horizontal crosshair */}
      <Line
        points={[
          new THREE.Vector3(-radius, 0, 0),
          new THREE.Vector3(radius, 0, 0)
        ]}
        color="#263238"
        lineWidth={2}
        depthTest={false}
        depthWrite={false}
        renderOrder={999}
      />
      
      {/* Vertical crosshair */}
      <Line
        points={[
          new THREE.Vector3(0, -radius, 0),
          new THREE.Vector3(0, radius, 0)
        ]}
        color="#263238"
        lineWidth={2}
        depthTest={false}
        depthWrite={false}
        renderOrder={999}
      />
    </group>
  );
};
