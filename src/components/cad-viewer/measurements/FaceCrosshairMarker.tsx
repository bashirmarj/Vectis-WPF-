import { useEffect, useRef, useMemo } from 'react';
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

  // Create geometries once
  const circleGeometry = useMemo(() => {
    const points = new THREE.EllipseCurve(
      0, 0,           // center x, y
      radius, radius, // x radius, y radius
      0, 2 * Math.PI, // start angle, end angle
      false,          // clockwise
      0               // rotation
    ).getPoints(50);
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return geometry;
  }, [radius]);

  const horizontalGeometry = useMemo(() => {
    const points = [
      new THREE.Vector3(-radius, 0, 0),
      new THREE.Vector3(radius, 0, 0)
    ];
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [radius]);

  const verticalGeometry = useMemo(() => {
    const points = [
      new THREE.Vector3(0, -radius, 0),
      new THREE.Vector3(0, radius, 0)
    ];
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [radius]);

  const material = useMemo(() => {
    return new THREE.LineBasicMaterial({
      color: 0x263238,
      depthTest: false,
      depthWrite: false,
      transparent: false
    });
  }, []);

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
      <primitive object={new THREE.Line(circleGeometry, material)} />
      
      {/* Horizontal crosshair */}
      <primitive object={new THREE.Line(horizontalGeometry, material)} />
      
      {/* Vertical crosshair */}
      <primitive object={new THREE.Line(verticalGeometry, material)} />
    </group>
  );
}
