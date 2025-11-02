import { useState, useEffect, useMemo } from 'react';
import { useThree } from '@react-three/fiber';
import { Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { CrosshairMarker } from './measurements/CrosshairMarker';
import { MeasurementPanel } from './measurements/MeasurementPanel';
import { MarkerData, getFaceWorldNormal, calculateMarkerValues } from '@/lib/measurementCalculations';

interface ProfessionalMeasurementToolProps {
  enabled: boolean;
  meshRef: THREE.Object3D | null;
  boundingSphere?: THREE.Sphere | null;
}

export const ProfessionalMeasurementTool = ({ 
  enabled, 
  meshRef,
  boundingSphere 
}: ProfessionalMeasurementToolProps) => {
  const { camera, gl, raycaster } = useThree();
  const [markers, setMarkers] = useState<MarkerData[]>([]);
  const [tempMarker, setTempMarker] = useState<MarkerData | null>(null);
  const [measurements, setMeasurements] = useState<any>(null);

  // Calculate marker radius from bounding sphere (1/20th of the model radius)
  const markerRadius = useMemo(() => {
    return boundingSphere ? boundingSphere.radius / 20.0 : 5;
  }, [boundingSphere]);
  
  // Handle mouse move - show temp marker
  useEffect(() => {
    if (!enabled || !meshRef) return;
    
    const handlePointerMove = (event: PointerEvent) => {
      const rect = gl.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );
      
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, true);
      
      if (intersects.length > 0) {
        const intersection = intersects[0];
        const normal = getFaceWorldNormal(intersection);
        
        setTempMarker({
          position: intersection.point.clone(),
          normal: normal,
          intersection: intersection
        });
        
        gl.domElement.style.cursor = 'crosshair';
      } else {
        setTempMarker(null);
        gl.domElement.style.cursor = 'default';
      }
    };
    
    gl.domElement.addEventListener('pointermove', handlePointerMove);
    return () => {
      gl.domElement.removeEventListener('pointermove', handlePointerMove);
      gl.domElement.style.cursor = 'default';
    };
  }, [enabled, meshRef, camera, gl, raycaster]);
  
  // Handle click - add marker
  useEffect(() => {
    if (!enabled) return;
    
    const handleClick = () => {
      if (!tempMarker) return;
      
      if (markers.length === 2) {
        // Third click - reset
        setMarkers([]);
        setMeasurements(null);
      } else if (markers.length === 1) {
        // Second click - calculate measurements
        const newMarkers = [...markers, tempMarker];
        setMarkers(newMarkers);
        
        const calculated = calculateMarkerValues(newMarkers[0], newMarkers[1]);
        setMeasurements(calculated);
      } else {
        // First click
        setMarkers([tempMarker]);
      }
    };
    
    gl.domElement.addEventListener('click', handleClick);
    return () => {
      gl.domElement.removeEventListener('click', handleClick);
    };
  }, [enabled, tempMarker, markers, gl]);
  
  // Reset when disabled
  useEffect(() => {
    if (!enabled) {
      setMarkers([]);
      setTempMarker(null);
      setMeasurements(null);
    }
  }, [enabled]);
  
  if (!enabled) return null;
  
  return (
    <>
      {/* Permanent markers */}
      {markers.map((marker, index) => (
        <CrosshairMarker
          key={`marker-${index}`}
          position={marker.position}
          normal={marker.normal}
          radius={markerRadius}
        />
      ))}
      
      {/* Temp marker */}
      {tempMarker && (
        <CrosshairMarker
          position={tempMarker.position}
          normal={tempMarker.normal}
          radius={markerRadius}
          isTemp
        />
      )}
      
      {/* Connecting line between two markers */}
      {markers.length === 2 && (
        <Line
          points={[markers[0].position, markers[1].position]}
          color="#263238"
          lineWidth={2}
          depthTest={false}
          depthWrite={false}
          renderOrder={999}
        />
      )}
      
      {/* Measurement Panel */}
      <Html fullscreen>
        <MeasurementPanel
          markerCount={markers.length}
          measurements={measurements}
        />
      </Html>
    </>
  );
};
