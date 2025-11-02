import { useEffect, useState, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { FaceCrosshairMarker } from './measurements/FaceCrosshairMarker';
import { calculateMarkerValues, type MarkerValues } from '@/lib/faceMeasurementUtils';

interface MarkerData {
  intersection: THREE.Intersection;
}

interface FaceMeasurementToolProps {
  enabled: boolean;
  meshRef: THREE.Mesh | null;
  boundingSphere: {
    center: THREE.Vector3;
    radius: number;
  };
  onMeasurementsChange?: (measurements: MarkerValues | null, markerCount: number) => void;
  resetTrigger?: number;
}

export function FaceMeasurementTool({ 
  enabled, 
  meshRef,
  boundingSphere,
  onMeasurementsChange,
  resetTrigger
}: FaceMeasurementToolProps) {
  const { camera, raycaster, gl, scene } = useThree();
  
  const permanentMarkersRef = useRef<MarkerData[]>([]);
  const [permanentMarkers, setPermanentMarkers] = useState<MarkerData[]>([]);
  const [tempMarker, setTempMarker] = useState<MarkerData | null>(null);
  const [measurements, setMeasurements] = useState<MarkerValues | null>(null);
  const [connectingLine, setConnectingLine] = useState<THREE.Line | null>(null);
  
  const markerRadius = boundingSphere.radius / 20.0;

  // Handle pointer move (temp marker preview)
  useEffect(() => {
    if (!enabled || !meshRef) return;

    const canvas = gl.domElement;

    const handlePointerMove = (event: PointerEvent) => {
      // Only show temp marker if we haven't placed 2 markers yet
      if (permanentMarkersRef.current.length >= 2) {
        setTempMarker(null);
        canvas.style.cursor = 'default';
        return;
      }

      const rect = canvas.getBoundingClientRect();
      
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length > 0) {
        canvas.style.cursor = 'crosshair';
        setTempMarker({ intersection: intersects[0] });
      } else {
        canvas.style.cursor = 'default';
        setTempMarker(null);
      }
    };

    canvas.addEventListener('pointermove', handlePointerMove);
    return () => {
      canvas.removeEventListener('pointermove', handlePointerMove);
      canvas.style.cursor = 'default';
    };
  }, [enabled, meshRef, camera, raycaster, gl]);

  // Handle click (add permanent marker)
  useEffect(() => {
    if (!enabled || !meshRef) return;

    const canvas = gl.domElement;

    const handleClick = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length === 0) {
        // Ignore clicks on empty space - don't reset
        return;
      }

      const intersection = intersects[0];
      const currentMarkers = permanentMarkersRef.current;

      if (currentMarkers.length === 0) {
        // First marker
        const newMarkers = [{ intersection }];
        permanentMarkersRef.current = newMarkers;
        setPermanentMarkers(newMarkers);
        onMeasurementsChange?.(null, 1);
      } else if (currentMarkers.length === 1) {
        // Second marker - calculate measurements and draw line
        const newMarkers = [...currentMarkers, { intersection }];
        permanentMarkersRef.current = newMarkers;
        setPermanentMarkers(newMarkers);

        const values = calculateMarkerValues(
          currentMarkers[0].intersection,
          intersection
        );
        setMeasurements(values);
        onMeasurementsChange?.(values, 2);

        // Create connecting line
        const geometry = new THREE.BufferGeometry().setFromPoints([
          currentMarkers[0].intersection.point,
          intersection.point,
        ]);
        const material = new THREE.LineBasicMaterial({
          color: 0x263238,
          depthTest: false,
          depthWrite: false,
        });
        const line = new THREE.Line(geometry, material);
        line.renderOrder = 999;
        scene.add(line);
        setConnectingLine(line);
      }
      // If 2 markers already exist, do nothing (wait for reset button)
    };

    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [enabled, meshRef, camera, raycaster, scene, gl]);

  // Handle reset trigger
  useEffect(() => {
    if (resetTrigger === undefined || resetTrigger === 0) return;
    
    // Clear everything
    permanentMarkersRef.current = [];
    setPermanentMarkers([]);
    setMeasurements(null);
    setTempMarker(null);
    onMeasurementsChange?.(null, 0);
    if (connectingLine) {
      scene.remove(connectingLine);
      setConnectingLine(null);
    }
  }, [resetTrigger, scene]);

  // Cleanup on disable
  useEffect(() => {
    if (!enabled) {
      permanentMarkersRef.current = [];
      setPermanentMarkers([]);
      setTempMarker(null);
      setMeasurements(null);
      onMeasurementsChange?.(null, 0);
      if (connectingLine) {
        scene.remove(connectingLine);
        setConnectingLine(null);
      }
    }
  }, [enabled, scene, onMeasurementsChange]);

  if (!enabled) return null;

  return (
    <>
      {/* Permanent markers */}
      {permanentMarkers.map((marker, index) => (
        <FaceCrosshairMarker
          key={`permanent-${index}`}
          intersection={marker.intersection}
          radius={markerRadius}
          visible={true}
        />
      ))}

      {/* Temp marker (preview) - only show if less than 2 markers */}
      {tempMarker && permanentMarkers.length < 2 && (
        <FaceCrosshairMarker
          intersection={tempMarker.intersection}
          radius={markerRadius}
          visible={true}
        />
      )}
    </>
  );
}
