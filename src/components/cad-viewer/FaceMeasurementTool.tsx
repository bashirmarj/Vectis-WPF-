import { useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { FaceCrosshairMarker } from './measurements/FaceCrosshairMarker';
import { FaceMeasurementPanel } from './measurements/FaceMeasurementPanel';
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
}

export function FaceMeasurementTool({ 
  enabled, 
  meshRef,
  boundingSphere 
}: FaceMeasurementToolProps) {
  const { camera, raycaster, gl, scene } = useThree();
  
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
      if (permanentMarkers.length >= 2) {
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
  }, [enabled, meshRef, camera, raycaster, gl, permanentMarkers.length]);

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
        // Click on empty space - clear all
        setPermanentMarkers([]);
        setMeasurements(null);
        if (connectingLine) {
          scene.remove(connectingLine);
          setConnectingLine(null);
        }
        return;
      }

      const intersection = intersects[0];

      if (permanentMarkers.length === 0) {
        // First marker
        setPermanentMarkers([{ intersection }]);
      } else if (permanentMarkers.length === 1) {
        // Second marker - calculate measurements and draw line
        const newMarkers = [...permanentMarkers, { intersection }];
        setPermanentMarkers(newMarkers);

        const values = calculateMarkerValues(
          permanentMarkers[0].intersection,
          intersection
        );
        setMeasurements(values);

        // Create connecting line
        const geometry = new THREE.BufferGeometry().setFromPoints([
          permanentMarkers[0].intersection.point,
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
      } else {
        // Third click - clear all (reset)
        setPermanentMarkers([]);
        setMeasurements(null);
        if (connectingLine) {
          scene.remove(connectingLine);
          setConnectingLine(null);
        }
      }
    };

    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [enabled, meshRef, camera, raycaster, permanentMarkers, connectingLine, scene, gl]);

  // Cleanup on disable
  useEffect(() => {
    if (!enabled) {
      setPermanentMarkers([]);
      setTempMarker(null);
      setMeasurements(null);
      if (connectingLine) {
        scene.remove(connectingLine);
        setConnectingLine(null);
      }
    }
  }, [enabled, connectingLine, scene]);

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

      {/* Measurement panel (DOM overlay) */}
      <Html center>
        <FaceMeasurementPanel 
          markerCount={permanentMarkers.length}
          measurements={measurements}
        />
      </Html>
    </>
  );
}
