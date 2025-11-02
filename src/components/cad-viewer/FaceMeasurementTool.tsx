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
  
  const clickStartPos = useRef<{ x: number; y: number } | null>(null);
  const isRealClick = useRef(true);
  
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

  // Handle click with drag detection (add permanent marker)
  useEffect(() => {
    if (!enabled || !meshRef) return;

    const canvas = gl.domElement;
    const MAX_CLICK_DISTANCE = 3; // pixels

    const handlePointerDown = (event: PointerEvent) => {
      clickStartPos.current = { x: event.clientX, y: event.clientY };
      isRealClick.current = true;
    };

    const handlePointerMoveForClick = (event: PointerEvent) => {
      if (!clickStartPos.current) return;
      
      const dx = event.clientX - clickStartPos.current.x;
      const dy = event.clientY - clickStartPos.current.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance > MAX_CLICK_DISTANCE) {
        isRealClick.current = false; // It's a drag, not a click
      }
    };

    const handlePointerUp = (event: PointerEvent) => {
      if (!isRealClick.current) {
        // It was a drag, not a click - ignore
        clickStartPos.current = null;
        return;
      }

      // Only process if it's a real click (no drag)
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length === 0) {
        // Click on empty space - reset markers (like reference code)
        permanentMarkersRef.current = [];
        setPermanentMarkers([]);
        setMeasurements(null);
        onMeasurementsChange?.(null, 0);
        if (connectingLine) {
          scene.remove(connectingLine);
          setConnectingLine(null);
        }
        clickStartPos.current = null;
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
        // Second marker
        const newMarkers = [...currentMarkers, { intersection }];
        permanentMarkersRef.current = newMarkers;
        setPermanentMarkers(newMarkers);

        const values = calculateMarkerValues(
          currentMarkers[0].intersection,
          intersection
        );
        setMeasurements(values);
        onMeasurementsChange?.(values, 2);

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
      } else {
        // Already have 2 markers - click on empty space resets
        permanentMarkersRef.current = [];
        setPermanentMarkers([]);
        setMeasurements(null);
        onMeasurementsChange?.(null, 0);
        if (connectingLine) {
          scene.remove(connectingLine);
          setConnectingLine(null);
        }
      }

      clickStartPos.current = null;
    };

    canvas.addEventListener('pointerdown', handlePointerDown);
    canvas.addEventListener('pointermove', handlePointerMoveForClick);
    canvas.addEventListener('pointerup', handlePointerUp);
    
    return () => {
      canvas.removeEventListener('pointerdown', handlePointerDown);
      canvas.removeEventListener('pointermove', handlePointerMoveForClick);
      canvas.removeEventListener('pointerup', handlePointerUp);
    };
  }, [enabled, meshRef, camera, raycaster, scene, gl, onMeasurementsChange, connectingLine]);

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
