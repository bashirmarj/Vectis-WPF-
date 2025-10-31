import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { useMeasurementStore, Measurement, MeasurementPoint } from '@/stores/measurementStore';
import {
  calculateDistance,
  calculateAngle,
  calculateRadius,
  getMidpoint,
  generateMeasurementLabel,
  snapToVertex,
  formatCoordinate,
  generateMeasurementId
} from '@/lib/measurementUtils';

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
}

interface AdvancedMeasurementToolProps {
  meshData?: MeshData;
  meshRef?: React.RefObject<THREE.Mesh>;
  enabled: boolean;
}

/**
 * Advanced Measurement Tool Component
 * Provides high-precision distance, angle, radius, and diameter measurements
 * with snap-to-vertex functionality
 */
export function AdvancedMeasurementTool({ 
  meshData, 
  meshRef,
  enabled 
}: AdvancedMeasurementToolProps) {
  const { scene, gl, camera } = useThree();
  const { 
    activeTool, 
    tempPoints, 
    measurements,
    snapEnabled,
    snapDistance,
    addTempPoint, 
    clearTempPoints,
    addMeasurement,
    setActiveTool
  } = useMeasurementStore();
  
  const markersRef = useRef<THREE.Mesh[]>([]);
  
  // Handle click to place measurement points
  useEffect(() => {
    if (!enabled || !activeTool || !meshData) return;
    
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    
    const onClick = (event: MouseEvent) => {
      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      
      raycaster.setFromCamera(mouse, camera);
      
      // Intersect with mesh only (not with markers)
      const targets = meshRef?.current ? [meshRef.current] : scene.children.filter(
        child => child.type === 'Mesh' && !markersRef.current.includes(child as THREE.Mesh)
      );
      
      const intersects = raycaster.intersectObjects(targets, true);
      
      if (intersects.length > 0) {
        let point = intersects[0].point.clone();
        const normal = intersects[0].face?.normal || new THREE.Vector3(0, 1, 0);
        
        // Apply snap-to-vertex if enabled
        if (snapEnabled && meshData) {
          const snapped = snapToVertex(point, meshData, snapDistance);
          if (snapped) {
            point = snapped.position;
          }
        }
        
        // Create measurement point
        const measurementPoint: MeasurementPoint = {
          id: `point-${Date.now()}-${Math.random()}`,
          position: point,
          normal: normal,
          surfaceType: 'vertex'
        };
        
        addTempPoint(measurementPoint);
        
        // Complete measurement when enough points are collected
        completeMeasurementIfReady();
      }
    };
    
    const canvas = gl.domElement;
    canvas.addEventListener('click', onClick);
    
    return () => {
      canvas.removeEventListener('click', onClick);
    };
  }, [enabled, activeTool, meshData, scene, camera, gl, tempPoints, snapEnabled, snapDistance, meshRef]);
  
  // Handle Escape key to cancel measurement
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && activeTool) {
        clearTempPoints();
        setActiveTool(null);
      }
    };
    
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeTool]);
  
  // Complete measurement when enough points are collected
  const completeMeasurementIfReady = () => {
    const requiredPoints = getRequiredPointsCount(activeTool);
    
    if (tempPoints.length === requiredPoints - 1) {
      // Next point will complete the measurement
      setTimeout(() => {
        const points = useMeasurementStore.getState().tempPoints;
        if (points.length === requiredPoints) {
          createMeasurement(points);
        }
      }, 50);
    }
  };
  
  // Determine required points for each measurement type
  const getRequiredPointsCount = (tool: string | null): number => {
    switch (tool) {
      case 'distance':
        return 2;
      case 'angle':
      case 'radius':
      case 'diameter':
        return 3;
      default:
        return 2;
    }
  };
  
  // Create and store completed measurement
  const createMeasurement = (points: MeasurementPoint[]) => {
    if (!activeTool) return;
    
    let value = 0;
    let unit: 'mm' | 'deg' = 'mm';
    
    // For edge-select, calculate distance between points
    if (activeTool === 'edge-select' && points.length >= 2) {
      value = calculateDistance(points[0].position, points[1].position);
    }
    
    const measurement: Measurement = {
      id: generateMeasurementId(),
      type: activeTool,
      points: points,
      value: value,
      unit: unit,
      label: generateMeasurementLabel(activeTool, value, unit),
      color: '#2563EB', // Blue
      visible: true,
      createdAt: new Date()
    };
    
    addMeasurement(measurement);
  };
  
  // Render measurement markers (temp points)
  const renderMarkers = () => {
    return tempPoints.map((point, index) => (
      <mesh
        key={`temp-marker-${index}`}
        position={[point.position.x, point.position.y, point.position.z]}
        ref={(mesh) => {
          if (mesh && !markersRef.current.includes(mesh)) {
            markersRef.current[index] = mesh;
          }
        }}
      >
        <sphereGeometry args={[0.8, 16, 16]} />
        <meshBasicMaterial color="#2563EB" transparent opacity={0.8} />
      </mesh>
    ));
  };
  
  // Render lines between temp points
  const renderLines = () => {
    if (tempPoints.length < 2) return null;
    
    const linePoints: THREE.Vector3[] = tempPoints.map(p => p.position);
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(linePoints);
    
    return (
      <primitive object={new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: "#2563EB" }))} />
    );
  };
  
  // Render coordinate labels for temp points
  const renderCoordinateLabels = () => {
    return tempPoints.map((point, index) => (
      <Html
        key={`coord-label-${index}`}
        position={[point.position.x, point.position.y + 2, point.position.z]}
        center
        distanceFactor={10}
      >
        <div className="bg-white/95 backdrop-blur-sm px-2 py-1 rounded shadow-lg border border-gray-200 text-xs font-mono">
          <div className="text-blue-600 font-semibold">Point {index + 1}</div>
          <div className="text-gray-700">
            X: {formatCoordinate(point.position.x)}<br/>
            Y: {formatCoordinate(point.position.y)}<br/>
            Z: {formatCoordinate(point.position.z)}
          </div>
        </div>
      </Html>
    ));
  };
  
  // Render completed measurements
  const renderCompletedMeasurements = () => {
    return measurements.filter(m => m.visible).map((measurement) => (
      <group key={measurement.id}>
        {/* Render measurement points */}
        {measurement.points.map((point, index) => (
          <mesh
            key={`${measurement.id}-point-${index}`}
            position={[point.position.x, point.position.y, point.position.z]}
          >
            <sphereGeometry args={[0.6, 16, 16]} />
            <meshBasicMaterial color={measurement.color} transparent opacity={0.7} />
          </mesh>
        ))}
        
        {/* Render measurement line */}
        {measurement.points.length >= 2 && (() => {
          const points = measurement.points.map(p => p.position);
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const material = new THREE.LineBasicMaterial({ color: measurement.color });
          return <primitive object={new THREE.Line(geometry, material)} />;
        })()}
        
        {/* Render measurement label */}
        {renderMeasurementLabel(measurement)}
      </group>
    ));
  };
  
  // Render measurement label at midpoint
  const renderMeasurementLabel = (measurement: Measurement) => {
    let labelPosition: THREE.Vector3;
    
    if (measurement.points.length >= 2) {
      labelPosition = getMidpoint(measurement.points[0].position, measurement.points[1].position);
    } else {
      return null;
    }
    
    return (
      <Html
        position={[labelPosition.x, labelPosition.y, labelPosition.z]}
        center
        distanceFactor={10}
      >
        <div className="bg-blue-600 text-white px-3 py-1.5 rounded-full shadow-lg text-sm font-semibold whitespace-nowrap">
          {measurement.label}
        </div>
      </Html>
    );
  };
  
  if (!enabled) return null;
  
  return (
    <group>
      {/* Temporary measurement visualization */}
      {renderMarkers()}
      {renderLines()}
      {renderCoordinateLabels()}
      
      {/* Completed measurements */}
      {renderCompletedMeasurements()}
    </group>
  );
}