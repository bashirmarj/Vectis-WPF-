import { useEffect, useRef, useState } from 'react';
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
  snapToEdge,
  formatCoordinate,
  generateMeasurementId,
  extractEdges,
  classifyEdge,
  EdgeClassification,
} from '@/lib/measurementUtils';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
}

interface ProfessionalMeasurementToolProps {
  meshData?: MeshData;
  meshRef?: React.RefObject<THREE.Mesh>;
  enabled: boolean;
}

/**
 * Professional Measurement Tool Component
 * ✅ NEW: Includes single-click edge selection with automatic line/arc/circle detection
 */
export function ProfessionalMeasurementTool({ 
  meshData, 
  meshRef,
  enabled 
}: ProfessionalMeasurementToolProps) {
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
    setActiveTool,
    hoverPoint,
    setHoverPoint,
  } = useMeasurementStore();
  
  const markersRef = useRef<THREE.Mesh[]>([]);
  const css2dRenderer = useRef<CSS2DRenderer | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<EdgeClassification | null>(null);
  
  // Initialize CSS2D Renderer for HTML labels
  useEffect(() => {
    if (!enabled) return;
    
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(gl.domElement.clientWidth, gl.domElement.clientHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none';
    labelRenderer.domElement.style.zIndex = '5'; // ✅ Prevent overlap with toolbar
    gl.domElement.parentElement?.appendChild(labelRenderer.domElement);
    
    css2dRenderer.current = labelRenderer;
    
    return () => {
      labelRenderer.domElement.remove();
      css2dRenderer.current = null;
    };
  }, [enabled, gl]);
  
  // Handle mouse move for hover feedback on edges
  useEffect(() => {
    if (!enabled || activeTool !== 'edge-select' || !meshData) return;
    
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    
    const onMouseMove = (event: MouseEvent) => {
      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      
      raycaster.setFromCamera(mouse, camera);
      
      const targets = meshRef?.current ? [meshRef.current] : scene.children.filter(
        child => child.type === 'Mesh' && !markersRef.current.includes(child as THREE.Mesh)
      );
      
      const intersects = raycaster.intersectObjects(targets, true);
      
      if (intersects.length > 0) {
        const point = intersects[0].point.clone();
        
        // Snap to edge and get classification
        if (snapEnabled && meshData) {
          const snapped = snapToEdge(point, meshData, snapDistance * 2); // Larger snap distance for edges
          if (snapped && snapped.metadata?.edgeType) {
            const edgeData: EdgeClassification = {
              type: snapped.metadata.edgeType,
              start: snapped.metadata.startPoint || new THREE.Vector3(),
              end: snapped.metadata.endPoint || new THREE.Vector3(),
              radius: snapped.metadata.radius,
              center: snapped.metadata.center,
            };
            
            setHoveredEdge(edgeData);
            setHoverPoint({
              id: `hover-${Date.now()}`,
              position: snapped.position,
              surfaceType: 'edge'
            });
            return;
          }
        }
      }
      
      setHoveredEdge(null);
      setHoverPoint(null);
    };
    
    const canvas = gl.domElement;
    canvas.addEventListener('mousemove', onMouseMove);
    
    return () => {
      canvas.removeEventListener('mousemove', onMouseMove);
    };
  }, [enabled, activeTool, meshData, scene, camera, gl, snapEnabled, snapDistance, meshRef]);
  
  // Handle click to place measurement points or select edges
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
      
      const targets = meshRef?.current ? [meshRef.current] : scene.children.filter(
        child => child.type === 'Mesh' && !markersRef.current.includes(child as THREE.Mesh)
      );
      
      const intersects = raycaster.intersectObjects(targets, true);
      
      if (intersects.length > 0) {
        let point = intersects[0].point.clone();
        const normal = intersects[0].face?.normal || new THREE.Vector3(0, 1, 0);
        
        // ✅ NEW: Handle edge-select mode (single-click automatic measurement)
        if (activeTool === 'edge-select') {
          if (snapEnabled && meshData) {
            const snapped = snapToEdge(point, meshData, snapDistance * 2);
            if (snapped && snapped.metadata?.edgeType) {
              point = snapped.position;
              
              // Create single-click measurement based on edge type
              createEdgeSelectMeasurement(snapped);
              return;
            }
          }
        }
        
        // Regular multi-point measurements
        if (snapEnabled && meshData) {
          const snapped = snapToVertex(point, meshData, snapDistance);
          if (snapped) {
            point = snapped.position;
          }
        }
        
        const measurementPoint: MeasurementPoint = {
          id: `point-${Date.now()}-${Math.random()}`,
          position: point,
          normal: normal,
          surfaceType: 'vertex'
        };
        
        addTempPoint(measurementPoint);
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
        setHoveredEdge(null);
        setHoverPoint(null);
      }
      
      // Undo/Redo
      if (event.ctrlKey || event.metaKey) {
        if (event.key === 'z' && !event.shiftKey) {
          useMeasurementStore.getState().undo();
          event.preventDefault();
        } else if (event.key === 'y' || (event.key === 'z' && event.shiftKey)) {
          useMeasurementStore.getState().redo();
          event.preventDefault();
        }
      }
    };
    
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeTool]);
  
  // ✅ NEW: Create edge-select measurement (single-click automatic)
  const createEdgeSelectMeasurement = (snapped: any) => {
    if (!snapped.metadata) return;
    
    const { edgeType, radius, center, startPoint, endPoint } = snapped.metadata;
    
    let value = 0;
    let label = '';
    let measurementType: 'distance' | 'radius' | 'diameter' = 'distance';
    
    if (edgeType === 'line') {
      // Straight line - show length
      if (startPoint && endPoint) {
        value = startPoint.distanceTo(endPoint);
        label = `Length: ${value.toFixed(2)} mm`;
        measurementType = 'distance';
      }
    } else if (edgeType === 'circle') {
      // Circle - show diameter
      if (radius) {
        value = radius * 2;
        label = `⊙ Diameter: ${value.toFixed(2)} mm`;
        measurementType = 'diameter';
      }
    } else if (edgeType === 'arc') {
      // Arc - show radius
      if (radius) {
        value = radius;
        label = `⌒ Radius: ${value.toFixed(2)} mm`;
        measurementType = 'radius';
      }
    }
    
    const measurement: Measurement = {
      id: generateMeasurementId(),
      type: 'edge-select',
      points: [{
        id: `point-${Date.now()}`,
        position: snapped.position,
        normal: new THREE.Vector3(0, 1, 0),
        surfaceType: 'edge',
      }],
      value: value,
      unit: 'mm',
      label: label,
      color: '#10B981', // Green for smart selection
      visible: true,
      createdAt: new Date(),
      metadata: {
        edgeType: edgeType,
        arcRadius: radius,
        arcCenter: center,
        edgeStart: startPoint,
        edgeEnd: endPoint,
      }
    };
    
    addMeasurement(measurement);
  };
  
  // Complete measurement when enough points are collected
  const completeMeasurementIfReady = () => {
    const requiredPoints = getRequiredPointsCount(activeTool);
    
    if (tempPoints.length === requiredPoints - 1) {
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
      case 'coordinate':
        return 1;
      case 'edge-select':
        return 1; // Single click
      default:
        return 2;
    }
  };
  
  // Create and store completed measurement (multi-point)
  const createMeasurement = (points: MeasurementPoint[]) => {
    if (!activeTool) return;
    
    let value = 0;
    let unit: 'mm' | 'deg' | 'coord' = 'mm';
    
    switch (activeTool) {
      case 'distance':
        if (points.length >= 2) {
          value = calculateDistance(points[0].position, points[1].position);
        }
        break;
      case 'angle':
        if (points.length >= 3) {
          value = calculateAngle(points[0].position, points[1].position, points[2].position);
          unit = 'deg';
        }
        break;
      case 'radius':
        if (points.length >= 3) {
          value = calculateRadius(points[0].position, points[1].position, points[2].position);
        }
        break;
      case 'diameter':
        if (points.length >= 3) {
          value = calculateRadius(points[0].position, points[1].position, points[2].position) * 2;
        }
        break;
      case 'coordinate':
        if (points.length >= 1) {
          const pos = points[0].position;
          value = 0; // Coordinate doesn't have a single value
          unit = 'coord';
        }
        break;
    }
    
    const measurement: Measurement = {
      id: generateMeasurementId(),
      type: activeTool as any,
      points: points,
      value: value,
      unit: unit,
      label: activeTool === 'coordinate' 
        ? `X: ${formatCoordinate(points[0].position.x)}  Y: ${formatCoordinate(points[0].position.y)}  Z: ${formatCoordinate(points[0].position.z)}`
        : generateMeasurementLabel(activeTool, value, unit),
      color: '#2563EB',
      visible: true,
      createdAt: new Date()
    };
    
    addMeasurement(measurement);
  };
  
  // Render hover indicator for edge-select mode
  const renderHoverIndicator = () => {
    if (!hoveredEdge || !hoverPoint || activeTool !== 'edge-select') return null;
    
    const position = hoverPoint.position;
    
    let hoverText = '';
    if (hoveredEdge.type === 'line') {
      hoverText = `📏 Line: ${hoveredEdge.length?.toFixed(2) || '0'} mm`;
    } else if (hoveredEdge.type === 'circle') {
      hoverText = `⊙ Circle: ø ${((hoveredEdge.radius || 0) * 2).toFixed(2)} mm`;
    } else if (hoveredEdge.type === 'arc') {
      hoverText = `⌒ Arc: R ${hoveredEdge.radius?.toFixed(2) || '0'} mm`;
    }
    
    return (
      <Html position={position} center distanceFactor={10}>
        <div className="bg-emerald-500/90 text-white px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap shadow-lg pointer-events-none">
          {hoverText}
        </div>
      </Html>
    );
  };
  
  // Render measurement annotations
  const renderMeasurements = () => {
    return measurements
      .filter(m => m.visible)
      .map(measurement => (
        <MeasurementAnnotation key={measurement.id} measurement={measurement} />
      ));
  };
  
  // Render temporary points
  const renderTempPoints = () => {
    return tempPoints.map((point, index) => (
      <mesh key={point.id} position={point.position}>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshBasicMaterial color="#3B82F6" />
      </mesh>
    ));
  };
  
  return (
    <>
      {renderTempPoints()}
      {renderMeasurements()}
      {renderHoverIndicator()}
    </>
  );
}

// Measurement Annotation Component
function MeasurementAnnotation({ measurement }: { measurement: Measurement }) {
  const getLabelPosition = () => {
    if (measurement.type === 'edge-select' && measurement.points.length === 1) {
      // For edge-select, position label at the clicked point
      return measurement.points[0].position;
    }
    
    if (measurement.type === 'distance' && measurement.points.length >= 2) {
      return getMidpoint(measurement.points[0].position, measurement.points[1].position);
    }
    
    if (measurement.type === 'angle' && measurement.points.length >= 3) {
      return measurement.points[1].position;
    }
    
    if (measurement.type === 'coordinate' && measurement.points.length >= 1) {
      return measurement.points[0].position;
    }
    
    return measurement.points[0]?.position || new THREE.Vector3();
  };
  
  const labelPosition = getLabelPosition();
  
  // Render lines connecting points (except for edge-select and coordinate)
  const renderLines = () => {
    if (measurement.type === 'edge-select' || measurement.type === 'coordinate') {
      return null; // No lines for single-point measurements
    }
    
    if (measurement.type === 'distance' && measurement.points.length >= 2) {
      return (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([
                ...measurement.points[0].position.toArray(),
                ...measurement.points[1].position.toArray(),
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={measurement.color} linewidth={2} />
        </line>
      );
    }
    
    if (measurement.type === 'angle' && measurement.points.length >= 3) {
      return (
        <>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([
                  ...measurement.points[0].position.toArray(),
                  ...measurement.points[1].position.toArray(),
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color={measurement.color} linewidth={2} />
          </line>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([
                  ...measurement.points[1].position.toArray(),
                  ...measurement.points[2].position.toArray(),
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color={measurement.color} linewidth={2} />
          </line>
        </>
      );
    }
    
    return null;
  };
  
  return (
    <group>
      {/* Point markers */}
      {measurement.points.map((point, index) => (
        <mesh key={`${measurement.id}-point-${index}`} position={point.position}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color={measurement.color} />
        </mesh>
      ))}
      
      {/* Connecting lines */}
      {renderLines()}
      
      {/* Label */}
      <Html position={labelPosition} center distanceFactor={10}>
        <div className={`${
          measurement.type === 'edge-select' ? 'bg-emerald-600' : 'bg-blue-600'
        }/90 text-white px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap shadow-lg`}>
          {measurement.label}
        </div>
      </Html>
    </group>
  );
}
