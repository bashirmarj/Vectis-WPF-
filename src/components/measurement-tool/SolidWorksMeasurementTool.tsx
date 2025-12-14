/**
 * SolidWorksMeasurementTool.tsx
 *
 * Three.js integration component for SolidWorks-style measurements.
 * This component handles 3D scene interaction, raycasting, and measurement visualization.
 *
 * Features:
 * - Unified measurement mode (single tool, multiple measurement types)
 * - Edge detection with hover highlights
 * - Face selection for area and face-to-face measurements
 * - Point-to-point distance measurement
 * - SolidWorks-style callouts and visualizations
 */

import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import {
  useSolidWorksMeasureStore,
  createMeasurement,
  type SWMeasurement,
} from '@/stores/solidworksMeasureStore';
import { measurementApi, localCalculations } from '@/services/measurementApiService';
import { toast } from '@/hooks/use-toast';

// === Type Definitions ===

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_face_ids?: number[];
  face_classifications?: Array<{
    face_id: number;
    type: string;
    center: [number, number, number];
    normal: [number, number, number];
    area: number;
    surface_type: string;
    radius?: number;
  }>;
  tagged_edges?: Array<{
    feature_id: number;
    start: [number, number, number];
    end: [number, number, number];
    type: 'line' | 'circle' | 'arc';
    diameter?: number;
    radius?: number;
    length?: number;
  }>;
}

interface SolidWorksMeasurementToolProps {
  meshData: MeshData | null;
  meshRef: THREE.Mesh | null;
  featureEdgesGeometry: THREE.BufferGeometry | null;
  enabled: boolean;
  boundingSphere?: { center: THREE.Vector3; radius: number };
}

interface HoverEdgeInfo {
  position: THREE.Vector3;
  edge: THREE.Line3;
  allSegments: THREE.Line3[];
  taggedEdge: NonNullable<MeshData['tagged_edges']>[0];
  label: string;
}

interface MarkerData {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  faceId?: number;
  faceInfo?: NonNullable<MeshData['face_classifications']>[0];
}

// === Main Component ===

export const SolidWorksMeasurementTool: React.FC<SolidWorksMeasurementToolProps> = ({
  meshData,
  meshRef,
  featureEdgesGeometry,
  enabled,
  boundingSphere,
}) => {
  const { camera, gl, raycaster } = useThree();

  // Store hooks
  const {
    measurements,
    isActive,
    unitSystem,
    addMeasurement,
    setHoverInfo,
  } = useSolidWorksMeasureStore();

  // Local state
  const [hoverEdge, setHoverEdge] = useState<HoverEdgeInfo | null>(null);
  const [markers, setMarkers] = useState<MarkerData[]>([]);
  const [tempMarker, setTempMarker] = useState<MarkerData | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  // Convert feature edges geometry to Line3 array
  const edgeLines = useMemo(() => {
    if (!featureEdgesGeometry) return [];
    const lines: THREE.Line3[] = [];
    const positions = featureEdgesGeometry.attributes.position;

    for (let i = 0; i < positions.count; i += 2) {
      lines.push(
        new THREE.Line3(
          new THREE.Vector3(positions.getX(i), positions.getY(i), positions.getZ(i)),
          new THREE.Vector3(positions.getX(i + 1), positions.getY(i + 1), positions.getZ(i + 1))
        )
      );
    }
    return lines;
  }, [featureEdgesGeometry]);

  // Find matching tagged edge for a given edge line
  const findTaggedEdge = useCallback((
    edge: THREE.Line3,
    taggedEdges: NonNullable<MeshData['tagged_edges']>
  ) => {
    const tolerance = 0.5;

    for (const tagged of taggedEdges) {
      const taggedStart = new THREE.Vector3(...tagged.start);
      const taggedEnd = new THREE.Vector3(...tagged.end);

      // Check if edge matches this tagged edge segment
      const matchesForward =
        edge.start.distanceTo(taggedStart) < tolerance &&
        edge.end.distanceTo(taggedEnd) < tolerance;
      const matchesReverse =
        edge.start.distanceTo(taggedEnd) < tolerance &&
        edge.end.distanceTo(taggedStart) < tolerance;

      if (matchesForward || matchesReverse) {
        return tagged;
      }
    }

    return null;
  }, []);

  // Get all segments belonging to a feature
  const getFeatureSegments = useCallback((
    featureId: number,
    taggedEdges: NonNullable<MeshData['tagged_edges']>
  ): THREE.Line3[] => {
    return taggedEdges
      .filter(e => e.feature_id === featureId)
      .map(e => new THREE.Line3(
        new THREE.Vector3(...e.start),
        new THREE.Vector3(...e.end)
      ));
  }, []);

  // Reset state when disabled
  useEffect(() => {
    if (!enabled) {
      setMarkers([]);
      setTempMarker(null);
      setHoverEdge(null);
      setIsDragging(false);
      dragStartRef.current = null;
      setHoverInfo(null);
    }
  }, [enabled, setHoverInfo]);

  // Handle pointer move for edge hover detection
  useEffect(() => {
    if (!enabled || !meshRef || edgeLines.length === 0) {
      setHoverEdge(null);
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length > 0) {
        const point = intersects[0].point;

        // Find closest edge
        let closestEdge: THREE.Line3 | null = null;
        let minDist = 2.0; // Detection threshold in mm

        for (const edge of edgeLines) {
          const closestPoint = new THREE.Vector3();
          edge.closestPointToPoint(point, true, closestPoint);
          const dist = point.distanceTo(closestPoint);

          if (dist < minDist) {
            minDist = dist;
            closestEdge = edge;
          }
        }

        if (closestEdge && meshData?.tagged_edges) {
          const taggedEdge = findTaggedEdge(closestEdge, meshData.tagged_edges);

          if (taggedEdge) {
            const allSegments = getFeatureSegments(taggedEdge.feature_id, meshData.tagged_edges);

            let label = '';
            if (taggedEdge.type === 'circle' && taggedEdge.diameter) {
              label = `Ø ${taggedEdge.diameter.toFixed(2)} mm`;
            } else if (taggedEdge.type === 'arc' && taggedEdge.radius) {
              label = `R ${taggedEdge.radius.toFixed(2)} mm`;
            } else if (taggedEdge.type === 'line' && taggedEdge.length) {
              label = `${taggedEdge.length.toFixed(2)} mm`;
            } else {
              label = 'Edge detected';
            }

            setHoverEdge({
              position: point,
              edge: closestEdge,
              allSegments,
              taggedEdge,
              label,
            });

            setHoverInfo({
              type: 'edge',
              position: point,
              edgeInfo: taggedEdge,
              label,
            });
            return;
          }
        }
      }

      setHoverEdge(null);
      setHoverInfo(null);
    };

    gl.domElement.addEventListener('pointermove', handlePointerMove);
    return () => gl.domElement.removeEventListener('pointermove', handlePointerMove);
  }, [enabled, meshRef, meshData, edgeLines, camera, gl, raycaster, findTaggedEdge, getFeatureSegments, setHoverInfo]);

  // Handle click for measurements
  useEffect(() => {
    if (!enabled || !meshRef) return;

    const handlePointerDown = (event: PointerEvent) => {
      dragStartRef.current = { x: event.clientX, y: event.clientY };
    };

    const handlePointerUp = async (event: PointerEvent) => {
      if (!dragStartRef.current) return;

      // Check if this was a drag (camera rotation)
      const dx = event.clientX - dragStartRef.current.x;
      const dy = event.clientY - dragStartRef.current.y;
      const dragDistance = Math.sqrt(dx * dx + dy * dy);

      if (dragDistance > 5) {
        // Was a drag, not a click
        dragStartRef.current = null;
        setIsDragging(false);
        return;
      }

      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length === 0) {
        dragStartRef.current = null;
        setIsDragging(false);
        return;
      }

      // Priority 1: Edge measurement
      if (hoverEdge) {
        const { taggedEdge } = hoverEdge;

        let measurement: SWMeasurement;

        if (taggedEdge.type === 'circle' && taggedEdge.diameter) {
          measurement = createMeasurement('edge_diameter', taggedEdge.diameter, {
            edgeType: 'circle',
            diameter: taggedEdge.diameter,
            radius: taggedEdge.radius,
            point1: new THREE.Vector3(...taggedEdge.start),
            backendMatch: true,
          });

          toast({
            title: 'Diameter Measured',
            description: `Ø ${taggedEdge.diameter.toFixed(2)} mm`,
          });
        } else if (taggedEdge.type === 'arc' && taggedEdge.radius) {
          measurement = createMeasurement('edge_radius', taggedEdge.radius, {
            edgeType: 'arc',
            radius: taggedEdge.radius,
            point1: new THREE.Vector3(...taggedEdge.start),
            backendMatch: true,
          });

          toast({
            title: 'Radius Measured',
            description: `R ${taggedEdge.radius.toFixed(2)} mm`,
          });
        } else if (taggedEdge.type === 'line' && taggedEdge.length) {
          measurement = createMeasurement('edge_length', taggedEdge.length, {
            edgeType: 'line',
            point1: new THREE.Vector3(...taggedEdge.start),
            point2: new THREE.Vector3(...taggedEdge.end),
            backendMatch: true,
          });

          toast({
            title: 'Length Measured',
            description: `${taggedEdge.length.toFixed(2)} mm`,
          });
        } else {
          dragStartRef.current = null;
          return;
        }

        addMeasurement(measurement);
        dragStartRef.current = null;
        setIsDragging(false);
        return;
      }

      // Priority 2: Face/Point measurement
      const intersection = intersects[0];
      const point = intersection.point.clone();
      const normal = intersection.face?.normal.clone() || new THREE.Vector3(0, 1, 0);

      // Get face info if available
      let faceInfo: NonNullable<MeshData['face_classifications']>[0] | undefined;
      if (meshData?.vertex_face_ids && meshData?.face_classifications && intersection.face) {
        const vertexIndex = intersection.face.a;
        const faceId = meshData.vertex_face_ids[vertexIndex];
        faceInfo = meshData.face_classifications.find(f => f.face_id === faceId);
      }

      const newMarker: MarkerData = {
        point,
        normal,
        faceId: faceInfo?.face_id,
        faceInfo,
      };

      if (markers.length === 0) {
        // First point
        setMarkers([newMarker]);
        toast({
          title: 'First Point Placed',
          description: 'Click another point to measure distance',
        });
      } else if (markers.length === 1) {
        // Second point - create measurement
        const marker1 = markers[0];
        const marker2 = newMarker;

        // Use local calculation (fast) or API (if needed)
        const result = localCalculations.pointToPoint(marker1.point, marker2.point, unitSystem);

        // Calculate angle between face normals if available
        let angle: number | undefined;
        let isParallel: boolean | undefined;
        let perpDistance: number | undefined;

        if (marker1.faceInfo && marker2.faceInfo) {
          const n1 = new THREE.Vector3(...marker1.faceInfo.normal);
          const n2 = new THREE.Vector3(...marker2.faceInfo.normal);
          const dot = Math.abs(n1.dot(n2));
          angle = Math.acos(Math.min(1.0, Math.max(-1.0, dot))) * (180 / Math.PI);
          isParallel = angle < 5 || angle > 175;

          if (isParallel) {
            const diff = marker2.point.clone().sub(marker1.point);
            perpDistance = Math.abs(diff.dot(n1));
          }
        }

        const measurement = createMeasurement('point_to_point', result.value, {
          point1: marker1.point,
          point2: marker2.point,
          deltaX: result.delta_x,
          deltaY: result.delta_y,
          deltaZ: result.delta_z,
          angle,
          isParallel,
          perpendicularDistance: perpDistance,
          backendMatch: false,
        });

        addMeasurement(measurement);
        setMarkers([]);
        setTempMarker(null);

        let description = `${result.value.toFixed(2)} mm`;
        if (angle !== undefined) {
          description += ` (${angle.toFixed(1)}°)`;
        }

        toast({
          title: 'Distance Measured',
          description,
        });
      } else {
        // Reset and start new measurement
        setMarkers([newMarker]);
        setTempMarker(null);
      }

      dragStartRef.current = null;
      setIsDragging(false);
    };

    gl.domElement.addEventListener('pointerdown', handlePointerDown);
    gl.domElement.addEventListener('pointerup', handlePointerUp);

    return () => {
      gl.domElement.removeEventListener('pointerdown', handlePointerDown);
      gl.domElement.removeEventListener('pointerup', handlePointerUp);
    };
  }, [enabled, meshRef, meshData, markers, hoverEdge, addMeasurement, camera, gl, raycaster, unitSystem]);

  // Render helpers
  const markerRadius = (boundingSphere?.radius || 50) * 0.015;
  const lineWidth = 2;

  return (
    <>
      {/* Edge hover highlight - SolidWorks orange style */}
      {enabled && hoverEdge && (
        <group>
          {hoverEdge.allSegments.map((segment, idx) => (
            <Line
              key={idx}
              points={[segment.start, segment.end]}
              color="#ff8800"
              lineWidth={4}
            />
          ))}
        </group>
      )}

      {/* Edge hover label */}
      {enabled && hoverEdge && (
        <Html position={hoverEdge.position} center>
          <div
            style={{
              background: 'rgba(26, 95, 180, 0.95)',
              color: 'white',
              padding: '6px 12px',
              borderRadius: '4px',
              fontSize: '14px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
              fontFamily: '"Segoe UI", Arial, sans-serif',
            }}
          >
            {hoverEdge.label}
          </div>
        </Html>
      )}

      {/* Face point markers */}
      {enabled && markers.map((marker, index) => (
        <group key={index} position={marker.point}>
          {/* Sphere marker */}
          <mesh>
            <sphereGeometry args={[markerRadius, 16, 16]} />
            <meshBasicMaterial color="#1a5fb4" />
          </mesh>
          {/* Crosshair lines */}
          <Line
            points={[
              [marker.point.x - markerRadius * 2, marker.point.y, marker.point.z],
              [marker.point.x + markerRadius * 2, marker.point.y, marker.point.z],
            ]}
            color="#1a5fb4"
            lineWidth={2}
          />
          <Line
            points={[
              [marker.point.x, marker.point.y - markerRadius * 2, marker.point.z],
              [marker.point.x, marker.point.y + markerRadius * 2, marker.point.z],
            ]}
            color="#1a5fb4"
            lineWidth={2}
          />
        </group>
      ))}

      {/* Line between markers */}
      {enabled && markers.length === 2 && (
        <Line
          points={[markers[0].point, markers[1].point]}
          color="#1a5fb4"
          lineWidth={lineWidth}
          dashed
          dashSize={markerRadius}
          gapSize={markerRadius * 0.5}
        />
      )}

      {/* Measurement visualizations */}
      <SolidWorksMeasurementRenderer boundingSphere={boundingSphere} />
    </>
  );
};

// === Measurement Renderer Component ===

interface MeasurementRendererProps {
  boundingSphere?: { center: THREE.Vector3; radius: number };
}

const SolidWorksMeasurementRenderer: React.FC<MeasurementRendererProps> = ({
  boundingSphere,
}) => {
  const { measurements, unitSystem } = useSolidWorksMeasureStore();
  const visibleMeasurements = measurements.filter(m => m.visible);

  const markerRadius = (boundingSphere?.radius || 50) * 0.012;

  const formatValue = useCallback((value: number, unit: string) => {
    if (unit === 'deg') return `${value.toFixed(2)}°`;
    if (unit === 'mm²') {
      return unitSystem === 'imperial'
        ? `${(value * 0.00155).toFixed(3)} in²`
        : `${value.toFixed(3)} mm²`;
    }
    return unitSystem === 'imperial'
      ? `${(value * 0.0393701).toFixed(3)} in`
      : `${value.toFixed(3)} mm`;
  }, [unitSystem]);

  return (
    <>
      {visibleMeasurements.map((measurement) => {
        const color = '#1a5fb4';

        if (measurement.type === 'point_to_point' && measurement.point1 && measurement.point2) {
          const midpoint = new THREE.Vector3()
            .addVectors(measurement.point1, measurement.point2)
            .multiplyScalar(0.5);

          return (
            <group key={measurement.id}>
              {/* Points */}
              <mesh position={measurement.point1}>
                <sphereGeometry args={[markerRadius, 16, 16]} />
                <meshBasicMaterial color={color} />
              </mesh>
              <mesh position={measurement.point2}>
                <sphereGeometry args={[markerRadius, 16, 16]} />
                <meshBasicMaterial color={color} />
              </mesh>

              {/* Line */}
              <Line
                points={[measurement.point1, measurement.point2]}
                color={color}
                lineWidth={2}
              />

              {/* Label at midpoint */}
              <Html position={midpoint} center distanceFactor={1.5}>
                <div
                  style={{
                    background: 'rgba(255, 255, 255, 0.95)',
                    color: '#1a1a1a',
                    padding: '10px 16px',
                    borderRadius: '6px',
                    fontSize: '16px',
                    fontWeight: '600',
                    border: `2px solid ${color}`,
                    boxShadow: '0 3px 12px rgba(0,0,0,0.2)',
                    minWidth: '120px',
                    textAlign: 'center',
                    fontFamily: '"Segoe UI", Arial, sans-serif',
                  }}
                >
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color }}>
                    {formatValue(measurement.value, measurement.unit)}
                  </div>
                  {measurement.deltaX !== undefined && (
                    <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
                      <span style={{ color: '#dc2626' }}>ΔX:</span> {formatValue(measurement.deltaX, 'mm')}{' '}
                      <span style={{ color: '#16a34a' }}>ΔY:</span> {formatValue(measurement.deltaY || 0, 'mm')}{' '}
                      <span style={{ color: '#2563eb' }}>ΔZ:</span> {formatValue(measurement.deltaZ || 0, 'mm')}
                    </div>
                  )}
                </div>
              </Html>
            </group>
          );
        }

        if ((measurement.type === 'edge_diameter' || measurement.type === 'edge_radius' || measurement.type === 'edge_length') && measurement.point1) {
          const prefix = measurement.type === 'edge_diameter' ? 'Ø ' :
                        measurement.type === 'edge_radius' ? 'R ' : '';

          return (
            <group key={measurement.id}>
              <mesh position={measurement.point1}>
                <sphereGeometry args={[markerRadius, 16, 16]} />
                <meshBasicMaterial color={color} />
              </mesh>

              <Html position={measurement.point1} center distanceFactor={1.5}>
                <div
                  style={{
                    background: 'rgba(255, 255, 255, 0.95)',
                    color: '#1a1a1a',
                    padding: '10px 16px',
                    borderRadius: '6px',
                    fontSize: '16px',
                    fontWeight: '600',
                    border: `2px solid ${color}`,
                    boxShadow: '0 3px 12px rgba(0,0,0,0.2)',
                    minWidth: '100px',
                    textAlign: 'center',
                    fontFamily: '"Segoe UI", Arial, sans-serif',
                  }}
                >
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color }}>
                    {prefix}{formatValue(measurement.value, measurement.unit)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#666', marginTop: '2px', textTransform: 'uppercase' }}>
                    {measurement.edgeType}
                  </div>
                </div>
              </Html>
            </group>
          );
        }

        return null;
      })}
    </>
  );
};

export default SolidWorksMeasurementTool;
