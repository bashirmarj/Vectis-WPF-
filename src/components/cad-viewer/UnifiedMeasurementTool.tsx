import React, { useEffect, useState, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { MeasurementRenderer } from "./MeasurementRenderer";
import { FaceCrosshairMarker } from "./measurements/FaceCrosshairMarker";
import { toast } from "@/hooks/use-toast";
import { findEdgeByFeatureId, getFeatureSegments, TaggedFeatureEdge } from "@/lib/featureIdMatcher";
import { calculateMarkerValues, RadDeg } from "@/lib/faceMeasurementUtils";

interface MarkerData {
  point: THREE.Vector3;
  intersection: THREE.Intersection;
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  vertex_face_ids?: number[];
  face_classifications?: any[];
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: any[];
  tagged_edges?: TaggedFeatureEdge[];
}

interface UnifiedMeasurementToolProps {
  meshData: MeshData | null;
  meshRef: THREE.Mesh | null;
  featureEdgesGeometry: THREE.BufferGeometry | null;
  enabled: boolean;
  boundingSphere?: { center: THREE.Vector3; radius: number };
}

export const UnifiedMeasurementTool: React.FC<UnifiedMeasurementToolProps> = ({
  meshData,
  meshRef,
  featureEdgesGeometry,
  enabled,
  boundingSphere,
}) => {
  const { camera, gl, raycaster } = useThree();
  
  // Edge detection state
  const [hoverInfo, setHoverInfo] = useState<{
    position: THREE.Vector3;
    classification: any;
    edge: THREE.Line3;
    allSegments?: THREE.Line3[];
  } | null>(null);
  const [labelText, setLabelText] = useState<string>("");

  // Face point-to-point state
  const [markers, setMarkers] = useState<MarkerData[]>([]);
  const [tempMarker, setTempMarker] = useState<MarkerData | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  const addMeasurement = useMeasurementStore((state) => state.addMeasurement);
  const activeTool = useMeasurementStore((state) => state.activeTool);

  // Convert feature edges to Line3 array
  const edgeLines = React.useMemo(() => {
    if (!featureEdgesGeometry) return [];
    const lines: THREE.Line3[] = [];
    const positions = featureEdgesGeometry.attributes.position;
    
    for (let i = 0; i < positions.count; i += 2) {
      lines.push(
        new THREE.Line3(
          new THREE.Vector3(positions.getX(i), positions.getY(i), positions.getZ(i)),
          new THREE.Vector3(positions.getX(i + 1), positions.getY(i + 1), positions.getZ(i + 1)),
        ),
      );
    }
    return lines;
  }, [featureEdgesGeometry]);

  // Debug: Log tagged_edges availability
  useEffect(() => {
    if (enabled) {
      console.log('📏 UnifiedMeasurementTool enabled:', {
        hasTaggedEdges: !!meshData?.tagged_edges,
        taggedEdgesCount: meshData?.tagged_edges?.length || 0,
        hasFeatureEdgesGeometry: !!featureEdgesGeometry,
        edgeLinesCount: edgeLines.length,
        sampleTaggedEdge: meshData?.tagged_edges?.[0],
      });
    }
  }, [enabled, meshData?.tagged_edges, featureEdgesGeometry, edgeLines.length]);

  // Reset when disabled
  useEffect(() => {
    if (!enabled) {
      setMarkers([]);
      setTempMarker(null);
      setHoverInfo(null);
      setLabelText("");
      setIsDragging(false);
      dragStartRef.current = null;
    }
  }, [enabled]);

  // Handle hover detection for edges
  useEffect(() => {
    if (!enabled || !meshRef || edgeLines.length === 0) {
      setHoverInfo(null);
      setLabelText("");
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length > 0) {
        const point = intersects[0].point;

        let closestEdge: THREE.Line3 | null = null;
        let minDist = 1.5; // Detection threshold in mm

        edgeLines.forEach((edge) => {
          const closestPoint = new THREE.Vector3();
          edge.closestPointToPoint(point, true, closestPoint);
          const dist = point.distanceTo(closestPoint);

          if (dist <= minDist) {
            minDist = dist;
            closestEdge = edge;
          }
        });

        if (closestEdge) {
          // Check if this edge has backend classification
          const taggedEdge = findEdgeByFeatureId(closestEdge, meshData?.tagged_edges);

          if (taggedEdge) {
            // Get all segments belonging to this feature
            const allFeatureSegments = getFeatureSegments(
              taggedEdge.feature_id,
              meshData?.tagged_edges
            );
            
            const allSegmentLines = allFeatureSegments.map(seg => 
              new THREE.Line3(
                new THREE.Vector3(...seg.start),
                new THREE.Vector3(...seg.end)
              )
            );

            let label = "";
            if (taggedEdge.type === "circle" && taggedEdge.diameter) {
              label = `⊙ Diameter: ø${taggedEdge.diameter.toFixed(2)} mm`;
            } else if (taggedEdge.type === "arc" && taggedEdge.radius) {
              label = `⌒ Radius: R${taggedEdge.radius.toFixed(2)} mm`;
            } else if (taggedEdge.type === "line" && taggedEdge.length) {
              label = `📏 Length: ${taggedEdge.length.toFixed(2)} mm`;
            } else {
              label = "Feature detected";
            }

            setLabelText(label);
            setHoverInfo({
              position: point,
              classification: taggedEdge,
              edge: closestEdge,
              allSegments: allSegmentLines,
            });
          } else {
            setHoverInfo(null);
            setLabelText("");
          }
        } else {
          setHoverInfo(null);
          setLabelText("");
        }
      } else {
        setHoverInfo(null);
        setLabelText("");
      }
    };

    gl.domElement.addEventListener("pointermove", handlePointerMove);
    return () => gl.domElement.removeEventListener("pointermove", handlePointerMove);
  }, [enabled, meshRef, edgeLines, meshData?.tagged_edges, camera, gl, raycaster]);

  // Face point-to-point: Temporary marker preview on pointer move
  useEffect(() => {
    if (!enabled || !meshRef) return;

    const handlePointerMove = (event: PointerEvent) => {
      if (!isDragging) return;

      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length > 0) {
        setTempMarker({
          point: intersects[0].point.clone(),
          intersection: intersects[0],
        });
      }
    };

    gl.domElement.addEventListener("pointermove", handlePointerMove);
    return () => gl.domElement.removeEventListener("pointermove", handlePointerMove);
  }, [enabled, meshRef, isDragging, camera, gl, raycaster]);

  // Handle clicks: Priority 1 - Edge, Priority 2 - Face point
  useEffect(() => {
    if (!enabled || !meshRef) return;

    const handlePointerDown = (event: PointerEvent) => {
      dragStartRef.current = { x: event.clientX, y: event.clientY };
    };

    const handlePointerUp = (event: PointerEvent) => {
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

      // Valid click - check for edge first
      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length === 0) {
        dragStartRef.current = null;
        setIsDragging(false);
        return;
      }

      // Priority 1: Check if clicking on an edge
      if (hoverInfo && hoverInfo.classification) {
        const taggedEdge = hoverInfo.classification as TaggedFeatureEdge;
        
        // Create edge measurement
        if (taggedEdge.type === "circle" && taggedEdge.diameter) {
          addMeasurement({
            id: generateMeasurementId(),
            type: "measure",
            points: [
              {
                id: generateMeasurementId(),
                position: hoverInfo.edge.start,
                normal: new THREE.Vector3(0, 1, 0),
                surfaceType: "edge",
              },
            ],
            value: taggedEdge.diameter,
            unit: "mm",
            label: `Ø ${formatMeasurement(taggedEdge.diameter, "mm")}`,
            color: "#0066CC",
            visible: true,
            createdAt: new Date(),
            metadata: {
              measurementSubtype: "edge",
              edgeType: "circle",
              backendMatch: true,
            },
          });
          
          toast({
            title: "Edge Measured",
            description: `Circle diameter: ${formatMeasurement(taggedEdge.diameter, "mm")}`,
          });
        } else if (taggedEdge.type === "arc" && taggedEdge.radius) {
          addMeasurement({
            id: generateMeasurementId(),
            type: "measure",
            points: [
              {
                id: generateMeasurementId(),
                position: hoverInfo.edge.start,
                normal: new THREE.Vector3(0, 1, 0),
                surfaceType: "edge",
              },
            ],
            value: taggedEdge.radius,
            unit: "mm",
            label: `R ${formatMeasurement(taggedEdge.radius, "mm")}`,
            color: "#0066CC",
            visible: true,
            createdAt: new Date(),
            metadata: {
              measurementSubtype: "edge",
              edgeType: "arc",
              backendMatch: true,
            },
          });
          
          toast({
            title: "Edge Measured",
            description: `Arc radius: ${formatMeasurement(taggedEdge.radius, "mm")}`,
          });
        } else if (taggedEdge.type === "line" && taggedEdge.length) {
          addMeasurement({
            id: generateMeasurementId(),
            type: "measure",
            points: [
              {
                id: generateMeasurementId(),
                position: hoverInfo.edge.start,
                normal: new THREE.Vector3(0, 1, 0),
                surfaceType: "edge",
              },
            ],
            value: taggedEdge.length,
            unit: "mm",
            label: `📏 ${formatMeasurement(taggedEdge.length, "mm")}`,
            color: "#0066CC",
            visible: true,
            createdAt: new Date(),
            metadata: {
              measurementSubtype: "edge",
              edgeType: "line",
              backendMatch: true,
            },
          });
          
          toast({
            title: "Edge Measured",
            description: `Line length: ${formatMeasurement(taggedEdge.length, "mm")}`,
          });
        }

        dragStartRef.current = null;
        setIsDragging(false);
        return;
      }

      // Priority 2: No edge detected - treat as face point click
      const intersection = intersects[0];

      if (markers.length === 0) {
        // First marker
        setMarkers([{
          point: intersection.point.clone(),
          intersection: intersection,
        }]);
        setTempMarker(null);
        toast({
          title: "First Point Placed",
          description: "Click another point on the face",
        });
      } else if (markers.length === 1) {
        // Second marker - calculate measurement
        const marker1 = markers[0];
        const marker2 = {
          point: intersection.point.clone(),
          intersection: intersection,
        };

        const values = calculateMarkerValues(marker1.intersection, marker2.intersection);
        
        const label = values.parallelFacesDistance !== null
          ? `${values.pointsDistance.toFixed(2)} mm (${(values.facesAngle * RadDeg).toFixed(1)}°, ⊥${values.parallelFacesDistance.toFixed(2)} mm)`
          : `${values.pointsDistance.toFixed(2)} mm (${(values.facesAngle * RadDeg).toFixed(1)}°)`;

        addMeasurement({
          id: generateMeasurementId(),
          type: "measure",
          points: [
            {
              id: generateMeasurementId(),
              position: marker1.point,
              normal: marker1.intersection.face!.normal.clone(),
              surfaceType: "face",
            },
            {
              id: generateMeasurementId(),
              position: marker2.point,
              normal: marker2.intersection.face!.normal.clone(),
              surfaceType: "face",
            },
          ],
          value: values.pointsDistance,
          unit: "mm",
          label: label,
          color: "#00CC66",
          visible: true,
          createdAt: new Date(),
          metadata: {
            measurementSubtype: "face-point",
            facesAngle: values.facesAngle * RadDeg,
            parallelFacesDistance: values.parallelFacesDistance,
            backendMatch: false,
          },
        });

        setMarkers([]);
        setTempMarker(null);
        
        toast({
          title: "Measurement Created",
          description: label,
        });
      } else {
        // Reset and start new measurement
        setMarkers([{
          point: intersection.point.clone(),
          intersection: intersection,
        }]);
        setTempMarker(null);
      }

      dragStartRef.current = null;
      setIsDragging(false);
    };

    gl.domElement.addEventListener("pointerdown", handlePointerDown);
    gl.domElement.addEventListener("pointerup", handlePointerUp);

    return () => {
      gl.domElement.removeEventListener("pointerdown", handlePointerDown);
      gl.domElement.removeEventListener("pointerup", handlePointerUp);
    };
  }, [enabled, meshRef, markers, hoverInfo, addMeasurement, camera, gl, raycaster]);

  const radius = boundingSphere?.radius || 50;
  const markerRadius = radius * 0.01;

  return (
    <>
      {/* Edge hover highlight */}
      {enabled && hoverInfo && hoverInfo.allSegments && (
        <group>
          {hoverInfo.allSegments.map((segment, idx) => {
            const positions = new Float32Array([
              ...segment.start.toArray(),
              ...segment.end.toArray(),
            ]);
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            return (
              <primitive key={idx} object={new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: "#ff8800", linewidth: 3 }))} />
            );
          })}
        </group>
      )}

      {/* Edge hover label */}
      {enabled && hoverInfo && labelText && (
        <group position={hoverInfo.position}>
          <Html>
            <div
              style={{
                background: "rgba(0, 0, 0, 0.8)",
                color: "#fff",
                padding: "4px 8px",
                borderRadius: "4px",
                fontSize: "12px",
                whiteSpace: "nowrap",
                pointerEvents: "none",
              }}
            >
              {labelText}
            </div>
          </Html>
        </group>
      )}

      {/* Face markers */}
      {enabled && markers.map((marker, index) => (
        <FaceCrosshairMarker
          key={index}
          intersection={marker.intersection}
          radius={markerRadius}
          visible={true}
        />
      ))}

      {/* Temporary marker preview */}
      {enabled && tempMarker && (
        <FaceCrosshairMarker
          intersection={tempMarker.intersection}
          radius={markerRadius}
          visible={true}
        />
      )}

      {/* Line between two face markers */}
      {enabled && markers.length === 2 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([
                ...markers[0].point.toArray(),
                ...markers[1].point.toArray(),
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ff00" linewidth={2} />
        </line>
      )}

      {/* Render all measurements */}
      <MeasurementRenderer />
    </>
  );
};
