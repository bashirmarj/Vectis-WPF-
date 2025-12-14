import React, { useEffect, useState, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId, calculateXYZDeltas } from "@/lib/measurementUtils";
import { MeasurementRenderer } from "./MeasurementRenderer";
import { FaceCrosshairMarker } from "./measurements/FaceCrosshairMarker";
import { toast } from "@/hooks/use-toast";
import { findClosestTaggedEdge, TaggedFeatureEdge } from "@/lib/featureIdMatcher";
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
  const measurementMode = useMeasurementStore((state) => state.measurementMode);

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
  // Priority 1: Use backend tagged_edges if available (analytical data)
  // Priority 2: Use featureEdgesGeometry (rendered edge lines) as fallback
  // ONLY ACTIVE IN SINGLE MODE
  useEffect(() => {
    if (!enabled || !meshRef || measurementMode !== "single") {
      setHoverInfo(null);
      setLabelText("");
      return;
    }

    // Check if we have tagged edges from backend
    const taggedEdges = meshData?.tagged_edges;
    const hasTaggedEdges = taggedEdges && taggedEdges.length > 0;
    const hasEdgeLines = edgeLines.length > 0;

    // DEBUG: Log what data we have
    console.log("🔍 Edge Detection Debug:", {
      measurementMode,
      hasTaggedEdges,
      taggedEdgesCount: taggedEdges?.length || 0,
      hasEdgeLines,
      edgeLinesCount: edgeLines.length,
      featureEdgesGeometry: featureEdgesGeometry !== null,
    });

    if (!hasTaggedEdges && !hasEdgeLines) {
      // No edge data available at all
      console.warn("⚠️ No edge data available for measurement!");
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

        // Priority 1: Try backend tagged_edges first (has analytical data)
        if (hasTaggedEdges) {
          const taggedEdge = findClosestTaggedEdge(point, taggedEdges, 50); // 10mm threshold (coordinates are in mm)

          if (taggedEdge) {
            // Build highlight segments from snap_points
            const allSegmentLines: THREE.Line3[] = [];
            if (taggedEdge.snap_points && taggedEdge.snap_points.length >= 2) {
              for (let i = 0; i < taggedEdge.snap_points.length - 1; i++) {
                const p1 = taggedEdge.snap_points[i];
                const p2 = taggedEdge.snap_points[i + 1];
                allSegmentLines.push(
                  new THREE.Line3(
                    new THREE.Vector3(p1[0], p1[1], p1[2]),
                    new THREE.Vector3(p2[0], p2[1], p2[2])
                  )
                );
              }
              // For closed edges, add the closing segment
              if (taggedEdge.is_closed && taggedEdge.snap_points.length > 2) {
                const first = taggedEdge.snap_points[0];
                const last = taggedEdge.snap_points[taggedEdge.snap_points.length - 1];
                allSegmentLines.push(
                  new THREE.Line3(
                    new THREE.Vector3(last[0], last[1], last[2]),
                    new THREE.Vector3(first[0], first[1], first[2])
                  )
                );
              }
            } else {
              // Fallback to start/end
              allSegmentLines.push(
                new THREE.Line3(
                  new THREE.Vector3(...taggedEdge.start),
                  new THREE.Vector3(...taggedEdge.end)
                )
              );
            }

            // Generate context-aware label based on edge type
            let label = "";
            // Values are already in mm from backend (OpenCASCADE default unit)
            const diameterMM = taggedEdge.diameter;
            const radiusMM = taggedEdge.radius;
            const arcLengthMM = taggedEdge.arc_length;
            const lengthMM = taggedEdge.length;
            const majorRadiusMM = taggedEdge.major_radius;
            const minorRadiusMM = taggedEdge.minor_radius;

            if (taggedEdge.type === "circle") {
              if (taggedEdge.is_full_circle && diameterMM) {
                label = `⊙ Diameter: Ø${diameterMM.toFixed(2)} mm`;
              } else if (radiusMM) {
                label = `⌒ Radius: R${radiusMM.toFixed(2)} mm`;
                if (arcLengthMM) {
                  label += ` | Arc: ${arcLengthMM.toFixed(2)} mm`;
                }
              }
            } else if (taggedEdge.type === "line" && lengthMM) {
              label = `— Length: ${lengthMM.toFixed(2)} mm`;
            } else if (taggedEdge.type === "ellipse") {
              if (majorRadiusMM && minorRadiusMM) {
                label = `⬭ Ellipse: ${majorRadiusMM.toFixed(2)} × ${minorRadiusMM.toFixed(2)} mm`;
              }
            } else if (lengthMM) {
              label = `Edge: ${lengthMM.toFixed(2)} mm`;
            } else {
              label = `Edge (${taggedEdge.type})`;
            }

            setLabelText(label);
            setHoverInfo({
              position: point,
              classification: taggedEdge,
              edge: new THREE.Line3(
                new THREE.Vector3(...taggedEdge.start),
                new THREE.Vector3(...taggedEdge.end)
              ),
              allSegments: allSegmentLines,
            });
            return;
          }
        }

        // Priority 2: Use rendered edge geometry (featureEdgesGeometry)
        if (hasEdgeLines) {
          // Find the closest edge line segment to the click point
          let closestEdge: THREE.Line3 | null = null;
          let minDistance = 0.05; // 50mm threshold
          let closestPoint = new THREE.Vector3();

          for (const line of edgeLines) {
            const tempClosest = new THREE.Vector3();
            line.closestPointToPoint(point, true, tempClosest);
            const distance = point.distanceTo(tempClosest);

            if (distance < minDistance) {
              minDistance = distance;
              closestEdge = line;
              closestPoint = tempClosest.clone();
            }
          }

          if (closestEdge) {
            // Calculate edge length in mm (coordinates are already in mm from backend)
            const edgeLength = closestEdge.start.distanceTo(closestEdge.end);
            const label = `— Length: ${edgeLength.toFixed(2)} mm`;

            setLabelText(label);
            setHoverInfo({
              position: closestPoint,
              classification: { type: "line", length: edgeLength },
              edge: closestEdge,
              allSegments: [closestEdge],
            });
            return;
          }
        }

        // No edge found near cursor
        setHoverInfo(null);
        setLabelText("");
      } else {
        setHoverInfo(null);
        setLabelText("");
      }
    };

    gl.domElement.addEventListener("pointermove", handlePointerMove);
    return () => gl.domElement.removeEventListener("pointermove", handlePointerMove);
  }, [enabled, meshRef, meshData?.tagged_edges, edgeLines, camera, gl, raycaster, measurementMode]);

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

      // DEBUG: Log click state
      console.log("🖱️ CLICK DETECTED:", {
        hasIntersects: intersects.length > 0,
        hoverInfo: hoverInfo ? "exists" : "null",
        hoverClassification: hoverInfo?.classification ? "exists" : "null",
        measurementMode,
      });

      if (intersects.length === 0) {
        dragStartRef.current = null;
        setIsDragging(false);
        return;
      }

      // Priority 1: Check if clicking on an edge (from hover detection)
      if (hoverInfo && hoverInfo.classification) {
        const classification = hoverInfo.classification;

        console.log("✅ Edge click detected, classification:", classification);

        // Check if this is a TaggedFeatureEdge (from backend) or simple edge (from geometry)
        const isTaggedEdge = 'feature_id' in classification || 'start' in classification;

        if (isTaggedEdge) {
          // Handle TaggedFeatureEdge from backend
          const taggedEdge = classification as TaggedFeatureEdge;

          // Get center point for the measurement display
          const centerPoint = taggedEdge.center
            ? new THREE.Vector3(taggedEdge.center[0], taggedEdge.center[1], taggedEdge.center[2])
            : hoverInfo.position;

          // Create context-aware edge measurement based on edge type
          if (taggedEdge.type === "circle") {
            if (taggedEdge.is_full_circle && taggedEdge.diameter) {
              // Full circle → show diameter
              addMeasurement({
                id: generateMeasurementId(),
                type: "measure",
                points: [
                  {
                    id: generateMeasurementId(),
                    position: centerPoint,
                    normal: taggedEdge.normal
                      ? new THREE.Vector3(...taggedEdge.normal)
                      : new THREE.Vector3(0, 1, 0),
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
                  isFullCircle: true,
                },
              });

              toast({
                title: "Circle Measured",
                description: `Diameter: ${formatMeasurement(taggedEdge.diameter, "mm")}`,
              });
            } else if (taggedEdge.radius) {
              // Arc (partial circle) → show radius
              const label = taggedEdge.arc_length
                ? `R ${formatMeasurement(taggedEdge.radius, "mm")} | Arc: ${formatMeasurement(taggedEdge.arc_length, "mm")}`
                : `R ${formatMeasurement(taggedEdge.radius, "mm")}`;

              addMeasurement({
                id: generateMeasurementId(),
                type: "measure",
                points: [
                  {
                    id: generateMeasurementId(),
                    position: centerPoint,
                    normal: taggedEdge.normal
                      ? new THREE.Vector3(...taggedEdge.normal)
                      : new THREE.Vector3(0, 1, 0),
                    surfaceType: "edge",
                  },
                ],
                value: taggedEdge.radius,
                unit: "mm",
                label: label,
                color: "#00AACC",
                visible: true,
                createdAt: new Date(),
                metadata: {
                  measurementSubtype: "edge",
                  edgeType: "arc",
                  backendMatch: true,
                  isFullCircle: false,
                  arcLength: taggedEdge.arc_length,
                },
              });

              toast({
                title: "Arc Measured",
                description: `Radius: ${formatMeasurement(taggedEdge.radius, "mm")}`,
              });
            }
          } else if (taggedEdge.type === "line" && taggedEdge.length) {
            // Straight line → show length with XYZ deltas
            const startPoint = new THREE.Vector3(...taggedEdge.start);
            const endPoint = new THREE.Vector3(...taggedEdge.end);
            const xyzDeltas = calculateXYZDeltas(startPoint, endPoint);

            addMeasurement({
              id: generateMeasurementId(),
              type: "measure",
              points: [
                {
                  id: generateMeasurementId(),
                  position: startPoint,
                  normal: new THREE.Vector3(0, 1, 0),
                  surfaceType: "edge",
                },
                {
                  id: generateMeasurementId(),
                  position: endPoint,
                  normal: new THREE.Vector3(0, 1, 0),
                  surfaceType: "edge",
                },
              ],
              value: taggedEdge.length,
              unit: "mm",
              label: `${formatMeasurement(taggedEdge.length, "mm")}`,
              color: "#0066CC",
              visible: true,
              createdAt: new Date(),
              metadata: {
                measurementSubtype: "edge",
                edgeType: "line",
                backendMatch: true,
                // NEW: Add XYZ deltas
                deltaX: xyzDeltas.dX,
                deltaY: xyzDeltas.dY,
                deltaZ: xyzDeltas.dZ,
              },
            });

            toast({
              title: "Line Measured",
              description: `Length: ${formatMeasurement(taggedEdge.length, "mm")}`,
            });
          } else if (taggedEdge.type === "ellipse" && taggedEdge.major_radius && taggedEdge.minor_radius) {
            // Ellipse → show major/minor radii
            addMeasurement({
              id: generateMeasurementId(),
              type: "measure",
              points: [
                {
                  id: generateMeasurementId(),
                  position: centerPoint,
                  normal: new THREE.Vector3(0, 1, 0),
                  surfaceType: "edge",
                },
              ],
              value: taggedEdge.major_radius,
              unit: "mm",
              label: `Ellipse: ${formatMeasurement(taggedEdge.major_radius, "mm")} × ${formatMeasurement(taggedEdge.minor_radius, "mm")}`,
              color: "#9900CC",
              visible: true,
              createdAt: new Date(),
              metadata: {
                measurementSubtype: "edge",
                edgeType: "ellipse",
                backendMatch: true,
              },
            });

            toast({
              title: "Ellipse Measured",
              description: `Major: ${formatMeasurement(taggedEdge.major_radius, "mm")}, Minor: ${formatMeasurement(taggedEdge.minor_radius, "mm")}`,
            });
          } else if (taggedEdge.length) {
            // Generic edge with length
            addMeasurement({
              id: generateMeasurementId(),
              type: "measure",
              points: [
                {
                  id: generateMeasurementId(),
                  position: hoverInfo.position,
                  normal: new THREE.Vector3(0, 1, 0),
                  surfaceType: "edge",
                },
              ],
              value: taggedEdge.length,
              unit: "mm",
              label: `${formatMeasurement(taggedEdge.length, "mm")}`,
              color: "#666666",
              visible: true,
              createdAt: new Date(),
              metadata: {
                measurementSubtype: "edge",
                edgeType: taggedEdge.type,
                backendMatch: true,
              },
            });

            toast({
              title: "Edge Measured",
              description: `Length: ${formatMeasurement(taggedEdge.length, "mm")}`,
            });
          } else {
            // Edge exists but has no measurable properties - log for debugging
            console.warn("⚠️ Edge has no measurable properties:", taggedEdge);
            toast({
              title: "Edge Not Measurable",
              description: `Edge type "${taggedEdge.type}" has no dimension data`,
              variant: "destructive",
            });
          }
        } else {
          // Handle simple edge from geometry-based detection
          const simpleEdge = classification as { type: string; length: number };
          const edge = hoverInfo.edge;

          if (simpleEdge.length && edge) {
            const xyzDeltas = calculateXYZDeltas(edge.start, edge.end);

            addMeasurement({
              id: generateMeasurementId(),
              type: "measure",
              points: [
                {
                  id: generateMeasurementId(),
                  position: edge.start.clone(),
                  normal: new THREE.Vector3(0, 1, 0),
                  surfaceType: "edge",
                },
                {
                  id: generateMeasurementId(),
                  position: edge.end.clone(),
                  normal: new THREE.Vector3(0, 1, 0),
                  surfaceType: "edge",
                },
              ],
              value: simpleEdge.length,
              unit: "mm",
              label: `${formatMeasurement(simpleEdge.length, "mm")}`,
              color: "#0066CC",
              visible: true,
              createdAt: new Date(),
              metadata: {
                measurementSubtype: "edge",
                edgeType: "line",
                backendMatch: false, // From geometry, not backend
                // NEW: Add XYZ deltas
                deltaX: xyzDeltas.dX,
                deltaY: xyzDeltas.dY,
                deltaZ: xyzDeltas.dZ,
              },
            });

            toast({
              title: "Edge Measured",
              description: `Length: ${formatMeasurement(simpleEdge.length, "mm")}`,
            });
          }
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

      {/* Edge hover label - offset below cursor */}
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
                transform: "translate(-50%, 25px)", // Center horizontally, 25px below cursor
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
