import React, { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html, Line } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore, BackendFaceClassification } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { MeasurementRenderer } from "./MeasurementRenderer";
import { toast } from "@/hooks/use-toast";
import { findEdgeByFeatureId, getFeatureSegments, TaggedFeatureEdge } from "@/lib/featureIdMatcher";
import { getFaceFromIntersection, calculateFaceToFaceDistance } from "@/lib/faceDetection";

interface EdgeClassification {
  type: "line" | "circle" | "arc";
  diameter?: number;
  radius?: number;
  length?: number;
  center?: [number, number, number];
}

// Backend classification with ground truth
interface BackendEdgeClassification {
  id: number;
  feature_id?: number;
  type: "line" | "circle" | "arc";
  start_point: [number, number, number];
  end_point: [number, number, number];
  length?: number;
  radius?: number;
  diameter?: number;
  center?: [number, number, number];
  segment_count: number;
}

interface BackendFaceClassificationRaw {
  face_id: number;
  type: "internal" | "external" | "through" | "planar";
  center: [number, number, number];
  normal: [number, number, number];
  area: number;
  surface_type: "cylinder" | "plane" | "other";
  radius?: number;
}

interface ConnectedEdgeGroup {
  segments: THREE.Line3[];
  count: number;
  type: "line" | "circle" | "arc";
  backendClassification?: BackendEdgeClassification;
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  vertex_face_ids?: number[];
  face_classifications?: BackendFaceClassificationRaw[];
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: BackendEdgeClassification[];
  tagged_edges?: TaggedFeatureEdge[];
}

interface ProfessionalMeasurementToolProps {
  meshData: MeshData | null;
  meshRef: THREE.Mesh | null;
  featureEdgesGeometry: THREE.BufferGeometry | null;
  enabled: boolean;
}

// Removed: Replaced by feature_id based matching from featureIdMatcher.ts

export const ProfessionalMeasurementTool: React.FC<ProfessionalMeasurementToolProps> = ({
  meshData,
  meshRef,
  featureEdgesGeometry,
  enabled,
}) => {
  const { camera, gl, raycaster } = useThree();
  const [hoverInfo, setHoverInfo] = useState<{
    position: THREE.Vector3;
    classification: any;
    edge: THREE.Line3;
    allSegments?: THREE.Line3[];
  } | null>(null);
  const [labelText, setLabelText] = useState<string>("");
  
  // Local state for face-to-face measurement (no store updates until finalized)
  const [isDrawingMeasurement, setIsDrawingMeasurement] = useState(false);
  const [firstClickPoint, setFirstClickPoint] = useState<{
    position: THREE.Vector3;
    faceData: BackendFaceClassification;
  } | null>(null);
  const [previewLine, setPreviewLine] = useState<{
    start: THREE.Vector3;
    end: THREE.Vector3;
  } | null>(null);
  
  const addMeasurement = useMeasurementStore((state) => state.addMeasurement);
  const activeTool = useMeasurementStore((state) => state.activeTool);

  // Convert feature edges to Line3 array
  const edgeLines = React.useMemo(() => {
    if (!featureEdgesGeometry) {
      console.warn("⚠️ No featureEdgesGeometry provided for raycasting");
      return [];
    }
    const lines: THREE.Line3[] = [];
    const positions = featureEdgesGeometry.attributes.position;
    
    console.log("📐 Building edgeLines for raycasting:", {
      segmentCount: positions.count / 2,
      totalVertices: positions.count
    });
    
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

  // Reusable geometry for face highlighting (prevents WebGL context loss)
  const faceHighlightGeometry = React.useMemo(() => new THREE.SphereGeometry(2, 16, 16), []);
  const faceHighlightMaterial = React.useMemo(
    () => new THREE.MeshBasicMaterial({ color: 0x00cc66, opacity: 0.7, transparent: true }),
    []
  );
  const firstFaceMaterial = React.useMemo(
    () => new THREE.MeshBasicMaterial({ color: 0x0066cc }),
    []
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      faceHighlightGeometry.dispose();
      faceHighlightMaterial.dispose();
      firstFaceMaterial.dispose();
    };
  }, [faceHighlightGeometry, faceHighlightMaterial, firstFaceMaterial]);

  // Clear local measurement state when switching away from face-to-face mode
  useEffect(() => {
    if (activeTool !== "face-to-face") {
      setIsDrawingMeasurement(false);
      setFirstClickPoint(null);
      setPreviewLine(null);
    }
  }, [activeTool]);

  // Log available tagged edges from backend
  React.useEffect(() => {
    if (meshData?.tagged_edges) {
      console.log("✅ Tagged edges received from backend:", meshData.tagged_edges.length);
      console.log("   - Circles:", meshData.tagged_edges.filter(e => e.type === "circle").length);
      console.log("   - Arcs:", meshData.tagged_edges.filter(e => e.type === "arc").length);
      console.log("   - Lines:", meshData.tagged_edges.filter(e => e.type === "line").length);
    }
  }, [meshData?.tagged_edges]);

  // Handle hover detection
  useEffect(() => {
    if (!enabled || !meshRef) {
      setHoverInfo(null);
      setLabelText("");
      return;
    }

    // Only check edgeLines for edge-select mode
    if (activeTool === "edge-select" && edgeLines.length === 0) {
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

        // Face-to-face mode - update preview line if drawing
        if (activeTool === "face-to-face" && isDrawingMeasurement && firstClickPoint) {
          setPreviewLine({
            start: firstClickPoint.position,
            end: point.clone()
          });
          return;
        }
        
        // Skip edge hover detection in face-to-face mode when not drawing
        if (activeTool === "face-to-face") {
          return;
        }

        let closestEdge: THREE.Line3 | null = null;
        let minDist = 1.5; // Increased from 1.0 to 1.5 for better detection

        edgeLines.forEach((edge) => {
          const closestPoint = new THREE.Vector3();
          edge.closestPointToPoint(point, true, closestPoint);
          const dist = point.distanceTo(closestPoint);

          if (dist <= minDist) { // Changed from < to <= for inclusive threshold
            minDist = dist;
            closestEdge = edge;
          }
        });

        console.log("🎯 Closest edge distance:", minDist.toFixed(3), "threshold: 1.5 (inclusive)");

        if (closestEdge) {
          console.log("✅ Edge detected, checking feature_id...");
          // NEW: Use feature_id based matching
          const taggedEdge = findEdgeByFeatureId(closestEdge, meshData?.tagged_edges);

          if (taggedEdge) {
            // Get ALL segments belonging to this feature_id
            const allFeatureSegments = getFeatureSegments(
              taggedEdge.feature_id,
              meshData?.tagged_edges
            );
            
            // Convert to Line3 array for rendering
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
  }, [enabled, meshRef, edgeLines, meshData?.tagged_edges, activeTool, isDrawingMeasurement, firstClickPoint, camera, gl, raycaster]);

  // Handle click
  useEffect(() => {
    if (!enabled) return;

    // Drag detection to prevent clicks during rotation
    const dragThreshold = 5; // pixels
    let dragStartPos: { x: number; y: number } | null = null;
    let isDragging = false;

    const handleMouseDown = (event: MouseEvent) => {
      dragStartPos = { x: event.clientX, y: event.clientY };
      isDragging = false;
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (dragStartPos) {
        const dx = event.clientX - dragStartPos.x;
        const dy = event.clientY - dragStartPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > dragThreshold) {
          isDragging = true;
        }
      }
    };

    const handleMouseUp = (event: MouseEvent) => {
      // Only trigger measurement if not dragging
      if (!isDragging && dragStartPos) {
        handleClick(event as any);
      }
      
      dragStartPos = null;
      isDragging = false;
    };

    const handleClick = (event: PointerEvent) => {
      // Face-to-face mode - detect face on CLICK only
      if (activeTool === "face-to-face") {
        if (!meshRef) return;

        const canvas = gl.domElement;
        const rect = canvas.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1,
        );

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(meshRef, false);

        if (intersects.length === 0) {
          toast({
            title: "No face detected",
            description: "Click on a surface",
            variant: "destructive",
          });
          return;
        }

        // Detect face from intersection
        const faceData = getFaceFromIntersection(intersects[0], meshData);
        
        if (!faceData) {
          console.error("❌ Face detection failed:", {
            meshDataExists: !!meshData,
            hasVertexFaceIds: !!meshData?.vertex_face_ids,
            hasFaceClassifications: !!meshData?.face_classifications,
            intersectionFaceIndex: intersects[0].faceIndex,
          });
          
          toast({
            title: "Face detection failed",
            description: "This surface has no classification data. Try another surface.",
            variant: "destructive",
          });
          return;
        }

        // Convert arrays to Vector3 ONCE before storing
        const face: BackendFaceClassification = {
          ...faceData,
          center: new THREE.Vector3(...faceData.center),
          normal: new THREE.Vector3(...faceData.normal),
        };

        console.log("✅ Face clicked:", face.face_id, face.surface_type);

        // First click - start drawing (LOCAL STATE ONLY)
        if (!isDrawingMeasurement) {
          setFirstClickPoint({
            position: face.center,
            faceData: face
          });
          setIsDrawingMeasurement(true);
          setPreviewLine({
            start: face.center,
            end: face.center.clone()
          });
          
          toast({
            title: "First face selected",
            description: `${face.surface_type} - Click second face`,
          });
          console.log("✅ First face selected (local state):", face.face_id);
          return;
        }
        
        // Second click - finalize measurement (SINGLE STORE UPDATE)
        if (isDrawingMeasurement && firstClickPoint) {
          const face1 = firstClickPoint.faceData;
          const face2 = face;
          
          const { distance, type, description } = calculateFaceToFaceDistance(face1, face2);
          
          addMeasurement({
            id: generateMeasurementId(),
            type: "face-to-face",
            points: [
              {
                id: generateMeasurementId(),
                position: face1.center,
                normal: face1.normal,
                surfaceType: "face",
              },
              {
                id: generateMeasurementId(),
                position: face2.center,
                normal: face2.normal,
                surfaceType: "face",
              },
            ],
            value: distance,
            unit: "mm",
            label: `${formatMeasurement(distance, "mm")} (${description})`,
            color: "#00CC66",
            visible: true,
            createdAt: new Date(),
            metadata: {
              face1Id: face1.face_id,
              face2Id: face2.face_id,
              distanceType: type as "plane-to-plane" | "cylinder-to-cylinder" | "plane-to-cylinder" | "point-to-point",
              backendMatch: true,
            },
          });
          
          // Clear local state
          setFirstClickPoint(null);
          setIsDrawingMeasurement(false);
          setPreviewLine(null);
          
          toast({
            title: "Measurement created",
            description: `${description}: ${formatMeasurement(distance, "mm")}`,
          });
          
          console.log("✅ Measurement created between faces", face1.face_id, "and", face2.face_id);
          return;
        }
      }

      // Edge-select mode - verify click is on the mesh
      if (!meshRef) return;

      const canvas = gl.domElement;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      // Silent return if click is outside the part
      if (intersects.length === 0) {
        return;
      }

      // Click is on part, but no edge detected
      if (!hoverInfo) {
        toast({
          title: "No edge detected",
          description: "Hover over an edge feature first",
          variant: "destructive",
        });
        return;
      }

      const { edge, classification } = hoverInfo;
      const taggedEdge = classification as TaggedFeatureEdge;

      if (!taggedEdge) {
        toast({
          title: "No feature detected",
          description: "Click on a circle, arc, or line edge to measure",
          variant: "destructive",
        });
        return;
      }

      if (taggedEdge.type === "circle" && taggedEdge.diameter) {
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [
            {
              id: generateMeasurementId(),
              position: edge.start,
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
            edgeType: "circle",
            backendMatch: true,
          },
        });
      } else if (taggedEdge.type === "arc" && taggedEdge.radius) {
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [
            {
              id: generateMeasurementId(),
              position: edge.start,
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
            edgeType: "arc",
            backendMatch: true,
          },
        });
      } else if (taggedEdge.type === "line" && taggedEdge.length) {
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [
            {
              id: generateMeasurementId(),
              position: edge.start,
              normal: new THREE.Vector3(0, 1, 0),
              surfaceType: "edge",
            },
            {
              id: generateMeasurementId(),
              position: edge.end,
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
            edgeType: "line",
            backendMatch: true,
          },
        });
      } else {
        toast({
          title: "Incomplete feature data",
          description: "This feature doesn't have complete measurement data",
          variant: "destructive",
        });
      }
    };

    gl.domElement.addEventListener("mousedown", handleMouseDown);
    gl.domElement.addEventListener("mousemove", handleMouseMove);
    gl.domElement.addEventListener("mouseup", handleMouseUp);

    return () => {
      gl.domElement.removeEventListener("mousedown", handleMouseDown);
      gl.domElement.removeEventListener("mousemove", handleMouseMove);
      gl.domElement.removeEventListener("mouseup", handleMouseUp);
    };
  }, [enabled, hoverInfo, addMeasurement, gl, activeTool, isDrawingMeasurement, firstClickPoint, meshRef, meshData, camera, raycaster]);

  return (
    <>
      {hoverInfo && activeTool === "edge-select" && (
        <>
          {/* Highlight all segments of the feature */}
          {hoverInfo.allSegments && hoverInfo.allSegments.length > 0 ? (
            // Multiple segments (circle/arc) - render all
            hoverInfo.allSegments.map((segment, idx) => (
              <primitive
                key={idx}
                object={(() => {
                  const geometry = new THREE.BufferGeometry();
                  const positions = new Float32Array([
                    segment.start.x, segment.start.y, segment.start.z,
                    segment.end.x, segment.end.y, segment.end.z,
                  ]);
                  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

                  const material = new THREE.LineBasicMaterial({
                    color: 0xff8c00,
                    linewidth: 3,
                    transparent: true,
                    opacity: 0.9,
                    depthTest: false,
                  });

                  return new THREE.Line(geometry, material);
                })()}
              />
            ))
          ) : (
            // Single segment (line) - render normally
            <primitive
              object={(() => {
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array([
                  hoverInfo.edge.start.x,
                  hoverInfo.edge.start.y,
                  hoverInfo.edge.start.z,
                  hoverInfo.edge.end.x,
                  hoverInfo.edge.end.y,
                  hoverInfo.edge.end.z,
                ]);
                geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

                const material = new THREE.LineBasicMaterial({
                  color: 0xff8c00,
                  transparent: true,
                  opacity: 0.8,
                  depthTest: false,
                });

                return new THREE.Line(geometry, material);
              })()}
            />
          )}
        </>
      )}

      {/* Live preview for face-to-face measurement */}
      {activeTool === "face-to-face" && isDrawingMeasurement && previewLine && (
        <group>
          {/* Dashed preview line */}
          <Line
            points={[previewLine.start, previewLine.end]}
            color="#FFA500"
            lineWidth={2}
            dashed
            dashSize={2}
            gapSize={1}
          />
          
          {/* First point sphere */}
          <mesh position={previewLine.start}>
            <sphereGeometry args={[1.5, 16, 16]} />
            <meshBasicMaterial color="#FFA500" opacity={0.8} transparent />
          </mesh>
          
          {/* Live distance label */}
          <Html position={previewLine.start} center distanceFactor={2}>
            <div style={{
              background: 'rgba(255, 165, 0, 0.9)',
              color: 'white',
              padding: '8px 16px',
              borderRadius: '4px',
              fontSize: '16px',
              fontWeight: '600',
              pointerEvents: 'none',
            }}>
              {previewLine.start.distanceTo(previewLine.end).toFixed(2)} mm
            </div>
          </Html>
        </group>
      )}

      {/* Label display */}
      {hoverInfo && labelText && (
        <Html position={hoverInfo.position} center style={{ pointerEvents: "none", transform: "translateY(-40px)" }}>
          <div
            style={{
              color: "white",
              fontSize: "14px",
              fontWeight: "600",
              whiteSpace: "nowrap",
              textShadow: "1px 1px 2px black, -1px -1px 2px black",
              userSelect: "none",
            }}
          >
            {labelText}
          </div>
        </Html>
      )}

      <MeasurementRenderer />
    </>
  );
};
