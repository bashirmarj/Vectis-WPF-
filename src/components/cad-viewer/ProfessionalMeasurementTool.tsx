import React, { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { MeasurementRenderer } from "./MeasurementRenderer";
import { toast } from "@/hooks/use-toast";

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

interface BackendFaceClassification {
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
  face_classifications?: BackendFaceClassification[];
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: BackendEdgeClassification[];
}

interface ProfessionalMeasurementToolProps {
  meshData: MeshData | null;
  meshRef: THREE.Mesh | null;
  featureEdgesGeometry: THREE.BufferGeometry | null;
  enabled: boolean;
}

/**
 * Find backend edge classification that matches clicked edge segment
 */
function findMatchingBackendEdge(
  clickedEdge: THREE.Line3,
  backendEdges?: BackendEdgeClassification[],
): BackendEdgeClassification | undefined {
  if (!backendEdges || backendEdges.length === 0) return undefined;

  const threshold = 0.1;

  for (const backendEdge of backendEdges) {
    const backendStart = new THREE.Vector3(...backendEdge.start_point);
    const backendEnd = new THREE.Vector3(...backendEdge.end_point);

    // Check if segments match (both orientations)
    const startMatch = clickedEdge.start.distanceTo(backendStart) < threshold;
    const endMatch = clickedEdge.end.distanceTo(backendEnd) < threshold;
    const startMatchReverse = clickedEdge.start.distanceTo(backendEnd) < threshold;
    const endMatchReverse = clickedEdge.end.distanceTo(backendStart) < threshold;

    if ((startMatch && endMatch) || (startMatchReverse && endMatchReverse)) {
      return backendEdge;
    }
  }

  return undefined;
}

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
  } | null>(null);
  const [labelText, setLabelText] = useState<string>("");

  const addMeasurement = useMeasurementStore((state) => state.addMeasurement);

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

  // Helper to determine edge type from segment count
  const getEdgeType = (segmentCount: number): "line" | "circle" | "arc" => {
    if (segmentCount === 32 || segmentCount >= 30) return "circle";
    if (segmentCount === 2) return "line";
    return "arc";
  };

  // Simple edge grouping for visual feedback only
  const edgeGroupsCache = React.useMemo(() => {
    if (edgeLines.length === 0) return new Map<THREE.Line3, ConnectedEdgeGroup>();

    const cache = new Map<THREE.Line3, ConnectedEdgeGroup>();

    edgeLines.forEach((edge) => {
      const backendEdge = findMatchingBackendEdge(edge, meshData?.edge_classifications);
      
      const group: ConnectedEdgeGroup = {
        segments: [edge],
        count: 1,
        type: backendEdge?.type || "line",
        backendClassification: backendEdge,
      };

      cache.set(edge, group);
    });

    console.log("✅ Backend edge classifications found:", Array.from(cache.values()).filter((g) => g.backendClassification).length, "of", edgeLines.length);

    return cache;
  }, [edgeLines, meshData?.edge_classifications]);

  // Handle hover detection
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
        let minDist = 0.5;

        edgeLines.forEach((edge) => {
          const closestPoint = new THREE.Vector3();
          edge.closestPointToPoint(point, true, closestPoint);
          const dist = point.distanceTo(closestPoint);

          if (dist < minDist) {
            minDist = dist;
            closestEdge = edge;
          }
        });

        if (closestEdge) {
          const edgeGroup = edgeGroupsCache.get(closestEdge);
          const backendEdge = edgeGroup?.backendClassification;

          if (backendEdge) {
            let label = "";
            if (backendEdge.type === "circle" && backendEdge.diameter) {
              label = `⊙ Diameter: ø${backendEdge.diameter.toFixed(2)} mm`;
            } else if (backendEdge.type === "arc" && backendEdge.radius) {
              label = `⌒ Radius: R${backendEdge.radius.toFixed(2)} mm`;
            } else if (backendEdge.type === "line" && backendEdge.length) {
              label = `📏 Length: ${backendEdge.length.toFixed(2)} mm`;
            } else {
              label = "Feature detected";
            }

            setLabelText(label);
            setHoverInfo({
              position: point,
              classification: { type: backendEdge.type },
              edge: closestEdge,
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
  }, [enabled, meshRef, edgeLines, edgeGroupsCache, camera, gl, raycaster]);

  // Handle click
  useEffect(() => {
    if (!enabled || !hoverInfo) return;

    const handleClick = () => {
      const { edge } = hoverInfo;
      const edgeGroup = edgeGroupsCache.get(edge);
      const backendEdge = edgeGroup?.backendClassification;

      if (!backendEdge) {
        toast({
          title: "Backend classification required",
          description: "Unable to measure this feature without backend geometry data",
          variant: "destructive",
        });
        return;
      }

      if (backendEdge.type === "circle" && backendEdge.diameter) {
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
          value: backendEdge.diameter,
          unit: "mm",
          label: `⊙ ${formatMeasurement(backendEdge.diameter, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: {
            edgeType: "circle",
            center: backendEdge.center,
            backendMatch: true,
          },
        });
      } else if (backendEdge.type === "arc" && backendEdge.radius) {
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
          value: backendEdge.radius,
          unit: "mm",
          label: `R ${formatMeasurement(backendEdge.radius, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: {
            edgeType: "arc",
            center: backendEdge.center,
            backendMatch: true,
          },
        });
      } else if (backendEdge.type === "line" && backendEdge.length) {
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
          value: backendEdge.length,
          unit: "mm",
          label: `📏 ${formatMeasurement(backendEdge.length, "mm")}`,
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
          title: "Incomplete backend data",
          description: "This feature doesn't have complete measurement data",
          variant: "destructive",
        });
      }
    };

    gl.domElement.addEventListener("click", handleClick);
    return () => gl.domElement.removeEventListener("click", handleClick);
  }, [enabled, hoverInfo, addMeasurement, edgeGroupsCache, gl]);

  return (
    <>
      {hoverInfo && (
        <>
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
        </>
      )}

      <MeasurementRenderer />
    </>
  );
};
