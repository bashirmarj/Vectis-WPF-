import React, { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { classifyFeatureEdge } from "@/lib/featureEdgeClassification";
import { MeasurementRenderer } from "./MeasurementRenderer";

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
}

interface ProfessionalMeasurementToolProps {
  meshData: MeshData | null;
  meshRef: THREE.Mesh | null;
  featureEdgesGeometry: THREE.BufferGeometry | null;
  enabled: boolean;
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
      lines.push(new THREE.Line3(
        new THREE.Vector3(positions.getX(i), positions.getY(i), positions.getZ(i)),
        new THREE.Vector3(positions.getX(i + 1), positions.getY(i + 1), positions.getZ(i + 1))
      ));
    }
    return lines;
  }, [featureEdgesGeometry]);

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
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(meshRef, false);

      if (intersects.length > 0) {
        const point = intersects[0].point;

        // Find closest edge
        let closestEdge: THREE.Line3 | null = null;
        let minDist = 0.5;

        edgeLines.forEach(edge => {
          const closestPoint = new THREE.Vector3();
          edge.closestPointToPoint(point, true, closestPoint);
          const dist = point.distanceTo(closestPoint);

          if (dist < minDist) {
            minDist = dist;
            closestEdge = edge;
          }
        });

        if (closestEdge) {
          const classification = classifyFeatureEdge(closestEdge, edgeLines);

          let label = "";
          if (classification.type === "circle") {
            label = `⊙ Diameter: ø ${formatMeasurement(classification.diameter || 0, "mm")}`;
          } else if (classification.type === "arc") {
            label = `⌒ Radius: R ${formatMeasurement(classification.radius || 0, "mm")}`;
          } else {
            label = `📏 Length: ${formatMeasurement(classification.length || 0, "mm")}`;
          }

          setLabelText(label);
          setHoverInfo({ position: point, classification, edge: closestEdge });
        } else {
          setHoverInfo(null);
          setLabelText("");
        }
      } else {
        setHoverInfo(null);
        setLabelText("");
      }
    };

    canvas.addEventListener("pointermove", handlePointerMove);
    return () => canvas.removeEventListener("pointermove", handlePointerMove);
  }, [enabled, meshRef, edgeLines, camera, gl, raycaster]);

  // Handle click
  useEffect(() => {
    if (!enabled || !hoverInfo) return;

    const handleClick = () => {
      const { classification, edge } = hoverInfo;

      if (classification.type === "circle") {
        addMeasurement({
          id: generateMeasurementId(),
          type: "diameter",
          points: [{ position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }],
          value: classification.diameter || 0,
          unit: "mm",
          label: `⊙ ${formatMeasurement(classification.diameter || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          metadata: { edgeType: "circle", center: classification.center },
        });
      } else if (classification.type === "arc") {
        addMeasurement({
          id: generateMeasurementId(),
          type: "radius",
          points: [{ position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }],
          value: classification.radius || 0,
          unit: "mm",
          label: `R ${formatMeasurement(classification.radius || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          metadata: { edgeType: "arc", center: classification.center },
        });
      } else {
        addMeasurement({
          id: generateMeasurementId(),
          type: "distance",
          points: [
            { position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" },
            { position: edge.end, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }
          ],
          value: classification.length || 0,
          unit: "mm",
          label: `${formatMeasurement(classification.length || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          metadata: { edgeType: "line" },
        });
      }
    };

    canvas.addEventListener("click", handleClick);
    return () => canvas.removeEventListener("click", handleClick);
  }, [enabled, hoverInfo, addMeasurement]);

  return (
    <>
      {hoverInfo && (
        <>
          <line key="hover-edge">
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([
                  hoverInfo.edge.start.x, hoverInfo.edge.start.y, hoverInfo.edge.start.z,
                  hoverInfo.edge.end.x, hoverInfo.edge.end.y, hoverInfo.edge.end.z,
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#FF8C00" linewidth={2} transparent opacity={0.8} depthTest={false} />
          </line>

          <Html position={hoverInfo.position} center style={{ pointerEvents: 'none' }}>
            <div style={{
              color: "white",
              fontSize: "14px",
              fontWeight: "600",
              whiteSpace: "nowrap",
              textShadow: "1px 1px 2px black, -1px -1px 2px black",
              userSelect: "none",
            }}>
              {labelText}
            </div>
          </Html>
        </>
      )}

      <MeasurementRenderer />
    </>
  );
};
