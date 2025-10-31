import React, { useRef, useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import {
  classifyEdge,
  calculateDistance,
  calculateAngle,
  calculateRadius,
  detectCylindricalSurface,
} from "@/lib/measurementUtils";

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
  meshRef: React.RefObject<THREE.Mesh>;
  enabled: boolean;
}

interface HoverInfo {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  type: "face" | "edge" | "vertex";
  faceIndex?: number;
  edgeInfo?: {
    start: THREE.Vector3;
    end: THREE.Vector3;
    classification: any;
  };
}

export const ProfessionalMeasurementTool: React.FC<ProfessionalMeasurementToolProps> = ({
  meshData,
  meshRef,
  enabled,
}) => {
  const { camera, raycaster, scene } = useThree();
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [labelText, setLabelText] = useState<string>("");

  const { activeTool, tempPoints, addTempPoint, clearTempPoints, addMeasurement } = useMeasurementStore();

  // ✅ NEW: Detect all triangles belonging to the same planar face
  const getFullFaceTriangles = (faceIndex: number, meshData: MeshData, angleThreshold = 0.01): number[] => {
    const faceTriangles: number[] = [faceIndex];

    // Get normal of clicked face
    const fi = faceIndex * 3;
    const i0 = meshData.indices[fi] * 3;
    const i1 = meshData.indices[fi + 1] * 3;
    const i2 = meshData.indices[fi + 2] * 3;

    const v0 = new THREE.Vector3(meshData.vertices[i0], meshData.vertices[i0 + 1], meshData.vertices[i0 + 2]);
    const v1 = new THREE.Vector3(meshData.vertices[i1], meshData.vertices[i1 + 1], meshData.vertices[i1 + 2]);
    const v2 = new THREE.Vector3(meshData.vertices[i2], meshData.vertices[i2 + 1], meshData.vertices[i2 + 2]);

    const e1 = new THREE.Vector3().subVectors(v1, v0);
    const e2 = new THREE.Vector3().subVectors(v2, v0);
    const clickedNormal = new THREE.Vector3().crossVectors(e1, e2).normalize();

    // Find all coplanar triangles
    const totalTriangles = meshData.indices.length / 3;
    for (let i = 0; i < totalTriangles; i++) {
      if (i === faceIndex) continue;

      const ti = i * 3;
      const ti0 = meshData.indices[ti] * 3;
      const ti1 = meshData.indices[ti + 1] * 3;
      const ti2 = meshData.indices[ti + 2] * 3;

      const tv0 = new THREE.Vector3(meshData.vertices[ti0], meshData.vertices[ti0 + 1], meshData.vertices[ti0 + 2]);
      const tv1 = new THREE.Vector3(meshData.vertices[ti1], meshData.vertices[ti1 + 1], meshData.vertices[ti1 + 2]);
      const tv2 = new THREE.Vector3(meshData.vertices[ti2], meshData.vertices[ti2 + 1], meshData.vertices[ti2 + 2]);

      const te1 = new THREE.Vector3().subVectors(tv1, tv0);
      const te2 = new THREE.Vector3().subVectors(tv2, tv0);
      const testNormal = new THREE.Vector3().crossVectors(te1, te2).normalize();

      // Check if normals are parallel (same plane)
      const normalDot = Math.abs(clickedNormal.dot(testNormal));
      if (normalDot > 1 - angleThreshold) {
        // Check if triangles share at least one vertex (connected)
        const clickedVerts = [
          `${v0.x.toFixed(6)},${v0.y.toFixed(6)},${v0.z.toFixed(6)}`,
          `${v1.x.toFixed(6)},${v1.y.toFixed(6)},${v1.z.toFixed(6)}`,
          `${v2.x.toFixed(6)},${v2.y.toFixed(6)},${v2.z.toFixed(6)}`,
        ];
        const testVerts = [
          `${tv0.x.toFixed(6)},${tv0.y.toFixed(6)},${tv0.z.toFixed(6)}`,
          `${tv1.x.toFixed(6)},${tv1.y.toFixed(6)},${tv1.z.toFixed(6)}`,
          `${tv2.x.toFixed(6)},${tv2.y.toFixed(6)},${tv2.z.toFixed(6)}`,
        ];

        const hasSharedVertex = clickedVerts.some((cv) => testVerts.includes(cv));
        if (hasSharedVertex) {
          faceTriangles.push(i);
        }
      }
    }

    return faceTriangles;
  };

  // ✅ ENHANCED: Better edge extraction with proper indexing
  const extractEdges = (data: MeshData): Map<string, { start: THREE.Vector3; end: THREE.Vector3; faces: number[] }> => {
    const edgeMap = new Map<string, { start: THREE.Vector3; end: THREE.Vector3; faces: number[] }>();

    for (let i = 0; i < data.indices.length; i += 3) {
      const faceIndex = i / 3;
      const i0 = data.indices[i] * 3;
      const i1 = data.indices[i + 1] * 3;
      const i2 = data.indices[i + 2] * 3;

      const v0 = new THREE.Vector3(data.vertices[i0], data.vertices[i0 + 1], data.vertices[i0 + 2]);
      const v1 = new THREE.Vector3(data.vertices[i1], data.vertices[i1 + 1], data.vertices[i1 + 2]);
      const v2 = new THREE.Vector3(data.vertices[i2], data.vertices[i2 + 1], data.vertices[i2 + 2]);

      const edges = [
        { v1: v0, v2: v1, key: `${Math.min(i0, i1)}_${Math.max(i0, i1)}` },
        { v1: v1, v2: v2, key: `${Math.min(i1, i2)}_${Math.max(i1, i2)}` },
        { v1: v2, v2: v0, key: `${Math.min(i2, i0)}_${Math.max(i2, i0)}` },
      ];

      edges.forEach((edge) => {
        if (!edgeMap.has(edge.key)) {
          edgeMap.set(edge.key, {
            start: edge.v1.clone(),
            end: edge.v2.clone(),
            faces: [faceIndex],
          });
        } else {
          edgeMap.get(edge.key)!.faces.push(faceIndex);
        }
      });
    }

    return edgeMap;
  };

  // ✅ FIXED: Proper hover detection with face and edge priority
  useEffect(() => {
    if (!enabled || !meshData || !meshRef.current || !activeTool) return;

    const handlePointerMove = (event: PointerEvent) => {
      const rect = (event.target as HTMLElement).getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
      const intersects = raycaster.intersectObject(meshRef.current!);

      if (intersects.length > 0) {
        const intersection = intersects[0];
        const point = intersection.point;
        const faceIndex = intersection.faceIndex !== undefined ? intersection.faceIndex : -1;

        // Get face normal
        let normal = new THREE.Vector3(0, 1, 0);
        if (faceIndex >= 0) {
          const i = faceIndex * 3;
          const i0 = meshData.indices[i] * 3;
          const i1 = meshData.indices[i + 1] * 3;
          const i2 = meshData.indices[i + 2] * 3;

          const v0 = new THREE.Vector3(meshData.vertices[i0], meshData.vertices[i0 + 1], meshData.vertices[i0 + 2]);
          const v1 = new THREE.Vector3(meshData.vertices[i1], meshData.vertices[i1 + 1], meshData.vertices[i1 + 2]);
          const v2 = new THREE.Vector3(meshData.vertices[i2], meshData.vertices[i2 + 1], meshData.vertices[i2 + 2]);

          const e1 = new THREE.Vector3().subVectors(v1, v0);
          const e2 = new THREE.Vector3().subVectors(v2, v0);
          normal = new THREE.Vector3().crossVectors(e1, e2).normalize();
        }

        // ✅ ENHANCED: Check for edge proximity (within 3mm of edge)
        const edgeMap = extractEdges(meshData);
        let closestEdge: { start: THREE.Vector3; end: THREE.Vector3; distance: number } | null = null;
        let minEdgeDist = 3; // 3mm threshold for edge detection

        edgeMap.forEach((edgeData) => {
          const edge = new THREE.Line3(edgeData.start, edgeData.end);
          const closestOnEdge = new THREE.Vector3();
          edge.closestPointToPoint(point, true, closestOnEdge);
          const dist = point.distanceTo(closestOnEdge);

          if (dist < minEdgeDist) {
            minEdgeDist = dist;
            closestEdge = {
              start: edgeData.start,
              end: edgeData.end,
              distance: dist,
            };
          }
        });

        // Determine hover type and generate label
        if (closestEdge && activeTool === "edge-select") {
          // EDGE HOVER
          const edge = new THREE.Line3(closestEdge.start, closestEdge.end);
          const classification = classifyEdge(edge, meshData);

          let label = "";
          if (classification.type === "circle") {
            label = `⊙ Circle: ø ${(classification.radius! * 2).toFixed(2)}mm`;
          } else if (classification.type === "arc") {
            label = `⌒ Arc: R ${classification.radius!.toFixed(2)}mm`;
          } else {
            label = `📏 Line: ${edge.distance().toFixed(2)}mm`;
          }

          setHoverInfo({
            point,
            normal,
            type: "edge",
            edgeInfo: {
              start: closestEdge.start,
              end: closestEdge.end,
              classification,
            },
          });
          setLabelText(label);
        } else if (faceIndex >= 0) {
          // FACE HOVER
          setHoverInfo({
            point,
            normal,
            type: "face",
            faceIndex,
          });
          setLabelText("Click face");
        } else {
          setHoverInfo(null);
          setLabelText("");
        }
      } else {
        setHoverInfo(null);
        setLabelText("");
      }
    };

    const canvas = scene.children[0]?.parent as any;
    const domElement = canvas?.domElement || document.querySelector("canvas");

    if (domElement) {
      domElement.addEventListener("pointermove", handlePointerMove);
      return () => domElement.removeEventListener("pointermove", handlePointerMove);
    }
  }, [enabled, meshData, meshRef, camera, raycaster, scene, activeTool]);

  // ✅ FIXED: Click handler with proper edge-select
  useEffect(() => {
    if (!enabled || !meshData || !activeTool || !hoverInfo) return;

    const handleClick = (event: PointerEvent) => {
      if (!hoverInfo) return;

      const newPoint = {
        id: crypto.randomUUID(),
        position: hoverInfo.point.clone(),
        normal: hoverInfo.normal.clone(),
        surfaceType: hoverInfo.type as any,
        faceIndex: hoverInfo.faceIndex,
        metadata: hoverInfo.edgeInfo
          ? {
              startPoint: [hoverInfo.edgeInfo.start.x, hoverInfo.edgeInfo.start.y, hoverInfo.edgeInfo.start.z],
              endPoint: [hoverInfo.edgeInfo.end.x, hoverInfo.edgeInfo.end.y, hoverInfo.edgeInfo.end.z],
            }
          : undefined,
      };

      // Edge-select: complete on first click
      if (activeTool === "edge-select" && hoverInfo.type === "edge" && hoverInfo.edgeInfo) {
        const edge = new THREE.Line3(hoverInfo.edgeInfo.start, hoverInfo.edgeInfo.end);
        const classification = hoverInfo.edgeInfo.classification;

        let label = "";
        let value = 0;

        if (classification.type === "circle") {
          value = classification.radius! * 2; // Diameter
          label = `ø ${value.toFixed(2)}mm`;
        } else if (classification.type === "arc") {
          value = classification.radius!;
          label = `R ${value.toFixed(2)}mm`;
        } else {
          value = edge.distance();
          label = `${value.toFixed(2)}mm`;
        }

        addMeasurement({
          id: crypto.randomUUID(),
          type: "edge-select",
          points: [newPoint],
          value,
          unit: "mm",
          label,
          visible: true,
          color: "#00ff00",
          createdAt: new Date(),
          metadata: {
            edgeType: classification.type,
            arcRadius: classification.radius,
            arcCenter: classification.center,
            edgeStart: hoverInfo.edgeInfo.start,
            edgeEnd: hoverInfo.edgeInfo.end,
          },
        });
        clearTempPoints();
        return;
      }

      // Standard measurement flow
      addTempPoint(newPoint);
      const pointsCount = tempPoints.length + 1;

      // Complete measurement based on type
      if (
        (activeTool === "distance" && pointsCount === 2) ||
        (activeTool === "angle" && pointsCount === 3) ||
        (activeTool === "radius" && pointsCount === 3) ||
        (activeTool === "diameter" && pointsCount === 3) ||
        (activeTool === "edge-to-edge" && pointsCount === 2) ||
        (activeTool === "face-to-face" && pointsCount === 2) ||
        (activeTool === "coordinate" && pointsCount === 1)
      ) {
        const allPoints = [...tempPoints, newPoint];
        let value = 0;
        let label = "";

        switch (activeTool) {
          case "distance":
            // ✅ NEW: Detect cylindrical surface and use arc length if applicable
            const cylinderInfo = detectCylindricalSurface(allPoints[0].position, allPoints[1].position, meshData);
            if (cylinderInfo) {
              value = cylinderInfo.arcLength;
              label = `${value.toFixed(2)} mm (arc)`;
              // Store cylinder info in metadata
              addMeasurement({
                id: crypto.randomUUID(),
                type: activeTool,
                points: allPoints,
                value,
                unit: "mm",
                label,
                visible: true,
                color: "#0066ff",
                createdAt: new Date(),
                metadata: {
                  cylindrical: true,
                  radius: cylinderInfo.radius,
                  center: cylinderInfo.center,
                  axis: cylinderInfo.axis,
                },
              });
              clearTempPoints();
              return;
            } else {
              value = calculateDistance(allPoints[0].position, allPoints[1].position);
              label = `${value.toFixed(2)} mm`;
            }
            break;
          case "angle":
            value = calculateAngle(allPoints[0].position, allPoints[1].position, allPoints[2].position);
            label = `${value.toFixed(1)}°`;
            break;
          case "radius":
          case "diameter":
            value = calculateRadius(allPoints[0].position, allPoints[1].position, allPoints[2].position);
            if (activeTool === "diameter") value *= 2;
            label = `${value.toFixed(2)} mm`;
            break;
          case "edge-to-edge":
            value = calculateDistance(allPoints[0].position, allPoints[1].position);
            const deltas = {
              deltaX: Math.abs(allPoints[1].position.x - allPoints[0].position.x),
              deltaY: Math.abs(allPoints[1].position.y - allPoints[0].position.y),
              deltaZ: Math.abs(allPoints[1].position.z - allPoints[0].position.z),
            };
            label = `${value.toFixed(2)} mm`;

            // Add measurement with axis breakdowns
            addMeasurement({
              id: crypto.randomUUID(),
              type: activeTool,
              points: allPoints,
              value,
              unit: "mm",
              label,
              visible: true,
              color: "#00BCD4",
              createdAt: new Date(),
              metadata: deltas,
            });
            clearTempPoints();
            return;
          case "face-to-face":
            value = calculateDistance(allPoints[0].position, allPoints[1].position);
            label = `${value.toFixed(2)} mm`;
            break;
          case "coordinate":
            const pos = allPoints[0].position;
            label = `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`;
            value = 0;
            break;
        }

        addMeasurement({
          id: crypto.randomUUID(),
          type: activeTool,
          points: allPoints,
          value,
          unit: activeTool === "angle" ? "deg" : activeTool === "coordinate" ? "coord" : "mm",
          label,
          visible: true,
          color: "#0066ff",
          createdAt: new Date(),
          metadata:
            activeTool === "coordinate"
              ? {
                  deltaX: allPoints[0].position.x,
                  deltaY: allPoints[0].position.y,
                  deltaZ: allPoints[0].position.z,
                }
              : undefined,
        });

        clearTempPoints();
      }
    };

    const canvas = scene.children[0]?.parent as any;
    const domElement = canvas?.domElement || document.querySelector("canvas");

    if (domElement) {
      domElement.addEventListener("click", handleClick);
      return () => domElement.removeEventListener("click", handleClick);
    }
  }, [enabled, meshData, activeTool, hoverInfo, tempPoints, addTempPoint, clearTempPoints, addMeasurement, scene]);

  // Render hover indicators
  if (!enabled || !hoverInfo || !activeTool) return null;

  return (
    <>
      {/* ✅ ENHANCED: Full face highlighting (all coplanar triangles) */}
      {hoverInfo.type === "face" &&
        hoverInfo.faceIndex !== undefined &&
        (() => {
          // Get all triangles that belong to this planar face
          const faceTriangles = getFullFaceTriangles(hoverInfo.faceIndex, meshData!);

          // Create merged geometry from all coplanar triangles
          const allVertices: number[] = [];

          faceTriangles.forEach((triIndex) => {
            const i = triIndex * 3;
            const i0 = meshData!.indices[i] * 3;
            const i1 = meshData!.indices[i + 1] * 3;
            const i2 = meshData!.indices[i + 2] * 3;

            const v0 = new THREE.Vector3(
              meshData!.vertices[i0],
              meshData!.vertices[i0 + 1],
              meshData!.vertices[i0 + 2],
            );
            const v1 = new THREE.Vector3(
              meshData!.vertices[i1],
              meshData!.vertices[i1 + 1],
              meshData!.vertices[i1 + 2],
            );
            const v2 = new THREE.Vector3(
              meshData!.vertices[i2],
              meshData!.vertices[i2 + 1],
              meshData!.vertices[i2 + 2],
            );

            allVertices.push(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
          });

          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(allVertices), 3));

          return (
            <mesh geometry={geometry}>
              <meshBasicMaterial color="#4CAF50" transparent opacity={0.4} side={THREE.DoubleSide} depthTest={false} />
            </mesh>
          );
        })()}

      {/* ✅ FIXED: Edge highlighting (cylinder geometry along edge) - Professional sizing */}
      {hoverInfo.type === "edge" &&
        hoverInfo.edgeInfo &&
        (() => {
          const start = hoverInfo.edgeInfo.start;
          const end = hoverInfo.edgeInfo.end;
          const direction = new THREE.Vector3().subVectors(end, start);
          const length = direction.length();
          const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

          const quaternion = new THREE.Quaternion();
          quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());

          return (
            <mesh position={midpoint} quaternion={quaternion}>
              {/* ✅ ENHANCED: Larger, more visible edge highlight */}
              <cylinderGeometry args={[2.0, 2.0, length, 12]} />
              <meshBasicMaterial color="#FFB84D" transparent opacity={0.8} depthTest={false} />
            </mesh>
          );
        })()}

      {/* ✅ ENHANCED: Professional hover point marker with better sizing */}
      <mesh position={hoverInfo.point}>
        <sphereGeometry args={[2.0, 20, 20]} />
        <meshBasicMaterial color={hoverInfo.type === "edge" ? "#FFB84D" : "#4CAF50"} depthTest={false} />
      </mesh>

      {/* ✅ ENHANCED: Larger, more legible label */}
      {labelText && (
        <Html
          position={hoverInfo.point.clone().add(hoverInfo.normal.clone().multiplyScalar(5))}
          center
          distanceFactor={7}
        >
          <div
            style={{
              background: "rgba(0, 0, 0, 0.92)",
              color: hoverInfo.type === "edge" ? "#FFD700" : "#4CAF50",
              padding: "6px 12px",
              borderRadius: "6px",
              fontSize: "13px",
              fontWeight: "bold",
              whiteSpace: "nowrap",
              pointerEvents: "none",
              userSelect: "none",
              border: `2px solid ${hoverInfo.type === "edge" ? "#FFD700" : "#4CAF50"}`,
              boxShadow: "0 3px 12px rgba(0,0,0,0.4)",
              fontFamily: "system-ui, -apple-system, sans-serif",
            }}
          >
            {labelText}
          </div>
        </Html>
      )}

      {/* ✅ ENHANCED: Professional temp points with better visibility */}
      {tempPoints.map((point, index) => (
        <mesh key={index} position={point.position}>
          <sphereGeometry args={[2.5, 20, 20]} />
          <meshBasicMaterial color="#FF00FF" depthTest={false} />
        </mesh>
      ))}
    </>
  );
};
