import React, { useRef, useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import {
  performMultiSnap,
  classifyEdge,
  calculateDistance,
  calculateAngle,
  calculateRadius,
} from "@/lib/measurementUtils";
import type { MeshData, SnapResult, EdgeClassification } from "@/lib/measurementUtils";

interface ProfessionalMeasurementToolProps {
  meshData: MeshData | null;
  meshRef: React.RefObject<THREE.Mesh>;
  enabled: boolean;
}

export const ProfessionalMeasurementTool: React.FC<ProfessionalMeasurementToolProps> = ({
  meshData,
  meshRef,
  enabled,
}) => {
  const { camera, raycaster, scene } = useThree();
  const [hoverPoint, setHoverPoint] = useState<THREE.Vector3 | null>(null);
  const [labelText, setLabelText] = useState<string>("");
  const [snapInfo, setSnapInfo] = useState<SnapResult | null>(null);

  const { activeTool, tempPoints, addTempPoint, clearTempPoints, addMeasurement, snapDistance, activeSnapTypes } =
    useMeasurementStore();

  // Handle pointer move for hover feedback
  useEffect(() => {
    if (!enabled || !meshData || !meshRef.current) return;

    const handlePointerMove = (event: PointerEvent) => {
      const rect = (event.target as HTMLElement).getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
      const intersects = raycaster.intersectObject(meshRef.current!);

      if (intersects.length > 0) {
        const point = intersects[0].point;
        const snapped = performMultiSnap(point, meshData, activeSnapTypes, snapDistance);

        if (snapped) {
          setHoverPoint(snapped.position);
          setSnapInfo(snapped);

          // Generate label based on snap type
          if (activeTool === "edge-select") {
            if (snapped.surfaceType === "edge" && snapped.metadata) {
              const startPoint = new THREE.Vector3(
                snapped.metadata.startPoint[0],
                snapped.metadata.startPoint[1],
                snapped.metadata.startPoint[2],
              );
              const endPoint = new THREE.Vector3(
                snapped.metadata.endPoint[0],
                snapped.metadata.endPoint[1],
                snapped.metadata.endPoint[2],
              );
              const edge = new THREE.Line3(startPoint, endPoint);
              const classification = classifyEdge(edge, meshData);

              if (classification.type === "circle" || classification.type === "arc") {
                setLabelText(`${classification.type.toUpperCase()}: R=${classification.radius?.toFixed(2)}mm`);
              } else {
                const length = edge.distance();
                setLabelText(`LINE: ${length.toFixed(2)}mm`);
              }
            } else if (snapped.surfaceType === "face" && snapped.faceIndex !== undefined) {
              // Calculate face area for face snapping
              const i = snapped.faceIndex * 3;
              const i0 = meshData.indices[i] * 3;
              const i1 = meshData.indices[i + 1] * 3;
              const i2 = meshData.indices[i + 2] * 3;

              const v0 = new THREE.Vector3(meshData.vertices[i0], meshData.vertices[i0 + 1], meshData.vertices[i0 + 2]);
              const v1 = new THREE.Vector3(meshData.vertices[i1], meshData.vertices[i1 + 1], meshData.vertices[i1 + 2]);
              const v2 = new THREE.Vector3(meshData.vertices[i2], meshData.vertices[i2 + 1], meshData.vertices[i2 + 2]);

              const edge1 = new THREE.Vector3().subVectors(v1, v0);
              const edge2 = new THREE.Vector3().subVectors(v2, v0);
              const crossProduct = new THREE.Vector3().crossVectors(edge1, edge2);
              const area = crossProduct.length() / 2;

              setLabelText(`FACE: ${area.toFixed(2)}mm²`);
            } else {
              setLabelText("");
            }
          } else {
            setLabelText("");
          }
        } else {
          setHoverPoint(point);
          setSnapInfo(null);
          setLabelText("");
        }
      } else {
        setHoverPoint(null);
        setSnapInfo(null);
        setLabelText("");
      }
    };

    const canvas = scene.children[0]?.parent as any;
    const domElement = canvas?.domElement || document.querySelector("canvas");

    if (domElement) {
      domElement.addEventListener("pointermove", handlePointerMove);
      return () => domElement.removeEventListener("pointermove", handlePointerMove);
    }
  }, [enabled, meshData, meshRef, camera, raycaster, scene, activeTool, snapDistance, activeSnapTypes]);

  // Handle click for measurement
  useEffect(() => {
    if (!enabled || !meshData || !activeTool) return;

    const handleClick = (event: PointerEvent) => {
      if (!snapInfo || !hoverPoint) return;

      const newPoint = {
        id: crypto.randomUUID(),
        position: hoverPoint.clone(),
        normal: snapInfo.normal?.clone(),
        surfaceType: snapInfo.surfaceType,
        metadata: snapInfo.metadata,
      };

      // For edge-select, complete on first click
      if (activeTool === "edge-select") {
        if (snapInfo.surfaceType === "edge" && snapInfo.metadata) {
          const startPoint = new THREE.Vector3(
            snapInfo.metadata.startPoint[0],
            snapInfo.metadata.startPoint[1],
            snapInfo.metadata.startPoint[2],
          );
          const endPoint = new THREE.Vector3(
            snapInfo.metadata.endPoint[0],
            snapInfo.metadata.endPoint[1],
            snapInfo.metadata.endPoint[2],
          );
          const edge = new THREE.Line3(startPoint, endPoint);
          const classification = classifyEdge(edge, meshData);

          let label = "";
          let value = 0;

          if (classification.type === "circle" || classification.type === "arc") {
            value = classification.radius || 0;
            label = `R${value.toFixed(2)}`;
          } else {
            value = edge.distance();
            label = `${value.toFixed(2)}`;
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
              edgeStart: snapInfo.metadata.startPoint,
              edgeEnd: snapInfo.metadata.endPoint,
            },
          });
          clearTempPoints();
        }
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
            value = calculateDistance(allPoints[0].position, allPoints[1].position);
            label = `${value.toFixed(2)} mm`;
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
            // Calculate perpendicular distance between two edges
            value = calculateDistance(allPoints[0].position, allPoints[1].position);
            label = `${value.toFixed(2)} mm`;
            break;
          case "face-to-face":
            // Calculate distance between two parallel faces
            value = calculateDistance(allPoints[0].position, allPoints[1].position);
            label = `${value.toFixed(2)} mm`;
            break;
          case "coordinate":
            // Display XYZ coordinates of single point
            const pos = allPoints[0].position;
            label = `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`;
            value = 0; // No numeric value for coordinates
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
          color: "#ffff00",
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
  }, [
    enabled,
    meshData,
    activeTool,
    hoverPoint,
    snapInfo,
    tempPoints,
    addTempPoint,
    clearTempPoints,
    addMeasurement,
    scene,
  ]);

  // Render hover indicator
  if (!enabled || !hoverPoint) return null;

  return (
    <>
      {/* Edge highlight for edge-select tool - thick tube geometry */}
      {activeTool === "edge-select" &&
        snapInfo?.surfaceType === "edge" &&
        snapInfo.metadata &&
        (() => {
          const start = new THREE.Vector3(
            snapInfo.metadata.startPoint[0],
            snapInfo.metadata.startPoint[1],
            snapInfo.metadata.startPoint[2],
          );
          const end = new THREE.Vector3(
            snapInfo.metadata.endPoint[0],
            snapInfo.metadata.endPoint[1],
            snapInfo.metadata.endPoint[2],
          );
          const direction = new THREE.Vector3().subVectors(end, start);
          const length = direction.length();
          const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

          const quaternion = new THREE.Quaternion();
          quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());

          return (
            <mesh position={midpoint} quaternion={quaternion}>
              <cylinderGeometry args={[2, 2, length, 8]} />
              <meshStandardMaterial
                color="#FFB84D"
                emissive="#FFB84D"
                emissiveIntensity={0.5}
                transparent
                opacity={0.6}
              />
            </mesh>
          );
        })()}

      {/* Face highlight for edge-select tool */}
      {activeTool === "edge-select" &&
        snapInfo?.surfaceType === "face" &&
        snapInfo.faceIndex !== undefined &&
        (() => {
          const i = snapInfo.faceIndex * 3;
          const i0 = meshData!.indices[i] * 3;
          const i1 = meshData!.indices[i + 1] * 3;
          const i2 = meshData!.indices[i + 2] * 3;

          const v0 = new THREE.Vector3(meshData!.vertices[i0], meshData!.vertices[i0 + 1], meshData!.vertices[i0 + 2]);
          const v1 = new THREE.Vector3(meshData!.vertices[i1], meshData!.vertices[i1 + 1], meshData!.vertices[i1 + 2]);
          const v2 = new THREE.Vector3(meshData!.vertices[i2], meshData!.vertices[i2 + 1], meshData!.vertices[i2 + 2]);

          const geometry = new THREE.BufferGeometry();
          const vertices = new Float32Array([v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z]);
          geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

          return (
            <mesh geometry={geometry}>
              <meshBasicMaterial color="#FFB84D" transparent opacity={0.3} side={THREE.DoubleSide} />
            </mesh>
          );
        })()}

      {/* Label - minimal tooltip */}
      {activeTool === "edge-select" && labelText && (
        <Html position={hoverPoint} center distanceFactor={10}>
          <div
            style={{
              background: "rgba(0, 0, 0, 0.9)",
              color: "#00ff00",
              padding: "2px 6px",
              borderRadius: "3px",
              fontSize: "10px",
              fontWeight: "bold",
              whiteSpace: "nowrap",
              pointerEvents: "none",
              userSelect: "none",
              border: "1px solid rgba(0, 255, 0, 0.3)",
            }}
          >
            {labelText}
          </div>
        </Html>
      )}

      {/* Temp points */}
      {tempPoints.map((point, index) => (
        <mesh key={index} position={point.position}>
          <sphereGeometry args={[2.5, 16, 16]} />
          <meshBasicMaterial color="#ff00ff" />
        </mesh>
      ))}
    </>
  );
};
