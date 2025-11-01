import React, { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { MeasurementRenderer } from "./MeasurementRenderer";
import { validateMeasurement } from "@/lib/measurementValidation";

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
  isClosedLoop: boolean;
  totalLength: number;
  type: "line" | "circle" | "arc";
  diameter?: number;
  radius?: number;
  center?: THREE.Vector3;
  label: string;
  classification: any;
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
 * ✅ IMPROVED: Intelligent backend edge matching
 * Matches frontend edge groups to backend edges using geometric similarity
 */
function findMatchingBackendEdge(
  frontendGroup: {
    segments: THREE.Line3[];
    type: "line" | "circle" | "arc";
    center?: THREE.Vector3;
    totalLength: number;
  },
  backendEdges?: BackendEdgeClassification[],
): BackendEdgeClassification | undefined {
  if (!backendEdges || backendEdges.length === 0) return undefined;

  let bestMatch: BackendEdgeClassification | undefined;
  let bestScore = 0;

  for (const backendEdge of backendEdges) {
    // Must match type first
    if (backendEdge.type !== frontendGroup.type) continue;

    let score = 0;

    if (frontendGroup.type === "circle" && backendEdge.type === "circle") {
      // Circle matching: compare centers and diameter
      if (frontendGroup.center && backendEdge.center) {
        const frontendCenter = frontendGroup.center;
        const backendCenter = new THREE.Vector3(...backendEdge.center);
        const centerDist = frontendCenter.distanceTo(backendCenter);

        // Calculate diameter from frontend
        const frontendDiameter = frontendGroup.totalLength / Math.PI;
        const backendDiameter = backendEdge.diameter || (backendEdge.radius ? backendEdge.radius * 2 : 0);

        const diameterDiff = Math.abs(frontendDiameter - backendDiameter);
        const diameterError = diameterDiff / backendDiameter;

        // Score based on center proximity and diameter match
        // Perfect match: center < 1mm away, diameter < 2% error
        if (centerDist < 5 && diameterError < 0.05) {
          score = 100 - centerDist - diameterError * 100;
        }
      }
    } else if (frontendGroup.type === "line" && backendEdge.type === "line") {
      // Line matching: compare start/end points and length
      const frontendStart = frontendGroup.segments[0].start;
      const frontendEnd = frontendGroup.segments[frontendGroup.segments.length - 1].end;
      const backendStart = new THREE.Vector3(...backendEdge.start_point);
      const backendEnd = new THREE.Vector3(...backendEdge.end_point);

      // Check both orientations
      const dist1 = frontendStart.distanceTo(backendStart) + frontendEnd.distanceTo(backendEnd);
      const dist2 = frontendStart.distanceTo(backendEnd) + frontendEnd.distanceTo(backendStart);
      const minDist = Math.min(dist1, dist2);

      const lengthDiff = Math.abs(frontendGroup.totalLength - (backendEdge.length || 0));
      const lengthError = lengthDiff / (backendEdge.length || 1);

      // Perfect match: endpoints < 2mm away, length < 2% error
      if (minDist < 4 && lengthError < 0.05) {
        score = 100 - minDist - lengthError * 100;
      }
    } else if (frontendGroup.type === "arc" && backendEdge.type === "arc") {
      // Arc matching: compare endpoints, length, and radius
      const frontendStart = frontendGroup.segments[0].start;
      const frontendEnd = frontendGroup.segments[frontendGroup.segments.length - 1].end;
      const backendStart = new THREE.Vector3(...backendEdge.start_point);
      const backendEnd = new THREE.Vector3(...backendEdge.end_point);

      const dist1 = frontendStart.distanceTo(backendStart) + frontendEnd.distanceTo(backendEnd);
      const dist2 = frontendStart.distanceTo(backendEnd) + frontendEnd.distanceTo(backendStart);
      const minDist = Math.min(dist1, dist2);

      // Estimate frontend radius
      const chord = frontendStart.distanceTo(frontendEnd);
      const arcLength = frontendGroup.totalLength;
      const frontendRadius = (arcLength * arcLength + chord * chord) / (8 * arcLength);
      const backendRadius = backendEdge.radius || 0;

      const radiusDiff = Math.abs(frontendRadius - backendRadius);
      const radiusError = radiusDiff / backendRadius;

      // Perfect match: endpoints < 3mm away, radius < 5% error
      if (minDist < 6 && radiusError < 0.1) {
        score = 100 - minDist - radiusError * 100;
      }
    }

    if (score > bestScore) {
      bestScore = score;
      bestMatch = backendEdge;
    }
  }

  // Only return match if score is reasonably high
  return bestScore > 50 ? bestMatch : undefined;
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

  // Helper function to find connected segments
  const findFeatureSegments = React.useCallback((startEdge: THREE.Line3, allEdges: THREE.Line3[]): THREE.Line3[] => {
    const segments = [startEdge];
    const threshold = 0.01;
    const visited = new Set<THREE.Line3>([startEdge]);
    const maxSegments = 32;

    const areConnected = (edge1End: THREE.Vector3, edge2: THREE.Line3): { connected: boolean; reversed: boolean } => {
      if (edge2.start.distanceTo(edge1End) < threshold) {
        return { connected: true, reversed: false };
      }
      if (edge2.end.distanceTo(edge1End) < threshold) {
        return { connected: true, reversed: true };
      }
      return { connected: false, reversed: false };
    };

    const getDirection = (edge: THREE.Line3, reversed: boolean = false): THREE.Vector3 => {
      return reversed
        ? new THREE.Vector3().subVectors(edge.start, edge.end).normalize()
        : new THREE.Vector3().subVectors(edge.end, edge.start).normalize();
    };

    let currentEdge = startEdge;
    let currentDirection = getDirection(currentEdge);
    let foundNext = true;

    // Forward traversal
    while (foundNext && segments.length < maxSegments) {
      foundNext = false;
      let bestCandidate: THREE.Line3 | null = null;
      let bestReversed = false;
      let bestAlignment = -1;

      for (const edge of allEdges) {
        if (visited.has(edge)) continue;

        const connection = areConnected(currentEdge.end, edge);
        if (!connection.connected) continue;

        const nextDirection = getDirection(edge, connection.reversed);
        const alignment = currentDirection.dot(nextDirection);

        if (alignment > 0.5 && alignment > bestAlignment) {
          bestCandidate = edge;
          bestReversed = connection.reversed;
          bestAlignment = alignment;
        }
      }

      if (bestCandidate) {
        segments.push(bestCandidate);
        visited.add(bestCandidate);
        currentEdge = bestCandidate;
        currentDirection = getDirection(currentEdge, bestReversed);
        foundNext = true;
      }
    }

    // Backward traversal
    currentEdge = startEdge;
    currentDirection = getDirection(currentEdge).negate();
    foundNext = true;

    while (foundNext && segments.length < maxSegments) {
      foundNext = false;
      let bestCandidate: THREE.Line3 | null = null;
      let bestReversed = false;
      let bestAlignment = -1;

      for (const edge of allEdges) {
        if (visited.has(edge)) continue;

        const connection = areConnected(currentEdge.start, edge);
        if (!connection.connected) continue;

        const nextDirection = getDirection(edge, !connection.reversed).negate();
        const alignment = currentDirection.dot(nextDirection);

        if (alignment > 0.5 && alignment > bestAlignment) {
          bestCandidate = edge;
          bestReversed = connection.reversed;
          bestAlignment = alignment;
        }
      }

      if (bestCandidate) {
        segments.unshift(bestCandidate);
        visited.add(bestCandidate);
        currentEdge = bestCandidate;
        currentDirection = getDirection(currentEdge, bestReversed).negate();
        foundNext = true;
      }
    }

    return segments;
  }, []);

  // ✅ IMPROVED: Pre-compute edge groups with intelligent backend matching
  const edgeGroupsCache = React.useMemo(() => {
    if (edgeLines.length === 0) return new Map<THREE.Line3, ConnectedEdgeGroup>();

    const cache = new Map<THREE.Line3, ConnectedEdgeGroup>();
    const processed = new Set<THREE.Line3>();

    edgeLines.forEach((edge) => {
      if (processed.has(edge)) return;

      const segments = findFeatureSegments(edge, edgeLines);
      const segmentCount = segments.length;

      const isClosedLoop =
        segments.length > 0 && segments[0].start.distanceTo(segments[segments.length - 1].end) < 0.001;

      const totalLength = segments.reduce((sum, seg) => sum + seg.distance(), 0);

      let group: ConnectedEdgeGroup;

      // Circle: 32 segments OR closed loop with 30-32 segments
      if (segmentCount === 32 || (isClosedLoop && segmentCount >= 30)) {
        const circumference = totalLength;
        const diameter = circumference / Math.PI;

        const center = new THREE.Vector3();
        segments.forEach((seg) => {
          center.add(new THREE.Vector3().lerpVectors(seg.start, seg.end, 0.5));
        });
        center.divideScalar(segmentCount);

        // ✅ IMPROVED: Use geometric matching instead of simple distance
        const backendEdge = findMatchingBackendEdge(
          { segments, type: "circle", center, totalLength },
          meshData?.edge_classifications,
        );

        // ✅ CRITICAL FIX: Use backend diameter for label when available
        const displayDiameter = backendEdge?.diameter || diameter;

        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "circle",
          diameter: displayDiameter, // Use backend value if available
          radius: displayDiameter / 2,
          center,
          label: `⊙ Diameter: ø${displayDiameter.toFixed(2)} mm`,
          classification: {
            type: "circle",
            diameter: displayDiameter,
            radius: displayDiameter / 2,
            center,
            length: circumference,
          },
          backendClassification: backendEdge,
        };
      }
      // Line: 2 segments
      else if (segmentCount === 2) {
        const backendEdge = findMatchingBackendEdge(
          { segments, type: "line", totalLength },
          meshData?.edge_classifications,
        );

        // ✅ Use backend length for label when available
        const displayLength = backendEdge?.length || totalLength;

        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "line",
          label: `📏 Length: ${displayLength.toFixed(2)} mm`,
          classification: { type: "line", length: displayLength },
          backendClassification: backendEdge,
        };
      }
      // Arc: 3-31 segments
      else if (segmentCount >= 3 && segmentCount <= 31) {
        const startPoint = segments[0].start;
        const endPoint = segments[segmentCount - 1].end;
        const chord = startPoint.distanceTo(endPoint);

        const radius =
          chord > 0.001 ? (totalLength * totalLength + chord * chord) / (8 * totalLength) : totalLength / (2 * Math.PI);

        const midPoint = new THREE.Vector3().lerpVectors(startPoint, endPoint, 0.5);
        const direction = new THREE.Vector3().subVectors(endPoint, startPoint).normalize();
        const perpendicular = new THREE.Vector3(-direction.y, direction.x, direction.z);
        const h = Math.sqrt(Math.max(0, radius * radius - (chord / 2) * (chord / 2)));
        const center = midPoint.clone().add(perpendicular.multiplyScalar(h));

        const backendEdge = findMatchingBackendEdge(
          { segments, type: "arc", center, totalLength },
          meshData?.edge_classifications,
        );

        // ✅ CRITICAL FIX: Use backend radius for label when available
        const displayRadius = backendEdge?.radius || radius;

        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "arc",
          radius: displayRadius, // Use backend value if available
          center,
          label: `⌒ Radius: R${displayRadius.toFixed(2)} mm`,
          classification: { type: "arc", radius: displayRadius, length: totalLength, center },
          backendClassification: backendEdge,
        };
      }
      // Fallback
      else {
        const backendEdge = findMatchingBackendEdge(
          { segments, type: "line", totalLength },
          meshData?.edge_classifications,
        );

        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "line",
          label: `📏 Length: ${totalLength.toFixed(2)} mm`,
          classification: { type: "line", length: totalLength },
          backendClassification: backendEdge,
        };
      }

      segments.forEach((seg) => {
        cache.set(seg, group);
        processed.add(seg);
      });
    });

    console.log("✅ Edge groups created:", cache.size, "groups from", edgeLines.length, "edge lines");
    console.log(
      "📊 Backend matches:",
      Array.from(cache.values()).filter((g) => g.backendClassification).length,
      "of",
      cache.size,
    );

    return cache;
  }, [edgeLines, findFeatureSegments, meshData?.edge_classifications]);

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

          if (edgeGroup) {
            setLabelText(edgeGroup.label);
            setHoverInfo({
              position: point,
              classification: edgeGroup.classification,
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
      const { classification, edge } = hoverInfo;
      const edgeGroup = edgeGroupsCache.get(edge);
      const backendClassification = edgeGroup?.backendClassification;

      if (classification.type === "circle") {
        // ✅ CRITICAL FIX: Use backend ground truth diameter when available
        const actualDiameter = backendClassification?.diameter || classification.diameter || 0;

        const validation = validateMeasurement(
          actualDiameter,
          edgeGroup?.count || 0,
          "diameter",
          backendClassification,
        );

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
          value: actualDiameter,
          unit: "mm",
          label: `⊙ ${formatMeasurement(actualDiameter, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: {
            edgeType: "circle",
            center: classification.center,
            validation,
            segmentCount: edgeGroup?.count,
            backendMatch: !!backendClassification,
            backendDiameter: backendClassification?.diameter,
            frontendDiameter: classification.diameter,
          },
        });
      } else if (classification.type === "arc") {
        // ✅ CRITICAL FIX: Use backend ground truth radius when available
        const actualRadius = backendClassification?.radius || classification.radius || 0;

        const validation = validateMeasurement(actualRadius, edgeGroup?.count || 0, "radius", backendClassification);

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
          value: actualRadius,
          unit: "mm",
          label: `R ${formatMeasurement(actualRadius, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: {
            edgeType: "arc",
            center: classification.center,
            validation,
            segmentCount: edgeGroup?.count,
            backendMatch: !!backendClassification,
            backendRadius: backendClassification?.radius,
            frontendRadius: classification.radius,
          },
        });
      } else {
        // ✅ CRITICAL FIX: Use backend ground truth length when available
        const actualLength = backendClassification?.length || classification.length || 0;

        const validation = validateMeasurement(actualLength, edgeGroup?.count || 0, "length", backendClassification);

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
          value: actualLength,
          unit: "mm",
          label: `📏 ${formatMeasurement(actualLength, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: {
            edgeType: "line",
            validation,
            segmentCount: edgeGroup?.count,
            backendMatch: !!backendClassification,
            backendLength: backendClassification?.length,
            frontendLength: classification.length,
          },
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
