import React, { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { classifyFeatureEdge } from "@/lib/featureEdgeClassification";
import { MeasurementRenderer } from "./MeasurementRenderer";

interface EdgeClassification {
  type: 'line' | 'circle' | 'arc';
  diameter?: number;
  radius?: number;
  length?: number;
  center?: [number, number, number];
}

interface ConnectedEdgeGroup {
  segments: THREE.Line3[];
  count: number;
  isClosedLoop: boolean;
  totalLength: number;
  type: 'line' | 'circle' | 'arc';
  diameter?: number;
  radius?: number;
  center?: THREE.Vector3;
  label: string;
  classification: any;
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: EdgeClassification[];
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

  // Helper function to find all connected segments forming a continuous curve
  const findConnectedSegments = React.useCallback((startEdge: THREE.Line3, allEdges: THREE.Line3[]): THREE.Line3[] => {
    const connected = [startEdge];
    const threshold = 0.001; // 1 micron tolerance
    const visited = new Set<THREE.Line3>([startEdge]);
    
    // Find all edges that connect to this one
    let changed = true;
    while (changed) {
      changed = false;
      for (const edge of allEdges) {
        if (visited.has(edge)) continue;
        
        // Check if edge connects to any edge in our chain
        for (const chainEdge of connected) {
          const connects = 
            edge.start.distanceTo(chainEdge.end) < threshold ||
            edge.end.distanceTo(chainEdge.start) < threshold ||
            edge.start.distanceTo(chainEdge.start) < threshold ||
            edge.end.distanceTo(chainEdge.end) < threshold;
          
          if (connects) {
            connected.push(edge);
            visited.add(edge);
            changed = true;
            break;
          }
        }
      }
    }
    
    return connected;
  }, []);

  // Pre-compute all edge groups once (performance optimization)
  const edgeGroupsCache = React.useMemo(() => {
    if (edgeLines.length === 0) return new Map<THREE.Line3, ConnectedEdgeGroup>();
    
    const cache = new Map<THREE.Line3, ConnectedEdgeGroup>();
    const processed = new Set<THREE.Line3>();
    
    edgeLines.forEach(edge => {
      if (processed.has(edge)) return;
      
      // Find all connected segments for this edge
      const segments = findConnectedSegments(edge, edgeLines);
      const segmentCount = segments.length;
      
      // Check if it's a closed loop
      const isClosedLoop = segments.length > 0 &&
        segments[0].start.distanceTo(segments[segments.length - 1].end) < 0.001;
      
      // Calculate total length
      const totalLength = segments.reduce((sum, seg) => sum + seg.distance(), 0);
      
      let group: ConnectedEdgeGroup;
      
      // Circle: closed loop with exactly 32 segments (backend tessellation)
      if (isClosedLoop && segmentCount === 32) {
        const circumference = totalLength;
        const diameter = circumference / Math.PI;
        
        // Calculate center (average of all segment midpoints)
        const center = new THREE.Vector3();
        segments.forEach(seg => {
          center.add(new THREE.Vector3().lerpVectors(seg.start, seg.end, 0.5));
        });
        center.divideScalar(segmentCount);
        
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "circle",
          diameter,
          radius: diameter / 2,
          center,
          label: `⊙ Diameter: ø${diameter.toFixed(2)} mm`,
          classification: { type: "circle", diameter, radius: diameter / 2, center, length: circumference }
        };
      } 
      // Line: exactly 2 segments (straight edge endpoints)
      else if (segmentCount === 2) {
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "line",
          label: `📏 Length: ${totalLength.toFixed(2)} mm`,
          classification: { type: "line", length: totalLength }
        };
      } 
      // Arc: 3-31 segments (curved edges)
      else if (segmentCount >= 3 && segmentCount <= 31) {
        const startPoint = segments[0].start;
        const endPoint = segments[segmentCount - 1].end;
        const chord = startPoint.distanceTo(endPoint);
        
        // Estimate radius
        const radius = chord > 0.001 
          ? (totalLength * totalLength + chord * chord) / (8 * totalLength)
          : totalLength / (2 * Math.PI);
        
        // Calculate center
        const midPoint = new THREE.Vector3().lerpVectors(startPoint, endPoint, 0.5);
        const direction = new THREE.Vector3().subVectors(endPoint, startPoint).normalize();
        const perpendicular = new THREE.Vector3(-direction.y, direction.x, direction.z);
        const h = Math.sqrt(Math.max(0, radius * radius - (chord / 2) * (chord / 2)));
        const center = midPoint.clone().add(perpendicular.multiplyScalar(h));
        
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "arc",
          radius,
          center,
          label: `⌒ Radius: R${radius.toFixed(2)} mm`,
          classification: { type: "arc", radius, length: totalLength, center }
        };
      } 
      // Fallback for unexpected cases
      else {
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "line",
          label: `📏 Length: ${totalLength.toFixed(2)} mm`,
          classification: { type: "line", length: totalLength }
        };
      }
      
      // Cache the result for ALL segments in this group
      segments.forEach(seg => {
        cache.set(seg, group);
        processed.add(seg);
      });
    });
    
    return cache;
  }, [edgeLines, findConnectedSegments]);

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
          // Instant lookup from pre-computed cache!
          const edgeGroup = edgeGroupsCache.get(closestEdge);
          
          if (edgeGroup) {
            setLabelText(edgeGroup.label);
            setHoverInfo({ 
              position: point, 
              classification: edgeGroup.classification, 
              edge: closestEdge 
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

      if (classification.type === "circle") {
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [{ id: generateMeasurementId(), position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }],
          value: classification.diameter || 0,
          unit: "mm",
          label: `⊙ ${formatMeasurement(classification.diameter || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: { edgeType: "circle", center: classification.center },
        });
      } else if (classification.type === "arc") {
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [{ id: generateMeasurementId(), position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }],
          value: classification.radius || 0,
          unit: "mm",
          label: `R ${formatMeasurement(classification.radius || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: { edgeType: "arc", center: classification.center },
        });
      } else {
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [
            { id: generateMeasurementId(), position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" },
            { id: generateMeasurementId(), position: edge.end, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }
          ],
          value: classification.length || 0,
          unit: "mm",
          label: `${formatMeasurement(classification.length || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: { edgeType: "line" },
        });
      }
    };

    gl.domElement.addEventListener("click", handleClick);
    return () => gl.domElement.removeEventListener("click", handleClick);
  }, [enabled, hoverInfo, addMeasurement]);

  return (
    <>
      {hoverInfo && (
        <>
          <primitive object={(() => {
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array([
              hoverInfo.edge.start.x, hoverInfo.edge.start.y, hoverInfo.edge.start.z,
              hoverInfo.edge.end.x, hoverInfo.edge.end.y, hoverInfo.edge.end.z,
            ]);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const material = new THREE.LineBasicMaterial({ 
              color: 0xFF8C00, 
              transparent: true, 
              opacity: 0.8, 
              depthTest: false 
            });
            
            return new THREE.Line(geometry, material);
          })()} />

          <Html position={hoverInfo.position} center style={{ pointerEvents: 'none', transform: 'translateY(-40px)' }}>
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
