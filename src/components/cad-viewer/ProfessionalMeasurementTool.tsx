import React, { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import { formatMeasurement, generateMeasurementId } from "@/lib/measurementUtils";
import { classifyFeatureEdge } from "@/lib/featureEdgeClassification";
import { MeasurementRenderer } from "./MeasurementRenderer";
import { validateMeasurement, findClosestBackendEdge } from "@/lib/measurementValidation";
import { findEdgeByFeatureId, TaggedFeatureEdge } from "@/lib/featureIdMatcher";

interface EdgeClassification {
  type: 'line' | 'circle' | 'arc';
  diameter?: number;
  radius?: number;
  length?: number;
  center?: [number, number, number];
}

// Backend classification with ground truth
interface BackendEdgeClassification {
  id: number;
  feature_id?: number;  // Unique ID for this geometric feature (optional for backwards compatibility)
  type: 'line' | 'circle' | 'arc';
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
  type: 'line' | 'circle' | 'arc';
  diameter?: number;
  radius?: number;
  center?: THREE.Vector3;
  label: string;
  classification: any;
  backendClassification?: BackendEdgeClassification;  // Ground truth from backend
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  vertex_face_ids?: number[];  // Map vertex index to face_id
  face_classifications?: BackendFaceClassification[];  // Detailed face data
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: BackendEdgeClassification[];  // Backend ground truth
  tagged_feature_edges?: TaggedFeatureEdge[];  // NEW: Direct feature_id lookup
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

  // Helper function to find connected segments using directional traversal
  const findFeatureSegments = React.useCallback((startEdge: THREE.Line3, allEdges: THREE.Line3[]): THREE.Line3[] => {
    const segments = [startEdge];
    const threshold = 0.01; // Increased from 0.001 to handle tessellation gaps
    const visited = new Set<THREE.Line3>([startEdge]);
    
    const maxSegments = 32; // Max for a full circle
    
    // Helper: Check if two edges are connected (handles both directions)
    const areConnected = (edge1End: THREE.Vector3, edge2: THREE.Line3): { connected: boolean; reversed: boolean } => {
      if (edge2.start.distanceTo(edge1End) < threshold) {
        return { connected: true, reversed: false };
      }
      if (edge2.end.distanceTo(edge1End) < threshold) {
        return { connected: true, reversed: true };
      }
      return { connected: false, reversed: false };
    };
    
    // Helper: Get direction of edge (normalized)
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
        
        // Check tangent continuity (direction alignment)
        const nextDirection = getDirection(edge, connection.reversed);
        const alignment = currentDirection.dot(nextDirection);
        
        // Only accept edges that continue in roughly the same direction (>30° = 0.866)
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
    currentDirection = getDirection(currentEdge).negate(); // Opposite direction
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

  // Pre-compute all edge groups once (performance optimization)
  const edgeGroupsCache = React.useMemo(() => {
    if (edgeLines.length === 0) return new Map<THREE.Line3, ConnectedEdgeGroup>();
    
    const cache = new Map<THREE.Line3, ConnectedEdgeGroup>();
    const processed = new Set<THREE.Line3>();
    
    edgeLines.forEach(edge => {
      if (processed.has(edge)) return;
      
      // Find all connected segments for this edge
      const segments = findFeatureSegments(edge, edgeLines);
      const segmentCount = segments.length;
      
      // Check if it's a closed loop
      const isClosedLoop = segments.length > 0 &&
        segments[0].start.distanceTo(segments[segments.length - 1].end) < 0.001;
      
      // Calculate total length
      const totalLength = segments.reduce((sum, seg) => sum + seg.distance(), 0);
      
      let group: ConnectedEdgeGroup;
      
      // Circle: exactly 32 segments OR closed loop with 30-32 segments (backend tessellation)
      if (segmentCount === 32 || (isClosedLoop && segmentCount >= 30)) {
        const circumference = totalLength;
        const diameter = circumference / Math.PI;
        
        // Calculate center (average of all segment midpoints)
        const center = new THREE.Vector3();
        segments.forEach(seg => {
          center.add(new THREE.Vector3().lerpVectors(seg.start, seg.end, 0.5));
        });
        center.divideScalar(segmentCount);
        
        // Match with backend classification using first segment midpoint
            // TODO: Once backend tags edge segments with feature_id, use direct lookup
            const firstSegmentMid = new THREE.Vector3().lerpVectors(segments[0].start, segments[0].end, 0.5);
            const backendEdge = findClosestBackendEdge(firstSegmentMid, meshData?.edge_classifications);
        
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
          classification: { type: "circle", diameter, radius: diameter / 2, center, length: circumference },
          backendClassification: backendEdge
        };
      }
      // Line: exactly 2 segments (straight edge endpoints)
      else if (segmentCount === 2) {
        const midpoint = new THREE.Vector3().lerpVectors(segments[0].start, segments[0].end, 0.5);
        const backendEdge = findClosestBackendEdge(midpoint, meshData?.edge_classifications);
        
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "line",
          label: `📏 Length: ${totalLength.toFixed(2)} mm`,
          classification: { type: "line", length: totalLength },
          backendClassification: backendEdge
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
        
            // Match with backend classification using first segment midpoint  
            // TODO: Once backend tags edge segments with feature_id, use direct lookup
            const firstSegmentMid = new THREE.Vector3().lerpVectors(segments[0].start, segments[0].end, 0.5);
            const backendEdge = findClosestBackendEdge(firstSegmentMid, meshData?.edge_classifications);
        
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "arc",
          radius,
          center,
          label: `⌒ Radius: R${radius.toFixed(2)} mm`,
          classification: { type: "arc", radius, length: totalLength, center },
          backendClassification: backendEdge
        };
      }
      // Fallback for unexpected cases
      else {
        const midpoint = segments.length > 0 
          ? new THREE.Vector3().lerpVectors(segments[0].start, segments[segments.length - 1].end, 0.5)
          : new THREE.Vector3();
        const backendEdge = findClosestBackendEdge(midpoint, meshData?.edge_classifications);
        
        group = {
          segments,
          count: segmentCount,
          isClosedLoop,
          totalLength,
          type: "line",
          label: `📏 Length: ${totalLength.toFixed(2)} mm`,
          classification: { type: "line", length: totalLength },
          backendClassification: backendEdge
        };
      }
      
      // Cache the result for ALL segments in this group
      segments.forEach(seg => {
        cache.set(seg, group);
        processed.add(seg);
      });
    });
    
    return cache;
  }, [edgeLines, findFeatureSegments]);

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
      
      // NEW: Try direct feature_id lookup first
      const directMatch = findEdgeByFeatureId(edge, meshData?.tagged_feature_edges);
      
      // Get the edge group to access backend classification (fallback)
      const edgeGroup = edgeGroupsCache.get(edge);
      const backendClassification = directMatch ? {
        id: directMatch.feature_id,
        feature_id: directMatch.feature_id,
        type: directMatch.type,
        start_point: directMatch.start,
        end_point: directMatch.end,
        diameter: directMatch.diameter,
        radius: directMatch.radius,
        length: directMatch.length,
        segment_count: 32
      } as BackendEdgeClassification : edgeGroup?.backendClassification;

      if (classification.type === "circle") {
        // Validate against backend
        const validation = validateMeasurement(
          classification.diameter || 0,
          edgeGroup?.count || 0,
          "diameter",
          backendClassification
        );
        
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
          metadata: { 
            edgeType: "circle", 
            center: classification.center,
            validation,
            segmentCount: edgeGroup?.count
          },
        });
      } else if (classification.type === "arc") {
        // Validate against backend
        const validation = validateMeasurement(
          classification.radius || 0,
          edgeGroup?.count || 0,
          "radius",
          backendClassification
        );
        
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
          metadata: { 
            edgeType: "arc", 
            center: classification.center,
            validation,
            segmentCount: edgeGroup?.count
          },
        });
      } else {
        // Validate against backend
        const validation = validateMeasurement(
          classification.length || 0,
          edgeGroup?.count || 0,
          "length",
          backendClassification
        );
        
        addMeasurement({
          id: generateMeasurementId(),
          type: "edge-select",
          points: [
            { id: generateMeasurementId(), position: edge.start, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" },
            { id: generateMeasurementId(), position: edge.end, normal: new THREE.Vector3(0, 1, 0), surfaceType: "edge" }
          ],
          value: classification.length || 0,
          unit: "mm",
          label: `📏 ${formatMeasurement(classification.length || 0, "mm")}`,
          color: "#0066CC",
          visible: true,
          createdAt: new Date(),
          metadata: { 
            edgeType: "line",
            validation,
            segmentCount: edgeGroup?.count
          },
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
