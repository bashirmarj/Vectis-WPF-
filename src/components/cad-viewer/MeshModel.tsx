import { useEffect, useRef, forwardRef, useMemo } from "react";
import * as React from "react";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { SilhouetteEdges } from "./SilhouetteEdges";

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
  vertex_face_ids?: number[];
  face_mapping?: Record<number, { triangle_indices: number[]; triangle_range: [number, number] }>; // Maps BREP face index -> mesh triangles
  tagged_edges?: Array<{
    feature_id: number;
    start: [number, number, number];
    end: [number, number, number];
    type: string;
    iso_type?: string;
    snap_points?: [number, number, number][]; // For curved edges (circles, arcs)
  }>;
}

interface MeshModelProps {
  meshData: MeshData;
  sectionPlane: "none" | "xy" | "xz" | "yz";
  sectionPosition: number;
  showEdges: boolean; // For solid mode feature edges
  showHiddenEdges?: boolean; // For wireframe mode occlusion control
  displayStyle?: "solid" | "wireframe" | "translucent";
  topologyColors?: boolean;
  useSilhouetteEdges?: boolean;
  controlsRef?: React.RefObject<any>;
  highlightedFaceIds?: number[];
  highlightColor?: string;
  highlightIntensity?: number;
}

// Professional solid color for CAD rendering
const SOLID_COLOR = "#FF6B6B"; // Red color matching old working version

// Fusion 360 Analysis colors
const TOPOLOGY_COLORS = {
  internal: "#FF6B6B",
  cylindrical: "#FF6B6B",
  planar: "#FF6B6B",
  external: "#FF6B6B",
  through: "#FF6B6B",
  default: "#FF6B6B",
};

export interface MeshModelHandle {
  mesh: THREE.Mesh;
  featureEdgesGeometry: THREE.BufferGeometry | null;
}

export const MeshModel = forwardRef<MeshModelHandle, MeshModelProps>(
  (
    {
      meshData,
      sectionPlane,
      sectionPosition,
      showEdges,
      showHiddenEdges = false,
      displayStyle = "solid",
      topologyColors = false,
      useSilhouetteEdges = false,
      controlsRef,
      highlightedFaceIds = [],
      highlightColor = "#3B82F6", // Blue highlight color
      highlightIntensity = 0.8,
    },
    ref,
  ) => {
    const internalMeshRef = useRef<THREE.Mesh>(null);
    const meshRef = internalMeshRef;

    // Refs for geometry management (no memoization - manual change tracking)
    const geometryRef = useRef<THREE.BufferGeometry | null>(null);
    const prevMeshDataRef = useRef<MeshData | null>(null);

    // Create single unified geometry - ALWAYS USE INDEXED GEOMETRY
    // Only recreate when meshData reference changes
    if (meshData !== prevMeshDataRef.current) {
      // Dispose old geometry
      if (geometryRef.current) {
        geometryRef.current.dispose();
      }

      if (!meshData?.vertices || !meshData?.indices || !meshData?.normals) {
        console.warn("⚠️ Missing mesh data - creating fallback geometry");
        geometryRef.current = new THREE.BufferGeometry();
      } else {
        const geo = new THREE.BufferGeometry();

        // ALWAYS use indexed geometry (vertices shared across triangles)
        geo.setAttribute("position", new THREE.Float32BufferAttribute(meshData.vertices, 3));
        geo.setIndex(meshData.indices);
        geo.setAttribute("normal", new THREE.Float32BufferAttribute(meshData.normals, 3));

        // ✅ CRITICAL: DO NOT call computeVertexNormals() - trust backend normals!
        // The backend generates professional smooth normals that should not be overwritten
        // Calling computeVertexNormals() here destroys the smooth normals and creates visible horizontal lines

        geo.computeBoundingSphere();
        geometryRef.current = geo;
      }

      prevMeshDataRef.current = meshData;
    }

    const geometry = geometryRef.current!;

    // Apply vertex colors ONLY for topology mode (not for highlighting)
    useEffect(() => {
      if (!geometry) return;

      const vertexCount = meshData.vertices.length / 3;
      const colors = new Float32Array(vertexCount * 3);
      const baseColor = new THREE.Color(SOLID_COLOR);

      if (topologyColors) {
        // Topology color mode - use vertex_colors from backend
        if (meshData.vertex_colors && meshData.vertex_colors.length > 0) {
          for (let vertexIdx = 0; vertexIdx < vertexCount; vertexIdx++) {
            const faceType = meshData.vertex_colors[vertexIdx] || "default";
            const colorHex = TOPOLOGY_COLORS[faceType as keyof typeof TOPOLOGY_COLORS] || TOPOLOGY_COLORS.default;
            const color = new THREE.Color(colorHex);

            colors[vertexIdx * 3 + 0] = color.r;
            colors[vertexIdx * 3 + 1] = color.g;
            colors[vertexIdx * 3 + 2] = color.b;
          }
        }

        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geometry.attributes.color.needsUpdate = true;
      } else {
        // Solid color mode - use base color only (highlighting handled by separate mesh)
        for (let i = 0; i < vertexCount; i++) {
          colors[i * 3 + 0] = baseColor.r;
          colors[i * 3 + 1] = baseColor.g;
          colors[i * 3 + 2] = baseColor.b;
        }

        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geometry.attributes.color.needsUpdate = true;
      }
    }, [geometry, topologyColors, meshData.vertex_colors]);

    // 🔥 DUAL-MESH HIGHLIGHT: Build separate non-indexed geometry for highlighted faces
    const highlightGeometry = useMemo(() => {
      // Safety check: Only build if we have highlights
      if (!highlightedFaceIds || highlightedFaceIds.length === 0) {
        console.log("🔥 No highlighted faces - skipping highlight geometry");
        return null;
      }

      // Safety check: Required data
      if (!meshData.face_mapping || !meshData.indices || !meshData.vertices || !meshData.normals) {
        console.warn("⚠️ Missing data for highlight geometry:", {
          hasFaceMapping: !!meshData.face_mapping,
          hasIndices: !!meshData.indices,
          hasVertices: !!meshData.vertices,
          hasNormals: !!meshData.normals,
        });
        return null;
      }

      console.log("🔥 Building highlight geometry for faces:", highlightedFaceIds);

      // Collect triangle indices for highlighted faces
      const highlightTriIndices = new Set<number>();

      highlightedFaceIds.forEach(brepFaceId => {
        const mapping = meshData.face_mapping![brepFaceId];

        if (!mapping) {
          console.warn(`⚠️ No mapping for BREP face ${brepFaceId}`);
          return;
        }

        // Prioritize triangle_range for efficiency
        if (mapping.triangle_range) {
          const [start, end] = mapping.triangle_range;
          for (let i = start; i <= end; i++) {
            highlightTriIndices.add(i);
          }
        }
        // Fallback to triangle_indices
        else if (mapping.triangle_indices) {
          mapping.triangle_indices.forEach(idx => highlightTriIndices.add(idx));
        }
      });

      console.log("🔥 Collected triangles:", {
        triangleCount: highlightTriIndices.size,
        percentageOfMesh: ((highlightTriIndices.size / (meshData.indices!.length / 3)) * 100).toFixed(2) + '%'
      });

      // Safety check: Did we find any triangles?
      if (highlightTriIndices.size === 0) {
        console.warn("⚠️ No triangles found for highlighted faces");
        return null;
      }

      // Build non-indexed geometry from highlighted triangles
      const positions: number[] = [];
      const normals: number[] = [];

      highlightTriIndices.forEach(triIdx => {
        for (let i = 0; i < 3; i++) {
          const vIdx = meshData.indices![triIdx * 3 + i];

          // Safety check: Vertex index in bounds
          if (vIdx * 3 + 2 >= meshData.vertices!.length) {
            console.warn(`⚠️ Vertex index out of bounds: ${vIdx}`);
            return;
          }

          positions.push(
            meshData.vertices![vIdx * 3 + 0],
            meshData.vertices![vIdx * 3 + 1],
            meshData.vertices![vIdx * 3 + 2]
          );
          normals.push(
            meshData.normals![vIdx * 3 + 0],
            meshData.normals![vIdx * 3 + 1],
            meshData.normals![vIdx * 3 + 2]
          );
        }
      });

      // Safety check: Did we build any geometry?
      if (positions.length === 0) {
        console.warn("⚠️ No geometry built for highlight");
        return null;
      }

      console.log("✅ Highlight geometry built:", {
        vertices: positions.length / 3,
        triangles: positions.length / 9
      });

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
      geo.computeBoundingSphere();

      return geo;
    }, [highlightedFaceIds, meshData.face_mapping, meshData.indices, meshData.vertices, meshData.normals]);

    // Refs for feature edges geometry
    const featureEdgesRef = useRef<THREE.BufferGeometry | null>(null);
    const prevFeatureEdgesDataRef = useRef<{
      tagged_edges?: any[];
      feature_edges?: any[];
      indices?: number[];
      vertices?: number[];
    } | null>(null);

    // Pre-compute feature edges (for solid mode) - Only recreate when edge data changes
    const currentEdgesData = {
      tagged_edges: meshData?.tagged_edges,
      feature_edges: meshData?.feature_edges,
      indices: meshData?.indices,
      vertices: meshData?.vertices,
    };

    if (JSON.stringify(currentEdgesData) !== JSON.stringify(prevFeatureEdgesDataRef.current)) {
      // Dispose old geometry
      if (featureEdgesRef.current) {
        featureEdgesRef.current.dispose();
      }

      // PRIORITY: Use tagged_edges - backend handles all deduplication and filtering
      if (meshData?.tagged_edges && Array.isArray(meshData.tagged_edges) && meshData.tagged_edges.length > 0) {
        const featureEdgePositions: number[] = [];

        meshData.tagged_edges.forEach((edge) => {
          // Use snap_points for curved edges (circles, arcs) - creates polyline segments
          if (edge.snap_points && Array.isArray(edge.snap_points) && edge.snap_points.length >= 2) {
            for (let i = 0; i < edge.snap_points.length - 1; i++) {
              const p1 = edge.snap_points[i];
              const p2 = edge.snap_points[i + 1];
              featureEdgePositions.push(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
            }

            // Add closing segment for closed edges (circles, full ellipses)
            if (edge.is_closed && edge.snap_points.length > 2) {
              const first = edge.snap_points[0];
              const last = edge.snap_points[edge.snap_points.length - 1];
              featureEdgePositions.push(last[0], last[1], last[2], first[0], first[1], first[2]);
            }
          }
          // Fallback to start/end for straight edges
          else if (edge.start && edge.end && Array.isArray(edge.start) && Array.isArray(edge.end)) {
            featureEdgePositions.push(
              edge.start[0],
              edge.start[1],
              edge.start[2],
              edge.end[0],
              edge.end[1],
              edge.end[2],
            );
          }
        });

        if (featureEdgePositions.length > 0) {
          const geo = new THREE.BufferGeometry();
          geo.setAttribute("position", new THREE.Float32BufferAttribute(featureEdgePositions, 3));
          geo.computeBoundingSphere();
          featureEdgesRef.current = geo;
        }
      }
      // FALLBACK: Use feature_edges if tagged_edges not available
      else if (meshData?.feature_edges && Array.isArray(meshData.feature_edges) && meshData.feature_edges.length > 0) {
        const featureEdgePositions: number[] = [];

        meshData.feature_edges.forEach((polyline) => {
          for (let i = 0; i < polyline.length - 1; i++) {
            const p1 = polyline[i];
            const p2 = polyline[i + 1];
            featureEdgePositions.push(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
          }
        });

        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(featureEdgePositions, 3));
        geo.computeBoundingSphere();
        featureEdgesRef.current = geo;
      }
      // FALLBACK: Compute from mesh triangles (for STL files without BREP data)
      else if (meshData?.indices && meshData?.vertices) {
        const edgeMap = new Map<
          string,
          {
            v1: THREE.Vector3;
            v2: THREE.Vector3;
            normals: THREE.Vector3[];
          }
        >();

        const triangleCount = meshData.indices.length / 3;

        // Build edge map with face normals
        for (let i = 0; i < triangleCount; i++) {
          const i0 = meshData.indices[i * 3];
          const i1 = meshData.indices[i * 3 + 1];
          const i2 = meshData.indices[i * 3 + 2];

          const v0 = new THREE.Vector3(
            meshData.vertices[i0 * 3],
            meshData.vertices[i0 * 3 + 1],
            meshData.vertices[i0 * 3 + 2],
          );
          const v1 = new THREE.Vector3(
            meshData.vertices[i1 * 3],
            meshData.vertices[i1 * 3 + 1],
            meshData.vertices[i1 * 3 + 2],
          );
          const v2 = new THREE.Vector3(
            meshData.vertices[i2 * 3],
            meshData.vertices[i2 * 3 + 1],
            meshData.vertices[i2 * 3 + 2],
          );

          const e1 = new THREE.Vector3().subVectors(v1, v0);
          const e2 = new THREE.Vector3().subVectors(v2, v0);
          const normal = new THREE.Vector3().crossVectors(e1, e2).normalize();

          const getKey = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);

          const edges = [
            { v1: v0, v2: v1, key: getKey(i0, i1) },
            { v1: v1, v2: v2, key: getKey(i1, i2) },
            { v1: v2, v2: v0, key: getKey(i2, i0) },
          ];

          edges.forEach((edge) => {
            if (!edgeMap.has(edge.key)) {
              edgeMap.set(edge.key, {
                v1: edge.v1.clone(),
                v2: edge.v2.clone(),
                normals: [normal.clone()],
              });
            } else {
              edgeMap.get(edge.key)!.normals.push(normal.clone());
            }
          });
        }

        // Filter feature edges by angle threshold (20° - lowered to catch cylindrical edges)
        const featureEdgePositions: number[] = [];

        edgeMap.forEach((edgeData) => {
          // Boundary edges (only 1 face) - always show
          if (edgeData.normals.length === 1) {
            featureEdgePositions.push(
              edgeData.v1.x,
              edgeData.v1.y,
              edgeData.v1.z,
              edgeData.v2.x,
              edgeData.v2.y,
              edgeData.v2.z,
            );
            return;
          }

          // Feature edges: angle between adjacent faces > 20° (lowered from 45°)
          if (edgeData.normals.length === 2) {
            const n1 = edgeData.normals[0];
            const n2 = edgeData.normals[1];
            const normalAngle = Math.acos(Math.max(-1, Math.min(1, n1.dot(n2))));
            const normalAngleDeg = normalAngle * (180 / Math.PI);

            if (normalAngleDeg > 20) {
              featureEdgePositions.push(
                edgeData.v1.x,
                edgeData.v1.y,
                edgeData.v1.z,
                edgeData.v2.x,
                edgeData.v2.y,
                edgeData.v2.z,
              );
            }
          }
        });

        // Create geometry from filtered edges
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(featureEdgePositions, 3));
        featureEdgesRef.current = geo;
      }

      prevFeatureEdgesDataRef.current = currentEdgesData;
    }

    const featureEdgesGeometry = featureEdgesRef.current;

    // Refs for clean edges geometry
    const cleanEdgesRef = useRef<THREE.BufferGeometry | null>(null);
    const prevGeometryRef = useRef<THREE.BufferGeometry | null>(null);

    // Clean edge geometry for both solid and wireframe modes - Only recreate when geometry changes
    if (geometry !== prevGeometryRef.current) {
      if (cleanEdgesRef.current) {
        cleanEdgesRef.current.dispose();
      }

      cleanEdgesRef.current = geometry?.attributes?.position
        ? new THREE.EdgesGeometry(geometry, 1)
        : new THREE.BufferGeometry();

      prevGeometryRef.current = geometry;
    }

    const cleanEdgesGeometry = cleanEdgesRef.current!;

    // Refs for section plane
    const clippingPlaneRef = useRef<THREE.Plane[] | undefined>(undefined);
    const prevSectionPlaneRef = useRef(sectionPlane);
    const prevSectionPositionRef = useRef(sectionPosition);

    // Section plane - Only recreate when section props change
    if (sectionPlane !== prevSectionPlaneRef.current || sectionPosition !== prevSectionPositionRef.current) {
      if (sectionPlane === "none") {
        clippingPlaneRef.current = undefined;
      } else {
        let normal: THREE.Vector3;
        switch (sectionPlane) {
          case "xy":
            normal = new THREE.Vector3(0, 0, 1);
            break;
          case "xz":
            normal = new THREE.Vector3(0, 1, 0);
            break;
          case "yz":
            normal = new THREE.Vector3(1, 0, 0);
            break;
          default:
            normal = new THREE.Vector3(0, 0, 1);
        }

        clippingPlaneRef.current = [new THREE.Plane(normal, -sectionPosition)];
      }

      prevSectionPlaneRef.current = sectionPlane;
      prevSectionPositionRef.current = sectionPosition;
    }

    const clippingPlane = clippingPlaneRef.current;

    const { gl } = useThree();

    useEffect(() => {
      gl.localClippingEnabled = sectionPlane !== "none";
      gl.clippingPlanes = [];
    }, [sectionPlane, gl]);

    // Refs for material props
    const materialPropsRef = useRef<any>(null);
    const prevDisplayStyleRef = useRef(displayStyle);
    const prevClippingPlaneRef = useRef(clippingPlane);

    // Material props - Only recreate when display style or clipping plane changes
    if (displayStyle !== prevDisplayStyleRef.current || clippingPlane !== prevClippingPlaneRef.current) {
      const base = {
        color: SOLID_COLOR,
        side: THREE.DoubleSide,
        clippingPlanes: clippingPlane,
        clipIntersection: true,
        metalness: 0.0,
        roughness: 0.9,
        envMapIntensity: 0.5,
      };

      if (displayStyle === "wireframe") {
        materialPropsRef.current = { ...base, opacity: 0, transparent: true, wireframe: false };
      } else if (displayStyle === "translucent") {
        materialPropsRef.current = { ...base, transparent: true, opacity: 0.4, wireframe: false };
      } else {
        materialPropsRef.current = { ...base, transparent: false, opacity: 1, wireframe: false };
      }

      prevDisplayStyleRef.current = displayStyle;
      prevClippingPlaneRef.current = clippingPlane;
    }

    // Initialize material props on first render
    if (!materialPropsRef.current) {
      const base = {
        color: SOLID_COLOR,
        side: THREE.DoubleSide,
        clippingPlanes: clippingPlane,
        clipIntersection: true,
        metalness: 0.0,
        roughness: 0.9,
        envMapIntensity: 0.5,
      };
      materialPropsRef.current = { ...base, transparent: false, opacity: 1, wireframe: false };
    }

    const materialProps = materialPropsRef.current;

    // Cleanup geometries on unmount to prevent memory leaks
    useEffect(() => {
      return () => {
        if (geometryRef.current) geometryRef.current.dispose();
        if (featureEdgesRef.current) featureEdgesRef.current.dispose();
        if (cleanEdgesRef.current) cleanEdgesRef.current.dispose();
        if (highlightGeometry) highlightGeometry.dispose();
      };
    }, [highlightGeometry]);

    // Expose mesh and feature edges for external access
    React.useImperativeHandle(
      ref,
      () => ({
        mesh: meshRef.current!,
        featureEdgesGeometry: featureEdgesGeometry,
      }),
      [featureEdgesGeometry],
    );

    return (
      <group>
        {/* Mesh surface - DISABLED castShadow to prevent self-shadowing artifacts */}
        <mesh ref={meshRef} geometry={geometry} castShadow={false} receiveShadow>
          <meshStandardMaterial
            {...materialProps}
            color={topologyColors ? "#ffffff" : SOLID_COLOR}
            vertexColors={true}
            flatShading={true}
            toneMapped={false}
            metalness={0.1}
            roughness={0.6}
          />
        </mesh>

        {/* 🔥 HIGHLIGHT OVERLAY - Separate non-indexed mesh for precise feature highlighting */}
        {highlightGeometry && (
          <mesh
            geometry={highlightGeometry}
            renderOrder={1}
            castShadow={false}
          >
            <meshStandardMaterial
              color={highlightColor || "#3B82F6"}
              side={THREE.DoubleSide}
              toneMapped={false}
              metalness={0.1}
              roughness={0.6}
              emissive={highlightColor || "#3B82F6"}
              emissiveIntensity={highlightIntensity || 0.3}
              polygonOffset={true}
              polygonOffsetFactor={-2}
              polygonOffsetUnits={-2}
            />
          </mesh>
        )}

        {/* Invisible depth-writing mesh for wireframe occlusion */}
        {displayStyle === "wireframe" && !showHiddenEdges && (
          <mesh geometry={geometry} key={`depth-mesh-${showHiddenEdges}`}>
            <meshBasicMaterial
              colorWrite={false}
              depthWrite={true}
              depthTest={true}
              transparent={false}
              polygonOffset={true}
              polygonOffsetFactor={1}
              polygonOffsetUnits={1}
            />
          </mesh>
        )}

        {/* Solid mode: ONLY backend feature edges (SilhouetteEdges never active in solid) */}
        {displayStyle === "solid" && showEdges && featureEdgesGeometry && (
          <lineSegments geometry={featureEdgesGeometry} frustumCulled={true}>
            <lineBasicMaterial
              color="#000000"
              toneMapped={false}
              polygonOffset={true}
              polygonOffsetFactor={-15}
              polygonOffsetUnits={-10}
              depthTest={true}
              depthWrite={false}
              depthFunc={THREE.LessEqualDepth}
            />
          </lineSegments>
        )}

        {/* Wireframe mode - use feature edges for continuous curve rendering */}
        {displayStyle === "wireframe" &&
          (useSilhouetteEdges ? (
            <SilhouetteEdges
              geometry={geometry}
              mesh={meshRef.current}
              staticFeatureEdges={featureEdgesGeometry}
              showHiddenEdges={showHiddenEdges}
              controlsRef={controlsRef}
              displayMode={displayStyle}
            />
          ) : featureEdgesGeometry ? (
            <lineSegments geometry={featureEdgesGeometry} key={`wireframe-edges-${showHiddenEdges}`}>
              <lineBasicMaterial color="#000000" toneMapped={false} depthTest={!showHiddenEdges} depthWrite={false} />
            </lineSegments>
          ) : (
            <lineSegments geometry={cleanEdgesGeometry} key={`wireframe-fallback-${showHiddenEdges}`}>
              <lineBasicMaterial color="#000000" toneMapped={false} depthTest={!showHiddenEdges} depthWrite={false} />
            </lineSegments>
          ))}
      </group>
    );
  },
);
