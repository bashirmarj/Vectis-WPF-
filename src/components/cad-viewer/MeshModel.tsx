import { useEffect, useRef, forwardRef } from "react";
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
  tagged_edges?: Array<{
    feature_id: number;
    start: [number, number, number];
    end: [number, number, number];
    type: string;
    iso_type?: string;
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
    },
    ref,
  ) => {
    const internalMeshRef = useRef<THREE.Mesh>(null);
    const meshRef = internalMeshRef;

    // Create single unified geometry - ALWAYS USE INDEXED GEOMETRY
    // This is critical for smooth shading and vertex normal sharing
    const geometry = (() => {
      const geo = new THREE.BufferGeometry();

      // ALWAYS use indexed geometry (vertices shared across triangles)
      geo.setAttribute("position", new THREE.Float32BufferAttribute(meshData.vertices, 3));
      geo.setIndex(meshData.indices);
      geo.setAttribute("normal", new THREE.Float32BufferAttribute(meshData.normals, 3));

      // ✅ CRITICAL: DO NOT call computeVertexNormals() - trust backend normals!
      // The backend generates professional smooth normals that should not be overwritten
      // Calling computeVertexNormals() here destroys the smooth normals and creates visible horizontal lines

      geo.computeBoundingSphere();
      return geo;
    })();

    // Apply vertex colors to indexed geometry
    useEffect(() => {
      if (!geometry) return;

      if (topologyColors) {
        if (meshData.vertex_colors && meshData.vertex_colors.length > 0) {
          // Apply colors to vertices (not triangles)
          const vertexCount = meshData.vertices.length / 3;
          const colors = new Float32Array(vertexCount * 3);

          for (let vertexIdx = 0; vertexIdx < vertexCount; vertexIdx++) {
            const faceType = meshData.vertex_colors[vertexIdx] || "default";
            const colorHex = TOPOLOGY_COLORS[faceType as keyof typeof TOPOLOGY_COLORS] || TOPOLOGY_COLORS.default;
            const color = new THREE.Color(colorHex);

            colors[vertexIdx * 3 + 0] = color.r;
            colors[vertexIdx * 3 + 1] = color.g;
            colors[vertexIdx * 3 + 2] = color.b;
          }

          geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geometry.attributes.color.needsUpdate = true;
        } else {
          // No colors provided, use solid color
          const vertexCount = meshData.vertices.length / 3;
          const colors = new Float32Array(vertexCount * 3);
          const solidColor = new THREE.Color(SOLID_COLOR);

          for (let i = 0; i < vertexCount; i++) {
            colors[i * 3] = solidColor.r;
            colors[i * 3 + 1] = solidColor.g;
            colors[i * 3 + 2] = solidColor.b;
          }

          geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geometry.attributes.color.needsUpdate = true;
        }
      } else {
        if (geometry.attributes.color) {
          geometry.deleteAttribute("color");
        }
      }
    }, [topologyColors]);

    // Pre-compute feature edges (for solid mode) - NO CACHING
    const featureEdgesGeometry = (() => {
      // PRIORITY: Use tagged_edges with iso_type filtering to remove UIso/VIso construction lines
      if (meshData.tagged_edges && meshData.tagged_edges.length > 0) {
        const featureEdgePositions: number[] = [];

        meshData.tagged_edges.forEach((edge) => {
          // FILTER: Skip UIso/VIso parametric surface curves (interior construction lines)
          if (edge.iso_type === "uiso" || edge.iso_type === "viso") {
            return;
          }

          // Render all other edges (boundary, feature edges, fillets, etc.)
          featureEdgePositions.push(edge.start[0], edge.start[1], edge.start[2], edge.end[0], edge.end[1], edge.end[2]);
        });

        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(featureEdgePositions, 3));
        geo.computeBoundingSphere();
        return geo;
      }

      // FALLBACK: Use feature_edges if tagged_edges not available
      if (meshData.feature_edges && meshData.feature_edges.length > 0) {
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
        return geo;
      }

      // FALLBACK: Compute from mesh triangles (for STL files without BREP data)
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
      return geo;
    })();

    // Clean edge geometry for both solid and wireframe modes - NO CACHING
    const cleanEdgesGeometry = new THREE.EdgesGeometry(geometry, 1);

    // Section plane - NO CACHING
    const clippingPlane = (() => {
      if (sectionPlane === "none") return undefined;

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
          return undefined;
      }

      return [new THREE.Plane(normal, -sectionPosition)];
    })();

    const { gl } = useThree();

    useEffect(() => {
      gl.localClippingEnabled = sectionPlane !== "none";
      gl.clippingPlanes = [];
    }, [sectionPlane, gl]);

    const materialProps = (() => {
      const base = {
        color: SOLID_COLOR,
        side: THREE.DoubleSide,
        clippingPlanes: clippingPlane,
        clipIntersection: true,
        metalness: 0.1,
        roughness: 0.3,
        envMapIntensity: 0.5,
      };

      if (displayStyle === "wireframe") {
        return { ...base, opacity: 0, transparent: true, wireframe: false };
      } else if (displayStyle === "translucent") {
        return { ...base, transparent: true, opacity: 0.4, wireframe: false };
      }

      return { ...base, transparent: false, opacity: 1, wireframe: false };
    })();

    // Cleanup geometries on unmount to prevent memory leaks
    useEffect(() => {
      return () => {
        geometry.dispose();
        if (featureEdgesGeometry) featureEdgesGeometry.dispose();
        cleanEdgesGeometry.dispose();
      };
    }, []);

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
            vertexColors={topologyColors}
            flatShading={false}
            toneMapped={false}
            metalness={0.1}
            roughness={0.6}
          />
        </mesh>

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

        {/* Show only backend feature edges in solid mode (no tessellation lines) */}
        {displayStyle === "solid" && showEdges && featureEdgesGeometry && (
          <lineSegments geometry={featureEdgesGeometry} frustumCulled={true}>
            <lineBasicMaterial
              color="#000000"
              toneMapped={false}
              polygonOffset={true}
              polygonOffsetFactor={-10}
              polygonOffsetUnits={-10}
              depthTest={true}
              depthWrite={false}
            />
          </lineSegments>
        )}

        {/* Wireframe mode - use clean edges that show ALL mesh structure */}
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
          ) : (
            <lineSegments geometry={cleanEdgesGeometry} key={`wireframe-edges-${showHiddenEdges}`}>
              <lineBasicMaterial color="#000000" toneMapped={false} depthTest={!showHiddenEdges} depthWrite={false} />
            </lineSegments>
          ))}
      </group>
    );
  },
);
