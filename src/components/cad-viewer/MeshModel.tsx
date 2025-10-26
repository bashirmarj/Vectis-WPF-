import { useMemo, useEffect, useRef, forwardRef } from "react";
import * as THREE from "three";
import { useThree, useFrame } from "@react-three/fiber";

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
}

interface MeshModelProps {
  meshData: MeshData;
  sectionPlane: "none" | "xy" | "xz" | "yz";
  sectionPosition: number;
  showEdges: boolean;
  showHiddenEdges?: boolean;
  displayStyle?: "solid" | "wireframe" | "translucent";
  topologyColors?: #FF6B6B;
}

// Professional solid color for CAD rendering
const SOLID_COLOR = "#CCCCCC";

// Fusion 360 Analysis colors
const TOPOLOGY_COLORS = {
  internal: "#FF6B6B",
  cylindrical: "#FF6B6B",
  planar: "#FF6B6B",
  external: "#FF6B6B",
  through: "#FF6B6B",
  default: "#FF6B6B",
};

export const MeshModel = forwardRef<THREE.Mesh, MeshModelProps>(
  (
    {
      meshData,
      sectionPlane,
      sectionPosition,
      showEdges,
      showHiddenEdges = false,
      displayStyle = "solid",
      topologyColors = true, // RESTORED: Default to true as per system design
    },
    ref,
  ) => {
    const { camera } = useThree();
    const internalMeshRef = useRef<THREE.Mesh>(null);
    const meshRef = (ref as React.RefObject<THREE.Mesh>) || internalMeshRef;
    const dynamicEdgesRef = useRef<THREE.Group>(null);
    const wireframeEdgesRef = useRef<THREE.Group>(null);

    // ✅ FIXED: Create indexed geometry with smooth normals for ALL cases
    const geometry = useMemo(() => {
      const geo = new THREE.BufferGeometry();

      // Always use indexed geometry with backend smooth normals
      geo.setAttribute("position", new THREE.Float32BufferAttribute(meshData.vertices, 3));
      geo.setIndex(meshData.indices);
      geo.setAttribute("normal", new THREE.Float32BufferAttribute(meshData.normals, 3));

      // ✅ CRITICAL: DO NOT call computeVertexNormals() - trust backend normals!
      // The backend generates professional smooth normals that should not be overwritten

      geo.computeBoundingSphere();
      return geo;
    }, [meshData]);

    // ✅ FIXED: Apply vertex colors without changing geometry structure
    useEffect(() => {
      if (!geometry) return;

      if (topologyColors && meshData.vertex_colors && meshData.vertex_colors.length > 0) {
        // Create color attribute for indexed geometry
        const numVertices = meshData.vertices.length / 3;
        const colors = new Float32Array(numVertices * 3);

        for (let i = 0; i < numVertices; i++) {
          const colorStr = meshData.vertex_colors[i] || TOPOLOGY_COLORS.default;
          const color = new THREE.Color(colorStr);
          colors[i * 3] = color.r;
          colors[i * 3 + 1] = color.g;
          colors[i * 3 + 2] = color.b;
        }

        geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      } else {
        geometry.deleteAttribute("color");
      }
    }, [geometry, meshData, topologyColors]);

    // Dynamic edge rendering
    useFrame(() => {
      if (!meshRef.current || !dynamicEdgesRef.current) return;
      if (displayStyle === "wireframe") return;

      const mesh = meshRef.current;
      dynamicEdgesRef.current.clear();

      if (showEdges && meshData.feature_edges) {
        const cameraPosition = camera.position;
        const cameraDir = new THREE.Vector3();
        camera.getWorldDirection(cameraDir);

        for (const edge of meshData.feature_edges) {
          if (!Array.isArray(edge) || edge.length !== 2) continue;

          const [p1_arr, p2_arr] = edge;
          if (!Array.isArray(p1_arr) || !Array.isArray(p2_arr)) continue;
          if (p1_arr.length !== 3 || p2_arr.length !== 3) continue;

          const p1 = new THREE.Vector3(p1_arr[0], p1_arr[1], p1_arr[2]);
          const p2 = new THREE.Vector3(p2_arr[0], p2_arr[1], p2_arr[2]);

          const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
          const toCamera = new THREE.Vector3().subVectors(cameraPosition, midpoint).normalize();

          const edgeDir = new THREE.Vector3().subVectors(p2, p1).normalize();
          const perpendicular = new THREE.Vector3().crossVectors(edgeDir, cameraDir).normalize();

          const dotProduct = perpendicular.dot(toCamera);

          if (dotProduct > 0.1) {
            const positions = new Float32Array([p1.x, p1.y, p1.z, p2.x, p2.y, p2.z]);
            const edgeGeo = new THREE.BufferGeometry();
            edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));

            const edgeMat = new THREE.LineBasicMaterial({
              color: "#000000",
              linewidth: 2,
              toneMapped: false,
            });

            dynamicEdgesRef.current.add(new THREE.LineSegments(edgeGeo, edgeMat));
          }
        }
      }
    });

    // Wireframe edge rendering
    useEffect(() => {
      if (!wireframeEdgesRef.current) return;
      if (displayStyle !== "wireframe") return;

      wireframeEdgesRef.current.clear();

      if (meshData.feature_edges) {
        const visibleEdges: number[] = [];
        const hiddenEdges: number[] = [];

        for (const edge of meshData.feature_edges) {
          if (!Array.isArray(edge) || edge.length !== 2) continue;

          const [p1_arr, p2_arr] = edge;
          if (!Array.isArray(p1_arr) || !Array.isArray(p2_arr)) continue;
          if (p1_arr.length !== 3 || p2_arr.length !== 3) continue;

          visibleEdges.push(p1_arr[0], p1_arr[1], p1_arr[2], p2_arr[0], p2_arr[1], p2_arr[2]);
        }

        if (visibleEdges.length > 0) {
          const geo = new THREE.BufferGeometry();
          geo.setAttribute("position", new THREE.Float32BufferAttribute(visibleEdges, 3));
          const mat = new THREE.LineBasicMaterial({
            color: "#000000",
            linewidth: 2,
            toneMapped: false,
          });
          wireframeEdgesRef.current.add(new THREE.LineSegments(geo, mat));
        }

        if (showHiddenEdges && hiddenEdges.length > 0) {
          const geo = new THREE.BufferGeometry();
          geo.setAttribute("position", new THREE.Float32BufferAttribute(hiddenEdges, 3));
          const mat = new THREE.LineDashedMaterial({
            color: "#666666",
            linewidth: 1,
            dashSize: 3,
            gapSize: 2,
            toneMapped: false,
          });
          const lines = new THREE.LineSegments(geo, mat);
          lines.computeLineDistances();
          wireframeEdgesRef.current.add(lines);
        }
      }
    });

    // Section plane
    const clippingPlane = useMemo(() => {
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
    }, [sectionPlane, sectionPosition]);

    const { gl } = useThree();

    useEffect(() => {
      gl.localClippingEnabled = sectionPlane !== "none";
      gl.clippingPlanes = [];
    }, [sectionPlane, gl]);

    // Material properties with conditional color
    const materialProps = useMemo(() => {
      const base = {
        side: THREE.DoubleSide,
        clippingPlanes: clippingPlane,
        clipIntersection: false,
        metalness: 0.15,
        roughness: 0.6,
        envMapIntensity: 0.5,
        toneMapped: false,
        castShadow: true,
        receiveShadow: true,
      };

      if (displayStyle === "wireframe") {
        return { ...base, opacity: 0, transparent: true, wireframe: false };
      } else if (displayStyle === "translucent") {
        return { ...base, transparent: true, opacity: 0.4, wireframe: false };
      }

      return { ...base, transparent: false, opacity: 1, wireframe: false };
    }, [displayStyle, clippingPlane]);

    return (
      <group>
        {/* Mesh surface with smooth shading */}
        <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
          <meshStandardMaterial
            {...materialProps}
            color={topologyColors ? "#ffffff" : SOLID_COLOR}
            vertexColors={topologyColors}
            flatShading={false} // ✅ ALWAYS smooth shading - backend normals are smooth
            toneMapped={false}
          />
        </mesh>

        {/* Dynamic edges for solid mode */}
        {displayStyle !== "wireframe" && <group ref={dynamicEdgesRef} />}

        {/* Clean wireframe edges */}
        {displayStyle === "wireframe" && <group ref={wireframeEdgesRef} />}
      </group>
    );
  },
);
