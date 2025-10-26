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
  topologyColors?: boolean;
}

// Professional solid color for CAD rendering (FIXED: Changed from blue to gray)
const SOLID_COLOR = "#CCCCCC"; // Light gray - industry standard

// Fusion 360 Analysis colors (keeping existing topology colors)
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
      topologyColors = false, // CHANGED: Default to false for professional smooth rendering
    },
    ref,
  ) => {
    const { camera } = useThree();
    const internalMeshRef = useRef<THREE.Mesh>(null);
    const meshRef = (ref as React.RefObject<THREE.Mesh>) || internalMeshRef;
    const dynamicEdgesRef = useRef<THREE.Group>(null);
    const wireframeEdgesRef = useRef<THREE.Group>(null);

    // Create geometry with proper normal handling
    const geometry = useMemo(() => {
      const geo = new THREE.BufferGeometry();

      if (!topologyColors) {
        // ✅ FIXED: Trust backend normals for smooth rendering
        geo.setAttribute("position", new THREE.Float32BufferAttribute(meshData.vertices, 3));
        geo.setIndex(meshData.indices);
        geo.setAttribute("normal", new THREE.Float32BufferAttribute(meshData.normals, 3));

        // ❌ REMOVED: geo.computeVertexNormals() - This was overwriting backend smooth normals!
        // ❌ REMOVED: geo.normalizeNormals() - Backend normals are already normalized

        // Backend normals are already perfect - use them directly!
      } else {
        // Topology colors mode: Create non-indexed geometry for flat shading
        const triangleCount = meshData.indices.length / 3;
        const positions = new Float32Array(triangleCount * 9);

        for (let i = 0; i < triangleCount; i++) {
          const idx0 = meshData.indices[i * 3];
          const idx1 = meshData.indices[i * 3 + 1];
          const idx2 = meshData.indices[i * 3 + 2];

          positions[i * 9 + 0] = meshData.vertices[idx0 * 3];
          positions[i * 9 + 1] = meshData.vertices[idx0 * 3 + 1];
          positions[i * 9 + 2] = meshData.vertices[idx0 * 3 + 2];

          positions[i * 9 + 3] = meshData.vertices[idx1 * 3];
          positions[i * 9 + 4] = meshData.vertices[idx1 * 3 + 1];
          positions[i * 9 + 5] = meshData.vertices[idx1 * 3 + 2];

          positions[i * 9 + 6] = meshData.vertices[idx2 * 3];
          positions[i * 9 + 7] = meshData.vertices[idx2 * 3 + 1];
          positions[i * 9 + 8] = meshData.vertices[idx2 * 3 + 2];
        }

        geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        // For topology colors, let Three.js compute flat normals
        geo.computeVertexNormals();
      }

      geo.computeBoundingSphere();
      return geo;
    }, [meshData, topologyColors]);

    // Apply vertex colors (keeping existing logic)
    useEffect(() => {
      if (!geometry) return;

      if (topologyColors) {
        if (meshData.vertex_colors && meshData.vertex_colors.length > 0) {
          const triangleCount = meshData.indices.length / 3;
          const colors = new Float32Array(triangleCount * 9);

          for (let i = 0; i < triangleCount; i++) {
            const idx0 = meshData.indices[i * 3];
            const idx1 = meshData.indices[i * 3 + 1];
            const idx2 = meshData.indices[i * 3 + 2];

            const getColor = (idx: number) => {
              const colorStr = meshData.vertex_colors?.[idx] || TOPOLOGY_COLORS.default;
              const color = new THREE.Color(colorStr);
              return [color.r, color.g, color.b];
            };

            const color0 = getColor(idx0);
            const color1 = getColor(idx1);
            const color2 = getColor(idx2);

            colors[i * 9 + 0] = color0[0];
            colors[i * 9 + 1] = color0[1];
            colors[i * 9 + 2] = color0[2];

            colors[i * 9 + 3] = color1[0];
            colors[i * 9 + 4] = color1[1];
            colors[i * 9 + 5] = color1[2];

            colors[i * 9 + 6] = color2[0];
            colors[i * 9 + 7] = color2[1];
            colors[i * 9 + 8] = color2[2];
          }

          geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        }
      } else {
        geometry.deleteAttribute("color");
      }
    }, [geometry, meshData, topologyColors]);

    // Dynamic edge rendering for solid/translucent modes
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

    // Section plane (keeping existing logic)
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

    // ✅ FIXED: Professional material properties
    const materialProps = useMemo(() => {
      const base = {
        color: SOLID_COLOR, // ✅ Professional gray (not blue!)
        side: THREE.DoubleSide,
        clippingPlanes: clippingPlane,
        clipIntersection: false,
        metalness: 0.15, // ✅ Subtle metallic look (machined aluminum)
        roughness: 0.6, // ✅ Semi-gloss finish (not too matte)
        envMapIntensity: 0.5, // ✅ Environment reflections enabled
        toneMapped: false, // ✅ Preserve color accuracy
        castShadow: true, // ✅ Cast shadows
        receiveShadow: true, // ✅ Receive shadows
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
        {/* Mesh surface */}
        <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
          <meshStandardMaterial
            {...materialProps}
            color={topologyColors ? "#ffffff" : SOLID_COLOR}
            vertexColors={topologyColors}
            flatShading={false} // ✅ CRITICAL FIX: Always use smooth shading for proper normal rendering
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
