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

// Professional solid color for CAD rendering
const SOLID_COLOR = "#CCCCCC"; // Light gray

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
      showHiddenEdges = true,
      displayStyle = "solid",
      topologyColors = true,
    },
    ref,
  ) => {
    const { camera } = useThree();
    const internalMeshRef = useRef<THREE.Mesh>(null);
    const meshRef = (ref as React.RefObject<THREE.Mesh>) || internalMeshRef;
    const dynamicEdgesRef = useRef<THREE.Group>(null);
    const wireframeEdgesRef = useRef<THREE.Group>(null);

    // Create single unified geometry for professional solid rendering
    const geometry = useMemo(() => {
      const geo = new THREE.BufferGeometry();

      if (!topologyColors) {
        geo.setAttribute("position", new THREE.Float32BufferAttribute(meshData.vertices, 3));
        geo.setIndex(meshData.indices);
        geo.setAttribute("normal", new THREE.Float32BufferAttribute(meshData.normals, 3));
        geo.computeVertexNormals();
        geo.normalizeNormals();
      } else {
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
      }

      geo.computeBoundingSphere();
      return geo;
    }, [meshData, topologyColors]);

    // Apply vertex colors
    useEffect(() => {
      if (!geometry) return;

      if (topologyColors) {
        if (meshData.vertex_colors && meshData.vertex_colors.length > 0) {
          const triangleCount = meshData.indices.length / 3;
          const colors = new Float32Array(triangleCount * 9);

          for (let triIdx = 0; triIdx < triangleCount; triIdx++) {
            const vertexIdx = meshData.indices[triIdx * 3];
            const faceType = meshData.vertex_colors[vertexIdx] || "default";
            const colorHex = TOPOLOGY_COLORS[faceType as keyof typeof TOPOLOGY_COLORS] || TOPOLOGY_COLORS.default;
            const color = new THREE.Color(colorHex);

            for (let v = 0; v < 3; v++) {
              colors[triIdx * 9 + v * 3 + 0] = color.r;
              colors[triIdx * 9 + v * 3 + 1] = color.g;
              colors[triIdx * 9 + v * 3 + 2] = color.b;
            }
          }

          geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geometry.attributes.color.needsUpdate = true;
        } else {
          const triangleCount = meshData.indices.length / 3;
          const colors = new Float32Array(triangleCount * 9);
          const silverColor = new THREE.Color("#CCCCCC");

          for (let i = 0; i < triangleCount * 3; i++) {
            colors[i * 3] = silverColor.r;
            colors[i * 3 + 1] = silverColor.g;
            colors[i * 3 + 2] = silverColor.b;
          }

          geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geometry.attributes.color.needsUpdate = true;
        }
      } else {
        if (geometry.attributes.color) {
          geometry.deleteAttribute("color");
        }
      }
    }, [geometry, topologyColors, meshData]);

    // Pre-compute edge connectivity for ALL edges (used by both modes)
    const edgeMap = useMemo(() => {
      const map = new Map<
        string,
        {
          v1: THREE.Vector3;
          v2: THREE.Vector3;
          normals: THREE.Vector3[];
        }
      >();

      const triangleCount = meshData.indices.length / 3;

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
          if (!map.has(edge.key)) {
            map.set(edge.key, {
              v1: edge.v1.clone(),
              v2: edge.v2.clone(),
              normals: [normal.clone()],
            });
          } else {
            map.get(edge.key)!.normals.push(normal.clone());
          }
        });
      }

      return map;
    }, [meshData.vertices, meshData.indices]);

    // Dynamic edge rendering for BOTH solid and wireframe modes
    useFrame(() => {
      if (!edgeMap || !meshRef.current) return;

      const mesh = meshRef.current;
      const cameraWorldPos = new THREE.Vector3();
      camera.getWorldPosition(cameraWorldPos);

      const worldToLocal = new THREE.Matrix4().copy(mesh.matrixWorld).invert();
      const cameraLocalPos = cameraWorldPos.clone().applyMatrix4(worldToLocal);

      const visibleEdges: number[] = [];
      const hiddenEdges: number[] = [];

      // Check each edge for visibility
      edgeMap.forEach((edgeData) => {
        const v1World = edgeData.v1.clone().applyMatrix4(mesh.matrixWorld);
        const v2World = edgeData.v2.clone().applyMatrix4(mesh.matrixWorld);

        // Boundary edges (only 1 face) - always important
        if (edgeData.normals.length === 1) {
          const n = edgeData.normals[0];
          const edgeMidpoint = new THREE.Vector3().addVectors(edgeData.v1, edgeData.v2).multiplyScalar(0.5);
          const viewDir = new THREE.Vector3().subVectors(cameraLocalPos, edgeMidpoint).normalize();

          if (n.dot(viewDir) > 0) {
            visibleEdges.push(v1World.x, v1World.y, v1World.z, v2World.x, v2World.y, v2World.z);
          } else if (showHiddenEdges) {
            hiddenEdges.push(v1World.x, v1World.y, v1World.z, v2World.x, v2World.y, v2World.z);
          }
          return;
        }

        // Silhouette edges (2 adjacent faces with opposite facing)
        if (edgeData.normals.length === 2) {
          const n1 = edgeData.normals[0];
          const n2 = edgeData.normals[1];

          const edgeMidpoint = new THREE.Vector3().addVectors(edgeData.v1, edgeData.v2).multiplyScalar(0.5);
          const viewDir = new THREE.Vector3().subVectors(cameraLocalPos, edgeMidpoint).normalize();

          const dot1 = n1.dot(viewDir);
          const dot2 = n2.dot(viewDir);

          // Silhouette: one face visible, one hidden
          if ((dot1 > 0.01 && dot2 < -0.01) || (dot1 < -0.01 && dot2 > 0.01)) {
            visibleEdges.push(v1World.x, v1World.y, v1World.z, v2World.x, v2World.y, v2World.z);
          }
        }
      });

      // Update visible edges (solid lines)
      if (displayStyle === "solid" && showEdges && dynamicEdgesRef.current) {
        // Clear existing
        while (dynamicEdgesRef.current.children.length > 0) {
          const child = dynamicEdgesRef.current.children[0];
          dynamicEdgesRef.current.remove(child);
          if (child instanceof THREE.LineSegments) {
            child.geometry.dispose();
            (child.material as THREE.Material).dispose();
          }
        }

        if (visibleEdges.length > 0) {
          const geo = new THREE.BufferGeometry();
          geo.setAttribute("position", new THREE.Float32BufferAttribute(visibleEdges, 3));
          const mat = new THREE.LineBasicMaterial({
            color: "#000000",
            linewidth: 1.5,
            toneMapped: false,
            depthTest: true,
            depthWrite: false,
          });
          dynamicEdgesRef.current.add(new THREE.LineSegments(geo, mat));
        }
      }

      // Update wireframe edges (visible + hidden)
      if (displayStyle === "wireframe" && wireframeEdgesRef.current) {
        // Clear existing
        while (wireframeEdgesRef.current.children.length > 0) {
          const child = wireframeEdgesRef.current.children[0];
          wireframeEdgesRef.current.remove(child);
          if (child instanceof THREE.LineSegments) {
            child.geometry.dispose();
            (child.material as THREE.Material).dispose();
          }
        }

        // Visible edges - solid lines
        if (visibleEdges.length > 0) {
          const geo = new THREE.BufferGeometry();
          geo.setAttribute("position", new THREE.Float32BufferAttribute(visibleEdges, 3));
          const mat = new THREE.LineBasicMaterial({
            color: "#000000",
            linewidth: 1.5,
            toneMapped: false,
          });
          wireframeEdgesRef.current.add(new THREE.LineSegments(geo, mat));
        }

        // Hidden edges - dashed lines
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
          lines.computeLineDistances(); // Required for dashed lines
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

    const materialProps = useMemo(() => {
      const base = {
        color: "#5b9bd5",
        side: THREE.DoubleSide,
        clippingPlanes: clippingPlane,
        clipIntersection: true,
        metalness: 0,
        roughness: 0.8,
        envMapIntensity: 0,
      };

      if (displayStyle === "wireframe") {
        // In wireframe mode, hide the mesh surface completely
        return { ...base, opacity: 0, transparent: true, wireframe: false };
      } else if (displayStyle === "translucent") {
        return { ...base, transparent: true, opacity: 0.4, wireframe: false };
      }

      return { ...base, transparent: false, opacity: 1, wireframe: false };
    }, [displayStyle, clippingPlane]);

    return (
      <group>
        {/* Mesh surface (hidden in wireframe mode) */}
        <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
          <meshStandardMaterial
            {...materialProps}
            color={topologyColors ? "#ffffff" : SOLID_COLOR}
            vertexColors={topologyColors}
            flatShading={topologyColors}
            toneMapped={false}
          />
        </mesh>

        {/* Dynamic edges for solid mode */}
        {displayStyle !== "wireframe" && <group ref={dynamicEdgesRef} />}

        {/* Clean wireframe edges (visible + hidden dashed) */}
        {displayStyle === "wireframe" && <group ref={wireframeEdgesRef} />}
      </group>
    );
  },
);
