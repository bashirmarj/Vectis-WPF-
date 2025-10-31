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

    // Create single unified geometry - ALWAYS USE INDEXED GEOMETRY
    // This is critical for smooth shading and vertex normal sharing
    const geometry = useMemo(() => {
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
    }, [meshData]);

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

    // Pre-compute static feature edges for solid mode (calculated once with higher threshold)
    const staticFeatureEdges = useMemo(() => {
      if (!edgeMap) return [];
      
      const edges: number[] = [];
      const FEATURE_ANGLE_THRESHOLD = 50; // Higher threshold - only genuine sharp edges

      edgeMap.forEach((edgeData) => {
        // Only process internal edges (2 adjacent faces)
        if (edgeData.normals.length === 2) {
          const n1 = edgeData.normals[0];
          const n2 = edgeData.normals[1];
          
          // Calculate angle between normals
          const normalAngle = Math.acos(Math.max(-1, Math.min(1, n1.dot(n2))));
          const normalAngleDeg = normalAngle * (180 / Math.PI);
          
          // Only mark as feature edge if angle is significant (sharp edge)
          if (normalAngleDeg > FEATURE_ANGLE_THRESHOLD) {
            edges.push(
              edgeData.v1.x, edgeData.v1.y, edgeData.v1.z,
              edgeData.v2.x, edgeData.v2.y, edgeData.v2.z
            );
          }
        }
      });

      return edges;
    }, [edgeMap]);

    // Dynamic edge rendering for wireframe mode only (silhouette + feature edges)
    // Solid mode uses static feature edges computed once in useMemo
    useFrame(() => {
      if (!edgeMap || !meshRef.current) return;
      
      // Skip dynamic edge calculation for solid mode - use static edges only
      if (displayStyle === "solid") {
        // Update static feature edges for solid mode
        if (showEdges && dynamicEdgesRef.current) {
          // Clear existing
          while (dynamicEdgesRef.current.children.length > 0) {
            const child = dynamicEdgesRef.current.children[0];
            dynamicEdgesRef.current.remove(child);
            if (child instanceof THREE.LineSegments) {
              child.geometry.dispose();
              (child.material as THREE.Material).dispose();
            }
          }

          // Render static feature edges in world space
          if (staticFeatureEdges.length > 0) {
            const mesh = meshRef.current;
            const worldEdges: number[] = [];
            
            // Transform static edges to world space
            for (let i = 0; i < staticFeatureEdges.length; i += 6) {
              const v1 = new THREE.Vector3(
                staticFeatureEdges[i],
                staticFeatureEdges[i + 1],
                staticFeatureEdges[i + 2]
              ).applyMatrix4(mesh.matrixWorld);
              
              const v2 = new THREE.Vector3(
                staticFeatureEdges[i + 3],
                staticFeatureEdges[i + 4],
                staticFeatureEdges[i + 5]
              ).applyMatrix4(mesh.matrixWorld);
              
              worldEdges.push(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
            }

            if (worldEdges.length > 0) {
              const geo = new THREE.BufferGeometry();
              geo.setAttribute("position", new THREE.Float32BufferAttribute(worldEdges, 3));
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
        }
        return;
      }

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

          // ✅ ENHANCED: Silhouette detection with relaxed threshold for circular surfaces
          // Original: (dot1 > 0.01 && dot2 < -0.01) - too strict for cylinders/circles
          // New: Detect silhouette OR feature edges (sharp angle between normals)

          // Calculate angle between normals
          const normalAngle = Math.acos(Math.max(-1, Math.min(1, n1.dot(n2))));
          const normalAngleDeg = normalAngle * (180 / Math.PI);

          // Feature edge: sharp angle between adjacent faces (> 20 degrees)
          const isFeatureEdge = normalAngleDeg > 20;

          // Silhouette edge: one face visible, one hidden (relaxed threshold)
          const isSilhouette = (dot1 > -0.1 && dot2 < 0.1) || (dot1 < 0.1 && dot2 > -0.1);

          if (isSilhouette || isFeatureEdge) {
            // Show visible edges only if at least one face is somewhat visible
            if (dot1 > -0.3 || dot2 > -0.3) {
              visibleEdges.push(v1World.x, v1World.y, v1World.z, v2World.x, v2World.y, v2World.z);
            }
          }
        }
      });

      // Update wireframe edges (visible + hidden) - dynamic silhouette detection
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
        metalness: 0.0,
        roughness: 0.9,
        envMapIntensity: 0.3,
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
        {/* Mesh surface - DISABLED castShadow to prevent self-shadowing artifacts */}
        <mesh ref={meshRef} geometry={geometry} castShadow={false} receiveShadow>
          <meshStandardMaterial
            {...materialProps}
            color={topologyColors ? "#ffffff" : SOLID_COLOR}
            vertexColors={topologyColors}
            flatShading={false}
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
