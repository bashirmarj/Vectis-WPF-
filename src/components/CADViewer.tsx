// CADViewer.tsx
// ✅ FIXED VERSION - October 28, 2025
// ✅ Arrow rotation dependency fixed: [] instead of [boundingBox]

import { Canvas } from "@react-three/fiber";
import { TrackballControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense, useMemo, useEffect, useState, useRef, useCallback } from "react";
import { CardContent } from "@/components/ui/card";
import { Loader2, Box } from "lucide-react";
import { Button } from "@/components/ui/button";
import * as THREE from "three";
import { supabase } from "@/integrations/supabase/client";
import { MeshModel } from "./cad-viewer/MeshModel";
import { DimensionAnnotations } from "./cad-viewer/DimensionAnnotations";
import { OrientationCubeViewport } from "./cad-viewer/OrientationCubeViewport";
import { ProfessionalLighting } from "./cad-viewer/enhancements/ProfessionalLighting";
import { UnifiedCADToolbar } from "./cad-viewer/UnifiedCADToolbar";
import { useMeasurementStore } from "@/stores/measurementStore";

interface CADViewerProps {
  meshId?: string;
  fileUrl?: string;
  fileName?: string;
  onMeshLoaded?: (data: MeshData) => void;
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  face_types?: string[];
  feature_edges?: number[][][];
}

export function CADViewer({ meshId, fileUrl, fileName, onMeshLoaded }: CADViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [meshData, setMeshData] = useState<MeshData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<"solid" | "wireframe" | "translucent">("solid");
  const [showEdges, setShowEdges] = useState(true);
  const [shadowsEnabled, setShadowsEnabled] = useState(true);
  const [ssaoEnabled, setSSAOEnabled] = useState(false);
  const [sectionPlane, setSectionPlane] = useState<"xy" | "xz" | "yz" | null>(null);
  const [sectionPosition, setSectionPosition] = useState(0);

  const { activeTool, setActiveTool, clearAllMeasurements, measurements } = useMeasurementStore();
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<any>(null);
  const meshRef = useRef<THREE.Mesh>(null);

  // File extension detection
  const fileExtension = useMemo(() => {
    if (fileName) return fileName.split(".").pop()?.toLowerCase() || "";
    if (fileUrl) return fileUrl.split(".").pop()?.toLowerCase().split("?")[0] || "";
    return "";
  }, [fileName, fileUrl]);

  const isRenderableFormat = useMemo(() => {
    return ["step", "stp", "iges", "igs", "stl"].includes(fileExtension);
  }, [fileExtension]);

  // Load mesh data from Supabase
  useEffect(() => {
    if (!meshId || !isRenderableFormat) {
      setIsLoading(false);
      return;
    }

    const loadMeshData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const { data: mesh, error: meshError } = await supabase
          .from("cad_meshes")
          .select("*")
          .eq("id", meshId)
          .single();

        if (meshError) throw meshError;
        if (!mesh) throw new Error("Mesh not found");

        const vertices = Array.isArray(mesh.vertices) ? mesh.vertices : JSON.parse(mesh.vertices as string);
        const indices = Array.isArray(mesh.indices) ? mesh.indices : JSON.parse(mesh.indices as string);
        const normals = Array.isArray(mesh.normals) ? mesh.normals : JSON.parse(mesh.normals as string);
        const vertex_colors = mesh.vertex_colors
          ? Array.isArray(mesh.vertex_colors)
            ? mesh.vertex_colors
            : JSON.parse(mesh.vertex_colors as string)
          : undefined;
        const feature_edges = mesh.feature_edges
          ? Array.isArray(mesh.feature_edges)
            ? mesh.feature_edges
            : JSON.parse(mesh.feature_edges as string)
          : undefined;

        const loadedMeshData: MeshData = {
          vertices,
          indices,
          normals,
          vertex_colors,
          triangle_count: indices.length / 3,
          feature_edges,
        };

        setMeshData(loadedMeshData);
        onMeshLoaded?.(loadedMeshData);
        setIsLoading(false);
      } catch (error) {
        console.error("Error loading mesh:", error);
        setError(error instanceof Error ? error.message : "Failed to load mesh");
        setIsLoading(false);
      }
    };

    loadMeshData();
  }, [meshId, isRenderableFormat, onMeshLoaded]);

  // Calculate bounding box
  const boundingBox = useMemo(() => {
    if (!meshData) {
      return {
        min: new THREE.Vector3(-1, -1, -1),
        max: new THREE.Vector3(1, 1, 1),
        center: new THREE.Vector3(0, 0, 0),
        width: 2,
        height: 2,
        depth: 2,
      };
    }

    const vertices = meshData.vertices;
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const y = vertices[i + 1];
      const z = vertices[i + 2];

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }

    const center = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);

    return {
      min: new THREE.Vector3(minX, minY, minZ),
      max: new THREE.Vector3(maxX, maxY, maxZ),
      center,
      width: maxX - minX,
      height: maxY - minY,
      depth: maxZ - minZ,
    };
  }, [meshData]);

  // Calculate optimal camera position (isometric view)
  const initialCameraPosition = useMemo(() => {
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 2;
    const direction = new THREE.Vector3(1, 1, 1).normalize();

    return [
      boundingBox.center.x + direction.x * distance,
      boundingBox.center.y + direction.y * distance,
      boundingBox.center.z + direction.z * distance,
    ] as [number, number, number];
  }, [boundingBox]);

  // ✅ Ensure controls target is always set to model center
  useEffect(() => {
    if (controlsRef.current && boundingBox) {
      controlsRef.current.target.copy(boundingBox.center);
      controlsRef.current.update();
    }
  }, [boundingBox]);

  // Download handler
  const handleDownload = useCallback(() => {
    if (fileUrl) {
      const link = document.createElement("a");
      link.href = fileUrl;
      link.download = fileName || "download";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }, [fileUrl, fileName]);

  // Set predefined view
  const handleSetView = useCallback(
    (view: "front" | "back" | "left" | "right" | "top" | "bottom" | "isometric") => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = boundingBox.center;
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 2;

      let direction: THREE.Vector3;
      let up = new THREE.Vector3(0, 1, 0);

      switch (view) {
        case "front":
          direction = new THREE.Vector3(0, 0, 1);
          break;
        case "back":
          direction = new THREE.Vector3(0, 0, -1);
          break;
        case "left":
          direction = new THREE.Vector3(-1, 0, 0);
          break;
        case "right":
          direction = new THREE.Vector3(1, 0, 0);
          break;
        case "top":
          direction = new THREE.Vector3(0, 1, 0);
          up = new THREE.Vector3(0, 0, -1);
          break;
        case "bottom":
          direction = new THREE.Vector3(0, -1, 0);
          up = new THREE.Vector3(0, 0, 1);
          break;
        case "isometric":
          direction = new THREE.Vector3(1, 1, 1).normalize();
          break;
        default:
          return;
      }

      // ✅ ISSUE #3 FIXED: Use calculated direction instead of currentPosition
      const newPosition = target.clone().add(direction.multiplyScalar(distance));

      camera.position.copy(newPosition);
      camera.up.copy(up);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();
    },
    [boundingBox],
  );

  // ✅ CRITICAL FIX: Empty dependency array since function only uses refs
  const handleRotateCamera = useCallback(
    (direction: "up" | "down" | "left" | "right" | "cw" | "ccw") => {
      if (!cameraRef.current || !controlsRef.current) return;

      console.log("🎯 Rotating camera:", direction);

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = controls.target.clone();

      // Get current state
      const currentPosition = camera.position.clone();
      const currentUp = camera.up.clone();

      // Calculate view direction (from camera toward target)
      const viewDir = target.clone().sub(currentPosition).normalize();

      // 90-degree rotation
      const rotationAngle = Math.PI / 2;

      let newPosition: THREE.Vector3;
      let newUp: THREE.Vector3;

      switch (direction) {
        case "left":
        case "right": {
          // ✅ FIXED: Rotate around camera's current up vector (not world Y-axis)
          // This respects the camera's orientation after any previous rotations
          const rotDir = direction === "left" ? 1 : -1;

          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(currentUp, rotationAngle * rotDir);
          newPosition.add(target);

          // Up vector stays the same for left/right rotation
          newUp = currentUp.clone();
          break;
        }

        case "up":
        case "down": {
          // ✅ Use camera's own right vector (from quaternion)
          // This eliminates gimbal lock because we're using the camera's actual orientation
          const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
          const rotDir = direction === "up" ? 1 : -1;

          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(cameraRight, rotationAngle * rotDir);
          newPosition.add(target);
          newUp = currentUp.clone().applyAxisAngle(cameraRight, rotationAngle * rotDir);
          break;
        }

        case "cw":
        case "ccw": {
          // Roll around view direction
          const rotDir = direction === "cw" ? -1 : 1;
          newPosition = currentPosition.clone();
          newUp = currentUp.clone().applyAxisAngle(viewDir, rotationAngle * rotDir);
          break;
        }

        default:
          return;
      }

      // Apply rotation
      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      console.log("✅ Camera rotated successfully!");
      console.log("   New position:", {
        x: newPosition.x.toFixed(2),
        y: newPosition.y.toFixed(2),
        z: newPosition.z.toFixed(2),
      });
      console.log("   New up vector:", {
        x: newUp.x.toFixed(2),
        y: newUp.y.toFixed(2),
        z: newUp.z.toFixed(2),
      });
    },
    [], // ✅ CRITICAL FIX: Empty array - refs don't need dependencies
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key.toLowerCase()) {
        case " ":
          e.preventDefault();
          handleSetView("isometric");
          break;
        case "e":
          setShowEdges((prev) => !prev);
          break;
        case "m":
          setActiveTool(activeTool ? null : "distance");
          break;
        case "escape":
          setActiveTool(null);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [handleSetView, activeTool, setActiveTool]);

  return (
    <div className="w-full h-full relative">
      <CardContent className="p-0 h-full">
        {/* ✅ FIXED: OrientationCube outside conditional - stays mounted */}
        {meshData && isRenderableFormat && !isLoading && !error && (
          <OrientationCubeViewport
            mainCameraRef={cameraRef}
            controlsRef={controlsRef}
            onCubeClick={(direction) => {
              if (Math.abs(direction.x) > 0.5) {
                handleSetView(direction.x > 0 ? "right" : "left");
              } else if (Math.abs(direction.y) > 0.5) {
                handleSetView(direction.y > 0 ? "top" : "bottom");
              } else if (Math.abs(direction.z) > 0.5) {
                handleSetView(direction.z > 0 ? "front" : "back");
              }
            }}
            onRotateUp={() => handleRotateCamera("up")}
            onRotateDown={() => handleRotateCamera("down")}
            onRotateLeft={() => handleRotateCamera("left")}
            onRotateRight={() => handleRotateCamera("right")}
            onRotateClockwise={() => handleRotateCamera("cw")}
            onRotateCounterClockwise={() => handleRotateCamera("ccw")}
          />
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <Box className="h-16 w-16 text-destructive" />
            <div className="text-center space-y-2">
              <p className="text-sm font-medium text-destructive">Error loading 3D model</p>
              <p className="text-xs text-muted-foreground max-w-md">{error}</p>
            </div>
            {fileUrl && (
              <Button variant="outline" onClick={handleDownload}>
                Download Original File
              </Button>
            )}
          </div>
        ) : meshData && isRenderableFormat ? (
          <div className="relative w-full h-full">
            <UnifiedCADToolbar
              onHomeView={() => handleSetView("isometric")}
              onFrontView={() => handleSetView("front")}
              onTopView={() => handleSetView("top")}
              onIsometricView={() => handleSetView("isometric")}
              onFitView={() => handleSetView("isometric")}
              displayMode={displayMode}
              onDisplayModeChange={setDisplayMode}
              showEdges={showEdges}
              onToggleEdges={() => setShowEdges((prev) => !prev)}
              measurementMode={activeTool}
              onMeasurementModeChange={setActiveTool}
              measurementCount={measurements.length}
              onClearMeasurements={clearAllMeasurements}
              sectionPlane={sectionPlane}
              onSectionPlaneChange={setSectionPlane}
              sectionPosition={sectionPosition}
              onSectionPositionChange={setSectionPosition}
              shadowsEnabled={shadowsEnabled}
              onToggleShadows={() => setShadowsEnabled((prev) => !prev)}
              ssaoEnabled={ssaoEnabled}
              onToggleSSAO={() => setSSAOEnabled((prev) => !prev)}
              boundingBox={boundingBox}
            />

            <Canvas
              shadows
              gl={{
                antialias: true,
                alpha: true,
                preserveDrawingBuffer: true,
                powerPreference: "high-performance",
              }}
              onCreated={({ gl }) => {
                gl.domElement.addEventListener(
                  "webglcontextlost",
                  (e) => {
                    e.preventDefault();
                    console.warn("WebGL context lost");
                  },
                  false,
                );
                gl.domElement.addEventListener(
                  "webglcontextrestored",
                  () => {
                    console.log("WebGL context restored");
                  },
                  false,
                );
              }}
            >
              <color attach="background" args={["#f8f9fa"]} />

              <PerspectiveCamera ref={cameraRef} makeDefault position={initialCameraPosition} fov={50} />

              <Suspense fallback={null}>
                <ProfessionalLighting intensity={1.0} enableShadows={shadowsEnabled} shadowQuality="high" />

                <MeshModel
                  ref={meshRef}
                  meshData={meshData}
                  displayStyle={displayMode}
                  showEdges={showEdges}
                  sectionPlane={sectionPlane || "none"}
                  sectionPosition={sectionPosition}
                />

                <DimensionAnnotations boundingBox={boundingBox} />

                <TrackballControls
                  ref={controlsRef}
                  makeDefault
                  target={boundingBox.center}
                  dynamicDampingFactor={0.2}
                  minDistance={Math.max(boundingBox.width, boundingBox.height, boundingBox.depth) * 0.01}
                  maxDistance={Math.max(boundingBox.width, boundingBox.height, boundingBox.depth) * 5}
                  rotateSpeed={1.8}
                  panSpeed={0.8}
                  zoomSpeed={1.2}
                  staticMoving={false}
                  noPan={false}
                  noRotate={false}
                />
              </Suspense>
            </Canvas>
          </div>
        ) : isRenderableFormat ? (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <Box className="h-16 w-16 text-muted-foreground" />
            <p className="text-sm text-muted-foreground text-center">Upload a file to view 3D preview</p>
            <p className="text-xs text-muted-foreground text-center max-w-md">Supports STEP, IGES, and STL formats</p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <Box className="h-16 w-16 text-muted-foreground" />
            <p className="text-sm text-muted-foreground text-center">
              3D preview not available for {fileExtension.toUpperCase()} files
            </p>
            <Button variant="outline" onClick={handleDownload}>
              Download File
            </Button>
          </div>
        )}
      </CardContent>
    </div>
  );
}
