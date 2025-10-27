import { Canvas } from "@react-three/fiber";
import { TrackballControls, PerspectiveCamera, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { Suspense, useMemo, useEffect, useState, useRef, useCallback } from "react";
import { CardContent } from "@/components/ui/card";
import { Loader2, Box } from "lucide-react";
import { Button } from "@/components/ui/button";
import * as THREE from "three";
import { supabase } from "@/integrations/supabase/client";
import { MeshModel } from "./cad-viewer/MeshModel";
import { DimensionAnnotations } from "./cad-viewer/DimensionAnnotations";
import { OrientationCubePreview, OrientationCubeHandle } from "./cad-viewer/OrientationCubePreview";
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
  const orientationCubeRef = useRef<OrientationCubeHandle>(null);

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
          vertex_colors: vertex_colors as string[] | undefined,
          triangle_count: mesh.triangle_count || indices.length / 3,
          feature_edges: feature_edges as number[][][] | undefined,
        };

        setMeshData(loadedMeshData);
        onMeshLoaded?.(loadedMeshData);
      } catch (error) {
        console.error("Failed to load mesh:", error);
        setError(error instanceof Error ? error.message : "Failed to load 3D model");
      } finally {
        setIsLoading(false);
      }
    };

    loadMeshData();
  }, [meshId, isRenderableFormat, onMeshLoaded]);

  // Calculate bounding box
  const boundingBox = useMemo(() => {
    if (!meshData) {
      return {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
        center: [0, 0, 0] as [number, number, number],
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
      minX = Math.min(minX, vertices[i]);
      minY = Math.min(minY, vertices[i + 1]);
      minZ = Math.min(minZ, vertices[i + 2]);
      maxX = Math.max(maxX, vertices[i]);
      maxY = Math.max(maxY, vertices[i + 1]);
      maxZ = Math.max(maxZ, vertices[i + 2]);
    }

    const width = maxX - minX;
    const height = maxY - minY;
    const depth = maxZ - minZ;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    return {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
      center: [centerX, centerY, centerZ] as [number, number, number],
      width,
      height,
      depth,
    };
  }, [meshData]);

  // Initial camera position (isometric view)
  const initialCameraPosition = useMemo(() => {
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 1.5;
    return [
      boundingBox.center[0] + distance * 0.707,
      boundingBox.center[1] + distance * 0.707,
      boundingBox.center[2] + distance * 0.707,
    ] as [number, number, number];
  }, [boundingBox]);

  // Camera view change handler
  const handleViewChange = useCallback(
    (viewType: "front" | "top" | "side" | "isometric" | "home") => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      let newPosition: THREE.Vector3;
      let newUp = new THREE.Vector3(0, 1, 0);

      switch (viewType) {
        case "front":
          newPosition = new THREE.Vector3(target.x, target.y, target.z + distance);
          break;
        case "side":
          newPosition = new THREE.Vector3(target.x + distance, target.y, target.z);
          break;
        case "top":
          newPosition = new THREE.Vector3(target.x, target.y + distance, target.z);
          newUp = new THREE.Vector3(0, 0, 1);
          break;
        case "isometric":
          newPosition = new THREE.Vector3(
            target.x + distance * 0.707,
            target.y + distance * 0.707,
            target.z + distance * 0.707,
          );
          break;
        case "home":
        default:
          newPosition = new THREE.Vector3(
            target.x + distance * 0.707,
            target.y + distance * 0.707,
            target.z + distance * 0.707,
          );
          break;
      }

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      orientationCubeRef.current?.updateFromMainCamera(camera);
    },
    [boundingBox],
  );

  // Fit view to model
  const handleFitView = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return;

    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const target = new THREE.Vector3(...boundingBox.center);
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 1.5;
    const currentPosition = camera.position.clone().sub(target).normalize();
    const newPosition = target.clone().add(currentPosition.multiplyScalar(distance));

    camera.position.copy(newPosition);
    controls.target.copy(target);
    controls.update();

    orientationCubeRef.current?.updateFromMainCamera(camera);
  }, [boundingBox]);

  // Cube face click handler
  const handleCubeClick = useCallback(
    (direction: THREE.Vector3) => {
      if (!cameraRef.current || !controlsRef.current) return;

      console.log("🎯 Cube face clicked:", direction);

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      const newPosition = target.clone().add(direction.multiplyScalar(distance));

      // Determine appropriate up vector
      const newUp = new THREE.Vector3(0, 1, 0);
      if (Math.abs(direction.y) > 0.99) {
        newUp.set(0, 0, direction.y > 0 ? 1 : -1);
      }

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      orientationCubeRef.current?.updateFromMainCamera(camera);

      console.log("✅ Camera repositioned to face");
    },
    [boundingBox],
  );

  // ✅ DEBUGGED: Rotation handler with comprehensive logging
  const handleRotateCamera = useCallback(
    (rotationType: "up" | "down" | "left" | "right" | "cw" | "ccw") => {
      console.log(`🎯 handleRotateCamera called with: ${rotationType}`);

      if (!cameraRef.current || !controlsRef.current) {
        console.error("❌ Camera or controls ref not available!");
        return;
      }

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const currentDistance = camera.position.distanceTo(target);

      console.log("📊 Camera state before rotation:", {
        position: {
          x: camera.position.x.toFixed(2),
          y: camera.position.y.toFixed(2),
          z: camera.position.z.toFixed(2),
        },
        distance: currentDistance.toFixed(2),
        target: {
          x: target.x.toFixed(2),
          y: target.y.toFixed(2),
          z: target.z.toFixed(2),
        },
      });

      const ROTATION_ANGLE = Math.PI / 2; // 90 degrees

      // ✅ PURE VIEW-RELATIVE: Calculate screen axes from view direction
      const viewDirection = camera.position.clone().sub(target).normalize();
      const worldUp = new THREE.Vector3(0, 1, 0);

      // Calculate screen-right (horizontal on screen)
      let screenRight = new THREE.Vector3().crossVectors(worldUp, viewDirection).normalize();

      // Handle special case: looking straight up or down
      if (screenRight.length() < 0.001) {
        screenRight = new THREE.Vector3(1, 0, 0);
        console.log("⚠️ Using fallback screen-right (looking straight up/down)");
      }

      // Calculate screen-up (vertical on screen)
      const screenUp = new THREE.Vector3().crossVectors(viewDirection, screenRight).normalize();

      let newPosition: THREE.Vector3;
      let newUp: THREE.Vector3;

      switch (rotationType) {
        case "up": {
          console.log("⬆️ Rotating UP around screen-right axis");
          newPosition = viewDirection.clone();
          newPosition.applyAxisAngle(screenRight, -ROTATION_ANGLE);
          newPosition.normalize();

          const newViewDir = newPosition.clone();
          const newScreenRight = new THREE.Vector3().crossVectors(worldUp, newViewDir).normalize();
          if (newScreenRight.length() < 0.001) {
            newUp = screenRight.clone();
          } else {
            newUp = new THREE.Vector3().crossVectors(newViewDir, newScreenRight).normalize();
          }
          break;
        }

        case "down": {
          console.log("⬇️ Rotating DOWN around screen-right axis");
          newPosition = viewDirection.clone();
          newPosition.applyAxisAngle(screenRight, ROTATION_ANGLE);
          newPosition.normalize();

          const newViewDir = newPosition.clone();
          const newScreenRight = new THREE.Vector3().crossVectors(worldUp, newViewDir).normalize();
          if (newScreenRight.length() < 0.001) {
            newUp = screenRight.clone();
          } else {
            newUp = new THREE.Vector3().crossVectors(newViewDir, newScreenRight).normalize();
          }
          break;
        }

        case "left": {
          console.log("⬅️ Rotating LEFT around screen-up axis");
          newPosition = viewDirection.clone();
          newPosition.applyAxisAngle(screenUp, ROTATION_ANGLE);
          newPosition.normalize();

          newUp = screenUp.clone();
          break;
        }

        case "right": {
          console.log("➡️ Rotating RIGHT around screen-up axis");
          newPosition = viewDirection.clone();
          newPosition.applyAxisAngle(screenUp, -ROTATION_ANGLE);
          newPosition.normalize();

          newUp = screenUp.clone();
          break;
        }

        case "cw": {
          console.log("🔄 Rolling CLOCKWISE around view axis");
          newPosition = viewDirection.clone();
          newUp = screenUp.clone();
          newUp.applyAxisAngle(viewDirection, -ROTATION_ANGLE);
          newUp.normalize();
          break;
        }

        case "ccw": {
          console.log("🔄 Rolling COUNTER-CLOCKWISE around view axis");
          newPosition = viewDirection.clone();
          newUp = screenUp.clone();
          newUp.applyAxisAngle(viewDirection, ROTATION_ANGLE);
          newUp.normalize();
          break;
        }

        default:
          console.error(`❌ Unknown rotation type: ${rotationType}`);
          return;
      }

      // Apply new camera position and orientation
      const finalPosition = target.clone().add(newPosition.multiplyScalar(currentDistance));

      console.log("📊 Camera state after rotation:", {
        newPosition: {
          x: finalPosition.x.toFixed(2),
          y: finalPosition.y.toFixed(2),
          z: finalPosition.z.toFixed(2),
        },
        newUp: {
          x: newUp.x.toFixed(2),
          y: newUp.y.toFixed(2),
          z: newUp.z.toFixed(2),
        },
      });

      camera.position.copy(finalPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      console.log("✅ Controls updated");

      // Update orientation cube
      if (orientationCubeRef.current) {
        orientationCubeRef.current.updateFromMainCamera(camera);
        console.log("✅ Orientation cube notified");
      } else {
        console.warn("⚠️ Orientation cube ref not available");
      }
    },
    [boundingBox],
  );

  // Download handler
  const handleDownload = useCallback(async () => {
    if (!fileUrl) return;

    try {
      const response = await fetch(fileUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName || "model";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download failed:", error);
    }
  }, [fileUrl, fileName]);

  // ✅ Update orientation cube when camera moves via TrackballControls
  useEffect(() => {
    if (!cameraRef.current || !controlsRef.current) return;

    const controls = controlsRef.current;
    const handleControlsChange = () => {
      if (cameraRef.current) {
        console.log("🎮 Controls change event fired");
        orientationCubeRef.current?.updateFromMainCamera(cameraRef.current);
      }
    };

    controls.addEventListener("change", handleControlsChange);
    return () => {
      controls.removeEventListener("change", handleControlsChange);
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.code === "Space") {
        e.preventDefault();
        handleFitView();
      } else if (e.code === "KeyE") {
        e.preventDefault();
        setShowEdges(!showEdges);
      } else if (e.code === "Escape") {
        e.preventDefault();
        if (activeTool) {
          setActiveTool(null);
        }
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [handleFitView, showEdges, activeTool, setActiveTool]);

  return (
    <div className="relative w-full h-full">
      <CardContent className="p-0 h-full">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <p className="text-sm text-destructive text-center">{error}</p>
          </div>
        ) : meshData && isRenderableFormat ? (
          <div className="relative w-full h-full">
            {/* Unified Toolbar */}
            <UnifiedCADToolbar
              onHomeView={() => handleViewChange("home")}
              onFrontView={() => handleViewChange("front")}
              onTopView={() => handleViewChange("top")}
              onIsometricView={() => handleViewChange("isometric")}
              onFitView={handleFitView}
              displayMode={displayMode}
              onDisplayModeChange={setDisplayMode}
              showEdges={showEdges}
              onToggleEdges={() => setShowEdges(!showEdges)}
              measurementMode={activeTool}
              onMeasurementModeChange={setActiveTool}
              measurementCount={measurements.length}
              onClearMeasurements={clearAllMeasurements}
              sectionPlane={sectionPlane}
              onSectionPlaneChange={setSectionPlane}
              sectionPosition={sectionPosition}
              onSectionPositionChange={setSectionPosition}
              shadowsEnabled={shadowsEnabled}
              onToggleShadows={() => setShadowsEnabled(!shadowsEnabled)}
              ssaoEnabled={ssaoEnabled}
              onToggleSSAO={() => setSSAOEnabled(!ssaoEnabled)}
              boundingBox={{
                min: { x: boundingBox.min.x, y: boundingBox.min.y, z: boundingBox.min.z },
                max: { x: boundingBox.max.x, y: boundingBox.max.y, z: boundingBox.max.z },
                center: { x: boundingBox.center[0], y: boundingBox.center[1], z: boundingBox.center[2] },
              }}
            />

            {/* ✅ Orientation Cube with all rotation handlers */}
            <OrientationCubePreview
              ref={orientationCubeRef}
              onCubeClick={handleCubeClick}
              onRotateUp={() => handleRotateCamera("up")}
              onRotateDown={() => handleRotateCamera("down")}
              onRotateLeft={() => handleRotateCamera("left")}
              onRotateRight={() => handleRotateCamera("right")}
              onRotateClockwise={() => handleRotateCamera("cw")}
              onRotateCounterClockwise={() => handleRotateCamera("ccw")}
              displayMode={displayMode}
              onDisplayModeChange={setDisplayMode}
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

                {/* ✅ TrackballControls allows free rotation */}
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

              <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
                <GizmoViewport
                  axisColors={["#ff0000", "#00ff00", "#0000ff"]}
                  labelColor="white"
                  labels={["X", "Y", "Z"]}
                />
              </GizmoHelper>
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
