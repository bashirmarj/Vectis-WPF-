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
          vertex_colors: vertex_colors as string[] | undefined,
          triangle_count: indices.length / 3,
          face_types: mesh.face_types
            ? Array.isArray(mesh.face_types)
              ? mesh.face_types
              : JSON.parse(mesh.face_types as string)
            : undefined,
          feature_edges: feature_edges as number[][][] | undefined,
        };

        setMeshData(loadedMeshData);
        if (onMeshLoaded) {
          onMeshLoaded(loadedMeshData);
        }
      } catch (err) {
        console.error("Error loading mesh:", err);
        setError(err instanceof Error ? err.message : "Failed to load mesh");
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
        min: new THREE.Vector3(0, 0, 0),
        max: new THREE.Vector3(1, 1, 1),
        center: [0, 0, 0] as [number, number, number],
        width: 1,
        height: 1,
        depth: 1,
      };
    }

    const vertices = meshData.vertices;
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

    for (let i = 0; i < vertices.length; i += 3) {
      min.x = Math.min(min.x, vertices[i]);
      min.y = Math.min(min.y, vertices[i + 1]);
      min.z = Math.min(min.z, vertices[i + 2]);
      max.x = Math.max(max.x, vertices[i]);
      max.y = Math.max(max.y, vertices[i + 1]);
      max.z = Math.max(max.z, vertices[i + 2]);
    }

    const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);

    return {
      min,
      max,
      center: [center.x, center.y, center.z] as [number, number, number],
      width: max.x - min.x,
      height: max.y - min.y,
      depth: max.z - min.z,
    };
  }, [meshData]);

  // Calculate initial camera position
  const initialCameraPosition = useMemo(() => {
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 1.5;
    return new THREE.Vector3(
      boundingBox.center[0] + distance * 0.707,
      boundingBox.center[1] + distance * 0.707,
      boundingBox.center[2] + distance * 0.707,
    );
  }, [boundingBox]);

  // Camera view controls
  const handleSetView = useCallback(
    (view: string) => {
      if (!cameraRef.current || !controlsRef.current) {
        console.error("❌ Camera or controls ref not available");
        return;
      }

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      let newPosition: THREE.Vector3;

      switch (view) {
        case "front":
          newPosition = new THREE.Vector3(target.x, target.y, target.z + distance);
          break;
        case "top":
          newPosition = new THREE.Vector3(target.x, target.y + distance, target.z);
          break;
        case "isometric":
          newPosition = new THREE.Vector3(
            target.x + distance * 0.707,
            target.y + distance * 0.707,
            target.z + distance * 0.707,
          );
          break;
        case "home":
          newPosition = new THREE.Vector3(
            target.x + distance * 0.707,
            target.y + distance * 0.707,
            target.z + distance * 0.707,
          );
          break;
        default:
          return;
      }

      camera.position.copy(newPosition);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();
    },
    [boundingBox],
  );

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
  }, [boundingBox]);

  // ✅ Orientation cube face click handler
  const handleCubeClick = useCallback(
    (direction: THREE.Vector3) => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      const newPosition = target.clone().add(direction.multiplyScalar(distance));

      // Determine appropriate up vector
      let newUp = new THREE.Vector3(0, 1, 0);
      if (Math.abs(direction.y) > 0.9) {
        // Looking up or down - use Z as up
        newUp = new THREE.Vector3(0, 0, 1);
      }

      console.log("🎯 Setting camera to face:", {
        direction: { x: direction.x, y: direction.y, z: direction.z },
        newPosition: { x: newPosition.x.toFixed(2), y: newPosition.y.toFixed(2), z: newPosition.z.toFixed(2) },
      });

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();
    },
    [boundingBox],
  );

  // ✅ FIXED: Simplified rotation logic - eliminates gimbal lock issues
  const handleRotateCamera = useCallback(
    (direction: "up" | "down" | "left" | "right" | "cw" | "ccw") => {
      if (!cameraRef.current || !controlsRef.current) return;

      console.log("🎯 Rotating camera:", direction);

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = controls.target.clone();

      // Get current camera state
      const currentPosition = camera.position.clone();
      const currentUp = camera.up.clone();

      // Calculate view direction (from camera to target)
      const viewDir = target.clone().sub(currentPosition).normalize();

      // 90-degree rotation
      const rotationAngle = Math.PI / 2;

      let newPosition: THREE.Vector3;
      let newUp: THREE.Vector3;

      // ✅ SIMPLIFIED LOGIC: Always use view-relative rotation axes
      // This eliminates the gimbal lock complexity that was causing issues

      // Calculate screen-space axes relative to current view
      const worldUp = new THREE.Vector3(0, 1, 0);
      const screenRight = new THREE.Vector3().crossVectors(worldUp, viewDir).normalize();

      // If screenRight is near zero (looking straight up/down), use world X as fallback
      if (screenRight.length() < 0.01) {
        screenRight.set(1, 0, 0);
      }

      // Calculate screen up (perpendicular to both viewDir and screenRight)
      const screenUp = new THREE.Vector3().crossVectors(viewDir, screenRight).normalize();

      switch (direction) {
        case "left":
          // Rotate camera position around screenUp axis (horizontal rotation)
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenUp, rotationAngle);
          newPosition.add(target);
          // Up vector also rotates around the same axis
          newUp = currentUp.clone().applyAxisAngle(screenUp, rotationAngle);
          break;

        case "right":
          // Rotate camera position around screenUp axis (horizontal rotation, opposite direction)
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenUp, -rotationAngle);
          newPosition.add(target);
          // Up vector also rotates around the same axis
          newUp = currentUp.clone().applyAxisAngle(screenUp, -rotationAngle);
          break;

        case "up":
          // Rotate camera position around screenRight axis (vertical rotation)
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenRight, rotationAngle);
          newPosition.add(target);
          // Up vector also rotates around the same axis
          newUp = currentUp.clone().applyAxisAngle(screenRight, rotationAngle);
          break;

        case "down":
          // Rotate camera position around screenRight axis (vertical rotation, opposite direction)
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenRight, -rotationAngle);
          newPosition.add(target);
          // Up vector also rotates around the same axis
          newUp = currentUp.clone().applyAxisAngle(screenRight, -rotationAngle);
          break;

        case "cw":
          // Roll clockwise: rotate up vector around view direction
          newPosition = currentPosition.clone();
          newUp = currentUp.clone().applyAxisAngle(viewDir, -rotationAngle);
          break;

        case "ccw":
          // Roll counter-clockwise: rotate up vector around view direction
          newPosition = currentPosition.clone();
          newUp = currentUp.clone().applyAxisAngle(viewDir, rotationAngle);
          break;

        default:
          return;
      }

      // Apply new camera position and orientation
      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      console.log("✅ Camera rotated");
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
    [boundingBox],
  );

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
    } catch (err) {
      console.error("Error downloading file:", err);
    }
  }, [fileUrl, fileName]);

  return (
    <div className="relative w-full h-full">
      <CardContent className="p-0 h-full">
        {isLoading && (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center h-full gap-4 p-4">
            <div className="text-destructive text-center">
              <p className="font-semibold">Error loading 3D model</p>
              <p className="text-sm mt-2">{error}</p>
            </div>
            {fileUrl && (
              <Button variant="outline" onClick={handleDownload}>
                Download File
              </Button>
            )}
          </div>
        )}

        {!isLoading && !error && meshData && isRenderableFormat ? (
          <div className="relative w-full h-full">
            {/* Unified Toolbar */}
            <UnifiedCADToolbar
              onSetView={handleSetView}
              onFitView={handleFitView}
              activeTool={activeTool}
              onSetActiveTool={setActiveTool}
              onClearMeasurements={clearAllMeasurements}
              measurementCount={measurements.length}
              displayMode={displayMode}
              onDisplayModeChange={setDisplayMode}
              showEdges={showEdges}
              onToggleEdges={() => setShowEdges(!showEdges)}
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

            {/* ✅ PROFESSIONAL: Orientation Cube Viewport - Separate Canvas overlay */}
            <OrientationCubeViewport
              mainCameraRef={cameraRef}
              onCubeClick={handleCubeClick}
              onRotateUp={() => handleRotateCamera("up")}
              onRotateDown={() => handleRotateCamera("down")}
              onRotateLeft={() => handleRotateCamera("left")}
              onRotateRight={() => handleRotateCamera("right")}
              onRotateClockwise={() => handleRotateCamera("cw")}
              onRotateCounterClockwise={() => handleRotateCamera("ccw")}
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

              {/* ❌ REMOVED: Legacy GizmoHelper/GizmoViewport - replaced with professional OrientationCubeViewport */}
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
