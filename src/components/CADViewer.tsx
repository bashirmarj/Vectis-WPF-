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
// ✅ FIXED: Import both cube components
import { OrientationCubeControls } from "./cad-viewer/OrientationCubeControls";
import { OrientationCubeInScene } from "./cad-viewer/OrientationCubeInScene";
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

  // Determine coordinate system based on bounding box dimensions
  const coordinateSystem = useMemo(() => {
    const aspectRatio = boundingBox.width / boundingBox.height;
    if (aspectRatio > 1.2 || aspectRatio < 0.8) {
      return "y-up";
    }
    return "z-up";
  }, [boundingBox]);

  // Transform direction from cube space to model space
  const transformDirection = useCallback(
    (cubeDir: THREE.Vector3): THREE.Vector3 => {
      if (coordinateSystem === "z-up") {
        return new THREE.Vector3(cubeDir.x, -cubeDir.z, cubeDir.y);
      }
      return cubeDir.clone();
    },
    [coordinateSystem],
  );

  const modelBounds = useMemo(
    () => ({
      min: { x: boundingBox.min.x, y: boundingBox.min.y, z: boundingBox.min.z },
      max: { x: boundingBox.max.x, y: boundingBox.max.y, z: boundingBox.max.z },
      center: { x: boundingBox.center[0], y: boundingBox.center[1], z: boundingBox.center[2] },
    }),
    [boundingBox],
  );

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

  // ✅ FIXED: Orientation cube face click handler with coordinate transform
  const handleCubeClick = useCallback(
    (direction: THREE.Vector3) => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      // ✅ FIXED: Transform cube direction to match model coordinate system
      const transformedDirection = transformDirection(direction);
      const newPosition = target.clone().add(transformedDirection.multiplyScalar(distance));

      // Determine appropriate up vector
      let newUp = new THREE.Vector3(0, 1, 0);
      if (Math.abs(transformedDirection.y) > 0.9) {
        // Looking up or down - use Z as up
        newUp = new THREE.Vector3(0, 0, 1);
      }

      console.log("🎯 Cube face clicked:", {
        cubeDirection: { x: direction.x, y: direction.y, z: direction.z },
        modelDirection: {
          x: transformedDirection.x.toFixed(2),
          y: transformedDirection.y.toFixed(2),
          z: transformedDirection.z.toFixed(2),
        },
        coordinateSystem,
      });

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      console.log("✅ Camera repositioned to face");
    },
    [boundingBox, transformDirection, coordinateSystem],
  );

  // ✅ FIXED: Arrow rotation handler with 90-degree increments and gimbal lock fix
  const handleRotateCamera = useCallback((direction: "up" | "down" | "left" | "right" | "cw" | "ccw") => {
    if (!cameraRef.current || !controlsRef.current) return;

    console.log("🎯 handleRotateCamera called with:", direction);

    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const target = controls.target.clone();

    const currentPosition = camera.position.clone();
    const distance = currentPosition.distanceTo(target);
    const viewDir = currentPosition.clone().sub(target).normalize();

    console.log("📊 Camera state before rotation:", {
      position: {
        x: currentPosition.x.toFixed(2),
        y: currentPosition.y.toFixed(2),
        z: currentPosition.z.toFixed(2),
      },
      distance: distance.toFixed(2),
      target: { x: target.x.toFixed(2), y: target.y.toFixed(2), z: target.z.toFixed(2) },
    });

    // ✅ FIXED: Use 90-degree rotations (Math.PI / 2) instead of 15 degrees
    const rotationAngle = Math.PI / 2; // 90° rotation

    // ✅ FIXED: Detect gimbal lock condition
    const worldUp = new THREE.Vector3(0, 1, 0);
    const dotProduct = Math.abs(viewDir.dot(worldUp));
    const isGimbalLock = dotProduct > 0.98;

    let newPosition: THREE.Vector3;
    let newUp: THREE.Vector3;

    if (isGimbalLock) {
      console.log("⚠️ Gimbal lock detected - using world-space rotation");

      const worldZ = new THREE.Vector3(0, 0, 1);
      const isLookingDown = viewDir.y > 0;

      switch (direction) {
        case "left":
          console.log("⬅️ Rotating LEFT around world Y-axis (90°)");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(worldUp, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(worldUp, rotationAngle);
          break;

        case "right":
          console.log("➡️ Rotating RIGHT around world Y-axis (90°)");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(worldUp, -rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(worldUp, -rotationAngle);
          break;

        case "up":
          console.log("⬆️ Rotating UP - moving away from gimbal lock (90°)");
          const upAxis = isLookingDown ? worldZ : worldZ.clone().negate();
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(upAxis, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(upAxis, rotationAngle);
          break;

        case "down":
          console.log("⬇️ Rotating DOWN - moving away from gimbal lock (90°)");
          const downAxis = isLookingDown ? worldZ.clone().negate() : worldZ;
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(downAxis, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(downAxis, rotationAngle);
          break;

        case "cw":
          console.log("🔄 Rolling CLOCKWISE around view axis (90°)");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, -rotationAngle);
          break;

        case "ccw":
          console.log("🔄 Rolling COUNTER-CLOCKWISE around view axis (90°)");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, rotationAngle);
          break;

        default:
          return;
      }
    } else {
      // Normal rotation using screen-relative axes
      const screenRight = new THREE.Vector3();
      const screenUp = new THREE.Vector3();

      screenRight.crossVectors(worldUp, viewDir).normalize();
      screenUp.crossVectors(viewDir, screenRight).normalize();

      switch (direction) {
        case "left":
          console.log("⬅️ Rotating LEFT around screen UP-axis (90°)");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenUp, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(screenUp, rotationAngle);
          break;

        case "right":
          console.log("➡️ Rotating RIGHT around screen UP-axis (90°)");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenUp, -rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(screenUp, -rotationAngle);
          break;

        case "up":
          console.log("⬆️ Rotating UP around screen RIGHT-axis (90°)");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenRight, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(screenRight, rotationAngle);
          break;

        case "down":
          console.log("⬇️ Rotating DOWN around screen RIGHT-axis (90°)");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenRight, -rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(screenRight, -rotationAngle);
          break;

        case "cw":
          console.log("🔄 Rolling CLOCKWISE around view axis (90°)");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, -rotationAngle);
          break;

        case "ccw":
          console.log("🔄 Rolling COUNTER-CLOCKWISE around view axis (90°)");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, rotationAngle);
          break;

        default:
          return;
      }
    }

    camera.position.copy(newPosition);
    camera.up.copy(newUp);
    camera.lookAt(target);
    controls.target.copy(target);
    controls.update();

    console.log("✅ Camera rotated", {
      newPosition: {
        x: newPosition.x.toFixed(2),
        y: newPosition.y.toFixed(2),
        z: newPosition.z.toFixed(2),
      },
    });
  }, []);

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
            {/* ✅ Unified Toolbar - OUTSIDE Canvas */}
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
              sectionPlane={sectionPlane || "none"}
              onSectionPlaneChange={(plane) => setSectionPlane(plane === "none" ? null : plane)}
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

            {/* ✅ FIXED: Orientation Cube Controls - UI buttons OUTSIDE Canvas */}
            <OrientationCubeControls
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

                {/* ✅ FIXED: Orientation Cube INSIDE Canvas - syncs with camera */}
                <OrientationCubeInScene mainCameraRef={cameraRef} onCubeClick={handleCubeClick} />

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
