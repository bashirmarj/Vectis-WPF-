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

  // ✅ NEW: Auto-detect coordinate system based on part geometry
  const coordinateSystem = useMemo(() => {
    const { width, height, depth } = boundingBox;

    // Most CAD files (STEP/IGES) use Z-up coordinate system
    // If the part's Z dimension is significantly larger than Y, assume Z-up
    // If the part's Y dimension is significantly larger than Z, assume Y-up
    const isZUp = depth > height * 1.2;

    console.log("📊 Coordinate system detection:", {
      width: width.toFixed(2),
      height: height.toFixed(2),
      depth: depth.toFixed(2),
      ratio: (depth / height).toFixed(2),
      detected: isZUp ? "Z-up (STEP/IGES standard)" : "Y-up (Three.js standard)",
    });

    return isZUp ? "Z-up" : "Y-up";
  }, [boundingBox]);

  // ✅ NEW: Transform direction vector based on coordinate system
  const transformDirection = useCallback(
    (direction: THREE.Vector3): THREE.Vector3 => {
      if (coordinateSystem === "Z-up") {
        // Convert from Y-up (orientation cube) to Z-up (CAD model)
        // Y-up system: X=right, Y=up,      Z=forward
        // Z-up system: X=right, Y=backward, Z=up
        return new THREE.Vector3(
          direction.x, // X stays the same (right is right)
          -direction.z, // Y becomes -Z (forward becomes backward)
          direction.y, // Z becomes Y (up becomes up)
        );
      }
      // Y-up system: no transformation needed
      return direction.clone();
    },
    [coordinateSystem],
  );

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

      let direction: THREE.Vector3;
      let newUp = new THREE.Vector3(0, 1, 0);

      switch (viewType) {
        case "front":
          direction = new THREE.Vector3(0, 0, 1);
          break;
        case "side":
          direction = new THREE.Vector3(1, 0, 0);
          break;
        case "top":
          direction = new THREE.Vector3(0, 1, 0);
          newUp = new THREE.Vector3(0, 0, 1);
          break;
        case "isometric":
        case "home":
        default:
          direction = new THREE.Vector3(0.707, 0.707, 0.707);
          break;
      }

      // ✅ FIXED: Transform direction to match model coordinate system
      const transformedDirection = transformDirection(direction);
      const newPosition = target.clone().add(transformedDirection.multiplyScalar(distance));

      console.log(`📐 View change: ${viewType}`, {
        cubeDirection: { x: direction.x.toFixed(2), y: direction.y.toFixed(2), z: direction.z.toFixed(2) },
        modelDirection: {
          x: transformedDirection.x.toFixed(2),
          y: transformedDirection.y.toFixed(2),
          z: transformedDirection.z.toFixed(2),
        },
      });

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      // Update orientation cube
      orientationCubeRef.current?.updateFromMainCamera(camera);
    },
    [boundingBox, transformDirection],
  );

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

      // Update orientation cube
      orientationCubeRef.current?.updateFromMainCamera(camera);

      console.log("✅ Camera repositioned to face");
    },
    [boundingBox, transformDirection, coordinateSystem],
  );

  // ✅ Arrow rotation handler with gimbal lock fix
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

    // ✅ FIXED: Detect gimbal lock condition
    const worldUp = new THREE.Vector3(0, 1, 0);
    const dotProduct = Math.abs(viewDir.dot(worldUp));
    const isGimbalLock = dotProduct > 0.98;

    let newPosition: THREE.Vector3;
    let newUp: THREE.Vector3;
    const rotationAngle = Math.PI / 12; // 15° rotation

    if (isGimbalLock) {
      console.log("⚠️ Gimbal lock detected - using world-space rotation");

      const worldZ = new THREE.Vector3(0, 0, 1);
      const isLookingDown = viewDir.y > 0;

      switch (direction) {
        case "left":
          console.log("⬅️ Rotating LEFT around world Y-axis");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(worldUp, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(worldUp, rotationAngle);
          break;

        case "right":
          console.log("➡️ Rotating RIGHT around world Y-axis");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(worldUp, -rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(worldUp, -rotationAngle);
          break;

        case "up":
          console.log("⬆️ Rotating UP - moving away from gimbal lock");
          const upAxis = isLookingDown ? worldZ : worldZ.clone().negate();
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(upAxis, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(upAxis, rotationAngle);
          break;

        case "down":
          console.log("⬇️ Rotating DOWN - moving away from gimbal lock");
          const downAxis = isLookingDown ? worldZ.clone().negate() : worldZ;
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(downAxis, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(downAxis, rotationAngle);
          break;

        case "cw":
          console.log("🔄 Rolling CLOCKWISE around view axis");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, -rotationAngle);
          break;

        case "ccw":
          console.log("🔄 Rolling COUNTER-CLOCKWISE around view axis");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, rotationAngle);
          break;

        default:
          return;
      }
    } else {
      const screenUp = camera.up.clone().normalize();
      const screenRight = new THREE.Vector3().crossVectors(viewDir, screenUp).normalize();

      if (screenRight.lengthSq() < 0.01) {
        console.warn("⚠️ Degenerate screen-right vector");
        return;
      }

      switch (direction) {
        case "left":
          console.log("⬅️ Rotating LEFT around screen-up axis");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenUp, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone();
          break;

        case "right":
          console.log("➡️ Rotating RIGHT around screen-up axis");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenUp, -rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone();
          break;

        case "up":
          console.log("⬆️ Rotating UP around screen-right axis");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenRight, rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(screenRight, rotationAngle);
          break;

        case "down":
          console.log("⬇️ Rotating DOWN around screen-right axis");
          newPosition = currentPosition.clone().sub(target);
          newPosition.applyAxisAngle(screenRight, -rotationAngle);
          newPosition.add(target);
          newUp = camera.up.clone().applyAxisAngle(screenRight, -rotationAngle);
          break;

        case "cw":
          console.log("🔄 Rolling CLOCKWISE around view axis");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, -rotationAngle);
          break;

        case "ccw":
          console.log("🔄 Rolling COUNTER-CLOCKWISE around view axis");
          newPosition = currentPosition.clone();
          newUp = camera.up.clone().applyAxisAngle(viewDir, rotationAngle);
          break;

        default:
          return;
      }
    }

    console.log("📊 Camera state after rotation:", {
      newPosition: {
        x: newPosition.x.toFixed(2),
        y: newPosition.y.toFixed(2),
        z: newPosition.z.toFixed(2),
      },
      newUp: { x: newUp.x.toFixed(2), y: newUp.y.toFixed(2), z: newUp.z.toFixed(2) },
    });

    camera.position.copy(newPosition);
    camera.up.copy(newUp);
    camera.lookAt(target);
    controls.update();

    // Update orientation cube
    orientationCubeRef.current?.updateFromMainCamera(camera);

    console.log("✅ Controls updated");
    console.log("✅ Handler executed for:", direction.toUpperCase());
  }, []);

  const handleFitView = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return;

    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const target = new THREE.Vector3(...boundingBox.center);
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 1.5;

    // Maintain current viewing direction
    const currentDir = camera.position.clone().sub(target).normalize();
    const newPosition = target.clone().add(currentDir.multiplyScalar(distance));

    camera.position.copy(newPosition);
    controls.target.copy(target);
    controls.update();

    // Update orientation cube
    orientationCubeRef.current?.updateFromMainCamera(camera);
  }, [boundingBox]);

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
