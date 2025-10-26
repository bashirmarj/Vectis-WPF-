import { Canvas, useFrame } from "@react-three/fiber";
import { TrackballControls, PerspectiveCamera, ContactShadows, GizmoHelper, GizmoViewport, View, OrbitControls } from "@react-three/drei";
import { Suspense, useMemo, useEffect, useState, useRef, useCallback } from "react";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { CardContent } from "@/components/ui/card";
import { Loader2, Box } from "lucide-react";
import { Button } from "@/components/ui/button";
import * as THREE from "three";
import { supabase } from "@/integrations/supabase/client";
import { MeshModel } from "./cad-viewer/MeshModel";
import { DimensionAnnotations } from "./cad-viewer/DimensionAnnotations";
import { ProfessionalLighting } from "./cad-viewer/enhancements/ProfessionalLighting";
import { OrientationCubeMesh } from "./cad-viewer/OrientationCubeMesh";
import { UnifiedCADToolbar } from "./cad-viewer/UnifiedCADToolbar";
import { useMeasurementStore } from "@/stores/measurementStore";

// ✅ FIXED: Simplified interface with ONLY essential props
interface CADViewerProps {
  meshId?: string;
  fileUrl?: string;
  fileName?: string;
  onMeshLoaded?: (data: MeshData) => void;
}

// ✅ FIXED: Proper MeshData interface matching all components and database
interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[]; // ✅ Face type labels from database (always strings)
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
  // ✅ FIXED: Standardized sectionPlane type to match UnifiedCADToolbar
  const [sectionPlane, setSectionPlane] = useState<"xy" | "xz" | "yz" | null>(null);
  const [sectionPosition, setSectionPosition] = useState(0);

  const { activeTool, setActiveTool, clearAllMeasurements, measurements } = useMeasurementStore();
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<any>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const mainViewportRef = useRef<HTMLDivElement>(null);
  const orientationViewportRef = useRef<HTMLDivElement>(null);

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

        // ✅ Parse mesh data - handle both JSON and array formats
        const vertices = Array.isArray(mesh.vertices) ? mesh.vertices : JSON.parse(mesh.vertices as string);
        const indices = Array.isArray(mesh.indices) ? mesh.indices : JSON.parse(mesh.indices as string);
        const normals = Array.isArray(mesh.normals) ? mesh.normals : JSON.parse(mesh.normals as string);

        // ✅ Parse vertex_colors as string array (face type labels)
        const vertex_colors = mesh.vertex_colors
          ? Array.isArray(mesh.vertex_colors)
            ? mesh.vertex_colors
            : JSON.parse(mesh.vertex_colors as string)
          : undefined;

        // ✅ Parse feature_edges if available
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

  // Initial camera position
  const initialCameraPosition = useMemo(() => {
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 1.5;
    return [
      boundingBox.center[0] + distance * 0.707,
      boundingBox.center[1] + distance * 0.707,
      boundingBox.center[2] + distance * 0.707,
    ] as [number, number, number];
  }, [boundingBox]);
  // Dynamic fog distances based on bounding box
  const fogDistances = useMemo(() => {
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const cameraDistance = maxDim * 1.5;
    return {
      near: cameraDistance * 2,
      far: cameraDistance * 4,
    };
  }, [boundingBox]);
  const handleViewChange = useCallback(
    (viewType: "front" | "top" | "side" | "isometric" | "home") => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      let newPosition: THREE.Vector3;

      switch (viewType) {
        case "front":
          newPosition = new THREE.Vector3(target.x, target.y, target.z + distance);
          break;
        case "side":
          newPosition = new THREE.Vector3(target.x + distance, target.y, target.z);
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

  const handleCubeClick = useCallback(
    (direction: THREE.Vector3) => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      const newPosition = target.clone().add(direction.multiplyScalar(distance));

      camera.position.copy(newPosition);
      controls.target.copy(target);
      controls.update();
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
    } catch (error) {
      console.error("Download failed:", error);
    }
  }, [fileUrl, fileName]);

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

            {/* Main viewport - fills entire area */}
            <div ref={mainViewportRef} className="absolute inset-0" />

            {/* Orientation Cube Viewport - overlay */}
            <div 
              ref={orientationViewportRef}
              className="absolute top-4 right-4 w-[140px] h-[140px] border rounded bg-background/80 backdrop-blur-sm shadow-lg z-10"
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

              {/* Main Scene View */}
              <View index={0} track={mainViewportRef as React.MutableRefObject<HTMLElement>}>
                <PerspectiveCamera ref={cameraRef} makeDefault position={initialCameraPosition} fov={50} />

                <Suspense fallback={null}>
                  {/* ✅ Use ProfessionalLighting instead of missing LightingRig */}
                  <ProfessionalLighting intensity={1.0} enableShadows={shadowsEnabled} shadowQuality="high" />

                  {/* ✅ FIXED: Convert null to "none" for MeshModel */}
                  <MeshModel
                    ref={meshRef}
                    meshData={meshData}
                    displayStyle={displayMode}
                    showEdges={showEdges}
                    sectionPlane={sectionPlane || "none"}
                    sectionPosition={sectionPosition}
                  />

                  <DimensionAnnotations boundingBox={boundingBox} />

                  {/* ✅ Removed VisualEffects - not needed for basic rendering */}

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
              </View>

              {/* Orientation Cube View */}
              <View index={1} track={orientationViewportRef as React.MutableRefObject<HTMLElement>}>
                <PerspectiveCamera makeDefault={false} position={[5, 5, 5]} fov={50} />
                <ambientLight intensity={0.6} />
                <directionalLight position={[5, 5, 5]} intensity={0.8} />
                <OrientationCubeMesh onCubeClick={handleCubeClick} displayMode={displayMode} />
              </View>
            </Canvas>

            {/* ✅ Removed PerformanceSettingsPanel - not needed for basic functionality */}
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
