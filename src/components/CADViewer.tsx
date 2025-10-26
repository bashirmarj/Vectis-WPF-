import { Canvas } from "@react-three/fiber";
import { TrackballControls, PerspectiveCamera, ContactShadows, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { Suspense, useMemo, useEffect, useState, useRef, useCallback } from "react";
import { CardContent } from "@/components/ui/card";
import { Loader2, Box } from "lucide-react";
import { Button } from "@/components/ui/button";
import * as THREE from "three";
import { supabase } from "@/integrations/supabase/client";
import { MeshModel } from "./cad-viewer/MeshModel";
import { DimensionAnnotations } from "./cad-viewer/DimensionAnnotations";
import { OrientationCubePreview, OrientationCubeHandle } from "./cad-viewer/OrientationCubePreview";
import LightingRig from "./cad-viewer/LightingRig";
import VisualEffects from "./cad-viewer/VisualEffects";
import PerformanceSettingsPanel from "./cad-viewer/PerformanceSettingsPanel";
import { UnifiedCADToolbar } from "./cad-viewer/UnifiedCADToolbar";
import { useMeasurementStore } from "@/stores/measurementStore";

// ✅ FIXED: Simplified interface with ONLY essential props
interface CADViewerProps {
  meshId?: string;
  fileUrl?: string;
  fileName?: string;
  onMeshLoaded?: (data: MeshData) => void;
}

// ✅ FIXED: Proper MeshData interface with vertex_colors
interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: number[]; // ✅ FIXED: Changed from colors to vertex_colors
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
  const [ssaoEnabled, setSSAOEnabled] = useState(true);
  const [quality, setQuality] = useState<"performance" | "balanced" | "quality">("balanced"); // ✅ FIXED: Correct type
  const [sectionPlane, setSectionPlane] = useState<"xy" | "xz" | "yz" | "x" | "y" | "z" | null>(null);
  const [sectionPosition, setSectionPosition] = useState(0);

  // Measurement store
  const {
    activeTool: measurementMode,
    measurements,
    setActiveTool: setMeasurementMode,
    clearAllMeasurements,
  } = useMeasurementStore();

  const meshRef = useRef<THREE.Mesh>(null);
  const controlsRef = useRef<any>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const orientationCubeRef = useRef<OrientationCubeHandle>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);

  const fileExtension = useMemo(() => {
    if (!fileName) return "";
    return fileName.split(".").pop()?.toLowerCase() || "";
  }, [fileName]);

  const isRenderableFormat = useMemo(() => {
    return ["step", "stp", "iges", "igs", "stl"].includes(fileExtension);
  }, [fileExtension]);

  // Load mesh data from database
  useEffect(() => {
    if (!meshId && !fileUrl) {
      setIsLoading(false);
      return;
    }

    const loadMeshData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        if (meshId) {
          console.log(`📥 Fetching mesh from database: ${meshId}`);
          const { data, error: fetchError } = await supabase.from("cad_meshes").select("*").eq("id", meshId).single();

          if (fetchError) throw fetchError;

          if (data) {
            console.log("✅ Mesh loaded:", data.triangle_count, "triangles");
            const meshDataTyped: MeshData = {
              vertices: data.vertices,
              indices: data.indices,
              normals: data.normals,
              vertex_colors: data.vertex_colors, // ✅ FIXED: Use vertex_colors
              triangle_count: data.triangle_count,
              face_types: data.face_types,
              feature_edges: data.feature_edges,
            };
            setMeshData(meshDataTyped);
            onMeshLoaded?.(meshDataTyped);
          }
        }
      } catch (err: any) {
        console.error("❌ Error loading mesh:", err);
        setError(err.message || "Failed to load 3D model");
      } finally {
        setIsLoading(false);
      }
    };

    loadMeshData();
  }, [meshId, fileUrl, onMeshLoaded]);

  // Calculate bounding box
  const boundingBox = useMemo(() => {
    if (!meshData || !meshData.vertices || meshData.vertices.length === 0) {
      return {
        min: new THREE.Vector3(-50, -50, -50),
        max: new THREE.Vector3(50, 50, 50),
        center: [0, 0, 0] as [number, number, number],
        size: new THREE.Vector3(100, 100, 100),
        width: 100,
        height: 100,
        depth: 100,
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

    const center: [number, number, number] = [(min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2];

    const size = new THREE.Vector3(max.x - min.x, max.y - min.y, max.z - min.z);

    return {
      min,
      max,
      center,
      size,
      width: size.x,
      height: size.y,
      depth: size.z,
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

    const direction = new THREE.Vector3().subVectors(camera.position, target).normalize();

    const newPosition = target.clone().add(direction.multiplyScalar(distance));

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
        if (measurementMode) {
          setMeasurementMode(null);
        }
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [handleFitView, showEdges, measurementMode, setMeasurementMode]);

  if (isLoading) {
    return (
      <CardContent className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading 3D model...</p>
        </div>
      </CardContent>
    );
  }

  if (error) {
    return (
      <CardContent className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <Box className="h-16 w-16 text-destructive" />
          <div>
            <p className="text-sm font-medium text-destructive mb-2">Failed to load 3D model</p>
            <p className="text-xs text-muted-foreground">{error}</p>
          </div>
        </div>
      </CardContent>
    );
  }

  return (
    <div className="relative h-full overflow-hidden">
      <CardContent className="p-0 h-full">
        {meshData && isRenderableFormat ? (
          <div ref={canvasRef} className="relative h-full" style={{ background: "#f8f9fa" }}>
            {/* Unified CAD Toolbar */}
            <UnifiedCADToolbar
              onHomeView={() => handleSetView("home")}
              onFrontView={() => handleSetView("front")}
              onTopView={() => handleSetView("top")}
              onIsometricView={() => handleSetView("isometric")}
              onFitView={handleFitView}
              displayMode={displayMode}
              onDisplayModeChange={setDisplayMode}
              showEdges={showEdges}
              onToggleEdges={() => setShowEdges(!showEdges)}
              measurementMode={measurementMode}
              onMeasurementModeChange={setMeasurementMode}
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

            {/* Orientation Cube */}
            <OrientationCubePreview
              ref={orientationCubeRef}
              onCubeClick={handleCubeClick}
              displayMode={displayMode}
              onDisplayModeChange={setDisplayMode}
            />

            <Canvas
              shadows
              gl={{
                antialias: true,
                alpha: true,
                preserveDrawingBuffer: true,
              }}
            >
              <color attach="background" args={["#f8f9fa"]} />
              <fog attach="fog" args={["#f8f9fa", 100, 500]} />

              <PerspectiveCamera ref={cameraRef} makeDefault position={initialCameraPosition} fov={50} />

              <Suspense fallback={null}>
                <LightingRig quality={quality} shadowsEnabled={shadowsEnabled} />

                <MeshModel
                  ref={meshRef}
                  meshData={meshData}
                  displayMode={displayMode}
                  showEdges={showEdges}
                  sectionPlane={sectionPlane}
                  sectionPosition={sectionPosition}
                />

                <DimensionAnnotations boundingBox={boundingBox} />

                <VisualEffects enabled={ssaoEnabled} quality={quality} />

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

            {/* Performance Settings Panel */}
            <PerformanceSettingsPanel
              shadowsEnabled={shadowsEnabled}
              setShadowsEnabled={setShadowsEnabled}
              ssaoEnabled={ssaoEnabled}
              setSSAOEnabled={setSSAOEnabled}
              quality={quality}
              setQuality={setQuality}
              triangleCount={meshData.triangle_count}
            />
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
