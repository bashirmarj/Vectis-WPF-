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
import { SceneEnhancementWrapper } from "./cad-viewer/enhancements/SceneEnhancementWrapper";
import { VisualQualitySettings, DEFAULT_QUALITY_SETTINGS } from "./cad-viewer/enhancements/VisualQualityPanel";
import { CameraTransitionHandler } from "./cad-viewer/CameraTransitionHandler";
import { UnifiedCADToolbar } from "./cad-viewer/UnifiedCADToolbar";
import { useMeasurementStore } from "@/stores/measurementStore";

// ✅ FIXED: Simplified interface with ONLY essential props
interface CADViewerProps {
  meshId?: string;
  fileUrl?: string;
  fileName?: string;
  onMeshLoaded?: (data: MeshData) => void;
}

// ✅ FIXED: Proper MeshData interface matching database schema
interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[]; // ✅ Face type labels from database
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
  // ✅ FIXED: Changed quality type to match child components
  const [quality, setQuality] = useState<"low" | "medium" | "high">("medium");
  // ✅ FIXED: Changed sectionPlane type to match MeshModel
  const [sectionPlane, setSectionPlane] = useState<"xy" | "xz" | "yz" | null>(null);
  const [sectionPosition, setSectionPosition] = useState(0);

  // Visual quality settings for enhancements
  const [visualSettings, setVisualSettings] = useState<VisualQualitySettings>(DEFAULT_QUALITY_SETTINGS);

  // Camera transition state for declarative updates
  const [targetCameraPosition, setTargetCameraPosition] = useState<THREE.Vector3 | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);

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
              vertex_colors: data.vertex_colors as string[], // ✅ Cast Json to string[]
              triangle_count: data.triangle_count,
              face_types: data.face_types as string[],
              feature_edges: data.feature_edges as number[][][], // ✅ Cast Json to number[][][]
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
  }, [meshId, fileUrl]);

  // Bounding box calculation
  const boundingBox = useMemo(() => {
    if (!meshData) {
      return {
        min: new THREE.Vector3(),
        max: new THREE.Vector3(),
        center: [0, 0, 0],
        size: new THREE.Vector3(),
        width: 0,
        height: 0,
        depth: 0,
      };
    }

    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

    for (let i = 0; i < meshData.vertices.length; i += 3) {
      const x = meshData.vertices[i];
      const y = meshData.vertices[i + 1];
      const z = meshData.vertices[i + 2];

      min.x = Math.min(min.x, x);
      min.y = Math.min(min.y, y);
      min.z = Math.min(min.z, z);
      max.x = Math.max(max.x, x);
      max.y = Math.max(max.y, y);
      max.z = Math.max(max.z, z);
    }

    const center = [(min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2];
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

  // ✅ FIXED: Create modelBounds in correct format for LightingRig
  const modelBounds = useMemo(
    () => ({
      min: boundingBox.min,
      max: boundingBox.max,
      center: new THREE.Vector3(...boundingBox.center),
      size: boundingBox.size,
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

  // Camera view controls - DECLARATIVE approach
  const handleSetView = useCallback(
    (view: string) => {
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;

      let direction: THREE.Vector3;
      switch (view) {
        case "front":
          direction = new THREE.Vector3(0, 0, 1);
          break;
        case "back":
          direction = new THREE.Vector3(0, 0, -1);
          break;
        case "top":
          direction = new THREE.Vector3(0, 1, 0);
          break;
        case "bottom":
          direction = new THREE.Vector3(0, -1, 0);
          break;
        case "left":
          direction = new THREE.Vector3(-1, 0, 0);
          break;
        case "right":
          direction = new THREE.Vector3(1, 0, 0);
          break;
        case "isometric":
          direction = new THREE.Vector3(1, 1, 1).normalize();
          break;
        case "home":
          direction = new THREE.Vector3(1, 1, 1).normalize();
          break;
        default:
          return;
      }

      const newPosition = target.clone().add(direction.multiplyScalar(distance));
      setTargetCameraPosition(newPosition);
      setIsTransitioning(true);
    },
    [boundingBox],
  );

  const handleFitView = useCallback(() => {
    const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
    const distance = maxDim * 1.5;
    const target = new THREE.Vector3(...boundingBox.center);
    const currentDir =
      cameraRef.current?.position.clone().sub(target).normalize() || new THREE.Vector3(1, 1, 1).normalize();
    const newPosition = target.clone().add(currentDir.multiplyScalar(distance));
    setTargetCameraPosition(newPosition);
    setIsTransitioning(true);
  }, [boundingBox]);

  const handleCubeClick = useCallback(
    (direction: THREE.Vector3) => {
      const target = new THREE.Vector3(...boundingBox.center);
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 1.5;
      const newPosition = target.clone().add(direction.multiplyScalar(distance));
      setTargetCameraPosition(newPosition);
      setIsTransitioning(true);
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

  if (error) {
    return (
      <CardContent className="p-4">
        <div className="flex flex-col items-center justify-center h-full gap-4 text-destructive">
          <p className="font-semibold">Error Loading Model</p>
          <p className="text-sm">{error}</p>
          <Button variant="outline" onClick={() => window.location.reload()}>
            Retry
          </Button>
        </div>
      </CardContent>
    );
  }

  if (isLoading) {
    return (
      <CardContent className="p-4">
        <div className="flex flex-col items-center justify-center h-full gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading 3D model...</p>
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
              <fog 
                attach="fog" 
                args={[
                  "#f8f9fa", 
                  Math.max(
                    boundingBox.max[0] - boundingBox.min[0],
                    boundingBox.max[1] - boundingBox.min[1],
                    boundingBox.max[2] - boundingBox.min[2]
                  ) * 2,
                  Math.max(
                    boundingBox.max[0] - boundingBox.min[0],
                    boundingBox.max[1] - boundingBox.min[1],
                    boundingBox.max[2] - boundingBox.min[2]
                  ) * 8
                ]} 
              />

              <PerspectiveCamera ref={cameraRef} makeDefault position={initialCameraPosition} fov={50} />

              <CameraTransitionHandler
                targetPosition={targetCameraPosition}
                isTransitioning={isTransitioning}
                onTransitionComplete={() => {
                  setTargetCameraPosition(null);
                  setIsTransitioning(false);
                }}
                controlsRef={controlsRef}
                target={new THREE.Vector3(boundingBox.center[0], boundingBox.center[1], boundingBox.center[2])}
              />
              <Suspense fallback={null}>
                {/* DEBUGGING: Basic lighting without enhancements */}
                <ambientLight intensity={0.6} />
                <directionalLight position={[10, 10, 10]} intensity={0.8} castShadow />
                <directionalLight position={[-10, -10, -5]} intensity={0.3} />
                
                {/* ✅ EXPLICIT: Force topology colors ON for debugging */}
                <MeshModel
                  ref={meshRef}
                  meshData={meshData}
                  displayStyle={displayMode}
                  showEdges={showEdges}
                  sectionPlane={sectionPlane || "none"}
                  sectionPosition={sectionPosition}
                  topologyColors={true}
                />

                <DimensionAnnotations boundingBox={boundingBox} />

                <TrackballControls
                  ref={controlsRef}
                  makeDefault
                  target={new THREE.Vector3(boundingBox.center[0], boundingBox.center[1], boundingBox.center[2])}
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
