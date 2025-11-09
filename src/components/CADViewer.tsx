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
import { MeshModel, type MeshModelHandle } from "./cad-viewer/MeshModel";
import { DimensionAnnotations } from "./cad-viewer/DimensionAnnotations";
import { OrientationCubeViewport } from "./cad-viewer/OrientationCubeViewport";
import { AxisTriadInCanvas } from "./cad-viewer/AxisTriadInCanvas";
import { ProfessionalLighting } from "./cad-viewer/enhancements/ProfessionalLighting";
import { UnifiedCADToolbar } from "./cad-viewer/UnifiedCADToolbar";
import { ProfessionalMeasurementTool } from "./cad-viewer/ProfessionalMeasurementTool";
import { MeasurementRenderer } from "./cad-viewer/MeasurementRenderer";
import { MeasurementPanel } from "./cad-viewer/MeasurementPanel";
import { FaceMeasurementTool } from "./cad-viewer/FaceMeasurementTool";
import { FaceMeasurementPanel } from "./cad-viewer/measurements/FaceMeasurementPanel";
import { type MarkerValues } from "@/lib/faceMeasurementUtils";
import { useMeasurementStore } from "@/stores/measurementStore";

interface CADViewerProps {
  meshData?: MeshData; // ✅ Accept mesh data directly instead of meshId
  fileUrl?: string;
  fileName?: string;
  isSidebarCollapsed?: boolean;
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
  edge_classifications?: Array<{
    id: number;
    type: 'line' | 'circle' | 'arc';
    start_point: [number, number, number];
    end_point: [number, number, number];
    diameter?: number;
    radius?: number;
    length?: number;
    center?: [number, number, number];
    segment_count: number;
  }>;
  tagged_edges?: Array<{
    feature_id: number;
    start: [number, number, number];
    end: [number, number, number];
    type: 'line' | 'circle' | 'arc';
    diameter?: number;
    radius?: number;
    length?: number;
  }>;
  vertex_face_ids?: number[];
  face_classifications?: Array<{
    face_id: number;
    type: "internal" | "external" | "through" | "planar";
    center: [number, number, number];
    normal: [number, number, number];
    area: number;
    surface_type: "cylinder" | "plane" | "other";
    radius?: number;
  }>;
  ml_features?: {
    instances?: any[];
    [key: string]: any;
  };
}

export function CADViewer({ meshData: propMeshData, fileUrl, fileName, isSidebarCollapsed = false, onMeshLoaded }: CADViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [meshData, setMeshData] = useState<MeshData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<"solid" | "wireframe" | "translucent">("solid");
  const [showSolidEdges, setShowSolidEdges] = useState(true);
  const [showWireframeHiddenEdges, setShowWireframeHiddenEdges] = useState(false);
  
  const [shadowsEnabled, setShadowsEnabled] = useState(true);
  const [ssaoEnabled, setSSAOEnabled] = useState(false);
  const [sectionPlane, setSectionPlane] = useState<"xy" | "xz" | "yz" | null>(null);
  const [sectionPosition, setSectionPosition] = useState(0);

  const { activeTool, setActiveTool, clearAllMeasurements, measurements } = useMeasurementStore();
  const [faceMeasurements, setFaceMeasurements] = useState<MarkerValues | null>(null);
  const [faceMarkerCount, setFaceMarkerCount] = useState(0);
  const [faceResetTrigger, setFaceResetTrigger] = useState(0);
  
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<any>(null);
  const meshRef = useRef<MeshModelHandle>(null);

  // File extension detection
  const fileExtension = useMemo(() => {
    if (fileName) return fileName.split(".").pop()?.toLowerCase() || "";
    if (fileUrl) return fileUrl.split(".").pop()?.toLowerCase().split("?")[0] || "";
    return "";
  }, [fileName, fileUrl]);

  const isRenderableFormat = useMemo(() => {
    return ["step", "stp", "iges", "igs", "stl"].includes(fileExtension);
  }, [fileExtension]);

  // ✅ Use mesh data directly from props (no database fetch)
  useEffect(() => {
    if (!isRenderableFormat) {
      setIsLoading(false);
      return;
    }

    if (propMeshData) {
      console.log("✅ Using mesh data from props (no caching):", {
        hasVertices: !!propMeshData.vertices,
        hasIndices: !!propMeshData.indices,
        hasNormals: !!propMeshData.normals,
        triangleCount: propMeshData.triangle_count,
      });
      
      // 🔍 DIAGNOSTIC: Check edge and feature data
      const mlFeatures = propMeshData?.ml_features;
      const featureInstances = mlFeatures?.feature_instances;
      
      console.log("🔍 Backend mesh data received:", {
        hasTaggedEdges: !!propMeshData?.tagged_edges,
        taggedEdgesCount: propMeshData?.tagged_edges?.length || 0,
        taggedEdgesSample: propMeshData?.tagged_edges?.[0],
        
        hasMLFeatures: !!mlFeatures,
        mlFeaturesKeys: mlFeatures ? Object.keys(mlFeatures) : [],
        mlInstancesCount: Array.isArray(featureInstances) ? featureInstances.length : 0,
        
        hasEdgeClassifications: !!propMeshData?.edge_classifications,
        edgeClassificationsCount: propMeshData?.edge_classifications?.length || 0,
        
        allKeys: propMeshData ? Object.keys(propMeshData) : []
      });
      
      setMeshData(propMeshData);
      onMeshLoaded?.(propMeshData);
      setIsLoading(false);
    } else {
      setIsLoading(false);
    }
  }, [propMeshData, isRenderableFormat, onMeshLoaded]);

  // Calculate bounding box
  const boundingBox = useMemo(() => {
    if (!meshData || !meshData.vertices || meshData.vertices.length === 0) {
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

  // Handle orientation cube clicks - directly use direction vector for camera positioning
  const handleCubeClick = useCallback(
    (direction: THREE.Vector3) => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const target = boundingBox.center;
      const maxDim = Math.max(boundingBox.width, boundingBox.height, boundingBox.depth);
      const distance = maxDim * 2;

      // Determine appropriate up vector based on click
      let up = new THREE.Vector3(0, 1, 0);

      // Special case: top/bottom views need different up vectors
      if (Math.abs(direction.y) > 0.9) {
        up =
          direction.y > 0
            ? new THREE.Vector3(0, 0, -1) // Top view
            : new THREE.Vector3(0, 0, 1); // Bottom view
      }

      // Position camera in the direction we're looking FROM
      const newPosition = target.clone().add(direction.clone().multiplyScalar(distance));

      camera.position.copy(newPosition);
      camera.up.copy(up);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      console.log("📍 Camera moved to:", {
        direction: direction.toArray(),
        position: newPosition.toArray(),
      });
    },
    [boundingBox],
  );

  // ✅ Throttled camera rotation for performance
  const lastRotateTime = useRef(0);
  const THROTTLE_MS = 16; // 60fps
  
  const handleRotateCamera = useCallback(
    (direction: "up" | "down" | "left" | "right" | "cw" | "ccw") => {
      // Throttle camera updates
      const now = performance.now();
      if (now - lastRotateTime.current < THROTTLE_MS) return;
      lastRotateTime.current = now;
      
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

  // Handle new face measurement
  const handleNewFaceMeasurement = useCallback(() => {
    setFaceResetTrigger(prev => prev + 1);
  }, []);

  // Handle face measurements change (stable reference)
  const handleFaceMeasurementsChange = useCallback((measurements: MarkerValues | null, count: number) => {
    setFaceMeasurements(measurements);
    setFaceMarkerCount(count);
  }, []);

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
          if (displayMode === "wireframe") {
            setShowWireframeHiddenEdges((prev) => !prev);
          } else {
            setShowSolidEdges((prev) => !prev);
          }
          break;
        case "m":
          setActiveTool(activeTool ? null : "edge-select");
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
              showEdges={displayMode === "wireframe" ? showWireframeHiddenEdges : showSolidEdges}
              onToggleEdges={() => {
                if (displayMode === "wireframe") {
                  setShowWireframeHiddenEdges((prev) => !prev);
                } else {
                  setShowSolidEdges((prev) => !prev);
                }
              }}
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
              style={{ width: '100%', height: '100%' }}
              shadows
              gl={{
                antialias: true,
                alpha: true,
                preserveDrawingBuffer: true,
                powerPreference: "high-performance",
              }}
              onCreated={({ gl }) => {
                const handleContextLost = (e: Event) => {
                  e.preventDefault();
                  console.warn("⚠️ WebGL context lost - browser will auto-recover");
                };

                const handleContextRestored = () => {
                  console.log("✅ WebGL context restored automatically");
                };

                gl.domElement.addEventListener("webglcontextlost", handleContextLost, false);
                gl.domElement.addEventListener("webglcontextrestored", handleContextRestored, false);
              }}
            >
              <color attach="background" args={["#f8f9fa"]} />

              <PerspectiveCamera ref={cameraRef} makeDefault position={initialCameraPosition} fov={50} />

              <Suspense fallback={null}>
                <ProfessionalLighting intensity={0.85} enableShadows={shadowsEnabled} shadowQuality="high" />

              <MeshModel
                ref={meshRef}
                meshData={meshData}
                displayStyle={displayMode}
                showEdges={showSolidEdges}
                showHiddenEdges={showWireframeHiddenEdges}
                sectionPlane={sectionPlane || "none"}
                sectionPosition={sectionPosition}
                useSilhouetteEdges={displayMode === "wireframe"}
                controlsRef={controlsRef}
              />

                <DimensionAnnotations boundingBox={boundingBox} />

                {/* Professional Measurement Tool (Edge & Face-to-Face) */}
                <ProfessionalMeasurementTool 
                  meshData={meshData} 
                  meshRef={meshRef.current?.mesh || null} 
                  featureEdgesGeometry={meshRef.current?.featureEdgesGeometry || null}
                  enabled={activeTool === 'edge-select' || activeTool === 'face-to-face'} 
                />

                {/* Face Measurement Tool (Reference Implementation) */}
                <FaceMeasurementTool 
                  enabled={activeTool === 'measure'}
                  meshRef={meshRef.current?.mesh || null}
                  boundingSphere={{
                    center: boundingBox.center,
                    radius: Math.max(boundingBox.width, boundingBox.height, boundingBox.depth) / 2
                  }}
                  onMeasurementsChange={handleFaceMeasurementsChange}
                  resetTrigger={faceResetTrigger}
                />

                {/* Measurement Renderer */}
                <MeasurementRenderer />

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

            {/* ✅ Axis Triad - SolidWorks-style XYZ indicator (bottom-left) */}
            <AxisTriadInCanvas mainCameraRef={cameraRef} isSidebarCollapsed={isSidebarCollapsed} />

            {/* ✅ Orientation cube in separate Canvas (no WebGL conflicts) */}
            <OrientationCubeViewport
              mainCameraRef={cameraRef}
              controlsRef={controlsRef}
              onCubeClick={handleCubeClick}
              onRotateUp={() => handleRotateCamera("up")}
              onRotateDown={() => handleRotateCamera("down")}
              onRotateLeft={() => handleRotateCamera("left")}
              onRotateRight={() => handleRotateCamera("right")}
              onRotateClockwise={() => handleRotateCamera("cw")}
              onRotateCounterClockwise={() => handleRotateCamera("ccw")}
            />

            {/* ✅ Face Measurement Panel (outside Canvas) */}
            {activeTool === 'measure' && (
              <FaceMeasurementPanel 
                markerCount={faceMarkerCount}
                measurements={faceMeasurements}
                onNewMeasurement={handleNewFaceMeasurement}
              />
            )}

            {/* ✅ Measurement Panel - Shows measurement list and controls */}
            <MeasurementPanel />
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
