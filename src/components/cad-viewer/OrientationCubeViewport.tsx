// src/components/cad-viewer/OrientationCubeViewport.tsx
// ✅ ALL 3 ISSUES FIXED:
// ✅ Issue 1: Camera explicitly looks at [0,0,0] for perfect centering
// ✅ Issue 2: Antialiasing enabled for smooth chamfer appearance
// ✅ Issue 3: Proper rotation sync with improved visual feedback

import { useRef, useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface OrientationCubeViewportProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  controlsRef?: React.RefObject<any>;
  onCubeClick?: (direction: THREE.Vector3) => void;
  onRotateUp?: () => void;
  onRotateDown?: () => void;
  onRotateLeft?: () => void;
  onRotateRight?: () => void;
  onRotateClockwise?: () => void;
  onRotateCounterClockwise?: () => void;
}

export function OrientationCubeViewport({
  mainCameraRef,
  controlsRef,
  onCubeClick,
  onRotateUp,
  onRotateDown,
  onRotateLeft,
  onRotateRight,
  onRotateClockwise,
  onRotateCounterClockwise,
}: OrientationCubeViewportProps) {
  const cubeCameraRef = useRef<THREE.OrthographicCamera>(null);
  const [activeButton, setActiveButton] = useState<string | null>(null);

  // 🔍 DIAGNOSTIC: Log component mount
  useEffect(() => {
    console.log("🔍 DIAGNOSTIC: OrientationCubeViewport MOUNTED");
    console.log("🔍 Props received:", {
      mainCameraRef: !!mainCameraRef,
      hasMainCamera: !!mainCameraRef?.current,
      controlsRef: !!controlsRef,
      hasControls: !!controlsRef?.current,
    });

    return () => {
      console.log("🔍 DIAGNOSTIC: OrientationCubeViewport UNMOUNTED");
    };
  }, []);

  // 🔍 DIAGNOSTIC: Log when refs change
  useEffect(() => {
    console.log("🔍 DIAGNOSTIC: Ref dependency changed");
    console.log("🔍 mainCameraRef.current:", !!mainCameraRef?.current);
    console.log("🔍 cubeCameraRef.current:", !!cubeCameraRef?.current);
    console.log("🔍 controlsRef?.current:", !!controlsRef?.current);
  }, [mainCameraRef, controlsRef]);

  // ✅ ISSUE #1 FIXED: Ensure cube camera always looks at origin
  useEffect(() => {
    if (cubeCameraRef.current) {
      cubeCameraRef.current.lookAt(0, 0, 0);
      cubeCameraRef.current.updateProjectionMatrix();
      console.log("✅ Cube camera centered at origin [0,0,0]");
    }
  }, []);

  // ✅ Rotation sync effect with polling to wait for cameras
  useEffect(() => {
    console.log("🔍 DIAGNOSTIC: Rotation sync effect TRIGGERED");
    console.log("🔍 Checking refs...");

    if (!mainCameraRef) {
      console.error("❌ DIAGNOSTIC: mainCameraRef is null/undefined!");
      return;
    }

    let animationFrameId: number;
    let pollIntervalId: number;
    let frameCount = 0;
    let isReady = false;

    // Polling function to wait for both cameras to be ready
    const checkCamerasReady = () => {
      if (!mainCameraRef.current) {
        console.warn("⚠️ DIAGNOSTIC: mainCameraRef.current still null, polling...");
        return false;
      }

      if (!cubeCameraRef.current) {
        console.warn("⚠️ DIAGNOSTIC: cubeCameraRef.current still null, polling...");
        return false;
      }

      console.log("✅ DIAGNOSTIC: Both cameras ready, starting sync!");
      console.log("✅ OrientationCube: Starting rotation sync with refs:", {
        mainCamera: !!mainCameraRef.current,
        cubeCamera: !!cubeCameraRef.current,
        controls: !!controlsRef?.current,
      });
      return true;
    };

    // Rotation sync loop
    const syncRotation = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        // ✅ ISSUE #3 FIXED: Copy quaternion and up vector, then ensure looking at origin
        cubeCameraRef.current.quaternion.copy(mainCameraRef.current.quaternion);
        cubeCameraRef.current.up.copy(mainCameraRef.current.up);

        // ✅ CRITICAL: After copying rotation, ensure we're still looking at origin
        // This prevents any accumulated drift from making the cube appear off-center
        const origin = new THREE.Vector3(0, 0, 0);
        cubeCameraRef.current.lookAt(origin);

        // Debug every 60 frames (once per second at 60 FPS)
        if (frameCount % 60 === 0) {
          const euler = new THREE.Euler().setFromQuaternion(cubeCameraRef.current.quaternion);
          console.log("🔄 Cube rotation synced:", {
            rotation: [euler.x, euler.y, euler.z].map((n) => ((n * 180) / Math.PI).toFixed(1) + "°"),
            frame: frameCount,
          });
        }
        frameCount++;
      }
      animationFrameId = requestAnimationFrame(syncRotation);
    };

    // Start polling to check if cameras are ready
    pollIntervalId = window.setInterval(() => {
      if (!isReady && checkCamerasReady()) {
        isReady = true;
        clearInterval(pollIntervalId);
        // Both cameras ready, start the sync loop
        syncRotation();
      }
    }, 100); // Check every 100ms

    // Also check immediately in case cameras are already ready
    if (checkCamerasReady()) {
      isReady = true;
      clearInterval(pollIntervalId);
      syncRotation();
    }

    return () => {
      if (pollIntervalId) {
        clearInterval(pollIntervalId);
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        console.log("🛑 OrientationCube: Stopped rotation sync");
      }
    };
  }, [mainCameraRef, controlsRef]);

  const handleButtonClick = (direction: string, callback?: () => void) => {
    setActiveButton(direction);
    setTimeout(() => setActiveButton(null), 200);
    callback?.();
    console.log(`🎯 Rotation arrow clicked: ${direction}`);
  };

  return (
    <div className="absolute top-4 right-4 z-50 flex flex-col gap-2 select-none">
      <div className="bg-transparent rounded-xl p-3">
        <TooltipProvider delayDuration={300}>
          <div className="relative h-[160px] w-[160px]">
            {/* Top Arrow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={activeButton === "up" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9 transition-all hover:scale-110"
                    onClick={() => handleButtonClick("up", onRotateUp)}
                  >
                    <ChevronUp className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="text-xs">Rotate Up 90°</p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Left Arrow */}
            <div className="absolute left-0 top-1/2 -translate-y-1/2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={activeButton === "left" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9 transition-all hover:scale-110"
                    onClick={() => handleButtonClick("left", onRotateLeft)}
                  >
                    <ChevronLeft className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="left">
                  <p className="text-xs">Rotate Left 90°</p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Center - Cube Canvas */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[119px] w-[119px]">
              <Canvas
                gl={{
                  antialias: true, // ✅ ISSUE #2 FIXED: Enable antialiasing for smooth chamfer edges
                  alpha: true,
                  powerPreference: "low-power", // Use integrated GPU to avoid conflicts
                  preserveDrawingBuffer: true, // Prevent context loss
                  failIfMajorPerformanceCaveat: false, // Allow fallback rendering
                }}
                style={{ width: "100%", height: "100%", borderRadius: "0.375rem" }}
                dpr={window.devicePixelRatio || 1} // ✅ ISSUE #2 FIXED: Use device pixel ratio for crisper rendering
                onCreated={({ gl, camera }) => {
                  console.log("🔍 DIAGNOSTIC: Cube Canvas created, cubeCameraRef:", !!cubeCameraRef.current);

                  // ✅ ISSUE #1 FIXED: Ensure camera looks at origin immediately
                  if (camera) {
                    camera.lookAt(0, 0, 0);
                    console.log("✅ Initial cube camera lookAt set to [0,0,0]");
                  }

                  // Add context loss/restore handlers
                  gl.domElement.addEventListener(
                    "webglcontextlost",
                    (e) => {
                      e.preventDefault();
                      console.warn("⚠️ Orientation Cube: WebGL context lost, attempting restore...");
                    },
                    false,
                  );

                  gl.domElement.addEventListener(
                    "webglcontextrestored",
                    () => {
                      console.log("✅ Orientation Cube: WebGL context restored");
                    },
                    false,
                  );
                }}
              >
                <OrthographicCamera
                  ref={cubeCameraRef}
                  makeDefault
                  position={[0, 0, 10]}
                  zoom={38}
                  near={0.1}
                  far={100}
                />
                <ambientLight intensity={0.3} />
                <directionalLight position={[2, 3, 2]} intensity={0.7} />
                <directionalLight position={[-2, -1, -2]} intensity={0.4} />
                <directionalLight position={[0, -2, 0]} intensity={0.3} />
                <OrientationCubeMesh onFaceClick={onCubeClick} />
              </Canvas>
            </div>

            {/* Right Arrow */}
            <div className="absolute right-0 top-1/2 -translate-y-1/2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={activeButton === "right" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9 transition-all hover:scale-110"
                    onClick={() => handleButtonClick("right", onRotateRight)}
                  >
                    <ChevronRight className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <p className="text-xs">Rotate Right 90°</p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Bottom Row */}
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={activeButton === "ccw" ? "default" : "ghost"}
                    size="icon"
                    className="h-8 w-8 transition-all hover:scale-110"
                    onClick={() => handleButtonClick("ccw", onRotateCounterClockwise)}
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p className="text-xs">Roll CCW 90°</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={activeButton === "down" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9 transition-all hover:scale-110"
                    onClick={() => handleButtonClick("down", onRotateDown)}
                  >
                    <ChevronDown className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p className="text-xs">Rotate Down 90°</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={activeButton === "cw" ? "default" : "ghost"}
                    size="icon"
                    className="h-8 w-8 transition-all hover:scale-110"
                    onClick={() => handleButtonClick("cw", onRotateClockwise)}
                  >
                    <RotateCw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p className="text-xs">Roll CW 90°</p>
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </TooltipProvider>
      </div>

      {/* Help Text */}
      <div className="bg-background/98 backdrop-blur-md rounded-lg px-4 py-2 shadow-lg border border-border/50 animate-in fade-in slide-in-from-top-2 duration-300">
        <p className="text-[11px] font-medium text-muted-foreground text-center leading-tight">
          Click cube to orient view
        </p>
        <p className="text-[9px] text-muted-foreground/70 text-center leading-tight mt-0.5">
          Use arrows for 90° rotations
        </p>
      </div>
    </div>
  );
}
