// src/components/cad-viewer/OrientationCubeViewport.tsx
// ✅ FIXED: Removed lookAt from sync loop to allow proper 3D rotation
// ✅ FIXED: Only lookAt on initial setup, not during every frame

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

  // ✅ ISSUE #1 FIXED: Only set initial lookAt once, don't override quaternion sync
  useEffect(() => {
    if (cubeCameraRef.current) {
      cubeCameraRef.current.lookAt(0, 0, 0);
      cubeCameraRef.current.updateProjectionMatrix();
      console.log("✅ Cube camera initial setup: looking at origin [0,0,0]");
    }
  }, []);

  // ✅ Rotation sync effect with polling to wait for cameras
  useEffect(() => {
    if (!mainCameraRef) {
      console.error("❌ mainCameraRef is null/undefined!");
      return;
    }

    let animationFrameId: number;
    let pollIntervalId: number;
    let frameCount = 0;
    let isReady = false;

    const checkCamerasReady = () => {
      if (!mainCameraRef.current || !cubeCameraRef.current) {
        return false;
      }
      console.log("✅ Both cameras ready, starting sync!");
      return true;
    };

    // ✅ ISSUE #1 FIXED: Rotation sync without lookAt override
    const syncRotation = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        // Copy quaternion and up vector - this allows full 3D rotation
        cubeCameraRef.current.quaternion.copy(mainCameraRef.current.quaternion);
        cubeCameraRef.current.up.copy(mainCameraRef.current.up);

        // ❌ REMOVED: lookAt call that was preventing 3D rotation
        // The quaternion already contains the full rotation information

        // Debug every 60 frames
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

    pollIntervalId = window.setInterval(() => {
      if (!isReady && checkCamerasReady()) {
        isReady = true;
        clearInterval(pollIntervalId);
        syncRotation();
      }
    }, 100);

    if (checkCamerasReady()) {
      isReady = true;
      clearInterval(pollIntervalId);
      syncRotation();
    }

    return () => {
      if (pollIntervalId) clearInterval(pollIntervalId);
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
                  antialias: true,
                  alpha: true,
                  powerPreference: "low-power",
                  preserveDrawingBuffer: true,
                  failIfMajorPerformanceCaveat: false,
                }}
                style={{ width: "100%", height: "100%", borderRadius: "0.375rem" }}
                dpr={window.devicePixelRatio || 1}
                onCreated={({ gl, camera }) => {
                  if (camera) {
                    camera.lookAt(0, 0, 0);
                    console.log("✅ Cube Canvas created with camera looking at origin");
                  }

                  gl.domElement.addEventListener(
                    "webglcontextlost",
                    (e) => {
                      e.preventDefault();
                      console.warn("⚠️ Orientation Cube: WebGL context lost");
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
