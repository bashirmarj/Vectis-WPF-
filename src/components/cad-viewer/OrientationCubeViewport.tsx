// src/components/cad-viewer/OrientationCubeViewport.tsx
// Professional Orientation Cube Viewport - Industry Standard Pattern
// ✅ ISSUE #2 FIXED: Proper viewing direction sync (not quaternion copy)
// Separate Canvas with orthographic camera for clean rotation representation

import { useRef, useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrthographicCamera, Environment } from "@react-three/drei";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface OrientationCubeViewportProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  controlsRef?: React.RefObject<any>; // TrackballControls ref
  onCubeClick?: (direction: THREE.Vector3) => void;
  onRotateUp?: () => void;
  onRotateDown?: () => void;
  onRotateLeft?: () => void;
  onRotateRight?: () => void;
  onRotateClockwise?: () => void;
  onRotateCounterClockwise?: () => void;
}

/**
 * Professional Orientation Cube Viewport
 *
 * Architecture Pattern:
 * ✅ Separate Canvas with own orthographic camera
 * ✅ CSS-positioned fixed overlay (top-right corner)
 * ✅ FIXED: Proper viewing direction sync (shows what camera is looking at)
 * ✅ No complex coordinate transforms
 *
 * This matches industry implementations:
 * - Autodesk Fusion 360
 * - PTC Onshape
 * - Dassault SolidWorks
 * - Siemens NX
 */
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
  const cubeViewportRef = useRef<HTMLDivElement>(null);
  const cubeCameraRef = useRef<THREE.OrthographicCamera>(null);
  const [activeButton, setActiveButton] = useState<string | null>(null);

  // ✅ ISSUE #2 FIXED: Real-time viewing direction sync (60 FPS)
  // Instead of copying quaternion, we calculate the viewing direction
  // and make the cube camera look FROM that direction (inverse)
  useEffect(() => {
    if (!mainCameraRef.current || !cubeCameraRef.current) {
      console.warn("⚠️ OrientationCube: Camera refs not ready");
      return;
    }

    let animationFrameId: number;
    let frameCount = 0;

    const syncRotation = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        // Get the viewing direction from main camera
        // If we have controls ref, use its target, otherwise calculate from camera
        let target = new THREE.Vector3();

        if (controlsRef?.current?.target) {
          target.copy(controlsRef.current.target);
        } else {
          // Calculate target from camera's forward direction
          const forward = new THREE.Vector3(0, 0, -1);
          forward.applyQuaternion(mainCameraRef.current.quaternion);
          target.copy(mainCameraRef.current.position).add(forward);
        }

        // Calculate viewing direction (from camera to target)
        const viewDirection = new THREE.Vector3().subVectors(target, mainCameraRef.current.position).normalize();

        // The cube camera should look FROM the viewing direction
        // So we position it opposite to where the main camera is looking
        // Cube camera is at [0, 0, 10], we rotate it to look at -viewDirection
        const cubeTarget = new THREE.Vector3().copy(viewDirection).multiplyScalar(-10);

        cubeCameraRef.current.lookAt(cubeTarget);

        // Also copy the up vector to match camera tilt
        cubeCameraRef.current.up.copy(mainCameraRef.current.up);

        // Debug every 60 frames (once per second at 60 FPS)
        if (frameCount % 60 === 0) {
          const euler = new THREE.Euler().setFromQuaternion(cubeCameraRef.current.quaternion);
          console.log("🔄 Cube viewing direction synced:", {
            viewDir: {
              x: viewDirection.x.toFixed(2),
              y: viewDirection.y.toFixed(2),
              z: viewDirection.z.toFixed(2),
            },
            cubeRotation: [euler.x, euler.y, euler.z].map((n) => ((n * 180) / Math.PI).toFixed(1) + "°"),
          });
        }
        frameCount++;
      }
      animationFrameId = requestAnimationFrame(syncRotation);
    };

    console.log("✅ OrientationCube: Starting viewing direction sync");
    syncRotation();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        console.log("🛑 OrientationCube: Stopped viewing direction sync");
      }
    };
  }, [mainCameraRef, controlsRef]);

  // Handle button clicks with visual feedback
  const handleButtonClick = (direction: string, callback?: () => void) => {
    setActiveButton(direction);
    setTimeout(() => setActiveButton(null), 200); // Visual feedback duration
    callback?.();
    console.log(`🎯 Rotation arrow clicked: ${direction}`);
  };

  return (
    <div className="absolute top-4 right-4 z-50 flex flex-col gap-2 select-none">
      {/* Rotation Arrow Controls */}
      <div className="bg-transparent rounded-xl p-3">
        <TooltipProvider delayDuration={300}>
          <div className="relative h-[160px] w-[160px]">
            {/* Top Arrow - Edge positioned */}
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

            {/* Left Arrow - Edge positioned */}
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

            {/* Center - Large Cube Viewport */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[119px] w-[119px]">
              <Canvas
                gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
                style={{ width: "100%", height: "100%", borderRadius: "0.375rem" }}
                dpr={[1, 2]}
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

            {/* Right Arrow - Edge positioned */}
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

            {/* Bottom Row - Edge positioned flex group */}
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

      {/* Help Text with Animation */}
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
