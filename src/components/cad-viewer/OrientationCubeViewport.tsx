// src/components/cad-viewer/OrientationCubeViewport.tsx
// Professional Orientation Cube Viewport - Industry Standard Pattern
// Separate Canvas with orthographic camera for clean rotation sync

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
 * ✅ Simple rotation sync via quaternion copy (60 FPS)
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

  // ✅ Real-time rotation sync with main camera (60 FPS)
  useEffect(() => {
    if (!mainCameraRef.current || !cubeCameraRef.current) {
      console.warn("⚠️ OrientationCube: Camera refs not ready");
      return;
    }

    let animationFrameId: number;
    let frameCount = 0;

    const syncRotation = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        // ✅ CRITICAL: Single line for rotation sync
        // Copy quaternion from main camera to cube camera
        cubeCameraRef.current.quaternion.copy(mainCameraRef.current.quaternion);

        // Debug every 60 frames (1 second at 60 FPS)
        if (frameCount % 60 === 0) {
          console.log("🔄 Cube rotation synced:", {
            mainQuat: mainCameraRef.current.quaternion.toArray().map((n) => n.toFixed(3)),
            cubeQuat: cubeCameraRef.current.quaternion.toArray().map((n) => n.toFixed(3)),
          });
        }
        frameCount++;
      }
      animationFrameId = requestAnimationFrame(syncRotation);
    };

    console.log("✅ OrientationCube: Starting rotation sync");
    syncRotation();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        console.log("🛑 OrientationCube: Stopped rotation sync");
      }
    };
  }, [mainCameraRef]);

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
      <div className="bg-background/98 backdrop-blur-md rounded-xl p-3 shadow-2xl border-2 border-border/50 hover:border-border/70 transition-all">
        <TooltipProvider delayDuration={300}>
          <div className="grid grid-cols-3 gap-1.5">
            {/* Top Row */}
            <div />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={activeButton === "up" ? "default" : "ghost"}
                  size="icon"
                  className="h-9 w-9 transition-all hover:scale-110"
                  onClick={() => handleButtonClick("up", onRotateUp)}
                >
                  <ChevronUp className="h-15 w-15" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p className="text-xs">Rotate Up 90°</p>
              </TooltipContent>
            </Tooltip>
            <div />

            {/* Middle Row */}
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

            {/* Center - Cube viewport in the grid */}
            <div className="h-14 w-14 relative -m-1">
              <Canvas
                gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
                style={{ width: "100%", height: "100%", borderRadius: "0.375rem" }}
                dpr={[1, 2]}
              >
                <OrthographicCamera
                  ref={cubeCameraRef}
                  makeDefault
                  position={[0, 0, 10]}
                  zoom={35}
                  near={0.1}
                  far={100}
                />
                <color attach="background" args={["#f8fafc"]} />
                <ambientLight intensity={0.3} />
                <directionalLight position={[2, 3, 2]} intensity={0.7} />
                <directionalLight position={[-2, -1, -2]} intensity={0.4} />
                <directionalLight position={[0, -2, 0]} intensity={0.3} />
                <OrientationCubeMesh onFaceClick={onCubeClick} />
              </Canvas>
            </div>

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

            {/* Bottom Row */}
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
