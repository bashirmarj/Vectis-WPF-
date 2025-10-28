// src/components/cad-viewer/OrientationCubeViewport.tsx
// ✅ FIXED: Both horizontal AND vertical rotation account for camera orientation

import { useRef, useEffect, useState, useCallback } from "react";
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

function CubeSyncWrapper({
  mainCameraRef,
  controlsRef,
  onCubeClick,
}: {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  controlsRef?: React.RefObject<any>;
  onCubeClick?: (direction: THREE.Vector3) => void;
}) {
  const cubeGroupRef = useRef<THREE.Group>(null);

  // ✅ Rotation sync effect
  useEffect(() => {
    if (!mainCameraRef?.current || !cubeGroupRef.current) return;

    let animationFrameId: number;
    let frameCount = 0;

    const syncRotation = () => {
      if (mainCameraRef.current && cubeGroupRef.current) {
        cubeGroupRef.current.quaternion.copy(mainCameraRef.current.quaternion);

        if (frameCount % 60 === 0) {
          console.log("🔄 Cube mesh synced with camera orientation");
        }
        frameCount++;
      }
      animationFrameId = requestAnimationFrame(syncRotation);
    };

    syncRotation();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [mainCameraRef]);

  // ✅ FIXED: Handle drag-to-rotate with BOTH horizontal AND vertical inversion
  const handleDragRotate = useCallback(
    (deltaX: number, deltaY: number) => {
      if (!mainCameraRef.current || !controlsRef?.current) return;

      const camera = mainCameraRef.current;
      const controls = controlsRef.current;
      const target = controls.target.clone();

      // ✅ CRITICAL FIX: Detect if camera is upside down
      const isUpsideDown = camera.up.y < 0;

      const rotationSpeed = 0.005;

      // ✅ FIXED: Invert BOTH horizontal and vertical when upside down
      const horizontalMultiplier = isUpsideDown ? 1 : -1;
      const verticalMultiplier = isUpsideDown ? 1 : -1;

      const deltaAzimuth = deltaX * rotationSpeed * horizontalMultiplier;
      const deltaPolar = deltaY * rotationSpeed * verticalMultiplier;

      const currentPosition = camera.position.clone();
      const currentUp = camera.up.clone();

      // Step 1: Horizontal rotation (around world Y-axis)
      const worldYAxis = new THREE.Vector3(0, 1, 0);
      let newPosition = currentPosition.clone().sub(target);
      newPosition.applyAxisAngle(worldYAxis, deltaAzimuth);
      newPosition.add(target);

      let newUp = currentUp.clone().applyAxisAngle(worldYAxis, deltaAzimuth);

      // Step 2: Vertical rotation (around camera's right vector)
      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);

      const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      newPosition = camera.position.clone().sub(target);
      newPosition.applyAxisAngle(cameraRight, deltaPolar);
      newPosition.add(target);

      newUp = camera.up.clone().applyAxisAngle(cameraRight, deltaPolar);

      camera.position.copy(newPosition);
      camera.up.copy(newUp);
      camera.lookAt(target);
      controls.target.copy(target);
      controls.update();

      // Debug logging
      if ((Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5) && Math.random() < 0.1) {
        console.log("🔄 Drag rotation:", {
          isUpsideDown,
          upY: camera.up.y.toFixed(3),
          deltaX,
          deltaY,
          deltaAzimuth: deltaAzimuth.toFixed(4),
          deltaPolar: deltaPolar.toFixed(4),
        });
      }
    },
    [mainCameraRef, controlsRef],
  );

  return (
    <group ref={cubeGroupRef} position={[0, 0, 0]} rotation={[0, 0, 0]}>
      <OrientationCubeMesh groupRef={cubeGroupRef} onFaceClick={onCubeClick} onDragRotate={handleDragRotate} />
    </group>
  );
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
  const [activeButton, setActiveButton] = useState<string | null>(null);

  const handleButtonClick = (direction: string, callback?: () => void) => {
    setActiveButton(direction);
    setTimeout(() => setActiveButton(null), 200);
    if (callback) {
      callback();
    }
  };

  return (
    <div className="absolute top-4 right-4 flex flex-col gap-2 select-none" style={{ zIndex: 40 }}>
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

            {/* Center Canvas with Cube */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[119px] w-[119px]">
              <Canvas
                gl={{
                  antialias: true,
                  alpha: true,
                  powerPreference: "low-power",
                  preserveDrawingBuffer: true,
                }}
                style={{ width: "100%", height: "100%", borderRadius: "0.375rem" }}
                dpr={window.devicePixelRatio || 1}
              >
                <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={38} near={0.1} far={100} />

                <ambientLight intensity={0.3} />
                <directionalLight position={[2, 3, 2]} intensity={0.7} />
                <directionalLight position={[-2, -1, -2]} intensity={0.4} />
                <directionalLight position={[0, -2, 0]} intensity={0.3} />

                <CubeSyncWrapper mainCameraRef={mainCameraRef} controlsRef={controlsRef} onCubeClick={onCubeClick} />
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

            {/* Bottom Arrows */}
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

      <div className="bg-background/98 backdrop-blur-md rounded-lg px-4 py-2 shadow-lg border border-border/50 animate-in fade-in slide-in-from-top-2 duration-300">
        <p className="text-[11px] font-medium text-muted-foreground text-center leading-tight">
          Click cube to orient view
        </p>
        <p className="text-[9px] text-muted-foreground/70 text-center leading-tight mt-0.5">
          Drag to rotate • Use arrows for 90°
        </p>
      </div>
    </div>
  );
}
