// src/components/cad-viewer/OrientationCubeViewport.tsx
// ✅ FIXED: Inverted camera quaternion for correct part-relative orientation
// The cube should show the part's orientation relative to the viewer,
// not the camera's orientation in world space

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
  controlsRef?: React.ReffObject<any>;
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

  // ✅ CRITICAL FIX: Use INVERTED camera quaternion
  // In CAD viewers, the cube shows how you're looking AT the part,
  // not the camera's orientation itself
  useEffect(() => {
    let animationFrameId: number;
    let retryCount = 0;
    const maxRetries = 10;

    const checkAndSync = () => {
      if (!mainCameraRef?.current || !cubeGroupRef.current) {
        if (retryCount < maxRetries) {
          retryCount++;
          setTimeout(() => {
            animationFrameId = requestAnimationFrame(checkAndSync);
          }, 100);
        } else {
          console.warn("⚠️ Rotation sync: Missing refs after retries", {
            hasMainCamera: !!mainCameraRef?.current,
            hasCubeGroup: !!cubeGroupRef.current,
          });
        }
        return;
      }

      console.log("✅ Rotation sync initialized with INVERTED quaternion");

      const syncRotation = () => {
        if (mainCameraRef.current && cubeGroupRef.current) {
          // ✅ KEY FIX: Use inverse() to show part-relative orientation
          // This makes the cube show the part's coordinate system as seen from the camera,
          // not the camera's coordinate system
          const invertedQuaternion = mainCameraRef.current.quaternion.clone().invert();
          cubeGroupRef.current.quaternion.copy(invertedQuaternion);
        }
        animationFrameId = requestAnimationFrame(syncRotation);
      };

      syncRotation();
    };

    checkAndSync();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [mainCameraRef]);

  // ✅ Handle drag-to-rotate with orientation awareness
  const handleDragRotate = useCallback(
    (deltaX: number, deltaY: number) => {
      if (!mainCameraRef.current || !controlsRef?.current) return;

      const camera = mainCameraRef.current;
      const controls = controlsRef.current;
      const target = controls.target.clone();

      // Detect if camera is upside down
      const isUpsideDown = camera.up.y < 0;

      const rotationSpeed = 0.005;

      // Invert both horizontal and vertical when upside down
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
    },
    [mainCameraRef, controlsRef],
  );

  return (
    <group>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <directionalLight position={[-3, -3, -3]} intensity={0.3} />

      <group ref={cubeGroupRef} scale={0.37}>
        <OrientationCubeMesh groupRef={cubeGroupRef} onFaceClick={onCubeClick} onDragRotate={handleDragRotate} />
      </group>
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
    <div
      className="absolute top-4 right-4 flex flex-col gap-2 select-none"
      style={{ zIndex: 40, pointerEvents: "auto" }}
      onPointerMove={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
      onPointerUp={(e) => e.stopPropagation()}
    >
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
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[119px] w-[119px]"
              onPointerMove={(e) => e.stopPropagation()}
              onPointerDown={(e) => e.stopPropagation()}
              onPointerUp={(e) => e.stopPropagation()}
              onPointerEnter={(e) => e.stopPropagation()}
              onPointerLeave={(e) => e.stopPropagation()}
              style={{ pointerEvents: "auto" }}
            >
              <Canvas
                gl={{
                  antialias: true,
                  alpha: true,
                  powerPreference: "high-performance",
                  preserveDrawingBuffer: true,
                  failIfMajorPerformanceCaveat: false,
                }}
                style={{ width: "100%", height: "100%", borderRadius: "0.375rem" }}
                dpr={Math.min(window.devicePixelRatio || 1, 2)}
                onCreated={({ gl }) => {
                  gl.domElement.addEventListener(
                    "webglcontextlost",
                    (e) => {
                      e.preventDefault();
                      console.warn("⚠️ Orientation cube: WebGL context lost");
                    },
                    false,
                  );
                  gl.domElement.addEventListener(
                    "webglcontextrestored",
                    () => {
                      console.log("✅ Orientation cube: WebGL context restored");
                    },
                    false,
                  );
                }}
              >
                <color attach="background" args={["#f8f9fa"]} />
                <OrthographicCamera position={[0, 0, 10]} zoom={127} near={0.1} far={1000} makeDefault />
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

            {/* Bottom Arrow */}
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2">
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
            </div>

            {/* Clockwise Rotation */}
            <div className="absolute bottom-2 right-2">
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
                  <p className="text-xs">Roll Clockwise 90°</p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Counter-Clockwise Rotation */}
            <div className="absolute bottom-2 left-2">
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
                  <p className="text-xs">Roll Counter-Clockwise 90°</p>
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </TooltipProvider>
      </div>
    </div>
  );
}
