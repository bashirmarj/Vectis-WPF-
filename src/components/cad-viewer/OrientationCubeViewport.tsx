// src/components/cad-viewer/OrientationCubeViewport.tsx
// Industry-standard orientation cube using separate viewport pattern
// Based on Onshape/Fusion 360/SolidWorks implementations

import { useRef, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";
import { 
  ChevronUp, 
  ChevronDown, 
  ChevronLeft, 
  ChevronRight, 
  RotateCw, 
  RotateCcw 
} from "lucide-react";
import { Button } from "@/components/ui/button";

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
 * OrientationCubeViewport - Professional CAD-style orientation indicator
 * 
 * Architecture:
 * - Separate Canvas with own orthographic camera (no perspective distortion)
 * - CSS-positioned overlay in top-right corner
 * - Simple rotation sync via quaternion copy
 * - No complex 3D math or coordinate transforms needed
 * 
 * This matches industry standard implementations in:
 * - Autodesk Fusion 360
 * - Onshape
 * - SolidWorks Web
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

  // ✅ Sync cube rotation with main camera - runs every frame
  useEffect(() => {
    if (!mainCameraRef.current || !cubeCameraRef.current) return;

    let animationFrameId: number;

    const syncRotation = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        // Copy main camera's rotation to cube camera
        // This is the ONLY line needed for rotation sync
        cubeCameraRef.current.quaternion.copy(mainCameraRef.current.quaternion);
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

  return (
    <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
      {/* Rotation Controls */}
      <div className="bg-background/95 backdrop-blur-sm rounded-lg p-2 shadow-xl border-2 border-border/50">
        <div className="grid grid-cols-3 gap-1">
          {/* Top Row */}
          <div />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onRotateUp}
            title="Rotate Up (90°)"
          >
            <ChevronUp className="h-4 w-4" />
          </Button>
          <div />

          {/* Middle Row */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onRotateLeft}
            title="Rotate Left (90°)"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <div className="h-8 w-8" /> {/* Center spacer */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onRotateRight}
            title="Rotate Right (90°)"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          {/* Bottom Row */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-xs"
            onClick={onRotateCounterClockwise}
            title="Roll Counter-Clockwise (90°)"
          >
            <RotateCcw className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onRotateDown}
            title="Rotate Down (90°)"
          >
            <ChevronDown className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-xs"
            onClick={onRotateClockwise}
            title="Roll Clockwise (90°)"
          >
            <RotateCw className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Cube Viewport - Separate Canvas */}
      <div
        ref={cubeViewportRef}
        className="bg-background/95 backdrop-blur-sm rounded-lg shadow-xl border-2 border-border/50 overflow-hidden"
        style={{
          width: "120px",
          height: "120px",
          pointerEvents: "auto",
        }}
      >
        <Canvas
          gl={{
            antialias: true,
            alpha: true,
          }}
          style={{
            width: "100%",
            height: "100%",
          }}
        >
          {/* Orthographic camera - no perspective distortion */}
          <OrthographicCamera
            ref={cubeCameraRef}
            makeDefault
            position={[0, 0, 5]}
            zoom={50}
            near={0.1}
            far={100}
          />

          {/* Light background */}
          <color attach="background" args={["#f8f9fa"]} />

          {/* Simple ambient + directional lighting */}
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={0.5} />
          <directionalLight position={[-5, -5, -5]} intensity={0.3} />

          {/* The orientation cube mesh */}
          <OrientationCubeMesh onFaceClick={onCubeClick} />
        </Canvas>
      </div>

      {/* Help Text */}
      <div className="bg-background/95 backdrop-blur-sm rounded-lg px-3 py-1.5 shadow-lg border border-border/50">
        <p className="text-[10px] text-muted-foreground text-center">
          Click cube face to orient view
        </p>
      </div>
    </div>
  );
}