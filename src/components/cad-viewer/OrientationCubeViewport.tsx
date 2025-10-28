import { Canvas, useThree } from "@react-three/fiber";
import { OrthographicCamera, Environment } from "@react-three/drei";
import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrientationCubeMesh } from "./OrientationCubeMesh";
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCcw, RotateCw } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface OrientationCubeViewportProps {
  mainCameraRef: React.RefObject<THREE.Camera>;
  onRotateCamera?: (direction: string) => void;
}

export const OrientationCubeViewport = ({ mainCameraRef, onRotateCamera }: OrientationCubeViewportProps) => {
  const cubeCameraRef = useRef<THREE.OrthographicCamera>(null);
  const [activeButton, setActiveButton] = useState<string | null>(null);

  // ✅ FIXED: Rotation sync WITHOUT lookAt - allows full 3D rotation
  useEffect(() => {
    if (!mainCameraRef) return;

    let animationFrameId: number;
    let frameCount = 0;

    const syncRotation = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        // ✅ Copy rotation and up vector from main camera
        cubeCameraRef.current.quaternion.copy(mainCameraRef.current.quaternion);
        cubeCameraRef.current.up.copy(mainCameraRef.current.up);

        // ✅ CRITICAL FIX: Removed lookAt(0,0,0) call
        // lookAt() was overriding the quaternion, constraining rotation to one axis
        // Now cube rotates freely in full 3D matching main camera orientation

        cubeCameraRef.current.updateProjectionMatrix();

        if (frameCount % 60 === 0) {
          console.log("🔄 Cube synced with full 3D rotation");
        }
        frameCount++;
      }
      animationFrameId = requestAnimationFrame(syncRotation);
    };

    // Wait for both cameras
    const startSync = () => {
      if (mainCameraRef.current && cubeCameraRef.current) {
        syncRotation();
        return true;
      }
      return false;
    };

    const checkInterval = setInterval(() => {
      if (startSync()) {
        clearInterval(checkInterval);
      }
    }, 100);

    if (startSync()) {
      clearInterval(checkInterval);
    }

    return () => {
      clearInterval(checkInterval);
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [mainCameraRef]);

  const handleButtonClick = (direction: string, callback?: () => void) => {
    setActiveButton(direction);
    setTimeout(() => setActiveButton(null), 200);
    if (callback) callback();
  };

  const ArrowButton = ({
    direction,
    icon: Icon,
    tooltip,
    onClick,
  }: {
    direction: string;
    icon: typeof ChevronUp;
    tooltip: string;
    onClick: () => void;
  }) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            className={`
              w-10 h-10 flex items-center justify-center
              bg-white/90 hover:bg-white
              border border-gray-300 rounded-md
              transition-all duration-150
              hover:scale-110 hover:shadow-md
              ${activeButton === direction ? "bg-blue-100 scale-95" : ""}
            `}
            onClick={() => handleButtonClick(direction, onClick)}
          >
            <Icon className="w-5 h-5 text-gray-700" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  return (
    <div
      className="fixed z-50 pointer-events-auto"
      style={{
        top: "80px",
        right: "20px",
      }}
    >
      {/* Canvas Container */}
      <div
        className="relative rounded-lg overflow-hidden shadow-xl"
        style={{
          width: "140px",
          height: "140px",
          backdropFilter: "blur(10px)",
          background: "rgba(255, 255, 255, 0.95)",
          border: "2px solid rgba(0,0,0,0.1)",
        }}
      >
        <Canvas
          gl={{
            antialias: true,
            alpha: true,
            powerPreference: "high-performance",
            preserveDrawingBuffer: true,
          }}
          dpr={window.devicePixelRatio || 1}
        >
          {/* Orthographic camera positioned looking at origin */}
          <OrthographicCamera ref={cubeCameraRef} makeDefault position={[0, 0, 10]} zoom={38} near={0.1} far={100} />

          {/* Enhanced lighting for depth perception */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 8, 5]} intensity={0.8} castShadow />
          <directionalLight position={[-3, 3, -3]} intensity={0.3} />
          <directionalLight position={[0, -5, -5]} intensity={0.2} />
          <Environment preset="city" />

          {/* Orientation Cube */}
          <OrientationCubeMesh />

          {/* Ground plane for shadow */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.2, 0]} receiveShadow>
            <planeGeometry args={[10, 10]} />
            <shadowMaterial opacity={0.15} />
          </mesh>
        </Canvas>
      </div>

      {/* Arrow Controls Grid */}
      <div
        className="mt-2 grid grid-cols-3 gap-1 p-2 rounded-lg"
        style={{
          background: "rgba(255, 255, 255, 0.95)",
          border: "2px solid rgba(0,0,0,0.1)",
          backdropFilter: "blur(10px)",
        }}
      >
        {/* Top Row */}
        <div />
        <ArrowButton direction="up" icon={ChevronUp} tooltip="Rotate Up" onClick={() => onRotateCamera?.("up")} />
        <div />

        {/* Middle Row */}
        <ArrowButton
          direction="left"
          icon={ChevronLeft}
          tooltip="Rotate Left"
          onClick={() => onRotateCamera?.("left")}
        />
        <div className="w-10 h-10" />
        <ArrowButton
          direction="right"
          icon={ChevronRight}
          tooltip="Rotate Right"
          onClick={() => onRotateCamera?.("right")}
        />

        {/* Bottom Row */}
        <ArrowButton
          direction="ccw"
          icon={RotateCcw}
          tooltip="Roll Counter-Clockwise"
          onClick={() => onRotateCamera?.("ccw")}
        />
        <ArrowButton
          direction="down"
          icon={ChevronDown}
          tooltip="Rotate Down"
          onClick={() => onRotateCamera?.("down")}
        />
        <ArrowButton direction="cw" icon={RotateCw} tooltip="Roll Clockwise" onClick={() => onRotateCamera?.("cw")} />
      </div>
    </div>
  );
};
