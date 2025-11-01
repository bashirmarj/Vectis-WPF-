import { Hud, OrthographicCamera } from "@react-three/drei";
import { AxisTriad } from "./AxisTriad";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { useMemo } from "react";

interface AxisTriadInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  isSidebarCollapsed?: boolean;
}

function TriadPositioner({ mainCameraRef }: { mainCameraRef: React.RefObject<THREE.PerspectiveCamera> }) {
  const { viewport } = useThree();

  const triadPosition = useMemo(() => {
    const zoom = 45; // Must match OrthographicCamera zoom prop
    const aspect = viewport.width / viewport.height;

    // Calculate orthographic frustum boundaries
    const left = -aspect / zoom;
    const bottom = -1 / zoom;

    // Use FIXED offset from edges (not percentage)
    // Small margin to prevent clipping while staying in corner
    const fixedOffsetX = 0.002; // Small margin from left edge
    const fixedOffsetY = 0.002; // Small margin from bottom edge

    const x = left + fixedOffsetX;
    const y = bottom + fixedOffsetY;

    return [x, y, 0] as [number, number, number];
  }, [viewport.width, viewport.height]);

  return (
    <group position={triadPosition}>
      <ambientLight intensity={0.8} />
      <AxisTriad mainCameraRef={mainCameraRef} />
    </group>
  );
}

export function AxisTriadInCanvas({ mainCameraRef, isSidebarCollapsed = false }: AxisTriadInCanvasProps) {
  return (
    <Hud renderPriority={1}>
      <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={45} />
      <TriadPositioner mainCameraRef={mainCameraRef} />
    </Hud>
  );
}
