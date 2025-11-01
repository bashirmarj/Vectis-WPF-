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
    
    // Position as percentage from edges
    const offsetFromLeft = 0.08; // 8% from left edge
    const offsetFromBottom = 0.10; // 10% from bottom edge
    
    const frustumWidth = (aspect * 2) / zoom;
    const frustumHeight = 2 / zoom;
    
    const x = left + (frustumWidth * offsetFromLeft);
    const y = bottom + (frustumHeight * offsetFromBottom);
    
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
