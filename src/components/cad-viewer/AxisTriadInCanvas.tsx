import { Hud, OrthographicCamera } from "@react-three/drei";
import { AxisTriad } from "./AxisTriad";
import * as THREE from "three";

interface AxisTriadInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  isSidebarCollapsed?: boolean;
}

export function AxisTriadInCanvas({ mainCameraRef, isSidebarCollapsed = false }: AxisTriadInCanvasProps) {
  // Base position when sidebar is expanded (30% sidebar, 70% canvas)
  const baseX = -7.5;
  const baseY = -9;
  
  // When sidebar collapses, canvas width goes from 70% to 100%
  // So we need to adjust X by factor of 100/70 = 10/7
  const widthRatio = isSidebarCollapsed ? (10 / 7) : 1;
  const adjustedX = baseX * widthRatio;
  
  const triadPosition: [number, number, number] = [adjustedX, baseY, 0];

  return (
    <Hud renderPriority={1}>
      <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={45} />

      {/* Position dynamically calculated to maintain bottom-left corner */}
      <group position={triadPosition}>
        <ambientLight intensity={0.8} />
        <AxisTriad mainCameraRef={mainCameraRef} />
      </group>
    </Hud>
  );
}
