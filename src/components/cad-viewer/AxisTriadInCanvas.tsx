import { Hud, OrthographicCamera } from "@react-three/drei";
import { AxisTriad } from "./AxisTriad";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { useMemo } from "react";

interface AxisTriadInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
}

export function AxisTriadInCanvas({ mainCameraRef }: AxisTriadInCanvasProps) {
  // Fixed position in bottom-left corner (Hud uses screen-space coordinates)
  // No adjustment needed - viewport only changes width, not height
  const triadPosition: [number, number, number] = [-7.5, -9, 0];

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
