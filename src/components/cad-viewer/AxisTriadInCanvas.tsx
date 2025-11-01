import { Hud, OrthographicCamera } from "@react-three/drei";
import { AxisTriad } from "./AxisTriad";
import * as THREE from "three";

interface AxisTriadInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
}

export function AxisTriadInCanvas({ mainCameraRef }: AxisTriadInCanvasProps) {
  return (
    <Hud renderPriority={1}>
      <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={45} />

      {/* Position in bottom-left corner of screen space */}
      <group position={[-7.5, -5.5, 0]}>
        <ambientLight intensity={0.8} />
        <AxisTriad mainCameraRef={mainCameraRef} />
      </group>
    </Hud>
  );
}
