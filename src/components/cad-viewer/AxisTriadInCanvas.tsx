import { Canvas } from "@react-three/fiber";
import { AxisTriad } from "./AxisTriad";
import { OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";

interface AxisTriadInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
  isSidebarCollapsed?: boolean;
}

export function AxisTriadInCanvas({ mainCameraRef }: AxisTriadInCanvasProps) {
  return (
    <div className="absolute bottom-4 left-4 w-20 h-20 pointer-events-none z-10">
      <Canvas
        style={{ width: '100%', height: '100%' }}
        gl={{ alpha: true }}
      >
        <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={20} />
        <ambientLight intensity={0.8} />
        <AxisTriad mainCameraRef={mainCameraRef} />
      </Canvas>
    </div>
  );
}
