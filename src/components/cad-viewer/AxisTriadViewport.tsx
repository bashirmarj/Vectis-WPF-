import { Canvas } from "@react-three/fiber";
import { OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { AxisTriad } from "./AxisTriad";

interface AxisTriadViewportProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
}

export function AxisTriadViewport({ mainCameraRef }: AxisTriadViewportProps) {
  return (
    <div className="absolute bottom-4 left-4 w-[100px] h-[100px] pointer-events-none z-10">
      <Canvas>
        <OrthographicCamera makeDefault position={[0, 0, 5]} zoom={50} />
        <ambientLight intensity={0.8} />
        <AxisTriad mainCameraRef={mainCameraRef} />
      </Canvas>
    </div>
  );
}
