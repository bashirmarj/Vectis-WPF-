import { Hud, OrthographicCamera } from "@react-three/drei";
import { AxisTriad } from "./AxisTriad";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { useMemo } from "react";

interface AxisTriadInCanvasProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
}

export function AxisTriadInCanvas({ mainCameraRef }: AxisTriadInCanvasProps) {
  const { viewport } = useThree();
  
  // Calculate position based on viewport aspect ratio to maintain corner position
  const triadPosition = useMemo(() => {
    const aspectRatio = viewport.width / viewport.height;
    
    // Base position for standard aspect ratio
    const baseX = -7.5;
    const baseY = -6.75;
    
    // Adjust X based on aspect ratio to maintain relative position
    const adjustedX = baseX * (aspectRatio / 1.5);
    
    return [adjustedX, baseY, 0] as [number, number, number];
  }, [viewport.width, viewport.height]);

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
