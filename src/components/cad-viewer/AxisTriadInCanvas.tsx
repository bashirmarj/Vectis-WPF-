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
    
    // Original working position (bottom-left corner)
    const baseX = -7.5;
    const baseY = -6.75;
    
    // Reference aspect ratio for 70% panel width (typical expanded state)
    const referenceAspectRatio = 1.3;
    
    // Adjust X position to maintain corner placement across different aspect ratios
    const adjustedX = baseX * (aspectRatio / referenceAspectRatio);
    
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
