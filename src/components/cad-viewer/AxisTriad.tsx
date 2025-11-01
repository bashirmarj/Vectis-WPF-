import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";

interface AxisTriadProps {
  mainCameraRef: React.RefObject<THREE.PerspectiveCamera>;
}

export function AxisTriad({ mainCameraRef }: AxisTriadProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Sync rotation with main camera
  useFrame(() => {
    if (mainCameraRef.current && groupRef.current) {
      groupRef.current.quaternion.copy(mainCameraRef.current.quaternion);
    }
  });

  // SolidWorks-style arrow proportions
  const arrowLength = 1.2;
  const shaftRadius = 0.03;
  const coneRadius = 0.1;
  const coneLength = 0.25;

  const createArrow = (
    direction: THREE.Vector3,
    color: string,
    label: string
  ) => {
    const shaftGeometry = new THREE.CylinderGeometry(
      shaftRadius,
      shaftRadius,
      arrowLength - coneLength,
      8
    );
    const coneGeometry = new THREE.ConeGeometry(coneRadius, coneLength, 8);

    const material = new THREE.MeshBasicMaterial({ color });

    // Position shaft
    const shaft = new THREE.Mesh(shaftGeometry, material);
    const shaftOffset = (arrowLength - coneLength) / 2;

    // Position cone at arrow tip
    const cone = new THREE.Mesh(coneGeometry, material);
    const coneOffset = arrowLength - coneLength / 2;

    // Rotate to align with direction
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

    const shaftPos = direction.clone().multiplyScalar(shaftOffset);
    shaft.position.copy(shaftPos);
    shaft.quaternion.copy(quaternion);

    const conePos = direction.clone().multiplyScalar(coneOffset);
    cone.position.copy(conePos);
    cone.quaternion.copy(quaternion);

    const labelPos = direction.clone().multiplyScalar(arrowLength + 0.3);

    return { shaft, cone, labelPos, label, color };
  };

  // Create arrows - SolidWorks colors
  const xAxis = createArrow(new THREE.Vector3(1, 0, 0), "#E74C3C", "X");
  const yAxis = createArrow(new THREE.Vector3(0, 1, 0), "#2ECC71", "Y");
  const zAxis = createArrow(new THREE.Vector3(0, 0, 1), "#3498DB", "Z");

  const axes = [xAxis, yAxis, zAxis];

  return (
    <group ref={groupRef}>
      {axes.map((axis, idx) => (
        <group key={idx}>
          <primitive object={axis.shaft} />
          <primitive object={axis.cone} />
          <Html position={axis.labelPos} center>
            <div
              style={{
                color: axis.color,
                fontSize: "14px",
                fontWeight: "bold",
                fontFamily: "sans-serif",
                userSelect: "none",
                pointerEvents: "none",
              }}
            >
              {axis.label}
            </div>
          </Html>
        </group>
      ))}
    </group>
  );
}
