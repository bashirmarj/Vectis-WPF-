import { useRef, useState, forwardRef, useImperativeHandle, useEffect } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Button } from "@/components/ui/button";
import { Box, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import orientationCubeSTL from "@/assets/orientation-cube.stl?url";

interface OrientationCubePreviewProps {
  onCubeClick?: (direction: THREE.Vector3) => void;
  onOrientationChange?: (direction: THREE.Vector3) => void;
  onUpVectorChange?: (upVector: THREE.Vector3) => void;
  displayMode?: "solid" | "wireframe" | "translucent";
  onDisplayModeChange?: (style: "solid" | "wireframe" | "translucent") => void;
}

export interface OrientationCubeHandle {
  rotateCube: (direction: THREE.Vector3) => void;
  rotateClockwise: () => void;
  rotateCounterClockwise: () => void;
}

// Helper function to classify click region (face, edge, or corner)
const classifyClickRegion = (localPoint: THREE.Vector3) => {
  const faceDistance = 1.0;
  const edgeThreshold = 0.25;

  const absX = Math.abs(localPoint.x);
  const absY = Math.abs(localPoint.y);
  const absZ = Math.abs(localPoint.z);

  const nearMaxX = absX > faceDistance - edgeThreshold;
  const nearMaxY = absY > faceDistance - edgeThreshold;
  const nearMaxZ = absZ > faceDistance - edgeThreshold;

  const edgeCount = [nearMaxX, nearMaxY, nearMaxZ].filter(Boolean).length;

  // CORNER: All 3 dimensions near maximum
  if (edgeCount === 3) {
    const direction = new THREE.Vector3(
      Math.sign(localPoint.x),
      Math.sign(localPoint.y),
      Math.sign(localPoint.z),
    ).normalize();

    return {
      type: "corner" as const,
      direction,
      description: `Corner (${Math.sign(localPoint.x) > 0 ? "+" : "-"}X, ${Math.sign(localPoint.y) > 0 ? "+" : "-"}Y, ${Math.sign(localPoint.z) > 0 ? "+" : "-"}Z)`,
    };
  }

  // EDGE: Exactly 2 dimensions near maximum
  else if (edgeCount === 2) {
    const direction = new THREE.Vector3(
      nearMaxX ? Math.sign(localPoint.x) : 0,
      nearMaxY ? Math.sign(localPoint.y) : 0,
      nearMaxZ ? Math.sign(localPoint.z) : 0,
    ).normalize();

    return {
      type: "edge" as const,
      direction,
      description: "Edge view",
    };
  }

  // FACE: Only 1 dimension near maximum
  else {
    if (absX > absY && absX > absZ) {
      return {
        type: "face" as const,
        direction: new THREE.Vector3(Math.sign(localPoint.x), 0, 0),
        description: Math.sign(localPoint.x) > 0 ? "Right" : "Left",
      };
    } else if (absY > absX && absY > absZ) {
      return {
        type: "face" as const,
        direction: new THREE.Vector3(0, Math.sign(localPoint.y), 0),
        description: Math.sign(localPoint.y) > 0 ? "Top" : "Bottom",
      };
    } else {
      return {
        type: "face" as const,
        direction: new THREE.Vector3(0, 0, Math.sign(localPoint.z)),
        description: Math.sign(localPoint.z) > 0 ? "Front" : "Back",
      };
    }
  }
};

// Inner component that uses R3F hooks
function OrientationCubeMesh({
  onCubeClick,
  displayStyle,
  onHover,
  cameraRef,
}: {
  onCubeClick?: (direction: THREE.Vector3) => void;
  displayStyle: "solid" | "wireframe" | "translucent";
  onHover: (region: { type: string; description: string } | null) => void;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
}) {
  const { camera, gl } = useThree();
  const cubeRef = useRef<THREE.Mesh | null>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  // Store camera ref for imperative methods
  useEffect(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      cameraRef.current = camera;
    }
  }, [camera, cameraRef]);

  // Load STL geometry
  useEffect(() => {
    const loader = new STLLoader();
    loader.load(orientationCubeSTL, (loadedGeometry) => {
      loadedGeometry.computeBoundingBox();
      loadedGeometry.center();
      setGeometry(loadedGeometry);
    });
  }, []);

  // Handle click
  const handleClick = (event: any) => {
    if (!cubeRef.current || !onCubeClick) return;
    event.stopPropagation();

    const localPoint = cubeRef.current.worldToLocal(event.point.clone());
    const region = classifyClickRegion(localPoint);
    onCubeClick(region.direction);
  };

  // Handle hover
  const handlePointerMove = (event: any) => {
    if (!cubeRef.current) return;
    event.stopPropagation();

    const localPoint = cubeRef.current.worldToLocal(event.point.clone());
    const region = classifyClickRegion(localPoint);
    onHover({ type: region.type, description: region.description });
    gl.domElement.style.cursor = "pointer";
  };

  const handlePointerOut = () => {
    onHover(null);
    gl.domElement.style.cursor = "default";
  };

  if (!geometry) return null;

  return (
    <mesh
      ref={cubeRef}
      geometry={geometry}
      onClick={handleClick}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    >
      <meshBasicMaterial
        color={0x4a5568}
        transparent={displayStyle === "translucent"}
        opacity={displayStyle === "translucent" ? 0.5 : 1}
        wireframe={displayStyle === "wireframe"}
      />
      
      {/* Edge lines */}
      <lineSegments>
        <edgesGeometry args={[geometry, 30]} />
        <lineBasicMaterial color={0x1f2937} opacity={0.6} transparent />
      </lineSegments>

      {/* Face labels */}
      {[
        { text: "Front", position: [0, 0, 1], rotation: [0, 0, 0] },
        { text: "Back", position: [0, 0, -1], rotation: [0, Math.PI, 0] },
        { text: "Right", position: [1, 0, 0], rotation: [0, Math.PI / 2, 0] },
        { text: "Left", position: [-1, 0, 0], rotation: [0, -Math.PI / 2, 0] },
        { text: "Top", position: [0, 1, 0], rotation: [-Math.PI / 2, 0, 0] },
        { text: "Bottom", position: [0, -1, 0], rotation: [Math.PI / 2, 0, Math.PI] },
      ].map((face, index) => {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d")!;
        canvas.width = 512;
        canvas.height = 512;

        context.shadowColor = "rgba(255, 255, 255, 0.8)";
        context.shadowBlur = 8;
        context.shadowOffsetX = 2;
        context.shadowOffsetY = 2;

        context.fillStyle = "#000000";
        context.font = "bold 80px Arial";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(face.text, 256, 256);

        const texture = new THREE.CanvasTexture(canvas);

        return (
          <mesh
            key={index}
            position={[face.position[0] * 1.05, face.position[1] * 1.05, face.position[2] * 1.05]}
            rotation={[face.rotation[0], face.rotation[1], face.rotation[2]]}
          >
            <planeGeometry args={[0.8, 0.8]} />
            <meshBasicMaterial map={texture} transparent depthWrite={false} side={THREE.DoubleSide} />
          </mesh>
        );
      })}
    </mesh>
  );
}

export const OrientationCubePreview = forwardRef<OrientationCubeHandle, OrientationCubePreviewProps>(
  (
    { onCubeClick, onOrientationChange, onUpVectorChange, displayMode: externalDisplayMode, onDisplayModeChange },
    ref,
  ) => {
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const [hoveredRegion, setHoveredRegion] = useState<{
      type: string;
      description: string;
    } | null>(null);
    const [displayStyle, setDisplayStyle] = useState<"solid" | "wireframe" | "translucent">(
      externalDisplayMode || "solid",
    );

    // Update internal state when external prop changes
    useEffect(() => {
      if (externalDisplayMode) {
        setDisplayStyle(externalDisplayMode);
      }
    }, [externalDisplayMode]);

    // Imperative methods for parent component
    useImperativeHandle(ref, () => ({
      rotateCube: (direction: THREE.Vector3) => {
        if (!cameraRef.current) return;
        const distance = 5;
        cameraRef.current.position.copy(direction.clone().multiplyScalar(distance));
        cameraRef.current.lookAt(0, 0, 0);
      },
      rotateClockwise: () => {
        rotateCameraClockwise();
      },
      rotateCounterClockwise: () => {
        rotateCameraCounterClockwise();
      },
    }));

    const rotateCameraClockwise = () => {
      if (!cameraRef.current) return;
      const currentPos = cameraRef.current.position.clone().normalize();
      const currentUp = cameraRef.current.up.clone();

      const axis = currentPos;
      const angle = -Math.PI / 2;

      const newUp = currentUp.clone().applyAxisAngle(axis, angle);
      cameraRef.current.up.copy(newUp);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();

      if (onUpVectorChange) {
        onUpVectorChange(newUp);
      }
    };

    const rotateCameraCounterClockwise = () => {
      if (!cameraRef.current) return;
      const currentPos = cameraRef.current.position.clone().normalize();
      const currentUp = cameraRef.current.up.clone();

      const axis = currentPos;
      const angle = Math.PI / 2;

      const newUp = currentUp.clone().applyAxisAngle(axis, angle);
      cameraRef.current.up.copy(newUp);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();

      if (onUpVectorChange) {
        onUpVectorChange(newUp);
      }
    };

    const rotateCameraUp = () => {
      if (!cameraRef.current) return;
      const currentPos = cameraRef.current.position.clone().normalize();
      const distance = 5;

      const faces = [
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(0, -1, 0),
        new THREE.Vector3(0, 0, -1),
      ];

      let closestIndex = 0;
      let maxDot = -Infinity;

      faces.forEach((face, index) => {
        const dot = currentPos.dot(face);
        if (dot > maxDot) {
          maxDot = dot;
          closestIndex = index;
        }
      });

      const nextIndex = (closestIndex + 1) % faces.length;
      const nextFace = faces[nextIndex];

      cameraRef.current.position.copy(nextFace.clone().multiplyScalar(distance));

      const up = new THREE.Vector3(0, 1, 0);
      if (Math.abs(nextFace.y) > 0.99) {
        up.set(0, 0, nextFace.y > 0 ? 1 : -1);
      }
      cameraRef.current.up.copy(up);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();

      if (onOrientationChange) {
        const newDirection = cameraRef.current.position.clone().normalize();
        onOrientationChange(newDirection);
      }
    };

    const rotateCameraDown = () => {
      if (!cameraRef.current) return;
      const currentPos = cameraRef.current.position.clone().normalize();
      const distance = 5;

      const faces = [
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, -1),
        new THREE.Vector3(0, -1, 0),
        new THREE.Vector3(0, 0, 1),
      ];

      let closestIndex = 0;
      let maxDot = -Infinity;

      faces.forEach((face, index) => {
        const dot = currentPos.dot(face);
        if (dot > maxDot) {
          maxDot = dot;
          closestIndex = index;
        }
      });

      const nextIndex = (closestIndex + 1) % faces.length;
      const nextFace = faces[nextIndex];

      cameraRef.current.position.copy(nextFace.clone().multiplyScalar(distance));

      const up = new THREE.Vector3(0, 1, 0);
      if (Math.abs(nextFace.y) > 0.99) {
        up.set(0, 0, nextFace.y > 0 ? 1 : -1);
      }
      cameraRef.current.up.copy(up);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();

      if (onOrientationChange) {
        const newDirection = cameraRef.current.position.clone().normalize();
        onOrientationChange(newDirection);
      }
    };

    const rotateCameraLeft = () => {
      if (!cameraRef.current) return;
      const currentPos = cameraRef.current.position.clone().normalize();
      const distance = 5;

      const faces = [
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 0, -1),
        new THREE.Vector3(-1, 0, 0),
      ];

      let closestIndex = 0;
      let maxDot = -Infinity;

      faces.forEach((face, index) => {
        const dot = currentPos.dot(face);
        if (dot > maxDot) {
          maxDot = dot;
          closestIndex = index;
        }
      });

      const nextIndex = (closestIndex + 1) % faces.length;
      const nextFace = faces[nextIndex];

      cameraRef.current.position.copy(nextFace.clone().multiplyScalar(distance));
      cameraRef.current.up.set(0, 1, 0);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();

      if (onOrientationChange) {
        const newDirection = cameraRef.current.position.clone().normalize();
        onOrientationChange(newDirection);
      }
    };

    const rotateCameraRight = () => {
      if (!cameraRef.current) return;
      const currentPos = cameraRef.current.position.clone().normalize();
      const distance = 5;

      const faces = [
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(-1, 0, 0),
        new THREE.Vector3(0, 0, -1),
        new THREE.Vector3(1, 0, 0),
      ];

      let closestIndex = 0;
      let maxDot = -Infinity;

      faces.forEach((face, index) => {
        const dot = currentPos.dot(face);
        if (dot > maxDot) {
          maxDot = dot;
          closestIndex = index;
        }
      });

      const nextIndex = (closestIndex + 1) % faces.length;
      const nextFace = faces[nextIndex];

      cameraRef.current.position.copy(nextFace.clone().multiplyScalar(distance));
      cameraRef.current.up.set(0, 1, 0);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();

      if (onOrientationChange) {
        const newDirection = cameraRef.current.position.clone().normalize();
        onOrientationChange(newDirection);
      }
    };

    const handleDisplayStyleChange = (newStyle: "solid" | "wireframe" | "translucent") => {
      setDisplayStyle(newStyle);
      if (onDisplayModeChange) {
        onDisplayModeChange(newStyle);
      }
    };

    return (
      <div className="absolute top-4 right-4 z-10">
        <div className="bg-background/80 backdrop-blur-sm rounded-lg p-3 shadow-lg border">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 mb-2">
              <Box className="w-4 h-4" />
              <span className="text-sm font-medium">Orientation</span>
            </div>

            {/* Canvas container */}
            <div className="w-[110px] h-[110px] border rounded">
              <Canvas
                camera={{ position: [5, 5, 5], fov: 50, near: 0.1, far: 1000 }}
                gl={{ 
                  antialias: true, 
                  alpha: true,
                  preserveDrawingBuffer: true,
                }}
              >
                <OrientationCubeMesh
                  onCubeClick={onCubeClick}
                  displayStyle={displayStyle}
                  onHover={setHoveredRegion}
                  cameraRef={cameraRef}
                />
              </Canvas>
            </div>

            {/* Hover tooltip */}
            {hoveredRegion && (
              <div className="text-xs text-center py-1 px-2 bg-primary/10 rounded">
                {hoveredRegion.description}
              </div>
            )}

            {/* Direction controls */}
            <div className="grid grid-cols-3 gap-1">
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={rotateCameraUp}
                className="h-8 w-8 p-0"
              >
                <ChevronUp className="w-4 h-4" />
              </Button>
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={rotateCameraLeft}
                className="h-8 w-8 p-0"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={rotateCameraRight}
                className="h-8 w-8 p-0"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={rotateCameraDown}
                className="h-8 w-8 p-0"
              >
                <ChevronDown className="w-4 h-4" />
              </Button>
              <div />
            </div>

            {/* Rotation controls */}
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={rotateCameraCounterClockwise}
                className="flex-1 h-8"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={rotateCameraClockwise}
                className="flex-1 h-8"
              >
                <RotateCw className="w-4 h-4" />
              </Button>
            </div>

            {/* Display style selector */}
            <Select value={displayStyle} onValueChange={handleDisplayStyleChange}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="solid">Solid</SelectItem>
                <SelectItem value="wireframe">Wireframe</SelectItem>
                <SelectItem value="translucent">Translucent</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    );
  },
);

OrientationCubePreview.displayName = "OrientationCubePreview";
