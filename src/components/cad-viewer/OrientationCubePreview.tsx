import { useRef, useState, forwardRef, useImperativeHandle, useEffect, useMemo } from "react";
import * as THREE from "three";
import { Button } from "@/components/ui/button";
import { Box, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface OrientationCubePreviewProps {
  onCubeClick?: (direction: THREE.Vector3) => void;
  onRotateUp?: () => void;
  onRotateDown?: () => void;
  onRotateLeft?: () => void;
  onRotateRight?: () => void;
  onRotateClockwise?: () => void;
  onRotateCounterClockwise?: () => void;
  displayMode?: "solid" | "wireframe" | "translucent";
  onDisplayModeChange?: (style: "solid" | "wireframe" | "translucent") => void;
}

export interface OrientationCubeHandle {
  updateFromMainCamera: (camera: THREE.Camera) => void;
}

// ✅ OPTIMIZED: Create textures once and cache them
const createFaceTextures = () => {
  const faces = [
    { text: "FRONT", color: "#4A5568" },
    { text: "BACK", color: "#4A5568" },
    { text: "RIGHT", color: "#4A5568" },
    { text: "LEFT", color: "#4A5568" },
    { text: "TOP", color: "#4A5568" },
    { text: "BOTTOM", color: "#4A5568" },
  ];

  return faces.map(({ text, color }) => {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext("2d")!;

    // Draw background
    context.fillStyle = color;
    context.fillRect(0, 0, 256, 256);

    // Draw text with shadow
    context.shadowColor = "rgba(0, 0, 0, 0.5)";
    context.shadowBlur = 4;
    context.shadowOffsetX = 2;
    context.shadowOffsetY = 2;

    context.fillStyle = "#FFFFFF";
    context.font = "bold 32px Arial";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, 128, 128);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  });
};

// ✅ Cache textures globally to prevent recreation
let cachedTextures: THREE.Texture[] | null = null;

const getFaceTextures = () => {
  if (!cachedTextures) {
    cachedTextures = createFaceTextures();
  }
  return cachedTextures;
};

export const OrientationCubePreview = forwardRef<OrientationCubeHandle, OrientationCubePreviewProps>(
  (
    {
      onCubeClick,
      onRotateUp,
      onRotateDown,
      onRotateLeft,
      onRotateRight,
      onRotateClockwise,
      onRotateCounterClockwise,
      displayMode: externalDisplayMode,
      onDisplayModeChange,
    },
    ref,
  ) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const cubeRef = useRef<THREE.Mesh | null>(null);
    const animationFrameRef = useRef<number | null>(null);

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

    // ✅ OPTIMIZED: Initialize Three.js scene once
    useEffect(() => {
      if (!canvasRef.current) return;

      // Create renderer
      const renderer = new THREE.WebGLRenderer({
        canvas: canvasRef.current,
        antialias: true,
        alpha: true,
      });
      renderer.setSize(110, 110);
      renderer.setPixelRatio(window.devicePixelRatio);
      rendererRef.current = renderer;

      // Create scene
      const scene = new THREE.Scene();
      sceneRef.current = scene;

      // Create camera
      const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
      camera.position.set(3, 3, 3);
      camera.lookAt(0, 0, 0);
      cameraRef.current = camera;

      // Create cube with cached textures
      const geometry = new THREE.BoxGeometry(2, 2, 2);
      const textures = getFaceTextures();

      const materials = [
        new THREE.MeshBasicMaterial({ map: textures[2], transparent: true }), // Right
        new THREE.MeshBasicMaterial({ map: textures[3], transparent: true }), // Left
        new THREE.MeshBasicMaterial({ map: textures[4], transparent: true }), // Top
        new THREE.MeshBasicMaterial({ map: textures[5], transparent: true }), // Bottom
        new THREE.MeshBasicMaterial({ map: textures[0], transparent: true }), // Front
        new THREE.MeshBasicMaterial({ map: textures[1], transparent: true }), // Back
      ];

      const cube = new THREE.Mesh(geometry, materials);
      scene.add(cube);
      cubeRef.current = cube;

      // Add edge lines
      const edges = new THREE.EdgesGeometry(geometry);
      const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 1 }));
      cube.add(line);

      // Add subtle ambient light
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
      scene.add(ambientLight);

      // Animation loop
      const animate = () => {
        animationFrameRef.current = requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };
      animate();

      // Cleanup
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        renderer.dispose();
        geometry.dispose();
        materials.forEach((mat) => mat.dispose());
        edges.dispose();
      };
    }, []);

    // ✅ Update cube material based on display style
    useEffect(() => {
      if (!cubeRef.current) return;

      const materials = cubeRef.current.material as THREE.MeshBasicMaterial[];
      materials.forEach((mat) => {
        mat.wireframe = displayStyle === "wireframe";
        mat.opacity = displayStyle === "translucent" ? 0.5 : 1;
        mat.transparent = displayStyle === "translucent" || displayStyle === "wireframe";
        mat.needsUpdate = true;
      });
    }, [displayStyle]);

    // ✅ CRITICAL: Imperative method to sync with main camera
    useImperativeHandle(ref, () => ({
      updateFromMainCamera: (mainCamera: THREE.Camera) => {
        if (!cubeRef.current) return;

        // Get main camera's rotation
        const quaternion = mainCamera.quaternion.clone();

        // Apply inverse rotation to cube (so it appears to track camera)
        cubeRef.current.quaternion.copy(quaternion.invert());
      },
    }));

    // ✅ OPTIMIZED: Mouse interaction with raycasting
    const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current || !cameraRef.current || !cubeRef.current || !onCubeClick) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(x, y), cameraRef.current);

      const intersects = raycaster.intersectObject(cubeRef.current);
      if (intersects.length > 0) {
        const point = intersects[0].point;
        const localPoint = cubeRef.current.worldToLocal(point.clone());

        // Classify click region
        const absX = Math.abs(localPoint.x);
        const absY = Math.abs(localPoint.y);
        const absZ = Math.abs(localPoint.z);

        let direction: THREE.Vector3;
        if (absX > absY && absX > absZ) {
          direction = new THREE.Vector3(Math.sign(localPoint.x), 0, 0);
        } else if (absY > absX && absY > absZ) {
          direction = new THREE.Vector3(0, Math.sign(localPoint.y), 0);
        } else {
          direction = new THREE.Vector3(0, 0, Math.sign(localPoint.z));
        }

        onCubeClick(direction);
      }
    };

    const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current || !cameraRef.current || !cubeRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(x, y), cameraRef.current);

      const intersects = raycaster.intersectObject(cubeRef.current);
      if (intersects.length > 0) {
        const point = intersects[0].point;
        const localPoint = cubeRef.current.worldToLocal(point.clone());

        const absX = Math.abs(localPoint.x);
        const absY = Math.abs(localPoint.y);
        const absZ = Math.abs(localPoint.z);

        let description = "";
        if (absX > absY && absX > absZ) {
          description = Math.sign(localPoint.x) > 0 ? "Right" : "Left";
        } else if (absY > absX && absY > absZ) {
          description = Math.sign(localPoint.y) > 0 ? "Top" : "Bottom";
        } else {
          description = Math.sign(localPoint.z) > 0 ? "Front" : "Back";
        }

        setHoveredRegion({ type: "face", description });
        canvasRef.current.style.cursor = "pointer";
      } else {
        setHoveredRegion(null);
        canvasRef.current.style.cursor = "default";
      }
    };

    const handleCanvasMouseLeave = () => {
      setHoveredRegion(null);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = "default";
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

            {/* ✅ OPTIMIZED: Single canvas element */}
            <div className="w-[110px] h-[110px] border rounded overflow-hidden">
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
                style={{ display: "block", width: "100%", height: "100%" }}
              />
            </div>

            {/* Hover tooltip */}
            {hoveredRegion && (
              <div className="text-xs text-center py-1 px-2 bg-primary/10 rounded">{hoveredRegion.description}</div>
            )}

            {/* ✅ Direction controls - now control MAIN camera */}
            <div className="grid grid-cols-3 gap-1">
              <div />
              <Button variant="outline" size="sm" onClick={onRotateUp} className="h-8 w-8 p-0" title="Rotate camera up">
                <ChevronUp className="w-4 h-4" />
              </Button>
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={onRotateLeft}
                className="h-8 w-8 p-0"
                title="Rotate camera left"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={onRotateRight}
                className="h-8 w-8 p-0"
                title="Rotate camera right"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
              <div />
              <Button
                variant="outline"
                size="sm"
                onClick={onRotateDown}
                className="h-8 w-8 p-0"
                title="Rotate camera down"
              >
                <ChevronDown className="w-4 h-4" />
              </Button>
              <div />
            </div>

            {/* ✅ Rotation controls - now control MAIN camera */}
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={onRotateCounterClockwise}
                className="flex-1 h-8"
                title="Roll camera counter-clockwise"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={onRotateClockwise}
                className="flex-1 h-8"
                title="Roll camera clockwise"
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
