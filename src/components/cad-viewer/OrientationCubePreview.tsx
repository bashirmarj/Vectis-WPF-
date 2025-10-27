import { useRef, useState, forwardRef, useImperativeHandle, useEffect } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Button } from "@/components/ui/button";
import { Box, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import orientationCubeSTL from "@/assets/orientation-cube.stl?url";

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

// Cache STL geometry globally
let cachedSTLGeometry: THREE.BufferGeometry | null = null;
let isLoadingSTL = false;
const stlLoadCallbacks: ((geometry: THREE.BufferGeometry) => void)[] = [];

const loadSTLGeometry = (callback: (geometry: THREE.BufferGeometry) => void) => {
  if (cachedSTLGeometry) {
    callback(cachedSTLGeometry);
    return;
  }

  stlLoadCallbacks.push(callback);

  if (isLoadingSTL) {
    return;
  }

  isLoadingSTL = true;
  const loader = new STLLoader();

  loader.load(
    orientationCubeSTL,
    (geometry) => {
      geometry.computeBoundingBox();
      geometry.center();
      geometry.computeVertexNormals();

      cachedSTLGeometry = geometry;
      isLoadingSTL = false;

      console.log("✅ Orientation cube STL loaded successfully");
      stlLoadCallbacks.forEach((cb) => cb(geometry));
      stlLoadCallbacks.length = 0;
    },
    undefined,
    (error) => {
      console.error("❌ Error loading orientation cube STL:", error);
      isLoadingSTL = false;
      stlLoadCallbacks.length = 0;
    },
  );
};

// Professional face textures
const createFaceTextures = () => {
  const faces = [
    { text: "FRONT", color: "#e8eaed", textColor: "#3c4043" },
    { text: "BACK", color: "#e8eaed", textColor: "#3c4043" },
    { text: "RIGHT", color: "#e8eaed", textColor: "#3c4043" },
    { text: "LEFT", color: "#e8eaed", textColor: "#3c4043" },
    { text: "TOP", color: "#e8eaed", textColor: "#3c4043" },
    { text: "BOTTOM", color: "#e8eaed", textColor: "#3c4043" },
  ];

  return faces.map(({ text, color, textColor }) => {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext("2d")!;

    // Professional gradient background
    const gradient = context.createLinearGradient(0, 0, 256, 256);
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, "#d0d3d7");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 256, 256);

    // Subtle border
    context.strokeStyle = "#9aa0a6";
    context.lineWidth = 2;
    context.strokeRect(1, 1, 254, 254);

    // Professional text
    context.shadowColor = "rgba(0, 0, 0, 0.15)";
    context.shadowBlur = 3;
    context.shadowOffsetX = 1;
    context.shadowOffsetY = 1;

    context.fillStyle = textColor;
    context.font = "bold 28px -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, 128, 128);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  });
};

let cachedFaceTextures: THREE.Texture[] | null = null;

const getFaceTextures = () => {
  if (!cachedFaceTextures) {
    cachedFaceTextures = createFaceTextures();
  }
  return cachedFaceTextures;
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
    const labelGroupRef = useRef<THREE.Group | null>(null);
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

    // Initialize Three.js scene
    useEffect(() => {
      if (!canvasRef.current) return;

      console.log("🎬 Initializing orientation cube scene");

      const renderer = new THREE.WebGLRenderer({
        canvas: canvasRef.current,
        antialias: true,
        alpha: true,
      });
      renderer.setSize(150, 150);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      rendererRef.current = renderer;

      const scene = new THREE.Scene();
      sceneRef.current = scene;

      const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
      camera.position.set(3, 3, 3);
      camera.lookAt(0, 0, 0);
      cameraRef.current = camera;

      // Load STL geometry
      loadSTLGeometry((geometry) => {
        if (!sceneRef.current) return;

        const boundingBox = geometry.boundingBox;
        if (boundingBox) {
          const size = boundingBox.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 2 / maxDim;
          geometry.scale(scale, scale, scale);
        }

        const textures = getFaceTextures();

        // ✅ FIX: Professional semi-transparent material
        const material = new THREE.MeshStandardMaterial({
          color: 0xf5f5f5,
          metalness: 0.2,
          roughness: 0.4,
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
        });

        const cube = new THREE.Mesh(geometry, material);
        sceneRef.current.add(cube);
        cubeRef.current = cube;

        console.log("✅ Orientation cube mesh created");

        // ✅ FIX: Professional colored edges
        const edges = new THREE.EdgesGeometry(geometry, 15);
        const line = new THREE.LineSegments(
          edges,
          new THREE.LineBasicMaterial({
            color: 0x5f6368,
            linewidth: 1.5,
            transparent: true,
            opacity: 0.7,
          }),
        );
        cube.add(line);

        // Create label group
        const labelGroup = new THREE.Group();

        const facePositions = [
          { position: [0, 0, 1], rotation: [0, 0, 0], texture: textures[0] }, // Front
          { position: [0, 0, -1], rotation: [0, Math.PI, 0], texture: textures[1] }, // Back
          { position: [1, 0, 0], rotation: [0, Math.PI / 2, 0], texture: textures[2] }, // Right
          { position: [-1, 0, 0], rotation: [0, -Math.PI / 2, 0], texture: textures[3] }, // Left
          { position: [0, 1, 0], rotation: [-Math.PI / 2, 0, 0], texture: textures[4] }, // Top
          { position: [0, -1, 0], rotation: [Math.PI / 2, 0, Math.PI], texture: textures[5] }, // Bottom
        ];

        facePositions.forEach(({ position, rotation, texture }) => {
          const labelMesh = new THREE.Mesh(
            new THREE.PlaneGeometry(0.9, 0.9),
            new THREE.MeshBasicMaterial({
              map: texture,
              transparent: true,
              opacity: 0.9,
              side: THREE.DoubleSide,
            }),
          );
          labelMesh.position.set(position[0] * 1.02, position[1] * 1.02, position[2] * 1.02);
          labelMesh.rotation.set(rotation[0], rotation[1], rotation[2]);
          labelGroup.add(labelMesh);
        });

        cube.add(labelGroup);
        labelGroupRef.current = labelGroup;
      });

      // ✅ FIX: Enhanced professional lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
      keyLight.position.set(5, 5, 5);
      scene.add(keyLight);

      const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
      fillLight.position.set(-3, 2, -2);
      scene.add(fillLight);

      // ✅ CRITICAL FIX: Continuous animation loop that always renders
      const animate = () => {
        animationFrameRef.current = requestAnimationFrame(animate);

        // ALWAYS render - this ensures the cube updates are visible
        if (rendererRef.current && sceneRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };
      animate();

      console.log("✅ Orientation cube animation loop started");

      // Cleanup
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        renderer.dispose();
      };
    }, []);

    // Update cube material based on display style
    useEffect(() => {
      if (!cubeRef.current) return;

      const material = cubeRef.current.material as THREE.MeshStandardMaterial;

      if (displayStyle === "wireframe") {
        material.wireframe = true;
        material.opacity = 1;
      } else if (displayStyle === "translucent") {
        material.wireframe = false;
        material.opacity = 0.4;
      } else {
        material.wireframe = false;
        material.opacity = 0.85;
      }

      material.transparent = true;
      material.needsUpdate = true;

      console.log(`🎨 Display style changed to: ${displayStyle}`);
    }, [displayStyle]);

    // ✅ CRITICAL FIX: Update cube orientation from main camera
    useImperativeHandle(ref, () => ({
      updateFromMainCamera: (mainCamera: THREE.Camera) => {
        if (!cubeRef.current) {
          console.warn("⚠️ Cube ref not ready for update");
          return;
        }

        // Clone and invert camera quaternion to rotate cube opposite to camera
        const cameraQuat = mainCamera.quaternion.clone();
        const invertedQuat = cameraQuat.invert();

        // Apply to cube - this will be visible in next animation frame
        cubeRef.current.quaternion.copy(invertedQuat);

        console.log("🔄 Orientation cube updated from camera", {
          cameraQuat: {
            x: mainCamera.quaternion.x.toFixed(3),
            y: mainCamera.quaternion.y.toFixed(3),
            z: mainCamera.quaternion.z.toFixed(3),
            w: mainCamera.quaternion.w.toFixed(3),
          },
          cubeQuat: {
            x: cubeRef.current.quaternion.x.toFixed(3),
            y: cubeRef.current.quaternion.y.toFixed(3),
            z: cubeRef.current.quaternion.z.toFixed(3),
            w: cubeRef.current.quaternion.w.toFixed(3),
          },
        });
      },
    }));

    // Mouse interaction
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

        console.log("🎯 Cube face clicked:", direction);
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

    // ✅ FIX: Arrow click handlers with debugging
    const handleArrowClick = (direction: string, handler?: () => void) => {
      console.log(`⬆️ Arrow clicked: ${direction}`);
      if (handler) {
        handler();
        console.log(`✅ Handler executed for: ${direction}`);
      } else {
        console.warn(`⚠️ No handler provided for: ${direction}`);
      }
    };

    return (
      <div className="absolute top-4 right-4 z-10">
        <div className="bg-background/95 backdrop-blur-sm rounded-lg p-3 shadow-xl border-2 border-border/50">
          <div className="flex flex-col gap-2">
            {/* Header */}
            <div className="flex items-center gap-2 mb-1">
              <Box className="w-4 h-4 text-primary" />
              <span className="text-sm font-semibold">View Orientation</span>
            </div>

            {/* ✅ FIX: Larger canvas with integrated arrow controls */}
            <div className="relative w-[150px] h-[150px] border-2 border-border/30 rounded-lg overflow-hidden bg-gradient-to-br from-muted/20 to-background shadow-inner">
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
                className="w-full h-full"
                style={{ display: "block" }}
              />

              {/* ✅ FIX: Arrow controls overlaid on canvas */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Top Arrow */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleArrowClick("UP", onRotateUp)}
                  className="absolute top-1 left-1/2 -translate-x-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                  title="Rotate view up (90°)"
                >
                  <ChevronUp className="w-4 h-4" />
                </Button>

                {/* Bottom Arrow */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleArrowClick("DOWN", onRotateDown)}
                  className="absolute bottom-1 left-1/2 -translate-x-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                  title="Rotate view down (90°)"
                >
                  <ChevronDown className="w-4 h-4" />
                </Button>

                {/* Left Arrow */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleArrowClick("LEFT", onRotateLeft)}
                  className="absolute left-1 top-1/2 -translate-y-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                  title="Rotate view left (90°)"
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>

                {/* Right Arrow */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleArrowClick("RIGHT", onRotateRight)}
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                  title="Rotate view right (90°)"
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>

                {/* Roll Controls */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleArrowClick("CCW", onRotateCounterClockwise)}
                  className="absolute bottom-1 left-1 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                  title="Roll view counter-clockwise (90°)"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleArrowClick("CW", onRotateClockwise)}
                  className="absolute bottom-1 right-1 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                  title="Roll view clockwise (90°)"
                >
                  <RotateCw className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>

            {/* Hover tooltip */}
            {hoveredRegion && (
              <div className="text-xs text-center py-1.5 px-3 bg-primary/15 text-primary font-medium rounded-md shadow-sm">
                {hoveredRegion.description}
              </div>
            )}

            {/* Display style selector */}
            <Select value={displayStyle} onValueChange={handleDisplayStyleChange}>
              <SelectTrigger className="h-8 text-xs font-medium">
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
