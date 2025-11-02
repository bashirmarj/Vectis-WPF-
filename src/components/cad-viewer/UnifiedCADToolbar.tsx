import { useState, useEffect } from "react";
import {
  Home,
  Eye,
  Ruler,
  Scissors,
  Settings,
  Box,
  Grid3x3,
  Package,
  Circle,
  ZoomIn,
  Layers,
  X,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";

interface UnifiedCADToolbarProps {
  // View Controls
  onHomeView: () => void;
  onFrontView: () => void;
  onTopView: () => void;
  onIsometricView: () => void;
  onFitView: () => void;

  // Display Mode
  displayMode: "solid" | "wireframe" | "translucent";
  onDisplayModeChange: (mode: "solid" | "wireframe" | "translucent") => void;

  // Edge Display
  showEdges: boolean;
  onToggleEdges: () => void;

  // Measurement - Smart Edge Select, Face-to-Face, and Face Measurement
  measurementMode: "edge-select" | "face-to-face" | "measure" | null;
  onMeasurementModeChange: (mode: "edge-select" | "face-to-face" | "measure" | null) => void;
  measurementCount?: number;
  onClearMeasurements?: () => void;

  // Section Planes
  sectionPlane: "xy" | "xz" | "yz" | null;
  onSectionPlaneChange: (plane: "xy" | "xz" | "yz" | null) => void;
  sectionPosition?: number;
  onSectionPositionChange?: (position: number) => void;

  // Visual Settings
  shadowsEnabled: boolean;
  onToggleShadows: () => void;
  ssaoEnabled: boolean;
  onToggleSSAO: () => void;

  // Bounding Box for section range calculation
  boundingBox?: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
    center: { x: number; y: number; z: number };
  };

  // Silhouette edges
  useSilhouettes?: boolean;
  onToggleSilhouettes?: () => void;
}

export function UnifiedCADToolbar({
  onHomeView,
  onFrontView,
  onTopView,
  onIsometricView,
  onFitView,
  displayMode,
  onDisplayModeChange,
  showEdges,
  onToggleEdges,
  measurementMode,
  onMeasurementModeChange,
  measurementCount = 0,
  onClearMeasurements,
  sectionPlane,
  onSectionPlaneChange,
  sectionPosition = 0,
  onSectionPositionChange,
  shadowsEnabled,
  onToggleShadows,
  ssaoEnabled,
  onToggleSSAO,
  boundingBox,
  useSilhouettes = false,
  onToggleSilhouettes,
}: UnifiedCADToolbarProps) {
  const [showSectionPanel, setShowSectionPanel] = useState(false);

  // Auto-show section panel when section tool activated
  useEffect(() => {
    if (sectionPlane) {
      setShowSectionPanel(true);
    }
  }, [sectionPlane]);

  // Handle section tool button click
  const handleSectionToolClick = () => {
    if (!sectionPlane) {
      // Activate default section plane
      onSectionPlaneChange("xy");
      setShowSectionPanel(true);
    } else {
      // Toggle panel visibility
      setShowSectionPanel(!showSectionPanel);
    }
  };

  // Calculate slider range based on bounding box
  const getSectionRange = (): [number, number] => {
    if (!boundingBox) return [-100, 100];

    switch (sectionPlane) {
      case "xy":
        return [boundingBox.min.z - 10, boundingBox.max.z + 10];
      case "xz":
        return [boundingBox.min.y - 10, boundingBox.max.y + 10];
      case "yz":
        return [boundingBox.min.x - 10, boundingBox.max.x + 10];
      default:
        return [-100, 100];
    }
  };

  const [minRange, maxRange] = getSectionRange();

  return (
    <>
      {/* Main Toolbar */}
      <div className="absolute top-5 left-1/2 transform -translate-x-1/2 z-20">
        <div className="flex items-center gap-2 bg-white/95 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg border border-gray-200">
          {/* View Controls Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-9 w-9 p-0" title="View Controls">
                <Home className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-48">
              <DropdownMenuItem onClick={onHomeView}>
                <Home className="mr-2 h-4 w-4" />
                Home View
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onFrontView}>
                <Box className="mr-2 h-4 w-4" />
                Front View
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onTopView}>
                <Grid3x3 className="mr-2 h-4 w-4" />
                Top View
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onIsometricView}>
                <Package className="mr-2 h-4 w-4" />
                Isometric View
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onFitView}>
                <ZoomIn className="mr-2 h-4 w-4" />
                Fit to View
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Separator orientation="vertical" className="h-6" />

          {/* Display Mode Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-9 w-9 p-0" title="Display Mode">
                <Eye className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-48">
              <DropdownMenuItem
                onClick={() => onDisplayModeChange("solid")}
                className={cn(displayMode === "solid" && "bg-accent")}
              >
                <Box className="mr-2 h-4 w-4" />
                Solid
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => onDisplayModeChange("wireframe")}
                className={cn(displayMode === "wireframe" && "bg-accent")}
              >
                <Grid3x3 className="mr-2 h-4 w-4" />
                Wireframe
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => onDisplayModeChange("translucent")}
                className={cn(displayMode === "translucent" && "bg-accent")}
              >
                <Circle className="mr-2 h-4 w-4" />
                Translucent
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onToggleEdges}>
                <Layers className="mr-2 h-4 w-4" />
                {displayMode === "wireframe" 
                  ? (showEdges ? "Hide Hidden" : "Show Hidden")
                  : (showEdges ? "Hide" : "Show")
                } Edges
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Separator orientation="vertical" className="h-6" />

          {/* Measurement Tools */}
          <Button
            variant={measurementMode === "edge-select" ? "default" : "ghost"}
            size="sm"
            className="h-9 w-9 p-0 relative"
            title="Edge Measurement (Click edges to measure)"
            onClick={() => onMeasurementModeChange(measurementMode === "edge-select" ? null : "edge-select")}
          >
            <Ruler className="h-4 w-4" />
            {measurementMode === "edge-select" && measurementCount > 0 && (
              <Badge
                variant="destructive"
                className="absolute -top-1 -right-1 h-4 w-4 p-0 flex items-center justify-center text-[10px]"
              >
                {measurementCount}
              </Badge>
            )}
          </Button>

          <Button
            variant={measurementMode === "face-to-face" ? "default" : "ghost"}
            size="sm"
            className="h-9 w-9 p-0 relative"
            title="Face-to-Face Measurement (Click two faces)"
            onClick={() => onMeasurementModeChange(measurementMode === "face-to-face" ? null : "face-to-face")}
          >
            <Square className="h-4 w-4" />
            {measurementMode === "face-to-face" && measurementCount > 0 && (
              <Badge
                variant="destructive"
                className="absolute -top-1 -right-1 h-4 w-4 p-0 flex items-center justify-center text-[10px]"
              >
                {measurementCount}
              </Badge>
            )}
          </Button>

          <Button
            variant={measurementMode === "measure" ? "default" : "ghost"}
            size="sm"
            className="h-9 w-9 p-0 relative"
            title="Face Measurement (Point-to-point with angle)"
            onClick={() => onMeasurementModeChange(measurementMode === "measure" ? null : "measure")}
          >
            <Circle className="h-4 w-4" />
            {measurementMode === "measure" && measurementCount > 0 && (
              <Badge
                variant="destructive"
                className="absolute -top-1 -right-1 h-4 w-4 p-0 flex items-center justify-center text-[10px]"
              >
                {measurementCount}
              </Badge>
            )}
          </Button>

          <Separator orientation="vertical" className="h-6" />

          {/* Section Plane Tool */}
          <Button
            variant={sectionPlane ? "default" : "ghost"}
            size="sm"
            className="h-9 w-9 p-0"
            title="Section Planes"
            onClick={handleSectionToolClick}
          >
            <Scissors className="h-4 w-4" />
          </Button>

          <Separator orientation="vertical" className="h-6" />

          {/* Visual Settings Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-9 w-9 p-0" title="Visual Settings">
                <Settings className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={onToggleShadows}>
                <Eye className="mr-2 h-4 w-4" />
                {shadowsEnabled ? "Disable" : "Enable"} Shadows
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onToggleSSAO}>
                <Eye className="mr-2 h-4 w-4" />
                {ssaoEnabled ? "Disable" : "Enable"} SSAO
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Section Panel - Auto-shows when section tool active */}
      {showSectionPanel && sectionPlane && (
        <div className="absolute top-20 left-1/2 transform -translate-x-1/2 z-20">
          <div className="bg-white/95 backdrop-blur-sm px-4 py-3 rounded-lg shadow-lg border border-gray-200 w-80">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-700">Section Plane</h3>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={() => {
                  onSectionPlaneChange(null);
                  setShowSectionPanel(false);
                }}
              >
                <X className="h-3 w-3" />
              </Button>
            </div>

            {/* Plane Selection */}
            <div className="flex gap-2 mb-3">
              <Button
                variant={sectionPlane === "xy" ? "default" : "outline"}
                size="sm"
                onClick={() => onSectionPlaneChange("xy")}
                className="flex-1"
              >
                XY
              </Button>
              <Button
                variant={sectionPlane === "xz" ? "default" : "outline"}
                size="sm"
                onClick={() => onSectionPlaneChange("xz")}
                className="flex-1"
              >
                XZ
              </Button>
              <Button
                variant={sectionPlane === "yz" ? "default" : "outline"}
                size="sm"
                onClick={() => onSectionPlaneChange("yz")}
                className="flex-1"
              >
                YZ
              </Button>
            </div>

            {/* Position Slider - ✅ FIXED: Uses bounding box for range */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-gray-600">
                <span>Position</span>
                <span className="font-mono">{sectionPosition.toFixed(1)}mm</span>
              </div>
              <Slider
                value={[sectionPosition]}
                onValueChange={(values) => onSectionPositionChange?.(values[0])}
                min={minRange}
                max={maxRange}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
