import { useState, useEffect } from "react";
import {
  Home,
  Eye,
  Ruler,
  Scissors,
  Settings,
  ChevronDown,
  Box,
  Grid3x3,
  Package,
  Circle,
  Move,
  RotateCcw,
  ZoomIn,
  Layers,
  X,
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

  // Measurement - ✅ FIXED: Added all measurement types including edge-select
  measurementMode: "distance" | "angle" | "radius" | "diameter" | "edge-to-edge" | "face-to-face" | "coordinate" | "edge-select" | null;
  onMeasurementModeChange: (mode: "distance" | "angle" | "radius" | "diameter" | "edge-to-edge" | "face-to-face" | "coordinate" | "edge-select" | null) => void;
  measurementCount?: number;
  onClearMeasurements?: () => void;

  // Section Planes - ✅ FIXED: Removed individual axes ("x", "y", "z")
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
}: UnifiedCADToolbarProps) {
  const [showSectionPanel, setShowSectionPanel] = useState(false);

  // Auto-show section panel when section tool activated
  useEffect(() => {
    if (sectionPlane !== null) {
      setShowSectionPanel(true);
    } else {
      setShowSectionPanel(false);
    }
  }, [sectionPlane]);

  // ✅ FIXED: Calculate proportional slider range and step size (removed individual axis cases)
  const getSectionRange = () => {
    if (!boundingBox) return { min: -50, max: 50, center: 0, step: 1 };

    const { min, max, center } = boundingBox;

    // Determine range based on active section plane
    let rangeMin: number, rangeMax: number, rangeCenter: number;

    switch (sectionPlane) {
      case "xy":
        rangeMin = min.z;
        rangeMax = max.z;
        rangeCenter = center.z;
        break;
      case "xz":
        rangeMin = min.y;
        rangeMax = max.y;
        rangeCenter = center.y;
        break;
      case "yz":
        rangeMin = min.x;
        rangeMax = max.x;
        rangeCenter = center.x;
        break;
      default:
        rangeMin = -50;
        rangeMax = 50;
        rangeCenter = 0;
    }

    // Calculate step size as 0.5% of total dimension
    // This ensures smooth movement regardless of part size
    const totalDimension = Math.abs(rangeMax - rangeMin);
    const step = Math.max(0.1, totalDimension * 0.005); // 0.5% of dimension, min 0.1mm

    return {
      min: rangeMin,
      max: rangeMax,
      center: rangeCenter,
      step: step,
    };
  };

  const sectionRange = getSectionRange();

  // Initialize section position to center when plane activated
  useEffect(() => {
    if (sectionPlane !== null && onSectionPositionChange) {
      onSectionPositionChange(sectionRange.center);
    }
  }, [sectionPlane]);

  const handleSectionToolClick = () => {
    if (sectionPlane !== null) {
      onSectionPlaneChange(null);
      setShowSectionPanel(false);
    } else {
      onSectionPlaneChange("xy");
      setShowSectionPanel(true);
    }
  };

  // ✅ FIXED: Updated function signature to only accept plane types
  const handleSectionPlaneSelect = (plane: "xy" | "xz" | "yz") => {
    onSectionPlaneChange(plane);
  };

  const handleCloseSectionPanel = () => {
    onSectionPlaneChange(null);
    setShowSectionPanel(false);
  };

  return (
    <>
      {/* Main Toolbar */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50">
        <div className="flex items-center gap-2 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80 border rounded-lg shadow-lg px-3 py-2">
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
                {showEdges ? "Hide" : "Show"} Edges
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Separator orientation="vertical" className="h-6" />

          {/* Measurement Tools Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant={measurementMode ? "default" : "ghost"}
                size="sm"
                className="h-9 w-9 p-0 relative"
                title="Measurement Tools"
              >
                <Ruler className="h-4 w-4" />
                {measurementCount > 0 && (
                  <Badge
                    variant="destructive"
                    className="absolute -top-1 -right-1 h-4 w-4 p-0 flex items-center justify-center text-[10px]"
                  >
                    {measurementCount}
                  </Badge>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-48">
              <DropdownMenuItem
                onClick={() => onMeasurementModeChange(measurementMode === "distance" ? null : "distance")}
                className={cn(measurementMode === "distance" && "bg-accent")}
              >
                <Ruler className="mr-2 h-4 w-4" />
                Distance
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => onMeasurementModeChange(measurementMode === "angle" ? null : "angle")}
                className={cn(measurementMode === "angle" && "bg-accent")}
              >
                <RotateCcw className="mr-2 h-4 w-4" />
                Angle
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => onMeasurementModeChange(measurementMode === "radius" ? null : "radius")}
                className={cn(measurementMode === "radius" && "bg-accent")}
              >
                <Circle className="mr-2 h-4 w-4" />
                Radius
              </DropdownMenuItem>
              {/* ✅ FIXED: Added diameter measurement option */}
              <DropdownMenuItem
                onClick={() => onMeasurementModeChange(measurementMode === "diameter" ? null : "diameter")}
                className={cn(measurementMode === "diameter" && "bg-accent")}
              >
                <Circle className="mr-2 h-4 w-4" />
                Diameter
              </DropdownMenuItem>
              {measurementMode && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => onMeasurementModeChange(null)}>
                    <X className="mr-2 h-4 w-4" />
                    Cancel Measurement
                  </DropdownMenuItem>
                </>
              )}
              {measurementCount > 0 && onClearMeasurements && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={onClearMeasurements} className="text-destructive">
                    <X className="mr-2 h-4 w-4" />
                    Clear All ({measurementCount})
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>

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
                <Layers className="mr-2 h-4 w-4" />
                {shadowsEnabled ? "Disable" : "Enable"} Shadows
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onToggleSSAO}>
                <Circle className="mr-2 h-4 w-4" />
                {ssaoEnabled ? "Disable" : "Enable"} SSAO
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Section Control Panel - appears below toolbar when active */}
      {showSectionPanel && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-40">
          <div className="bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80 border rounded-lg shadow-lg p-4 min-w-[400px]">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold">Section Plane Controls</h3>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={handleCloseSectionPanel} title="Close">
                <X className="h-3 w-3" />
              </Button>
            </div>

            {/* ✅ FIXED: Plane Selection - Removed individual X button */}
            <div className="flex gap-2 mb-4">
              <Button
                variant={sectionPlane === "xy" ? "default" : "outline"}
                size="sm"
                onClick={() => handleSectionPlaneSelect("xy")}
                className="flex-1"
              >
                XY Plane
              </Button>
              <Button
                variant={sectionPlane === "xz" ? "default" : "outline"}
                size="sm"
                onClick={() => handleSectionPlaneSelect("xz")}
                className="flex-1"
              >
                XZ Plane
              </Button>
              <Button
                variant={sectionPlane === "yz" ? "default" : "outline"}
                size="sm"
                onClick={() => handleSectionPlaneSelect("yz")}
                className="flex-1"
              >
                YZ Plane
              </Button>
            </div>

            {/* Position Slider - Uses proportional steps */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Position</span>
                <span className="font-mono">{sectionPosition?.toFixed(2)} mm</span>
              </div>

              <Slider
                value={[sectionPosition || 0]}
                onValueChange={(values) => onSectionPositionChange?.(values[0])}
                min={sectionRange.min}
                max={sectionRange.max}
                step={sectionRange.step}
                className="w-full"
              />

              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>{sectionRange.min.toFixed(1)}</span>
                <span className="text-center text-muted-foreground/70">Step: {sectionRange.step.toFixed(2)}mm</span>
                <span>{sectionRange.max.toFixed(1)}</span>
              </div>
            </div>

            <div className="mt-3 text-[10px] text-muted-foreground">
              💡 Tip: Slider step size is proportional to part size ({sectionRange.step.toFixed(2)}mm per step)
            </div>
          </div>
        </div>
      )}
    </>
  );
}
