// src/components/cad-viewer/OrientationCubeControls.tsx
import { Button } from "@/components/ui/button";
import { Box, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface OrientationCubeControlsProps {
  onRotateUp?: () => void;
  onRotateDown?: () => void;
  onRotateLeft?: () => void;
  onRotateRight?: () => void;
  onRotateClockwise?: () => void;
  onRotateCounterClockwise?: () => void;
  displayMode?: "solid" | "wireframe" | "translucent";
  onDisplayModeChange?: (style: "solid" | "wireframe" | "translucent") => void;
}

export function OrientationCubeControls({
  onRotateUp,
  onRotateDown,
  onRotateLeft,
  onRotateRight,
  onRotateClockwise,
  onRotateCounterClockwise,
  displayMode = "solid",
  onDisplayModeChange,
}: OrientationCubeControlsProps) {
  const handleArrowClick = (direction: string, handler?: () => void) => {
    console.log(`⬆️ Arrow clicked: ${direction}`);
    handler?.();
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

          {/* Cube view area with arrow controls */}
          <div className="relative w-[150px] h-[150px] border-2 border-border/30 rounded-lg overflow-hidden bg-gradient-to-br from-muted/20 to-background shadow-inner">
            {/* Note: The actual 3D cube renders in the main Canvas via OrientationCubeInScene */}
            {/* This area shows where the cube appears */}
            <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground pointer-events-none">
              <span className="bg-background/80 px-2 py-1 rounded">Cube View</span>
            </div>

            {/* Arrow controls overlay */}
            <div className="absolute inset-0 pointer-events-none">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleArrowClick("UP", onRotateUp)}
                className="absolute top-1 left-1/2 -translate-x-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                title="Rotate view up (90°)"
              >
                <ChevronUp className="w-4 h-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleArrowClick("DOWN", onRotateDown)}
                className="absolute bottom-1 left-1/2 -translate-x-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                title="Rotate view down (90°)"
              >
                <ChevronDown className="w-4 h-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleArrowClick("LEFT", onRotateLeft)}
                className="absolute left-1 top-1/2 -translate-y-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                title="Rotate view left (90°)"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleArrowClick("RIGHT", onRotateRight)}
                className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                title="Rotate view right (90°)"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleArrowClick("CCW", onRotateCounterClockwise)}
                className="absolute bottom-1 left-1 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                title="Roll counter-clockwise (90°)"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleArrowClick("CW", onRotateClockwise)}
                className="absolute bottom-1 right-1 h-7 w-7 p-0 pointer-events-auto bg-background/80 hover:bg-background/95 backdrop-blur-sm border border-border/50 shadow-md transition-all hover:scale-110"
                title="Roll clockwise (90°)"
              >
                <RotateCw className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>

          {/* Display style selector */}
          <Select value={displayMode} onValueChange={onDisplayModeChange}>
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
}