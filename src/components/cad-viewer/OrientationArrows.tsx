// OrientationArrows.tsx
// HTML overlay for orientation cube rotation controls

import { useState } from "react";
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface OrientationArrowsProps {
  onRotateUp?: () => void;
  onRotateDown?: () => void;
  onRotateLeft?: () => void;
  onRotateRight?: () => void;
  onRotateClockwise?: () => void;
  onRotateCounterClockwise?: () => void;
}

export function OrientationArrows({
  onRotateUp,
  onRotateDown,
  onRotateLeft,
  onRotateRight,
  onRotateClockwise,
  onRotateCounterClockwise,
}: OrientationArrowsProps) {
  const [activeButton, setActiveButton] = useState<string | null>(null);

  const handleClick = (direction: string, callback?: () => void) => {
    setActiveButton(direction);
    callback?.();
    setTimeout(() => setActiveButton(null), 150);
  };

  return (
    <TooltipProvider>
      <div className="absolute top-4 right-4 z-50 pointer-events-auto">
        <div className="relative w-[160px] h-[160px]">
          {/* Up Arrow */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleClick("up", onRotateUp)}
                className={`absolute top-0 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-background/80 backdrop-blur-sm border shadow-sm transition-all hover:scale-110 hover:bg-primary/10 ${
                  activeButton === "up" ? "scale-95 bg-primary/20" : ""
                }`}
              >
                <ChevronUp className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">Rotate Up (90°)</TooltipContent>
          </Tooltip>

          {/* Down Arrow */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleClick("down", onRotateDown)}
                className={`absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-background/80 backdrop-blur-sm border shadow-sm transition-all hover:scale-110 hover:bg-primary/10 ${
                  activeButton === "down" ? "scale-95 bg-primary/20" : ""
                }`}
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Rotate Down (90°)</TooltipContent>
          </Tooltip>

          {/* Left Arrow */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleClick("left", onRotateLeft)}
                className={`absolute left-0 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-background/80 backdrop-blur-sm border shadow-sm transition-all hover:scale-110 hover:bg-primary/10 ${
                  activeButton === "left" ? "scale-95 bg-primary/20" : ""
                }`}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">Rotate Left (90°)</TooltipContent>
          </Tooltip>

          {/* Right Arrow */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleClick("right", onRotateRight)}
                className={`absolute right-0 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-background/80 backdrop-blur-sm border shadow-sm transition-all hover:scale-110 hover:bg-primary/10 ${
                  activeButton === "right" ? "scale-95 bg-primary/20" : ""
                }`}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Rotate Right (90°)</TooltipContent>
          </Tooltip>

          {/* Clockwise Arrow (Top-Right) */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleClick("cw", onRotateClockwise)}
                className={`absolute top-3 right-3 w-7 h-7 rounded-full bg-background/80 backdrop-blur-sm border shadow-sm transition-all hover:scale-110 hover:bg-primary/10 ${
                  activeButton === "cw" ? "scale-95 bg-primary/20" : ""
                }`}
              >
                <RotateCw className="h-3.5 w-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Rotate Clockwise (90°)</TooltipContent>
          </Tooltip>

          {/* Counter-Clockwise Arrow (Top-Left) */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleClick("ccw", onRotateCounterClockwise)}
                className={`absolute top-3 left-3 w-7 h-7 rounded-full bg-background/80 backdrop-blur-sm border shadow-sm transition-all hover:scale-110 hover:bg-primary/10 ${
                  activeButton === "ccw" ? "scale-95 bg-primary/20" : ""
                }`}
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">Rotate Counter-Clockwise (90°)</TooltipContent>
          </Tooltip>

          {/* Help text */}
          <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap">
            <p className="text-xs text-muted-foreground">Click arrows to rotate view</p>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
