import { useMeasurementStore } from "@/stores/measurementStore";
import { Button } from "@/components/ui/button";
import {
  Ruler,
  Triangle,
  Circle,
  Eye,
  EyeOff,
  X,
  Download,
  Trash2,
  ChevronDown,
  ChevronUp,
  Minimize2,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  XCircle,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { useState, useEffect } from "react";
import { MeasurementOptions } from "./measurements/MeasurementOptions";
import { SelectionDisplay } from "./measurements/SelectionDisplay";
import { MeasurementModeSelector } from "./measurements/MeasurementModeSelector";



/**
 * Compact Measurement Panel Component
 * Professional, space-efficient measurement management
 * Positioned at TOP-LEFT to avoid overlapping orientation cube (top-right)
 * Only visible when measurement tool is active
 */
export function MeasurementPanel() {
  const {
    activeTool,
    measurements,
    setActiveTool,
    removeMeasurement,
    toggleMeasurementVisibility,
    clearAllMeasurements,
  } = useMeasurementStore();

  const [isExpanded, setIsExpanded] = useState(true);
  const [showList, setShowList] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const [position, setPosition] = useState({ x: 20, y: 96 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // ✅ CRITICAL FIX: Only show panel when measurement tool is active OR measurements exist
  const shouldShowPanel = activeTool !== null || measurements.length > 0;

  // Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.drag-handle')) {
      setIsDragging(true);
      setDragOffset({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      });
    }
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      setPosition({
        x: e.clientX - dragOffset.x,
        y: e.clientY - dragOffset.y,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  // Export measurements to CSV
  const exportToCSV = () => {
    if (measurements.length === 0) return;

    const headers = ["Type", "Value", "Unit", "Points", "Created"];
    const rows = measurements.map((m) => [
      m.type,
      m.value.toFixed(3),
      m.unit,
      m.points.length,
      m.createdAt.toLocaleString(),
    ]);

    const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `measurements-${Date.now()}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  // ✅ Don't render at all if no active tool and no measurements
  if (!shouldShowPanel) return null;

  // Minimized view - just a button (TOP-LEFT)
  if (isMinimized) {
    return (
      <div
        className="absolute z-30"
        style={{ left: `${position.x}px`, top: `${position.y}px` }}
      >
        <Button variant="default" size="sm" onClick={() => setIsMinimized(false)} className="shadow-lg">
          <Ruler className="w-4 h-4 mr-2" />
          Measurements ({measurements.length})
        </Button>
      </div>
    );
  }

  return (
    <div
      className="absolute z-30 bg-[#f0f0f0] backdrop-blur-sm rounded-lg shadow-2xl border border-[#c8c8c8]"
      style={{
        width: '280px', // Increased from 240px for SolidWorks-style spaciousness
        left: `${position.x}px`,
        top: `${position.y}px`,
        cursor: isDragging ? 'grabbing' : 'default'
      }}
      onMouseDown={handleMouseDown}
    >
      {/* Header - SolidWorks style */}
      <div className="px-4 py-2.5 border-b border-[#c8c8c8] bg-[#e5e5e5] drag-handle cursor-grab active:cursor-grabbing rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Ruler className="w-5 h-5 text-[#0066cc]" />
            <h3 className="text-sm font-semibold text-[#222222]">Measure</h3>
            <Badge variant="secondary" className="text-xs h-5 bg-[#0066cc] text-white border-none">
              {measurements.length}
            </Badge>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => setIsExpanded(!isExpanded)}
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => setIsMinimized(true)}
              title="Minimize"
            >
              <Minimize2 className="w-3 h-3" />
            </Button>
          </div>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Measurement Mode Selector */}
          <MeasurementModeSelector />

          {/* Selection Display */}
          <SelectionDisplay />

          {/* Measurements List Header */}
          {measurements.length > 0 && (
            <div className="px-4 py-2 border-b border-[#c8c8c8] bg-[#f5f5f5]">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setShowList(!showList)}
                  className="flex items-center gap-1 text-xs font-medium text-[#222222] hover:text-[#0066cc]"
                >
                  {showList ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
                  List ({measurements.length})
                </button>
                <div className="flex gap-1">
                  <Button variant="ghost" size="icon" className="h-6 w-6 hover:bg-[#e0e0e0]" onClick={exportToCSV} title="Export CSV">
                    <Download className="w-3 h-3 text-[#666666]" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-red-600 hover:text-red-700 hover:bg-red-50"
                    onClick={clearAllMeasurements}
                    title="Clear All"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Measurements List */}
          {showList && measurements.length > 0 && (
            <ScrollArea className="h-64">
              <div className="p-3 space-y-2">
                {measurements.map((measurement, index) => (
                  <div
                    key={measurement.id}
                    className="p-2.5 bg-white rounded border border-[#d0d0d0] hover:border-[#0066cc] transition-colors shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-xs font-medium text-[#666666]">#{index + 1}</span>
                          <Badge variant="outline" className="text-xs h-4 px-1 capitalize border-[#c8c8c8] text-[#666666]">
                            {measurement.type}
                          </Badge>
                        </div>
                        {/* Primary Value - Large and Bold like SolidWorks */}
                        <div className="text-base font-bold text-[#0066cc] truncate mb-0.5">{measurement.label}</div>

                        {/* Backend source indicator */}
                        {measurement.metadata?.backendMatch && (
                          <div className="mt-1">
                            <Badge variant="default" className="text-[10px] h-4 px-1 bg-green-600 hover:bg-green-600">
                              <CheckCircle2 className="w-2.5 h-2.5 mr-0.5" />
                              CAD Data
                            </Badge>
                          </div>
                        )}

                        {/* Show edge type or face-point metadata */}
                        {measurement.type === "measure" && measurement.metadata?.measurementSubtype === "edge" && measurement.metadata?.edgeType && (
                          <div className="text-xs text-[#666666] mt-1">
                            Type: {measurement.metadata.edgeType.toUpperCase()}
                            {measurement.metadata.arcRadius && ` | R: ${measurement.metadata.arcRadius.toFixed(2)}mm`}
                          </div>
                        )}

                        {measurement.type === "measure" && measurement.metadata?.measurementSubtype === "face-point" && (
                          <div className="text-xs text-muted-foreground mt-1">
                            Angle: {measurement.metadata.facesAngle?.toFixed(1)}°
                            {measurement.metadata.parallelFacesDistance !== null && measurement.metadata.parallelFacesDistance !== undefined &&
                              ` | ⊥: ${measurement.metadata.parallelFacesDistance.toFixed(2)}mm`}
                          </div>
                        )}

                        {/* NEW: Show XYZ deltas when enabled and available */}
                        {measurement.metadata?.deltaX !== undefined && measurement.metadata?.deltaY !== undefined && measurement.metadata?.deltaZ !== undefined && (
                          <div className="text-xs text-muted-foreground mt-1 font-mono">
                            dX: {measurement.metadata.deltaX.toFixed(2)}mm | dY: {measurement.metadata.deltaY.toFixed(2)}mm | dZ: {measurement.metadata.deltaZ.toFixed(2)}mm
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-0.5 flex-shrink-0">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => toggleMeasurementVisibility(measurement.id)}
                        >
                          {measurement.visible ? (
                            <Eye className="w-3 h-3" />
                          ) : (
                            <EyeOff className="w-3 h-3 text-gray-400" />
                          )}
                        </Button>

                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-red-600 hover:text-red-700 hover:bg-red-50"
                          onClick={() => removeMeasurement(measurement.id)}
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}

          {/* Empty State */}
          {measurements.length === 0 && (
            <div className="p-4 text-center">
              <Ruler className="w-8 h-8 mx-auto mb-2 opacity-30 text-gray-400" />
              <p className="text-xs text-gray-500">No measurements</p>
              <p className="text-xs text-gray-400 mt-0.5">Select a tool above</p>
            </div>
          )}

          {/* Measurement Options */}
          <MeasurementOptions />
        </>
      )}
    </div>
  );
}
