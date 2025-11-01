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
import { useState } from "react";


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

  // ✅ CRITICAL FIX: Only show panel when measurement tool is active OR measurements exist
  const shouldShowPanel = activeTool !== null || measurements.length > 0;

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
      <div className="absolute top-24 left-5 z-30">
        <Button variant="default" size="sm" onClick={() => setIsMinimized(false)} className="shadow-lg">
          <Ruler className="w-4 h-4 mr-2" />
          Measurements ({measurements.length})
        </Button>
      </div>
    );
  }

  return (
    <div className="absolute top-24 left-5 z-30 w-60 bg-white/95 backdrop-blur-sm rounded-lg shadow-2xl border border-gray-200">
      {/* Header */}
      <div className="p-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Ruler className="w-4 h-4 text-gray-600" />
            <h3 className="text-xs font-semibold text-gray-900">Measurements</h3>
            <Badge variant="secondary" className="text-xs h-5">
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
          {/* Measurements List Header */}
          {measurements.length > 0 && (
            <div className="px-3 py-2 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setShowList(!showList)}
                  className="flex items-center gap-1 text-xs font-medium text-gray-700 hover:text-gray-900"
                >
                  {showList ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
                  List ({measurements.length})
                </button>
                <div className="flex gap-1">
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={exportToCSV} title="Export CSV">
                    <Download className="w-3 h-3" />
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
            <ScrollArea className="max-h-64">
              <div className="p-2 space-y-1.5">
                {measurements.map((measurement, index) => (
                  <div
                    key={measurement.id}
                    className="p-2 bg-gray-50 rounded border border-gray-200 hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1 mb-0.5">
                          <span className="text-xs font-medium text-gray-500">#{index + 1}</span>
                          <Badge variant="outline" className="text-xs h-4 px-1 capitalize">
                            {measurement.type}
                          </Badge>
                        </div>
                        <div className="text-xs font-bold text-gray-900 truncate">{measurement.label}</div>

                        {/* Backend source indicator */}
                        {measurement.metadata?.backendMatch && (
                          <div className="mt-1">
                            <Badge variant="default" className="text-[10px] h-4 px-1">
                              <CheckCircle2 className="w-2.5 h-2.5 mr-0.5" />
                              Backend
                            </Badge>
                          </div>
                        )}

                        {/* Show edge type metadata for edge-select measurements */}
                        {measurement.type === "edge-select" && measurement.metadata?.edgeType && (
                          <div className="text-xs text-muted-foreground mt-1">
                            Type: {measurement.metadata.edgeType.toUpperCase()}
                            {measurement.metadata.arcRadius && ` | R: ${measurement.metadata.arcRadius.toFixed(2)}mm`}
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
        </>
      )}
    </div>
  );
}
