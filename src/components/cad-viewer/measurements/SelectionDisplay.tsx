import React from "react";
import { X } from "lucide-react";
import { useMeasurementStore } from "@/stores/measurementStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export function SelectionDisplay() {
    const { currentSelections, measurementMode, removeSelection, clearSelections } = useMeasurementStore();

    if (currentSelections.length === 0) {
        return null;
    }

    // Determine entity type label
    const getEntityLabel = (selection: any, index: number) => {
        const type = selection.surfaceType || "point";
        if (type === "edge") return `Edge<${index + 1}>`;
        if (type === "face") return `Face<${index + 1}>`;
        if (type === "vertex") return `Vertex<${index + 1}>`;
        return `Point<${index + 1}>`;
    };

    return (
        <div className="px-4 py-2 bg-[#f5f5f5] border-b border-[#c8c8c8]">
            <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-[#222222]">Selection</span>
                {currentSelections.length > 1 && (
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={clearSelections}
                        className="h-5 px-2 text-xs text-[#666666] hover:text-[#0066cc] hover:bg-[#e0e0e0]"
                    >
                        Clear All
                    </Button>
                )}
            </div>

            <div className="space-y-1.5">
                {currentSelections.map((selection, index) => (
                    <div
                        key={selection.id}
                        className="flex items-center justify-between p-1.5 bg-white rounded border border-[#d0d0d0] group hover:border-[#0066cc] transition-colors"
                    >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                            <Badge variant="outline" className="text-[10px] h-4 px-1 border-[#0066cc] text-[#0066cc] font-semibold">
                                {getEntityLabel(selection, index)}
                            </Badge>
                            <span className="text-xs text-[#666666] truncate font-mono">
                                {selection.surfaceType === "vertex"
                                    ? `(${(selection.position.x * 1000).toFixed(1)}, ${(selection.position.y * 1000).toFixed(1)}, ${(selection.position.z * 1000).toFixed(1)})`
                                    : selection.surfaceType === "edge"
                                        ? "Line"
                                        : "Selected"}
                            </span>
                        </div>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeSelection(selection.id)}
                            className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-50 hover:text-red-600"
                        >
                            <X className="w-3 h-3" />
                        </Button>
                    </div>
                ))}
            </div>

            {/* Mode indicator */}
            <div className="mt-2 text-[10px] text-[#999999]">
                Mode: <span className="font-medium text-[#666666] capitalize">{measurementMode}</span>
                {measurementMode === "distance" && " (select 2 entities)"}
                {measurementMode === "angle" && " (select 3 entities)"}
            </div>
        </div>
    );
}
