import React from "react";
import { Ruler, ArrowLeftRight, RotateCw, Move } from "lucide-react";
import { useMeasurementStore, MeasurementMode } from "@/stores/measurementStore";
import { Button } from "@/components/ui/button";

const modes: { value: MeasurementMode; icon: React.ReactNode; label: string; tooltip: string }[] = [
    {
        value: "single",
        icon: <Ruler className="w-4 h-4" />,
        label: "Single",
        tooltip: "Measure single entity (edge, face, vertex)",
    },
    {
        value: "distance",
        icon: <ArrowLeftRight className="w-4 h-4" />,
        label: "Distance",
        tooltip: "Measure distance between two entities",
    },
    {
        value: "angle",
        icon: <RotateCw className="w-4 h-4" />,
        label: "Angle",
        tooltip: "Measure angle between faces or edges",
    },
    {
        value: "point-to-point",
        icon: <Move className="w-4 h-4" />,
        label: "Point",
        tooltip: "Measure distance between two points",
    },
];

export function MeasurementModeSelector() {
    const { measurementMode, setMeasurementMode } = useMeasurementStore();

    return (
        <div className="px-4 py-2.5 bg-[#f5f5f5] border-b border-[#c8c8c8]">
            <div className="mb-1.5">
                <span className="text-xs font-medium text-[#222222]">Measurement Mode</span>
            </div>
            <div className="grid grid-cols-4 gap-1">
                {modes.map((mode) => (
                    <Button
                        key={mode.value}
                        variant="ghost"
                        size="sm"
                        onClick={() => setMeasurementMode(mode.value)}
                        title={mode.tooltip}
                        className={`
              h-9 flex flex-col items-center justify-center gap-0.5 px-2
              ${measurementMode === mode.value
                                ? "bg-[#0066cc] text-white hover:bg-[#005bb5] hover:text-white"
                                : "bg-white text-[#666666] hover:bg-[#e8e8e8] hover:text-[#222222] border border-[#d0d0d0]"
                            }
            `}
                    >
                        {mode.icon}
                        <span className="text-[10px] font-medium leading-none">{mode.label}</span>
                    </Button>
                ))}
            </div>
        </div>
    );
}
