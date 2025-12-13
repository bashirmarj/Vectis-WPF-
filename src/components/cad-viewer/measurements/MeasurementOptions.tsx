import React from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useMeasurementStore } from "@/stores/measurementStore";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";

export function MeasurementOptions() {
    const [isExpanded, setIsExpanded] = React.useState(false);
    const { selectedUnits, precision, showXYZ, setUnits, setPrecision, toggleShowXYZ } = useMeasurementStore();

    return (
        <div className="border-t border-[#c8c8c8]">
            {/* Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full px-4 py-2 bg-[#f5f5f5] flex items-center justify-between hover:bg-[#e8e8e8] transition-colors"
            >
                <span className="text-xs font-medium text-[#222222]">Options</span>
                {isExpanded ? (
                    <ChevronUp className="w-3.5 h-3.5 text-[#666666]" />
                ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-[#666666]" />
                )}
            </button>

            {/* Options Content */}
            {isExpanded && (
                <div className="p-4 space-y-4 bg-white">
                    {/* Units Selector */}
                    <div className="space-y-2">
                        <Label htmlFor="units" className="text-xs font-medium text-[#222222]">
                            Units
                        </Label>
                        <Select value={selectedUnits} onValueChange={(value: 'mm' | 'inch' | 'dual') => setUnits(value)}>
                            <SelectTrigger id="units" className="h-8 text-xs border-[#c8c8c8]">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="mm">Millimeters (mm)</SelectItem>
                                <SelectItem value="inch">Inches (in)</SelectItem>
                                <SelectItem value="dual">Dual (mm + in)</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Precision Slider */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <Label htmlFor="precision" className="text-xs font-medium text-[#222222]">
                                Precision
                            </Label>
                            <span className="text-xs text-[#666666] font-mono">{precision} decimals</span>
                        </div>
                        <Slider
                            id="precision"
                            min={0}
                            max={5}
                            step={1}
                            value={[precision]}
                            onValueChange={([value]) => setPrecision(value)}
                            className="w-full"
                        />
                        <div className="flex justify-between text-[10px] text-[#999999]">
                            <span>0</span>
                            <span>1</span>
                            <span>2</span>
                            <span>3</span>
                            <span>4</span>
                            <span>5</span>
                        </div>
                    </div>

                    {/* Show XYZ Deltas */}
                    <div className="flex items-center space-x-2 pt-2 border-t border-[#e8e8e8]">
                        <Checkbox
                            id="show-xyz"
                            checked={showXYZ}
                            onCheckedChange={toggleShowXYZ}
                            className="border-[#c8c8c8]"
                        />
                        <Label
                            htmlFor="show-xyz"
                            className="text-xs font-medium text-[#222222] cursor-pointer select-none"
                        >
                            Show XYZ Deltas (dX, dY, dZ)
                        </Label>
                    </div>

                    {/* Future: Projection Plane Selector */}
                    {/* 
          <div className="space-y-2 pt-2 border-t border-[#e8e8e8]">
            <Label className="text-xs font-medium text-[#222222]">
              Projection Plane
            </Label>
            <Select defaultValue="none">
              <SelectTrigger className="h-8 text-xs border-[#c8c8c8]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="xy">XY Plane</SelectItem>
                <SelectItem value="yz">YZ Plane</SelectItem>
                <SelectItem value="xz">XZ Plane</SelectItem>
              </SelectContent>
            </Select>
          </div>
          */}
                </div>
            )}
        </div>
    );
}
