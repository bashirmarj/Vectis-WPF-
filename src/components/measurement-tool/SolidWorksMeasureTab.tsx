/**
 * SolidWorksMeasureTab.tsx
 *
 * SolidWorks-style measurement tab component that replicates the exact UI/UX
 * of SolidWorks' Measure tool. This is a STANDALONE component that integrates
 * with the existing CAD viewer WITHOUT modifying any existing code.
 *
 * Features:
 * - Unified measurement tool (single tool, multiple measurement types)
 * - Point-to-point distance with XYZ deltas
 * - Edge measurements (lines, circles/diameters, arcs/radii)
 * - Face area calculations
 * - Face-to-face distance and angle
 * - Metric/Imperial unit toggle
 * - Professional SolidWorks-style UI
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Ruler,
  Circle,
  Square,
  Layers,
  Trash2,
  Download,
  Eye,
  EyeOff,
  X,
  ChevronDown,
  ChevronUp,
  Move3D,
  Copy,
  RotateCcw,
  Info,
  ArrowRight,
} from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useSolidWorksMeasureStore, type SWMeasurement, type UnitSystem } from '@/stores/solidworksMeasureStore';

interface SolidWorksMeasureTabProps {
  onClose?: () => void;
  onMeasurementModeChange?: (enabled: boolean) => void;
  isMeasuring?: boolean;
}

/**
 * SolidWorks-style Measure Tab Component
 * Positioned as a floating panel that matches SolidWorks' measure dialog
 */
export const SolidWorksMeasureTab: React.FC<SolidWorksMeasureTabProps> = ({
  onClose,
  onMeasurementModeChange,
  isMeasuring = false,
}) => {
  const {
    measurements,
    activeMeasurement,
    unitSystem,
    showDeltas,
    precision,
    setUnitSystem,
    setShowDeltas,
    setPrecision,
    removeMeasurement,
    toggleMeasurementVisibility,
    clearAllMeasurements,
    setActiveMeasurement,
  } = useSolidWorksMeasureStore();

  const [isExpanded, setIsExpanded] = useState(true);
  const [showMeasurementList, setShowMeasurementList] = useState(true);
  const [position, setPosition] = useState({ x: 20, y: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Drag handlers for floating panel
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.sw-drag-handle')) {
      setIsDragging(true);
      setDragOffset({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      });
    }
  }, [position]);

  React.useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      setPosition({
        x: Math.max(0, e.clientX - dragOffset.x),
        y: Math.max(0, e.clientY - dragOffset.y),
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

  // Format measurement value based on precision and unit system
  const formatValue = useCallback((value: number, unit: string) => {
    if (unit === 'deg') {
      return `${value.toFixed(precision)}°`;
    }
    if (unit === 'mm²' || unit === 'in²') {
      return `${value.toFixed(precision)} ${unit}`;
    }
    const displayUnit = unitSystem === 'imperial' ? 'in' : 'mm';
    const displayValue = unitSystem === 'imperial' ? value * 0.0393701 : value;
    return `${displayValue.toFixed(precision)} ${displayUnit}`;
  }, [precision, unitSystem]);

  // Get icon for measurement type
  const getMeasurementIcon = (type: SWMeasurement['type']) => {
    switch (type) {
      case 'point_to_point':
        return <Ruler className="w-4 h-4" />;
      case 'edge_diameter':
      case 'edge_radius':
        return <Circle className="w-4 h-4" />;
      case 'edge_length':
        return <ArrowRight className="w-4 h-4" />;
      case 'face_area':
        return <Square className="w-4 h-4" />;
      case 'face_to_face':
        return <Layers className="w-4 h-4" />;
      default:
        return <Ruler className="w-4 h-4" />;
    }
  };

  // Get type label for display
  const getTypeLabel = (type: SWMeasurement['type']) => {
    switch (type) {
      case 'point_to_point':
        return 'Distance';
      case 'edge_diameter':
        return 'Diameter';
      case 'edge_radius':
        return 'Radius';
      case 'edge_length':
        return 'Length';
      case 'face_area':
        return 'Area';
      case 'face_to_face':
        return 'Face Distance';
      default:
        return 'Measurement';
    }
  };

  // Export measurements to CSV
  const exportToCSV = useCallback(() => {
    if (measurements.length === 0) return;

    const headers = ['#', 'Type', 'Value', 'Unit', 'Delta X', 'Delta Y', 'Delta Z', 'Created'];
    const rows = measurements.map((m, idx) => [
      idx + 1,
      getTypeLabel(m.type),
      m.value.toFixed(precision),
      m.unit,
      m.deltaX?.toFixed(precision) || '',
      m.deltaY?.toFixed(precision) || '',
      m.deltaZ?.toFixed(precision) || '',
      m.createdAt.toLocaleString(),
    ]);

    const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `solidworks-measurements-${Date.now()}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, [measurements, precision]);

  // Copy measurement to clipboard
  const copyMeasurement = useCallback((measurement: SWMeasurement) => {
    const text = `${getTypeLabel(measurement.type)}: ${formatValue(measurement.value, measurement.unit)}`;
    navigator.clipboard.writeText(text);
  }, [formatValue]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    if (measurements.length === 0) return null;

    const distances = measurements.filter(m =>
      m.type === 'point_to_point' || m.type === 'edge_length'
    );

    if (distances.length === 0) return null;

    const values = distances.map(m => m.value);
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    return { count: distances.length, sum, avg, min, max };
  }, [measurements]);

  return (
    <TooltipProvider>
      <div
        className="fixed z-50 bg-white rounded-lg shadow-2xl border border-gray-300"
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
          width: '320px',
          cursor: isDragging ? 'grabbing' : 'default',
          fontFamily: '"Segoe UI", Arial, sans-serif',
        }}
        onMouseDown={handleMouseDown}
      >
        {/* SolidWorks-style Header */}
        <div className="sw-drag-handle cursor-grab active:cursor-grabbing bg-gradient-to-r from-[#1a5fb4] to-[#2472c8] text-white px-3 py-2 rounded-t-lg flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Ruler className="w-5 h-5" />
            <span className="font-semibold text-sm">Measure</span>
            {measurements.length > 0 && (
              <Badge variant="secondary" className="bg-white/20 text-white text-xs">
                {measurements.length}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-white/80 hover:text-white hover:bg-white/20"
                  onClick={() => setIsExpanded(!isExpanded)}
                >
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{isExpanded ? 'Collapse' : 'Expand'}</TooltipContent>
            </Tooltip>
            {onClose && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-white/80 hover:text-white hover:bg-white/20"
                    onClick={onClose}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Close</TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>

        {isExpanded && (
          <>
            {/* Unit System Toggle */}
            <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <Label className="text-xs font-medium text-gray-600">Units:</Label>
                    <div className="flex items-center gap-1 bg-white rounded border border-gray-300 p-0.5">
                      <button
                        onClick={() => setUnitSystem('metric')}
                        className={`px-2 py-0.5 text-xs rounded transition-colors ${
                          unitSystem === 'metric'
                            ? 'bg-[#1a5fb4] text-white'
                            : 'text-gray-600 hover:bg-gray-100'
                        }`}
                      >
                        mm
                      </button>
                      <button
                        onClick={() => setUnitSystem('imperial')}
                        className={`px-2 py-0.5 text-xs rounded transition-colors ${
                          unitSystem === 'imperial'
                            ? 'bg-[#1a5fb4] text-white'
                            : 'text-gray-600 hover:bg-gray-100'
                        }`}
                      >
                        in
                      </button>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-1">
                        <Switch
                          id="show-deltas"
                          checked={showDeltas}
                          onCheckedChange={setShowDeltas}
                          className="scale-75"
                        />
                        <Label htmlFor="show-deltas" className="text-xs text-gray-600 cursor-pointer">
                          XYZ
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>Show XYZ deltas</TooltipContent>
                  </Tooltip>
                </div>
              </div>
            </div>

            {/* Active Measurement Display (SolidWorks-style) */}
            {activeMeasurement && (
              <div className="px-3 py-3 bg-[#f8f9fa] border-b border-gray-200">
                <div className="flex items-start gap-2">
                  <div className="p-1.5 bg-[#1a5fb4] rounded text-white">
                    {getMeasurementIcon(activeMeasurement.type)}
                  </div>
                  <div className="flex-1">
                    <div className="text-xs text-gray-500 uppercase tracking-wide">
                      {getTypeLabel(activeMeasurement.type)}
                    </div>
                    <div className="text-2xl font-bold text-gray-900 tabular-nums">
                      {formatValue(activeMeasurement.value, activeMeasurement.unit)}
                    </div>

                    {/* Delta values (SolidWorks style) */}
                    {showDeltas && activeMeasurement.deltaX !== undefined && (
                      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                        <div className="bg-white px-2 py-1 rounded border border-gray-200">
                          <span className="text-red-600 font-medium">ΔX:</span>{' '}
                          <span className="tabular-nums">{formatValue(activeMeasurement.deltaX, 'mm')}</span>
                        </div>
                        <div className="bg-white px-2 py-1 rounded border border-gray-200">
                          <span className="text-green-600 font-medium">ΔY:</span>{' '}
                          <span className="tabular-nums">{formatValue(activeMeasurement.deltaY || 0, 'mm')}</span>
                        </div>
                        <div className="bg-white px-2 py-1 rounded border border-gray-200">
                          <span className="text-blue-600 font-medium">ΔZ:</span>{' '}
                          <span className="tabular-nums">{formatValue(activeMeasurement.deltaZ || 0, 'mm')}</span>
                        </div>
                      </div>
                    )}

                    {/* Additional info for face-to-face */}
                    {activeMeasurement.type === 'face_to_face' && activeMeasurement.angle !== undefined && (
                      <div className="mt-2 text-xs text-gray-600">
                        Angle: <span className="font-medium">{activeMeasurement.angle.toFixed(1)}°</span>
                        {activeMeasurement.isParallel && (
                          <Badge variant="outline" className="ml-2 text-[10px]">Parallel</Badge>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Instructions when no measurement */}
            {!activeMeasurement && measurements.length === 0 && (
              <div className="px-4 py-6 text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-gray-100 mb-3">
                  <Ruler className="w-6 h-6 text-gray-400" />
                </div>
                <p className="text-sm text-gray-600 mb-1">Click to measure</p>
                <p className="text-xs text-gray-400">
                  Select edges, faces, or points on the model
                </p>
              </div>
            )}

            {/* Measurements List */}
            {measurements.length > 0 && (
              <>
                <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                  <button
                    onClick={() => setShowMeasurementList(!showMeasurementList)}
                    className="flex items-center gap-1 text-xs font-medium text-gray-700 hover:text-gray-900"
                  >
                    {showMeasurementList ? (
                      <ChevronDown className="w-3 h-3" />
                    ) : (
                      <ChevronUp className="w-3 h-3" />
                    )}
                    Measurements ({measurements.length})
                  </button>
                  <div className="flex items-center gap-1">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-gray-500 hover:text-gray-700"
                          onClick={exportToCSV}
                        >
                          <Download className="w-3 h-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Export CSV</TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-red-500 hover:text-red-700 hover:bg-red-50"
                          onClick={clearAllMeasurements}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Clear All</TooltipContent>
                    </Tooltip>
                  </div>
                </div>

                {showMeasurementList && (
                  <ScrollArea className="max-h-60">
                    <div className="p-2 space-y-1">
                      {measurements.map((measurement, index) => (
                        <div
                          key={measurement.id}
                          className={`group flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${
                            activeMeasurement?.id === measurement.id
                              ? 'bg-[#e8f0fe] border border-[#1a5fb4]'
                              : 'bg-white border border-gray-200 hover:bg-gray-50'
                          }`}
                          onClick={() => setActiveMeasurement(measurement)}
                        >
                          <div className={`p-1 rounded ${
                            activeMeasurement?.id === measurement.id
                              ? 'bg-[#1a5fb4] text-white'
                              : 'bg-gray-100 text-gray-600'
                          }`}>
                            {getMeasurementIcon(measurement.type)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-gray-400">#{index + 1}</span>
                              <span className="text-xs text-gray-600">{getTypeLabel(measurement.type)}</span>
                            </div>
                            <div className="text-sm font-semibold text-gray-900 tabular-nums truncate">
                              {formatValue(measurement.value, measurement.unit)}
                            </div>
                          </div>
                          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    copyMeasurement(measurement);
                                  }}
                                >
                                  <Copy className="w-3 h-3" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>Copy</TooltipContent>
                            </Tooltip>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleMeasurementVisibility(measurement.id);
                                  }}
                                >
                                  {measurement.visible ? (
                                    <Eye className="w-3 h-3" />
                                  ) : (
                                    <EyeOff className="w-3 h-3 text-gray-400" />
                                  )}
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>{measurement.visible ? 'Hide' : 'Show'}</TooltipContent>
                            </Tooltip>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6 text-red-500 hover:text-red-700"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    removeMeasurement(measurement.id);
                                  }}
                                >
                                  <X className="w-3 h-3" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>Delete</TooltipContent>
                            </Tooltip>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}

                {/* Summary Statistics */}
                {stats && (
                  <div className="px-3 py-2 bg-gray-50 border-t border-gray-200">
                    <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1">
                      Summary ({stats.count} distances)
                    </div>
                    <div className="grid grid-cols-4 gap-1 text-xs">
                      <div className="text-center">
                        <div className="text-gray-400">Sum</div>
                        <div className="font-medium tabular-nums">{formatValue(stats.sum, 'mm')}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-400">Avg</div>
                        <div className="font-medium tabular-nums">{formatValue(stats.avg, 'mm')}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-400">Min</div>
                        <div className="font-medium tabular-nums">{formatValue(stats.min, 'mm')}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-400">Max</div>
                        <div className="font-medium tabular-nums">{formatValue(stats.max, 'mm')}</div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Footer with File Info */}
            <div className="px-3 py-2 bg-gray-100 rounded-b-lg border-t border-gray-200 flex items-center justify-between">
              <div className="flex items-center gap-1 text-[10px] text-gray-500">
                <Info className="w-3 h-3" />
                <span>Click edges or faces to measure</span>
              </div>
              <div className="text-[10px] text-gray-400">
                {precision}dp
              </div>
            </div>
          </>
        )}
      </div>
    </TooltipProvider>
  );
};

export default SolidWorksMeasureTab;
