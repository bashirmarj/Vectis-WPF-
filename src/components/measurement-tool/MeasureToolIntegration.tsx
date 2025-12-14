/**
 * MeasureToolIntegration.tsx
 *
 * Integration wrapper that demonstrates how to add the SolidWorks-style
 * measurement tool to the existing CAD viewer WITHOUT modifying any existing code.
 *
 * This component can be used as a drop-in replacement or overlay for the
 * existing CADViewer component.
 *
 * Usage:
 * ```tsx
 * <MeasureToolIntegration meshData={meshData}>
 *   <CADViewer meshData={meshData} />
 * </MeasureToolIntegration>
 * ```
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Ruler, X } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { SolidWorksMeasureTab } from './SolidWorksMeasureTab';
import { useSolidWorksMeasureStore } from '@/stores/solidworksMeasureStore';

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_face_ids?: number[];
  face_classifications?: any[];
  tagged_edges?: any[];
  [key: string]: any;
}

interface MeasureToolIntegrationProps {
  children: React.ReactNode;
  meshData?: MeshData | null;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  showToolbarButton?: boolean;
  initialEnabled?: boolean;
  onMeasurementCreated?: (measurement: any) => void;
  onMeasurementDeleted?: (id: string) => void;
}

/**
 * MeasureToolIntegration Component
 *
 * Wraps the existing CAD viewer and adds measurement functionality
 * as an overlay without modifying the original component.
 */
export const MeasureToolIntegration: React.FC<MeasureToolIntegrationProps> = ({
  children,
  meshData,
  position = 'top-left',
  showToolbarButton = true,
  initialEnabled = false,
  onMeasurementCreated,
  onMeasurementDeleted,
}) => {
  const [measurementEnabled, setMeasurementEnabled] = useState(initialEnabled);
  const { setIsActive, measurements } = useSolidWorksMeasureStore();

  // Track measurement count for callbacks
  const prevMeasurementCount = useRef(measurements.length);

  // Sync store active state with local state
  useEffect(() => {
    setIsActive(measurementEnabled);
  }, [measurementEnabled, setIsActive]);

  // Notify on measurement changes
  useEffect(() => {
    if (measurements.length > prevMeasurementCount.current) {
      // New measurement added
      const newMeasurement = measurements[measurements.length - 1];
      onMeasurementCreated?.(newMeasurement);
    }
    prevMeasurementCount.current = measurements.length;
  }, [measurements.length, onMeasurementCreated]);

  // Toggle measurement mode
  const toggleMeasurement = useCallback(() => {
    setMeasurementEnabled(prev => !prev);
  }, []);

  // Close measurement panel
  const closeMeasurement = useCallback(() => {
    setMeasurementEnabled(false);
  }, []);

  // Keyboard shortcut handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'm':
          if (!e.ctrlKey && !e.metaKey) {
            toggleMeasurement();
          }
          break;
        case 'escape':
          if (measurementEnabled) {
            closeMeasurement();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [measurementEnabled, toggleMeasurement, closeMeasurement]);

  // Position styles for the toolbar button
  const buttonPositionStyles: Record<string, React.CSSProperties> = {
    'top-left': { top: '16px', left: '16px' },
    'top-right': { top: '16px', right: '16px' },
    'bottom-left': { bottom: '16px', left: '16px' },
    'bottom-right': { bottom: '16px', right: '16px' },
  };

  return (
    <TooltipProvider>
      <div className="relative w-full h-full">
        {/* Original CAD Viewer (children) */}
        {children}

        {/* Measurement Toggle Button */}
        {showToolbarButton && (
          <div
            className="absolute z-40"
            style={buttonPositionStyles[position]}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={measurementEnabled ? 'default' : 'outline'}
                  size="sm"
                  onClick={toggleMeasurement}
                  className={`
                    gap-2 shadow-lg transition-all
                    ${measurementEnabled
                      ? 'bg-[#1a5fb4] hover:bg-[#144a8a] text-white'
                      : 'bg-white/90 hover:bg-white text-gray-700'
                    }
                  `}
                >
                  <Ruler className="w-4 h-4" />
                  <span className="hidden sm:inline">
                    {measurementEnabled ? 'Measuring' : 'Measure'}
                  </span>
                  {measurements.length > 0 && (
                    <span className={`
                      ml-1 px-1.5 py-0.5 text-xs rounded-full
                      ${measurementEnabled
                        ? 'bg-white/20 text-white'
                        : 'bg-[#1a5fb4] text-white'
                      }
                    `}>
                      {measurements.length}
                    </span>
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>{measurementEnabled ? 'Exit measurement mode' : 'Enter measurement mode'}</p>
                <p className="text-xs text-muted-foreground">Press M to toggle</p>
              </TooltipContent>
            </Tooltip>
          </div>
        )}

        {/* SolidWorks Measure Panel */}
        {measurementEnabled && (
          <SolidWorksMeasureTab
            onClose={closeMeasurement}
            onMeasurementModeChange={setMeasurementEnabled}
            isMeasuring={measurementEnabled}
          />
        )}

        {/* Measurement Mode Indicator */}
        {measurementEnabled && (
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-40">
            <div className="bg-[#1a5fb4]/90 text-white px-4 py-2 rounded-full shadow-lg backdrop-blur-sm flex items-center gap-2 text-sm">
              <Ruler className="w-4 h-4" />
              <span>Click edges or faces to measure</span>
              <button
                onClick={closeMeasurement}
                className="ml-2 p-1 hover:bg-white/20 rounded-full transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
};

/**
 * Hook to use measurement tool state outside of components
 */
export const useMeasurementTool = () => {
  const store = useSolidWorksMeasureStore();

  return {
    // State
    isActive: store.isActive,
    measurements: store.measurements,
    activeMeasurement: store.activeMeasurement,
    unitSystem: store.unitSystem,

    // Actions
    setActive: store.setIsActive,
    addMeasurement: store.addMeasurement,
    removeMeasurement: store.removeMeasurement,
    clearAll: store.clearAllMeasurements,
    setUnitSystem: store.setUnitSystem,

    // Undo/Redo
    undo: store.undo,
    redo: store.redo,
    canUndo: store.canUndo(),
    canRedo: store.canRedo(),

    // Utilities
    measurementCount: store.measurements.length,
    visibleMeasurements: store.getVisibleMeasurements(),
    exportData: store.exportMeasurements,
  };
};

export default MeasureToolIntegration;
