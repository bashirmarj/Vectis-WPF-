/**
 * solidworksMeasureStore.ts
 *
 * Zustand store for SolidWorks-style measurement tool.
 * This is a STANDALONE store that does not modify the existing measurementStore.ts
 *
 * Features:
 * - Full measurement state management
 * - Undo/redo support with command pattern
 * - Unit system management (metric/imperial)
 * - Measurement persistence and export
 */

import { create } from 'zustand';
import * as THREE from 'three';

// === Type Definitions ===

export type UnitSystem = 'metric' | 'imperial';

export type MeasurementType =
  | 'point_to_point'
  | 'edge_length'
  | 'edge_diameter'
  | 'edge_radius'
  | 'face_area'
  | 'face_to_face'
  | 'angle'
  | 'coordinate';

export type EdgeType = 'line' | 'circle' | 'arc' | 'ellipse' | 'spline';
export type SurfaceType = 'plane' | 'cylinder' | 'cone' | 'sphere' | 'torus' | 'other';

export interface MeasurementPoint {
  id: string;
  position: THREE.Vector3;
  normal?: THREE.Vector3;
  faceId?: number;
  edgeId?: number;
}

export interface SWMeasurement {
  id: string;
  type: MeasurementType;
  value: number;
  unit: string;
  label: string;
  displayValue: string;
  visible: boolean;
  createdAt: Date;

  // Points for visualization
  point1?: THREE.Vector3;
  point2?: THREE.Vector3;
  points?: MeasurementPoint[];

  // Delta values (for point-to-point)
  deltaX?: number;
  deltaY?: number;
  deltaZ?: number;

  // Edge-specific
  edgeType?: EdgeType;
  diameter?: number;
  radius?: number;
  center?: THREE.Vector3;
  edgeId?: number;

  // Face-specific
  faceId?: number;
  face1Id?: number;
  face2Id?: number;
  surfaceType?: SurfaceType;
  area?: number;
  normal?: THREE.Vector3;

  // Face-to-face specific
  angle?: number;
  isParallel?: boolean;
  perpendicularDistance?: number;

  // Metadata
  backendMatch: boolean;
  confidence: number;
  processingTimeMs?: number;
}

export interface MeasurementCommand {
  execute: () => void;
  undo: () => void;
  description: string;
}

interface SolidWorksMeasureStore {
  // State
  measurements: SWMeasurement[];
  activeMeasurement: SWMeasurement | null;
  tempPoints: MeasurementPoint[];
  isActive: boolean;

  // Settings
  unitSystem: UnitSystem;
  showDeltas: boolean;
  precision: number;
  snapEnabled: boolean;
  snapDistance: number;

  // Hover state
  hoverInfo: {
    type: 'edge' | 'face' | 'point' | null;
    position?: THREE.Vector3;
    edgeInfo?: any;
    faceInfo?: any;
    label?: string;
  } | null;

  // Undo/Redo
  undoStack: MeasurementCommand[];
  redoStack: MeasurementCommand[];

  // Actions - Measurements
  addMeasurement: (measurement: SWMeasurement) => void;
  removeMeasurement: (id: string) => void;
  updateMeasurement: (id: string, updates: Partial<SWMeasurement>) => void;
  toggleMeasurementVisibility: (id: string) => void;
  clearAllMeasurements: () => void;
  setActiveMeasurement: (measurement: SWMeasurement | null) => void;

  // Actions - Temp Points
  addTempPoint: (point: MeasurementPoint) => void;
  clearTempPoints: () => void;

  // Actions - Settings
  setIsActive: (active: boolean) => void;
  setUnitSystem: (system: UnitSystem) => void;
  setShowDeltas: (show: boolean) => void;
  setPrecision: (precision: number) => void;
  setSnapEnabled: (enabled: boolean) => void;
  setSnapDistance: (distance: number) => void;

  // Actions - Hover
  setHoverInfo: (info: SolidWorksMeasureStore['hoverInfo']) => void;

  // Actions - Undo/Redo
  executeCommand: (command: MeasurementCommand) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // Utility
  getMeasurementById: (id: string) => SWMeasurement | undefined;
  getVisibleMeasurements: () => SWMeasurement[];
  exportMeasurements: () => string;
}

// Generate unique ID
const generateId = () => `sw-measure-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// === Store Implementation ===

export const useSolidWorksMeasureStore = create<SolidWorksMeasureStore>((set, get) => ({
  // Initial State
  measurements: [],
  activeMeasurement: null,
  tempPoints: [],
  isActive: false,

  // Settings
  unitSystem: 'metric',
  showDeltas: true,
  precision: 3,
  snapEnabled: true,
  snapDistance: 2.0,

  // Hover state
  hoverInfo: null,

  // Undo/Redo stacks
  undoStack: [],
  redoStack: [],

  // === Measurement Actions ===

  addMeasurement: (measurement) => {
    const command: MeasurementCommand = {
      execute: () => {
        set((state) => ({
          measurements: [...state.measurements, measurement],
          activeMeasurement: measurement,
          tempPoints: [],
        }));
      },
      undo: () => {
        set((state) => ({
          measurements: state.measurements.filter((m) => m.id !== measurement.id),
          activeMeasurement:
            state.activeMeasurement?.id === measurement.id ? null : state.activeMeasurement,
        }));
      },
      description: `Add ${measurement.type} measurement`,
    };
    get().executeCommand(command);
  },

  removeMeasurement: (id) => {
    const measurement = get().measurements.find((m) => m.id === id);
    if (!measurement) return;

    const command: MeasurementCommand = {
      execute: () => {
        set((state) => ({
          measurements: state.measurements.filter((m) => m.id !== id),
          activeMeasurement: state.activeMeasurement?.id === id ? null : state.activeMeasurement,
        }));
      },
      undo: () => {
        set((state) => ({
          measurements: [...state.measurements, measurement],
        }));
      },
      description: `Remove measurement`,
    };
    get().executeCommand(command);
  },

  updateMeasurement: (id, updates) => {
    const oldMeasurement = get().measurements.find((m) => m.id === id);
    if (!oldMeasurement) return;

    const command: MeasurementCommand = {
      execute: () => {
        set((state) => ({
          measurements: state.measurements.map((m) =>
            m.id === id ? { ...m, ...updates } : m
          ),
          activeMeasurement:
            state.activeMeasurement?.id === id
              ? { ...state.activeMeasurement, ...updates }
              : state.activeMeasurement,
        }));
      },
      undo: () => {
        set((state) => ({
          measurements: state.measurements.map((m) =>
            m.id === id ? oldMeasurement : m
          ),
          activeMeasurement:
            state.activeMeasurement?.id === id ? oldMeasurement : state.activeMeasurement,
        }));
      },
      description: `Update measurement`,
    };
    get().executeCommand(command);
  },

  toggleMeasurementVisibility: (id) => {
    set((state) => ({
      measurements: state.measurements.map((m) =>
        m.id === id ? { ...m, visible: !m.visible } : m
      ),
    }));
  },

  clearAllMeasurements: () => {
    const measurements = get().measurements;
    if (measurements.length === 0) return;

    const command: MeasurementCommand = {
      execute: () => {
        set({
          measurements: [],
          activeMeasurement: null,
          tempPoints: [],
        });
      },
      undo: () => {
        set({ measurements });
      },
      description: `Clear all measurements`,
    };
    get().executeCommand(command);
  },

  setActiveMeasurement: (measurement) => {
    set({ activeMeasurement: measurement });
  },

  // === Temp Points Actions ===

  addTempPoint: (point) => {
    set((state) => ({
      tempPoints: [...state.tempPoints, point],
    }));
  },

  clearTempPoints: () => {
    set({ tempPoints: [] });
  },

  // === Settings Actions ===

  setIsActive: (active) => {
    set({
      isActive: active,
      tempPoints: active ? [] : get().tempPoints,
      hoverInfo: active ? get().hoverInfo : null,
    });
  },

  setUnitSystem: (system) => {
    set({ unitSystem: system });
  },

  setShowDeltas: (show) => {
    set({ showDeltas: show });
  },

  setPrecision: (precision) => {
    set({ precision: Math.max(0, Math.min(6, precision)) });
  },

  setSnapEnabled: (enabled) => {
    set({ snapEnabled: enabled });
  },

  setSnapDistance: (distance) => {
    set({ snapDistance: Math.max(0.1, Math.min(10, distance)) });
  },

  // === Hover Actions ===

  setHoverInfo: (info) => {
    set({ hoverInfo: info });
  },

  // === Undo/Redo Actions ===

  executeCommand: (command) => {
    command.execute();
    set((state) => ({
      undoStack: [...state.undoStack, command],
      redoStack: [],
    }));
  },

  undo: () => {
    const state = get();
    if (state.undoStack.length === 0) return;

    const command = state.undoStack[state.undoStack.length - 1];
    command.undo();

    set((state) => ({
      undoStack: state.undoStack.slice(0, -1),
      redoStack: [...state.redoStack, command],
    }));
  },

  redo: () => {
    const state = get();
    if (state.redoStack.length === 0) return;

    const command = state.redoStack[state.redoStack.length - 1];
    command.execute();

    set((state) => ({
      redoStack: state.redoStack.slice(0, -1),
      undoStack: [...state.undoStack, command],
    }));
  },

  canUndo: () => get().undoStack.length > 0,
  canRedo: () => get().redoStack.length > 0,

  // === Utility Functions ===

  getMeasurementById: (id) => {
    return get().measurements.find((m) => m.id === id);
  },

  getVisibleMeasurements: () => {
    return get().measurements.filter((m) => m.visible);
  },

  exportMeasurements: () => {
    const { measurements, unitSystem, precision } = get();

    const data = measurements.map((m, index) => ({
      index: index + 1,
      type: m.type,
      value: m.value,
      unit: m.unit,
      displayValue: m.displayValue,
      deltaX: m.deltaX,
      deltaY: m.deltaY,
      deltaZ: m.deltaZ,
      createdAt: m.createdAt.toISOString(),
    }));

    return JSON.stringify({
      exportDate: new Date().toISOString(),
      unitSystem,
      precision,
      measurementCount: measurements.length,
      measurements: data,
    }, null, 2);
  },
}));

// === Helper Functions ===

export const createMeasurement = (
  type: MeasurementType,
  value: number,
  options: Partial<Omit<SWMeasurement, 'id' | 'type' | 'value' | 'createdAt'>> = {}
): SWMeasurement => {
  const id = generateId();
  const unit = type === 'angle' ? 'deg' : type === 'face_area' ? 'mm²' : 'mm';

  let label = '';
  let displayValue = '';

  switch (type) {
    case 'point_to_point':
      label = 'Distance';
      displayValue = `${value.toFixed(3)} mm`;
      break;
    case 'edge_length':
      label = 'Length';
      displayValue = `${value.toFixed(3)} mm`;
      break;
    case 'edge_diameter':
      label = 'Diameter';
      displayValue = `Ø ${value.toFixed(3)} mm`;
      break;
    case 'edge_radius':
      label = 'Radius';
      displayValue = `R ${value.toFixed(3)} mm`;
      break;
    case 'face_area':
      label = 'Area';
      displayValue = `${value.toFixed(3)} mm²`;
      break;
    case 'face_to_face':
      label = 'Face Distance';
      displayValue = `${value.toFixed(3)} mm`;
      break;
    case 'angle':
      label = 'Angle';
      displayValue = `${value.toFixed(2)}°`;
      break;
    case 'coordinate':
      label = 'Coordinate';
      displayValue = `${value.toFixed(3)}`;
      break;
  }

  return {
    id,
    type,
    value,
    unit,
    label,
    displayValue,
    visible: true,
    createdAt: new Date(),
    backendMatch: false,
    confidence: 1.0,
    ...options,
  };
};

export const formatMeasurementValue = (
  value: number,
  unit: string,
  unitSystem: UnitSystem,
  precision: number = 3
): string => {
  if (unit === 'deg') {
    return `${value.toFixed(precision)}°`;
  }

  if (unitSystem === 'imperial') {
    if (unit === 'mm' || unit === 'mm²') {
      const converted = unit === 'mm' ? value * 0.0393701 : value * 0.00155;
      const displayUnit = unit === 'mm' ? 'in' : 'in²';
      return `${converted.toFixed(precision)} ${displayUnit}`;
    }
  }

  return `${value.toFixed(precision)} ${unit}`;
};

export default useSolidWorksMeasureStore;
