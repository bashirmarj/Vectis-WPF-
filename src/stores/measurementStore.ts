import { create } from "zustand";
import * as THREE from "three";

export type MeasurementType = "edge-select" | "face-to-face"; // Smart Edge Select: Auto-detect line/arc/circle | Face-to-face distance

export type SnapType = "vertex" | "edge" | "midpoint" | "center" | "intersection" | "face";

export interface MeasurementPoint {
  id: string;
  position: THREE.Vector3;
  normal?: THREE.Vector3;
  surfaceType?: SnapType;
  edgeIndex?: number;
  faceIndex?: number;
}

export interface Measurement {
  id: string;
  type: MeasurementType;
  points: MeasurementPoint[];
  value: number;
  unit: "mm" | "deg" | "coord";
  label: string;
  color: string;
  visible: boolean;
  createdAt: Date;
  metadata?: {
    deltaX?: number;
    deltaY?: number;
    deltaZ?: number;
    edgeType?: "line" | "arc" | "circle";
    arcRadius?: number;
    arcCenter?: THREE.Vector3;
    edgeStart?: THREE.Vector3;
    edgeEnd?: THREE.Vector3;
    cylindrical?: boolean;
    radius?: number;
    center?: THREE.Vector3 | [number, number, number];
    axis?: THREE.Vector3;
    faceVertices?: number[];
    backendMatch: boolean;
    // Face-to-face metadata
    faceId?: number;
    faceType?: "internal" | "external" | "through" | "planar";
    surfaceType?: "cylinder" | "plane" | "other";
    faceCenter?: [number, number, number];
    faceNormal?: [number, number, number];
    faceRadius?: number;
    face1Id?: number;
    face2Id?: number;
    distanceType?: "plane-to-plane" | "cylinder-to-cylinder" | "plane-to-cylinder" | "point-to-point";
  };
}

// Command pattern for undo/redo
export interface MeasurementCommand {
  execute: () => void;
  undo: () => void;
}

interface MeasurementStore {
  measurements: Measurement[];
  activeTool: MeasurementType | null;
  tempPoints: MeasurementPoint[];
  snapEnabled: boolean;
  snapDistance: number;
  activeSnapTypes: SnapType[];
  hoverPoint: MeasurementPoint | null;

  // Undo/Redo stacks
  undoStack: MeasurementCommand[];
  redoStack: MeasurementCommand[];

  // Actions
  setActiveTool: (tool: MeasurementType | null) => void;
  addTempPoint: (point: MeasurementPoint) => void;
  clearTempPoints: () => void;
  addMeasurement: (measurement: Measurement) => void;
  removeMeasurement: (id: string) => void;
  toggleMeasurementVisibility: (id: string) => void;
  clearAllMeasurements: () => void;
  updateMeasurement: (id: string, updates: Partial<Measurement>) => void;
  setSnapEnabled: (enabled: boolean) => void;
  setSnapDistance: (distance: number) => void;
  setActiveSnapTypes: (types: SnapType[]) => void;
  toggleSnapType: (type: SnapType) => void;
  setHoverPoint: (point: MeasurementPoint | null) => void;

  // Command pattern methods
  executeCommand: (command: MeasurementCommand) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
}

export const useMeasurementStore = create<MeasurementStore>((set, get) => ({
  measurements: [],
  activeTool: null,
  tempPoints: [],
  snapEnabled: true,
  snapDistance: 2,
  activeSnapTypes: ["vertex", "edge", "midpoint", "center"],
  hoverPoint: null,
  undoStack: [],
  redoStack: [],

  setActiveTool: (tool) => set({ activeTool: tool, tempPoints: [] }),

  addTempPoint: (point) =>
    set((state) => ({
      tempPoints: [...state.tempPoints, point],
    })),

  clearTempPoints: () => set({ tempPoints: [] }),

  addMeasurement: (measurement) => {
    const command: MeasurementCommand = {
      execute: () =>
        set((state) => ({
          measurements: [...state.measurements, measurement],
          tempPoints: [],
        })),
      undo: () =>
        set((state) => ({
          measurements: state.measurements.filter((m) => m.id !== measurement.id),
        })),
    };
    get().executeCommand(command);
  },

  removeMeasurement: (id) => {
    const measurement = get().measurements.find((m) => m.id === id);
    if (!measurement) return;

    const command: MeasurementCommand = {
      execute: () =>
        set((state) => ({
          measurements: state.measurements.filter((m) => m.id !== id),
        })),
      undo: () =>
        set((state) => ({
          measurements: [...state.measurements, measurement],
        })),
    };
    get().executeCommand(command);
  },

  toggleMeasurementVisibility: (id) =>
    set((state) => ({
      measurements: state.measurements.map((m) => (m.id === id ? { ...m, visible: !m.visible } : m)),
    })),

  clearAllMeasurements: () => {
    const measurements = get().measurements;
    const command: MeasurementCommand = {
      execute: () => set({ measurements: [], tempPoints: [] }),
      undo: () => set({ measurements }),
    };
    get().executeCommand(command);
  },

  updateMeasurement: (id, updates) => {
    const oldMeasurement = get().measurements.find((m) => m.id === id);
    if (!oldMeasurement) return;

    const command: MeasurementCommand = {
      execute: () =>
        set((state) => ({
          measurements: state.measurements.map((m) => (m.id === id ? { ...m, ...updates } : m)),
        })),
      undo: () =>
        set((state) => ({
          measurements: state.measurements.map((m) => (m.id === id ? oldMeasurement : m)),
        })),
    };
    get().executeCommand(command);
  },

  setSnapEnabled: (enabled) => set({ snapEnabled: enabled }),

  setSnapDistance: (distance) => set({ snapDistance: distance }),

  setActiveSnapTypes: (types) => set({ activeSnapTypes: types }),

  toggleSnapType: (type) =>
    set((state) => {
      const isActive = state.activeSnapTypes.includes(type);
      return {
        activeSnapTypes: isActive ? state.activeSnapTypes.filter((t) => t !== type) : [...state.activeSnapTypes, type],
      };
    }),

  setHoverPoint: (point) => set({ hoverPoint: point }),

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
}));
