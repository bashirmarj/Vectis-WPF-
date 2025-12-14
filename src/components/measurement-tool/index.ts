/**
 * Measurement Tool Module Index
 *
 * This module exports all components for the SolidWorks-style measurement tool.
 * All components are STANDALONE and do not modify any existing application code.
 *
 * Quick Start:
 * ```tsx
 * import {
 *   SolidWorksMeasureTab,
 *   SolidWorksMeasurementTool,
 *   useSolidWorksMeasureStore,
 *   MeasureToolIntegration,
 * } from '@/components/measurement-tool';
 * ```
 */

// === Main Components ===

export { SolidWorksMeasureTab } from './SolidWorksMeasureTab';
export { SolidWorksMeasurementTool } from './SolidWorksMeasurementTool';
export {
  SolidWorksMeasurementCallout,
  CompactCallout,
  HoverCallout,
} from './SolidWorksMeasurementCallout';
export {
  MeasureToolIntegration,
  useMeasurementTool,
} from './MeasureToolIntegration';

// === Store ===

export {
  useSolidWorksMeasureStore,
  createMeasurement,
  formatMeasurementValue,
  type UnitSystem,
  type MeasurementType,
  type EdgeType,
  type SurfaceType,
  type SWMeasurement,
  type MeasurementPoint,
  type MeasurementCommand,
} from '@/stores/solidworksMeasureStore';

// === API Service ===

export {
  measurementApi,
  localCalculations,
  MeasurementApiError,
  type MeasurementResultApi,
  type EdgeInfoApi,
  type FaceInfoApi,
  type GeometryAnalysisResult,
  type ConversionResult,
  type UnitSystemApi,
  type Point3DApi,
} from '@/services/measurementApiService';
