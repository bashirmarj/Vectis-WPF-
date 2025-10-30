import { useMeasurementStore } from '@/stores/measurementStore';
import { DimensionAnnotation } from './DimensionAnnotation';

export function MeasurementRenderer() {
  const measurements = useMeasurementStore((state) => state.measurements);

  return (
    <>
      {measurements.map((measurement) => {
        if (!measurement.visible) return null;

        // Use DimensionAnnotation component for all measurements
        return <DimensionAnnotation key={measurement.id} measurement={measurement} />;
      })}
    </>
  );
}
