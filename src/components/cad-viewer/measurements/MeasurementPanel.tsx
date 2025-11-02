import { Ruler, MoveVertical, Triangle } from 'lucide-react';

interface MeasurementPanelProps {
  markerCount: number;
  measurements?: {
    pointsDistance?: number;
    parallelFacesDistance?: number | null;
    facesAngle?: number;
  } | null;
}

export const MeasurementPanel = ({ markerCount, measurements }: MeasurementPanelProps) => {
  const getMessage = () => {
    if (markerCount === 0) return "Select a point.";
    if (markerCount === 1) return "Select another point.";
    return null;
  };

  const message = getMessage();

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-white border-2 border-red-500 rounded px-4 py-2 text-sm font-medium shadow-lg z-50 pointer-events-none">
      {message ? (
        <span className="text-gray-900">{message}</span>
      ) : (
        <div className="flex gap-6 items-center">
          {measurements?.pointsDistance !== undefined && (
            <div className="flex items-center gap-2">
              <Ruler className="w-4 h-4 text-gray-700" />
              <span className="text-gray-900">{measurements.pointsDistance.toFixed(3)} mm</span>
            </div>
          )}
          {measurements?.parallelFacesDistance !== undefined && measurements.parallelFacesDistance !== null && (
            <div className="flex items-center gap-2">
              <MoveVertical className="w-4 h-4 text-gray-700" />
              <span className="text-gray-900">{measurements.parallelFacesDistance.toFixed(3)} mm</span>
            </div>
          )}
          {measurements?.facesAngle !== undefined && (
            <div className="flex items-center gap-2">
              <Triangle className="w-4 h-4 text-gray-700" />
              <span className="text-gray-900">{measurements.facesAngle.toFixed(1)}°</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
