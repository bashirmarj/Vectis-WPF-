import { Ruler, MoveVertical, Triangle } from 'lucide-react';
import { RadDeg, type MarkerValues } from '@/lib/faceMeasurementUtils';

interface FaceMeasurementPanelProps {
  markerCount: number;
  measurements: MarkerValues | null;
}

export function FaceMeasurementPanel({ 
  markerCount, 
  measurements 
}: FaceMeasurementPanelProps) {
  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50 pointer-events-none">
      <div className="bg-white/90 backdrop-blur-sm border border-border rounded px-3 py-2 shadow-lg">
        {markerCount === 0 && (
          <p className="text-sm text-foreground">Select a point.</p>
        )}
        
        {markerCount === 1 && (
          <p className="text-sm text-foreground">Select another point.</p>
        )}
        
        {markerCount === 2 && measurements && (
          <div className="flex items-center gap-4 text-sm">
            {/* Point distance */}
            <div className="flex items-center gap-1.5">
              <Ruler className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">
                {measurements.pointsDistance.toFixed(3)}
              </span>
            </div>

            {/* Parallel face distance (only if parallel) */}
            {measurements.parallelFacesDistance !== null && (
              <div className="flex items-center gap-1.5">
                <MoveVertical className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">
                  {measurements.parallelFacesDistance.toFixed(3)}
                </span>
              </div>
            )}

            {/* Face angle */}
            <div className="flex items-center gap-1.5">
              <Triangle className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">
                {(measurements.facesAngle * RadDeg).toFixed(1)}°
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
