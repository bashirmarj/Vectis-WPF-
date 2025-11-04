import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  ChevronRight,
  ChevronDown,
  Circle,
  Box,
  Cylinder,
  Wrench,
  Drill,
  AlertCircle,
  Brain,
  Layers,
} from 'lucide-react';

// TypeScript Interfaces
interface ManufacturingFeature {
  diameter?: number;
  radius?: number;
  depth?: number;
  area?: number;
  position?: [number, number, number];
  axis?: [number, number, number];
}

interface FeatureSummary {
  through_holes: number;
  blind_holes: number;
  bores: number;
  bosses: number;
  total_holes: number;
  fillets: number;
  planar_faces: number;
  complexity_score: number;
}

interface ManufacturingFeatures {
  through_holes?: ManufacturingFeature[];
  blind_holes?: ManufacturingFeature[];
  bores?: ManufacturingFeature[];
  bosses?: ManufacturingFeature[];
  planar_faces?: ManufacturingFeature[];
  fillets?: ManufacturingFeature[];
  complex_surfaces?: ManufacturingFeature[];
}

interface MLFeatures {
  face_predictions: Array<{
    face_id: number;
    predicted_class: string;
    confidence: number;
    probabilities: number[];
  }>;
  feature_summary: {
    total_faces: number;
    [key: string]: number;
  };
}

interface FeatureTreeProps {
  mlFeatures?: MLFeatures | null;
}

const FeatureTree: React.FC<FeatureTreeProps> = ({ mlFeatures }) => {
  // State for expandable sections
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['mlFeatures'])
  );

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const expandAll = () => {
    setExpandedSections(new Set(['mlFeatures']));
  };

  const collapseAll = () => {
    setExpandedSections(new Set());
  };

  // Helper to format numbers
  const formatNumber = (num: number | undefined, decimals: number = 2): string => {
    if (num === undefined || num === null) return 'N/A';
    return num.toFixed(decimals);
  };

  // Get complexity color
  const getComplexityColor = (score: number): string => {
    if (score <= 3) return 'text-green-600';
    if (score <= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getComplexityBadge = (score: number): string => {
    if (score <= 3) return 'bg-green-100 text-green-800';
    if (score <= 6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getFeatureColor = (featureType: string): string => {
    const colorMap: Record<string, string> = {
      stock: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300',
      rectangular_through_slot: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
      chamfer: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
      triangular_pocket: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
      '6sides_passage': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
      slanted_through_step: 'bg-pink-100 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300',
      triangular_through_slot: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300',
      rectangular_blind_slot: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
      triangular_blind_step: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300',
      rectangular_blind_step: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
      rectangular_passage: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
      '2sides_through_step': 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300',
      '6sides_pocket': 'bg-slate-100 text-slate-800 dark:bg-slate-900/30 dark:text-slate-300',
      triangular_passage: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
      rectangular_pocket: 'bg-lime-100 text-lime-800 dark:bg-lime-900/30 dark:text-lime-300',
      rectangular_through_step: 'bg-fuchsia-100 text-fuchsia-800 dark:bg-fuchsia-900/30 dark:text-fuchsia-300',
    };
    return colorMap[featureType] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300';
  };

  // If no ML data, show empty state
  if (!mlFeatures) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-600" />
            ML Feature Recognition
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">No ML feature data available for this part.</p>
        </CardContent>
      </Card>
    );
  }

  // Render ML features
  return (
    <div className="space-y-4">
      {/* ML Features Detailed View */}
      {mlFeatures && (
          <Card>
            <CardHeader 
              className="cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => toggleSection('mlFeatures')}
            >
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-600" />
                  ML Feature Recognition (16 MFCAD Classes)
                </CardTitle>
                {expandedSections.has('mlFeatures') ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
              </div>
            </CardHeader>
            {expandedSections.has('mlFeatures') && (
              <CardContent>
                <div className="space-y-4">
                  {/* Summary Stats */}
                  <div className="p-3 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-800">
                    <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                      {mlFeatures.feature_summary.total_faces}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Faces Analyzed</div>
                  </div>

                  {/* Feature Type Distribution */}
                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm text-muted-foreground">Feature Distribution:</h4>
                    {Object.entries(mlFeatures.feature_summary)
                      .filter(([key]) => key !== 'total_faces')
                      .sort(([, a], [, b]) => (b as number) - (a as number))
                      .map(([featureType, count]) => (
                        <div key={featureType} className="flex items-center justify-between p-2 rounded hover:bg-accent/50">
                          <span className="capitalize text-sm">{featureType.replace(/_/g, ' ')}</span>
                          <Badge className={getFeatureColor(featureType)}>
                            {count as number}
                          </Badge>
                        </div>
                      ))}
                  </div>

                  {/* Individual Face Predictions */}
                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm text-muted-foreground">Face-Level Predictions:</h4>
                    <div className="max-h-64 overflow-y-auto space-y-1">
                      {mlFeatures.face_predictions.map((pred, idx) => (
                        <div key={idx} className="flex items-center justify-between p-2 bg-accent/30 rounded text-xs">
                          <span>Face {pred.face_id}</span>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className={getFeatureColor(pred.predicted_class)}>
                              {pred.predicted_class.replace(/_/g, ' ')}
                            </Badge>
                            <span className="text-muted-foreground">
                              {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            )}
          </Card>
        )}
      </div>
    );
};

export default FeatureTree;
