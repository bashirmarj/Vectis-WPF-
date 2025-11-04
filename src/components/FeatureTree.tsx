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
  features?: ManufacturingFeatures;
  featureSummary?: FeatureSummary;
  featureTree?: {
    oriented_sections?: any[];
    common_dimensions?: any;
  };
  mlFeatures?: MLFeatures;
}

const FeatureTree: React.FC<FeatureTreeProps> = ({ 
  features, 
  featureSummary, 
  featureTree,
  mlFeatures 
}) => {
  // State for expandable sections
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set([
      'comparison',
      'manufacturing',
      'surfaces',
      'through-holes',
      'blind-holes',
      'bores',
      'bosses',
      'mlFeatures'
    ])
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
    setExpandedSections(new Set([
      'comparison',
      'manufacturing',
      'surfaces',
      'through-holes',
      'blind-holes',
      'bores',
      'bosses',
      'planar-faces',
      'fillets',
      'mlFeatures'
    ]));
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

  const calculateAgreement = () => {
    if (!mlFeatures || !featureSummary) return 0;
    
    const heuristicHoles = (featureSummary.through_holes || 0) + (featureSummary.blind_holes || 0);
    const heuristicBosses = featureSummary.bosses || 0;
    const heuristicTotal = heuristicHoles + heuristicBosses;
    
    const mlHoles = (mlFeatures.feature_summary.hole || 0);
    const mlBosses = (mlFeatures.feature_summary.boss || 0);
    const mlTotal = mlHoles + mlBosses;
    
    if (heuristicTotal === 0 && mlTotal === 0) return 100;
    if (heuristicTotal === 0 || mlTotal === 0) return 0;
    
    const difference = Math.abs(heuristicTotal - mlTotal);
    const maxTotal = Math.max(heuristicTotal, mlTotal);
    return Math.max(0, Math.round(100 - (difference / maxTotal * 100)));
  };

  // Check if we have new format data
  const hasNewFormat = features || featureSummary;
  
  // If no data at all, show empty state
  if (!hasNewFormat && !featureTree) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            No Feature Data Available
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Upload and analyze a CAD file to see detected manufacturing features.
          </p>
        </CardContent>
      </Card>
    );
  }

  // Render new format (manufacturing features)
  if (hasNewFormat) {
    return (
      <div className="space-y-4">
        {/* Comparison View - Only show when both ML and heuristic data available */}
        {mlFeatures && featureSummary && (
          <Card>
            <CardHeader 
              className="cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => toggleSection('comparison')}
            >
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-purple-600" />
                  Heuristic vs ML Comparison
                </CardTitle>
                {expandedSections.has('comparison') ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
              </div>
            </CardHeader>
            {expandedSections.has('comparison') && (
              <CardContent>
                <div className="grid grid-cols-2 gap-6 mb-4">
                  {/* Heuristic column */}
                  <div className="space-y-3">
                    <h3 className="font-semibold text-sm text-muted-foreground flex items-center gap-2">
                      <Wrench className="w-4 h-4" />
                      Rule-Based Detection
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between p-2 bg-accent/30 rounded">
                        <span>Through-Holes:</span>
                        <span className="font-medium">{featureSummary.through_holes || 0}</span>
                      </div>
                      <div className="flex justify-between p-2 bg-accent/30 rounded">
                        <span>Blind-Holes:</span>
                        <span className="font-medium">{featureSummary.blind_holes || 0}</span>
                      </div>
                      <div className="flex justify-between p-2 bg-accent/30 rounded">
                        <span>Bosses:</span>
                        <span className="font-medium">{featureSummary.bosses || 0}</span>
                      </div>
                      <div className="flex justify-between p-2 bg-accent/30 rounded">
                        <span>Planar Faces:</span>
                        <span className="font-medium">{featureSummary.planar_faces || 0}</span>
                      </div>
                      <div className="flex justify-between p-2 bg-accent/30 rounded">
                        <span>Fillets:</span>
                        <span className="font-medium">{featureSummary.fillets || 0}</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* ML column */}
                  <div className="space-y-3">
                    <h3 className="font-semibold text-sm text-muted-foreground flex items-center gap-2">
                      <Brain className="w-4 h-4" />
                      ML Predictions (16 MFCAD Classes)
                    </h3>
                    <div className="space-y-2 text-sm max-h-64 overflow-y-auto">
                      {Object.entries(mlFeatures.feature_summary)
                        .filter(([key, count]) => key !== 'total_faces' && (count as number) > 0)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .map(([featureType, count]) => (
                          <div key={featureType} className="flex justify-between items-center p-2 bg-purple-50 dark:bg-purple-950/20 rounded">
                            <span className="capitalize text-xs">{featureType.replace(/_/g, ' ')}</span>
                            <Badge className={getFeatureColor(featureType)}>
                              {count as number}
                            </Badge>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
                
                {/* Agreement indicator */}
                <div className="mt-4 p-3 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Agreement Score:</span>
                    <Badge 
                      className={
                        calculateAgreement() >= 75 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' 
                          : calculateAgreement() >= 50 
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' 
                          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                      }
                    >
                      {calculateAgreement()}%
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Based on feature count similarity between heuristic and ML detection
                  </p>
                </div>
              </CardContent>
            )}
          </Card>
        )}

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

        {/* Header Card with Summary */}
        {featureSummary && (
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle>Manufacturing Features</CardTitle>
                <div className="flex gap-2">
                  <Button variant="ghost" size="sm" onClick={expandAll}>
                    Expand All
                  </Button>
                  <Button variant="ghost" size="sm" onClick={collapseAll}>
                    Collapse All
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Complexity Score */}
              <div className="flex items-center justify-between p-3 bg-accent/50 rounded-lg">
                <span className="font-medium">Complexity Score</span>
                <Badge className={getComplexityBadge(featureSummary.complexity_score)}>
                  {featureSummary.complexity_score} / 10
                </Badge>
              </div>

              {/* Quick Stats Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                  <div className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
                    {featureSummary.through_holes}
                  </div>
                  <div className="text-sm text-muted-foreground">Through-Holes</div>
                </div>
                <div className="p-3 bg-orange-50 dark:bg-orange-950/20 rounded-lg border border-orange-200 dark:border-orange-800">
                  <div className="text-2xl font-bold text-orange-700 dark:text-orange-300">
                    {featureSummary.blind_holes}
                  </div>
                  <div className="text-sm text-muted-foreground">Blind Holes</div>
                </div>
                <div className="p-3 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
                  <div className="text-2xl font-bold text-red-700 dark:text-red-300">
                    {featureSummary.bores}
                  </div>
                  <div className="text-sm text-muted-foreground">Bores</div>
                </div>
                <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                    {featureSummary.bosses}
                  </div>
                  <div className="text-sm text-muted-foreground">Bosses</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Detailed Features Card */}
        {features && (
          <Card>
            <CardHeader>
              <CardTitle>Detailed Feature Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {/* Manufacturing Features Section */}
              <div className="border rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleSection('manufacturing')}
                  className="w-full flex items-center justify-between p-3 bg-accent/50 hover:bg-accent transition-colors"
                >
                  <div className="flex items-center gap-2">
                    {expandedSections.has('manufacturing') ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                    <span className="font-medium">Manufacturing Features</span>
                  </div>
                  <Badge variant="secondary">
                    {(features.through_holes?.length || 0) + 
                     (features.blind_holes?.length || 0) + 
                     (features.bores?.length || 0) + 
                     (features.bosses?.length || 0)}
                  </Badge>
                </button>
                
                {expandedSections.has('manufacturing') && (
                  <div className="p-3 space-y-3">
                    {/* Through-Holes */}
                    {features.through_holes && features.through_holes.length > 0 && (
                      <div>
                        <button
                          onClick={() => toggleSection('through-holes')}
                          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 rounded transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            {expandedSections.has('through-holes') ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                            <Drill className="w-4 h-4 text-yellow-600" />
                            <span className="text-sm font-medium">Through-Holes</span>
                          </div>
                          <Badge variant="outline" className="bg-yellow-50 dark:bg-yellow-950/20">
                            {features.through_holes.length}
                          </Badge>
                        </button>
                        
                        {expandedSections.has('through-holes') && (
                          <div className="ml-6 mt-2 space-y-2">
                            {features.through_holes.map((hole, idx) => (
                              <div key={idx} className="p-2 bg-yellow-50 dark:bg-yellow-950/20 rounded text-sm">
                                <div className="font-medium">Hole {idx + 1}</div>
                                <div className="text-muted-foreground space-y-1 mt-1">
                                  <div>Diameter: {formatNumber(hole.diameter)} mm</div>
                                  {hole.area && <div>Area: {formatNumber(hole.area)} mm²</div>}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Blind Holes */}
                    {features.blind_holes && features.blind_holes.length > 0 && (
                      <div>
                        <button
                          onClick={() => toggleSection('blind-holes')}
                          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 rounded transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            {expandedSections.has('blind-holes') ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                            <Drill className="w-4 h-4 text-orange-600" />
                            <span className="text-sm font-medium">Blind Holes</span>
                          </div>
                          <Badge variant="outline" className="bg-orange-50 dark:bg-orange-950/20">
                            {features.blind_holes.length}
                          </Badge>
                        </button>
                        
                        {expandedSections.has('blind-holes') && (
                          <div className="ml-6 mt-2 space-y-2">
                            {features.blind_holes.map((hole, idx) => (
                              <div key={idx} className="p-2 bg-orange-50 dark:bg-orange-950/20 rounded text-sm">
                                <div className="font-medium">Blind Hole {idx + 1}</div>
                                <div className="text-muted-foreground space-y-1 mt-1">
                                  <div>Diameter: {formatNumber(hole.diameter)} mm</div>
                                  {hole.depth && <div>Depth: {formatNumber(hole.depth)} mm</div>}
                                  {hole.area && <div>Area: {formatNumber(hole.area)} mm²</div>}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Bores */}
                    {features.bores && features.bores.length > 0 && (
                      <div>
                        <button
                          onClick={() => toggleSection('bores')}
                          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 rounded transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            {expandedSections.has('bores') ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                            <Cylinder className="w-4 h-4 text-red-600" />
                            <span className="text-sm font-medium">Bores</span>
                          </div>
                          <Badge variant="outline" className="bg-red-50 dark:bg-red-950/20">
                            {features.bores.length}
                          </Badge>
                        </button>
                        
                        {expandedSections.has('bores') && (
                          <div className="ml-6 mt-2 space-y-2">
                            {features.bores.map((bore, idx) => (
                              <div key={idx} className="p-2 bg-red-50 dark:bg-red-950/20 rounded text-sm">
                                <div className="font-medium">Bore {idx + 1}</div>
                                <div className="text-muted-foreground space-y-1 mt-1">
                                  <div>Diameter: {formatNumber(bore.diameter)} mm</div>
                                  {bore.area && <div>Area: {formatNumber(bore.area)} mm²</div>}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Bosses */}
                    {features.bosses && features.bosses.length > 0 && (
                      <div>
                        <button
                          onClick={() => toggleSection('bosses')}
                          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 rounded transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            {expandedSections.has('bosses') ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                            <Box className="w-4 h-4 text-blue-600" />
                            <span className="text-sm font-medium">Bosses</span>
                          </div>
                          <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950/20">
                            {features.bosses.length}
                          </Badge>
                        </button>
                        
                        {expandedSections.has('bosses') && (
                          <div className="ml-6 mt-2 space-y-2">
                            {features.bosses.map((boss, idx) => (
                              <div key={idx} className="p-2 bg-blue-50 dark:bg-blue-950/20 rounded text-sm">
                                <div className="font-medium">Boss {idx + 1}</div>
                                <div className="text-muted-foreground space-y-1 mt-1">
                                  <div>Diameter: {formatNumber(boss.diameter)} mm</div>
                                  {boss.area && <div>Area: {formatNumber(boss.area)} mm²</div>}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Surface Features Section */}
              <div className="border rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleSection('surfaces')}
                  className="w-full flex items-center justify-between p-3 bg-accent/50 hover:bg-accent transition-colors"
                >
                  <div className="flex items-center gap-2">
                    {expandedSections.has('surfaces') ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                    <span className="font-medium">Surface Features</span>
                  </div>
                  <Badge variant="secondary">
                    {(features.planar_faces?.length || 0) + 
                     (features.fillets?.length || 0)}
                  </Badge>
                </button>
                
                {expandedSections.has('surfaces') && (
                  <div className="p-3 space-y-2">
                    {/* Planar Faces */}
                    {features.planar_faces && features.planar_faces.length > 0 && (
                      <div className="flex items-center justify-between p-2 bg-accent/50 rounded">
                        <div className="flex items-center gap-2">
                          <Wrench className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm">Planar Faces</span>
                        </div>
                        <Badge variant="outline">{features.planar_faces.length}</Badge>
                      </div>
                    )}

                    {/* Fillets */}
                    {features.fillets && features.fillets.length > 0 && (
                      <div className="flex items-center justify-between p-2 bg-accent/50 rounded">
                        <div className="flex items-center gap-2">
                          <Circle className="w-4 h-4 text-purple-600" />
                          <span className="text-sm">Fillets/Rounds</span>
                        </div>
                        <Badge variant="outline">{features.fillets.length}</Badge>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* ML-Based Feature Recognition Section */}
        {mlFeatures && (
          <Card>
            <CardHeader 
              className="cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => toggleSection('mlFeatures')}
            >
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-600" />
                  ML-Based Feature Recognition
                  <Badge variant="secondary" className="ml-2">
                    {mlFeatures.feature_summary.total_faces} faces
                  </Badge>
                </CardTitle>
                {expandedSections.has('mlFeatures') ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
              </div>
            </CardHeader>
            {expandedSections.has('mlFeatures') && (
              <CardContent>
                <div className="space-y-3">
                  <p className="text-sm text-muted-foreground mb-4">
                    Deep learning model analyzed {mlFeatures.feature_summary.total_faces} faces using UV-Net architecture
                  </p>
                  
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(mlFeatures.feature_summary)
                      .filter(([key]) => key !== 'total_faces')
                      .sort(([, a], [, b]) => (b as number) - (a as number))
                      .map(([featureType, count]) => {
                        if ((count as number) === 0) return null;
                        return (
                          <div key={featureType} className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-800">
                            <span className="text-sm font-medium capitalize">
                              {featureType.replace(/_/g, ' ')}
                            </span>
                            <Badge variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                              {count as number}
                            </Badge>
                          </div>
                        );
                      })}
                  </div>

                  {/* High-confidence predictions */}
                  {mlFeatures.face_predictions.filter(p => p.confidence > 0.8).length > 0 && (
                    <div className="mt-4 p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                      <div className="text-sm font-medium text-green-800 dark:text-green-300 mb-1">
                        High Confidence Detections
                      </div>
                      <div className="text-xs text-green-700 dark:text-green-400">
                        {mlFeatures.face_predictions.filter(p => p.confidence > 0.8).length} faces 
                        detected with &gt;80% confidence
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            )}
          </Card>
        )}
      </div>
    );
  }

  // Fallback to old format if new format not available
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Feature Tree (Legacy Format)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="p-4 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800 mb-4">
          <p className="text-sm text-yellow-800 dark:text-yellow-300">
            This file uses an older analysis format. For better feature detection, please re-upload the file.
          </p>
        </div>
        <pre className="text-xs overflow-auto max-h-96 bg-accent/50 p-4 rounded">
          {JSON.stringify(featureTree, null, 2)}
        </pre>
      </CardContent>
    </Card>
  );
};

export default FeatureTree;
