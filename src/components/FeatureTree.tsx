import React, { useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  ChevronDown, 
  ChevronRight, 
  Box, 
  Circle, 
  Square, 
  Triangle,
  Layers,
  Zap
} from 'lucide-react';

// Feature instance interface supporting both AAGNet and rule-based formats
interface FeatureInstance {
  type: string;
  subtype?: string;  // For rule-based: through_hole, blind_hole, etc.
  face_indices: number[];
  bottom_faces?: number[];  // AAGNet specific
  confidence: number;
  parameters?: Record<string, any>;  // Rule-based specific
}

interface RecognizedFeatures {
  instances: FeatureInstance[];
  semantic_labels?: number[];  // AAGNet specific
  extended_attributes?: {
    face_attributes: number[][];
    edge_attributes: number[][];
  };
  num_faces?: number;
  num_instances?: number;
  num_features_detected?: number;  // Rule-based specific
  feature_summary?: Record<string, number>;  // Rule-based specific
  processing_time?: number;
  recognition_method?: string;  // 'AAGNet' or 'rule_based'
}

interface FeatureTreeProps {
  features: RecognizedFeatures;
  featureSummary?: any;
  onFeatureSelect?: (featureInstance: FeatureInstance) => void;
}

// Feature type to display name mapping (supports both AAGNet and rule-based)
const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  // AAGNet feature types
  'chamfer': 'Chamfer',
  'through_hole': 'Through Hole',
  'triangular_passage': 'Triangular Passage',
  'rectangular_passage': 'Rectangular Passage',
  '6sides_passage': '6-Sides Passage',
  'triangular_through_slot': 'Triangular Through Slot',
  'rectangular_through_slot': 'Rectangular Through Slot',
  'circular_through_slot': 'Circular Through Slot',
  'rectangular_through_step': 'Rectangular Through Step',
  '2sides_through_step': '2-Sides Through Step',
  'slanted_through_step': 'Slanted Through Step',
  'Oring': 'O-Ring',
  'blind_hole': 'Blind Hole',
  'triangular_pocket': 'Triangular Pocket',
  'rectangular_pocket': 'Rectangular Pocket',
  '6sides_pocket': '6-Sides Pocket',
  'circular_end_pocket': 'Circular End Pocket',
  'rectangular_blind_slot': 'Rectangular Blind Slot',
  'v_circular_end_blind_slot': 'V-Circular End Blind Slot',
  'h_circular_end_blind_slot': 'H-Circular End Blind Slot',
  'triangular_blind_step': 'Triangular Blind Step',
  'circular_blind_step': 'Circular Blind Step',
  'rectangular_blind_step': 'Rectangular Blind Step',
  'round': 'Round',
  'stock': 'Stock',
  // Rule-based feature types
  'hole': 'Hole',
  'pocket': 'Pocket',
  'slot': 'Slot',
  'fillet': 'Fillet',
  'boss': 'Boss',
  'general_pocket': 'General Pocket',
  'rectangular_slot': 'Rectangular Slot',
  'constant_radius': 'Constant Radius',
  'variable_radius': 'Variable Radius',
  '45_degree': '45° Chamfer',
  'angled': 'Angled Chamfer',
  'partial_cylindrical': 'Partial Cylindrical'
};

// Feature category classification (works for both systems)
const FEATURE_CATEGORIES = {
  holes: ['through_hole', 'blind_hole', 'hole', 'partial_cylindrical'],
  pockets: ['triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket', 'pocket', 'general_pocket'],
  slots: [
    'triangular_through_slot', 
    'rectangular_through_slot', 
    'circular_through_slot',
    'rectangular_blind_slot',
    'v_circular_end_blind_slot',
    'h_circular_end_blind_slot',
    'slot',
    'rectangular_slot'
  ],
  steps: [
    'rectangular_through_step',
    '2sides_through_step',
    'slanted_through_step',
    'triangular_blind_step',
    'circular_blind_step',
    'rectangular_blind_step'
  ],
  passages: ['triangular_passage', 'rectangular_passage', '6sides_passage'],
  chamfers: ['chamfer', 'round', 'fillet', 'constant_radius', 'variable_radius', '45_degree', 'angled'],
  other: ['Oring', 'boss']
};

// Feature category display names
const CATEGORY_NAMES = {
  holes: 'Holes',
  pockets: 'Pockets',
  slots: 'Slots',
  steps: 'Steps',
  passages: 'Passages',
  chamfers: 'Chamfers & Rounds',
  other: 'Other Features'
};

// Category icons
const CategoryIcon: Record<string, React.ComponentType<any>> = {
  holes: Circle,
  pockets: Box,
  slots: Square,
  steps: Layers,
  passages: Triangle,
  chamfers: Zap,
  other: Box
};

// Confidence color coding
const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.9) return 'text-green-600 bg-green-50';
  if (confidence >= 0.7) return 'text-yellow-600 bg-yellow-50';
  return 'text-red-600 bg-red-50';
};

const getConfidenceBadge = (confidence: number): string => {
  if (confidence >= 0.9) return 'High';
  if (confidence >= 0.7) return 'Medium';
  return 'Low';
};

const FeatureTree: React.FC<FeatureTreeProps> = ({ 
  features, 
  featureSummary,
  onFeatureSelect 
}) => {
  // ✅ Add null check
  if (!features || !features.instances || features.instances.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Feature Recognition</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Box className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No features detected yet.</p>
            <p className="text-sm mt-2">Features will appear here after CAD analysis completes.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // ✅ FIXED: Proper state management with useMemo
  const [expandedCategories, setExpandedCategories] = React.useState<Set<string>>(
    new Set(Object.keys(CATEGORY_NAMES))
  );

  // Group features by category
  const categorizedFeatures = useMemo(() => {
    const grouped: Record<string, FeatureInstance[]> = {
      holes: [],
      pockets: [],
      slots: [],
      steps: [],
      passages: [],
      chamfers: [],
      other: []
    };

    features.instances.forEach(instance => {
      let categorized = false;
      
      for (const [category, types] of Object.entries(FEATURE_CATEGORIES)) {
        if (types.includes(instance.type)) {
          grouped[category].push(instance);
          categorized = true;
          break;
        }
      }
      
      if (!categorized) {
        grouped.other.push(instance);
      }
    });

    return grouped;
  }, [features.instances]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    const totalFeatures = features.num_instances || features.num_features_detected || features.instances.length;
    const totalFaces = features.num_faces || features.instances.reduce((sum, f) => sum + f.face_indices.length, 0);
    const avgConfidence = features.instances.length > 0
      ? features.instances.reduce((sum, f) => sum + f.confidence, 0) / features.instances.length
      : 0;
    
    return {
      totalFeatures,
      totalFaces,
      avgConfidence,
      highConfidence: features.instances.filter(f => f.confidence >= 0.9).length,
      mediumConfidence: features.instances.filter(f => f.confidence >= 0.7 && f.confidence < 0.9).length,
      lowConfidence: features.instances.filter(f => f.confidence < 0.7).length
    };
  }, [features]);

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  return (
    <div className="space-y-4">
      {/* Summary Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recognition Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{stats.totalFeatures}</div>
              <div className="text-xs text-gray-600">Features</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-600">{stats.totalFaces}</div>
              <div className="text-xs text-gray-600">Total Faces</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {(stats.avgConfidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-600">Avg Confidence</div>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <div className="text-xs text-gray-600 mb-1">Confidence</div>
              <div className="flex gap-1 justify-center">
                <span className="text-xs text-green-600">{stats.highConfidence}H</span>
                <span className="text-xs text-yellow-600">{stats.mediumConfidence}M</span>
                <span className="text-xs text-red-600">{stats.lowConfidence}L</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Tree by Category */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Detected Features</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[500px]">
            <div className="p-4 space-y-2">
              {Object.entries(categorizedFeatures).map(([category, instances]) => {
                if (instances.length === 0) return null;

                const isExpanded = expandedCategories.has(category);
                const Icon = CategoryIcon[category];

                return (
                  <div key={category} className="border rounded-lg">
                    {/* Category Header */}
                    <button
                      onClick={() => toggleCategory(category)}
                      className="w-full flex items-center justify-between p-3 hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <Icon className="h-5 w-5 text-gray-600" />
                        <span className="font-medium">
                          {CATEGORY_NAMES[category as keyof typeof CATEGORY_NAMES]}
                        </span>
                        <Badge variant="secondary">{instances.length}</Badge>
                      </div>
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4 text-gray-400" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-gray-400" />
                      )}
                    </button>

                    {/* Feature Instances */}
                    {isExpanded && (
                      <div className="border-t">
                        {instances.map((instance, idx) => (
                          <div
                            key={`${instance.type}-${idx}`}
                            onClick={() => onFeatureSelect?.(instance)}
                            className="p-3 border-b last:border-b-0 hover:bg-gray-50 cursor-pointer transition-colors"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-medium text-sm">
                                    {FEATURE_DISPLAY_NAMES[instance.type] || instance.type}
                                  </span>
                                  <Badge 
                                    variant="outline" 
                                    className={getConfidenceColor(instance.confidence)}
                                  >
                                    {getConfidenceBadge(instance.confidence)}
                                  </Badge>
                                </div>
                                
                                <div className="flex flex-wrap gap-3 text-xs text-gray-500">
                                  <span>
                                    {instance.face_indices.length} faces
                                  </span>
                                  {instance.bottom_faces && instance.bottom_faces.length > 0 && (
                                    <span>
                                      {instance.bottom_faces.length} bottom face{instance.bottom_faces.length !== 1 ? 's' : ''}
                                    </span>
                                  )}
                                  <span>
                                    {(instance.confidence * 100).toFixed(1)}% confidence
                                  </span>
                                </div>

                                {/* Face indices (collapsed) */}
                                <div className="mt-2 text-xs text-gray-400">
                                  Faces: [{instance.face_indices.slice(0, 5).join(', ')}
                                  {instance.face_indices.length > 5 && ` +${instance.face_indices.length - 5} more`}]
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Processing Info */}
      {features.processing_time && (
        <div className="text-xs text-gray-500 text-center">
          Analysis completed in {features.processing_time.toFixed(2)}s using {features.recognition_method === 'rule_based' ? 'Rule-Based Recognition' : features.recognition_method || 'Feature Recognition'}
        </div>
      )}
    </div>
  );
};

export default FeatureTree;
