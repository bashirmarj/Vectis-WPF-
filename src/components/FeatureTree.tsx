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
import './FeatureTree.css';

// ✅ FIXED: AAGNet multi-task output interface
interface AAGNetFeatureInstance {
  type: string;
  face_indices: number[];
  bottom_faces: number[];
  confidence: number;
}

interface AAGNetFeatures {
  instances: AAGNetFeatureInstance[];
  semantic_labels: number[];
  extended_attributes?: {
    face_attributes: number[][];
    edge_attributes: number[][];
  };
  num_faces: number;
  num_instances: number;
  processing_time?: number;
}

interface FeatureTreeProps {
  mlFeatures?: AAGNetFeatures;
  featureSummary?: any;
  onFeatureSelect?: (featureInstance: AAGNetFeatureInstance) => void;
  isLoading?: boolean;
}

// Feature type to display name mapping
const FEATURE_DISPLAY_NAMES: Record<string, string> = {
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
  'stock': 'Stock'
};

// Feature category classification
const FEATURE_CATEGORIES = {
  holes: ['through_hole', 'blind_hole'],
  pockets: ['triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket'],
  slots: [
    'triangular_through_slot', 
    'rectangular_through_slot', 
    'circular_through_slot',
    'rectangular_blind_slot',
    'v_circular_end_blind_slot',
    'h_circular_end_blind_slot'
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
  chamfers: ['chamfer', 'round'],
  other: ['Oring']
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
  mlFeatures, 
  featureSummary,
  onFeatureSelect,
  isLoading = false
}) => {
  // ✅ FIXED: Proper state management with useMemo
  const [expandedCategories, setExpandedCategories] = React.useState<Set<string>>(
    new Set(Object.keys(CATEGORY_NAMES))
  );

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Loading Features...</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!mlFeatures || !mlFeatures.instances || mlFeatures.instances.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Feature Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Upload a STEP file to see feature analysis</p>
        </CardContent>
      </Card>
    );
  }

  // Group features by category
  const categorizedFeatures = useMemo(() => {
    const grouped: Record<string, AAGNetFeatureInstance[]> = {
      holes: [],
      pockets: [],
      slots: [],
      steps: [],
      passages: [],
      chamfers: [],
      other: []
    };

    mlFeatures.instances.forEach(instance => {
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
  }, [mlFeatures.instances]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    const totalFeatures = mlFeatures.num_instances;
    const totalFaces = mlFeatures.num_faces;
    const avgConfidence = mlFeatures.instances.length > 0
      ? mlFeatures.instances.reduce((sum, f) => sum + f.confidence, 0) / mlFeatures.instances.length
      : 0;
    
    return {
      totalFeatures,
      totalFaces,
      avgConfidence,
      highConfidence: mlFeatures.instances.filter(f => f.confidence >= 0.9).length,
      mediumConfidence: mlFeatures.instances.filter(f => f.confidence >= 0.7 && f.confidence < 0.9).length,
      lowConfidence: mlFeatures.instances.filter(f => f.confidence < 0.7).length
    };
  }, [mlFeatures]);

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
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5" />
          AAGNet Feature Recognition
        </CardTitle>
        <div className="grid grid-cols-3 gap-2 mt-4">
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-2xl font-bold">{stats.totalFeatures}</div>
            <div className="text-xs text-muted-foreground">Features</div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-2xl font-bold">{stats.totalFaces}</div>
            <div className="text-xs text-muted-foreground">Faces</div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-2xl font-bold">{(stats.avgConfidence * 100).toFixed(0)}%</div>
            <div className="text-xs text-muted-foreground">Confidence</div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[600px]">
          <div className="space-y-4">
            {Object.entries(CATEGORY_NAMES).map(([categoryKey, categoryName]) => {
              const categoryFeatures = categorizedFeatures[categoryKey];
              if (categoryFeatures.length === 0) return null;

              const IconComponent = CategoryIcon[categoryKey];
              const isExpanded = expandedCategories.has(categoryKey);

              return (
                <div key={categoryKey} className="border rounded-lg p-3">
                  <button
                    onClick={() => toggleCategory(categoryKey)}
                    className="w-full flex items-center justify-between hover:bg-muted/50 p-2 rounded"
                  >
                    <div className="flex items-center gap-2">
                      <IconComponent className="h-4 w-4" />
                      <span className="font-medium">{categoryName}</span>
                      <Badge variant="secondary">{categoryFeatures.length}</Badge>
                    </div>
                    {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </button>

                  {isExpanded && (
                    <div className="mt-2 space-y-2">
                      {categoryFeatures.map((feature, idx) => (
                        <div
                          key={idx}
                          className="ml-6 p-2 border-l-2 border-primary/20 hover:bg-muted/30 rounded cursor-pointer"
                          onClick={() => onFeatureSelect?.(feature)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">
                              {FEATURE_DISPLAY_NAMES[feature.type] || feature.type}
                            </span>
                            <Badge className={getConfidenceColor(feature.confidence)} variant="outline">
                              {getConfidenceBadge(feature.confidence)} ({(feature.confidence * 100).toFixed(0)}%)
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {feature.face_indices.length} faces
                            {feature.bottom_faces.length > 0 && ` • ${feature.bottom_faces.length} bottom faces`}
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
  );
};

export default FeatureTree;
