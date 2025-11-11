import React, { useMemo, useState, useEffect } from 'react';
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
  Zap,
  Sparkles
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';

// Feature instance interface supporting both AAGNet and rule-based formats
interface FeatureInstance {
  type: string;
  subtype?: string;  // For rule-based: through_hole, blind_hole, etc.
  face_indices: number[];
  bottom_faces?: number[];  // AAGNet specific
  confidence: number;
  parameters?: Record<string, any>;  // Rule-based specific
}

// Backend geometric_features structure
interface GeometricFeatures {
  feature_instances: FeatureInstance[];
  num_features_detected: number;
  num_faces_analyzed: number;
  confidence_score: number;
  inference_time_sec: number;
  recognition_method: string;
  feature_summary?: Record<string, number>;
}

interface FeatureTreeProps {
  features: GeometricFeatures | null | undefined;
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
  'partial_cylindrical': 'Partial Cylindrical',
  'cylindrical_shaft': 'Cylindrical Shaft',
  // Production recognizer subtypes - HOLES
  'counterbore': 'Counterbored Hole',
  'countersink': 'Countersunk Hole',
  'tapped': 'Tapped Hole',
  'through': 'Through Hole',
  'blind': 'Blind Hole',
  // Production recognizer subtypes - POCKETS
  'rectangular': 'Rectangular Pocket',
  'circular': 'Circular Pocket',
  'complex_contour': 'Complex Pocket',
  'with_islands': 'Pocket with Islands',
  // Production recognizer subtypes - SLOTS
  'through_slot': 'Through Slot',
  'blind_slot': 'Blind Slot',
  't_slot': 'T-Slot',
  'keyway': 'Keyway',
  // Production recognizer subtypes - TURNING
  'base_cylinder': 'Base Cylinder',
  'step': 'Step',
  'groove': 'Groove',
  'taper': 'Taper',
  'turning_feature': 'Turning Feature'
};

// Feature category classification (works for both systems)
const FEATURE_CATEGORIES = {
  holes: ['through_hole', 'blind_hole', 'hole', 'partial_cylindrical', 'counterbore', 'countersink', 'tapped', 'through', 'blind'],
  pockets: ['triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket', 'pocket', 'general_pocket', 'rectangular', 'circular', 'complex_contour', 'with_islands'],
  slots: [
    'triangular_through_slot', 
    'rectangular_through_slot', 
    'circular_through_slot',
    'rectangular_blind_slot',
    'v_circular_end_blind_slot',
    'h_circular_end_blind_slot',
    'slot',
    'rectangular_slot',
    'through_slot',
    'blind_slot',
    't_slot',
    'keyway'
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
  turning: ['base_cylinder', 'step', 'groove', 'taper', 'turning_feature'],
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
  turning: 'Turning Features',
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
  turning: Circle,
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
  const [aiExplanation, setAiExplanation] = useState<string | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);

  // Fetch AI explanation when features load
  useEffect(() => {
    if (features && features.feature_instances && features.feature_instances.length > 0 && !aiExplanation && !loadingExplanation) {
      fetchAIExplanation();
    }
  }, [features]);

  const fetchAIExplanation = async () => {
    setLoadingExplanation(true);
    try {
      const { data, error } = await supabase.functions.invoke('explain-features', {
        body: {
          features: features.feature_instances.map(f => ({
            type: f.type,
            subtype: f.subtype,
            dimensions: f.parameters
          })),
          material: 'Aluminum 6061',
          volume_cm3: 0,
          part_name: 'CAD Part'
        }
      });

      if (error) {
        console.error('AI explanation failed:', error);
        return;
      }
      
      setAiExplanation(data.explanation);
    } catch (err) {
      console.error('AI explanation error:', err);
    } finally {
      setLoadingExplanation(false);
    }
  };

  
  // ✅ Add null check
  if (!features || !features.feature_instances || features.feature_instances.length === 0) {
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
      turning: [],
      other: []
    };

    features.feature_instances.forEach(instance => {
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
  }, [features.feature_instances]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    const totalFeatures = features.num_features_detected;
    const totalFaces = features.num_faces_analyzed;
    const avgConfidence = features.confidence_score;
    
    return {
      totalFeatures,
      totalFaces,
      avgConfidence,
      highConfidence: features.feature_instances.filter(f => f.confidence >= 0.9).length,
      mediumConfidence: features.feature_instances.filter(f => f.confidence >= 0.7 && f.confidence < 0.9).length,
      lowConfidence: features.feature_instances.filter(f => f.confidence < 0.7).length
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

  // Auto-number features by type for SolidWorks-style display
  const numberedFeatures = useMemo(() => {
    const counters: Record<string, number> = {};
    const numbered: Array<{
      instance: FeatureInstance;
      displayName: string;
      icon: React.ComponentType<any>;
      category: string;
    }> = [];

    Object.entries(categorizedFeatures).forEach(([category, instances]) => {
      instances.forEach(instance => {
        const baseType = FEATURE_DISPLAY_NAMES[instance.type] || instance.type;
        
        if (!counters[baseType]) {
          counters[baseType] = 0;
        }
        counters[baseType]++;
        
        numbered.push({
          instance,
          displayName: `${baseType}-${counters[baseType]}`,
          icon: CategoryIcon[category],
          category
        });
      });
    });

    return numbered;
  }, [categorizedFeatures]);

  const [showAI, setShowAI] = useState(false);

  return (
    <div className="space-y-0">
      {/* Compact Summary Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-muted/30 border-b">
        <div className="flex items-center gap-4 text-sm">
          <span className="font-medium text-foreground">{stats.totalFeatures} Features</span>
          <span className="text-muted-foreground">•</span>
          <span className="text-muted-foreground">{stats.totalFaces} Faces</span>
        </div>
        <Badge variant="outline" className="text-xs">
          {(stats.avgConfidence * 100).toFixed(0)}% Avg
        </Badge>
      </div>

      {/* Compact Feature List */}
      <ScrollArea className="h-[500px]">
        <div className="divide-y divide-border">
          {numberedFeatures.map(({ instance, displayName, icon: Icon }, idx) => (
            <div
              key={`${displayName}-${idx}`}
              onClick={() => onFeatureSelect?.(instance)}
              className="flex items-center gap-2 px-3 py-1.5 hover:bg-accent/50 cursor-pointer transition-colors"
            >
              <Icon className="w-4 h-4 text-muted-foreground flex-shrink-0" />
              <span className="text-sm font-medium text-foreground flex-1">
                {displayName}
              </span>
              {/* Space reserved for future dimensions */}
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Collapsible AI Analysis */}
      {aiExplanation && (
        <>
          <div className="border-t">
            <button
              onClick={() => setShowAI(!showAI)}
              className="w-full flex items-center justify-between px-4 py-2 text-sm hover:bg-muted/30 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-600" />
                <span className="font-medium">AI Manufacturing Analysis</span>
              </div>
              {showAI ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
          </div>
          
          {showAI && (
            <div className="px-4 py-3 bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border-t border-purple-200 dark:border-purple-800">
              <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
                {aiExplanation}
              </div>
            </div>
          )}
        </>
      )}

      {loadingExplanation && (
        <div className="px-4 py-3 border-t">
          <div className="flex items-center gap-2 text-sm text-purple-600 dark:text-purple-400">
            <Sparkles className="w-4 h-4 animate-pulse" />
            Generating AI insights...
          </div>
        </div>
      )}

      {/* Processing Info */}
      <div className="px-4 py-2 text-xs text-muted-foreground text-center border-t">
        Analysis: {features.inference_time_sec.toFixed(2)}s • {features.recognition_method || 'Feature Recognition'}
      </div>
    </div>
  );
};

export default FeatureTree;
