"""
Enhanced Feature Recognizer Wrapper
Wraps existing production recognizers with full prismatic taxonomy support
"""

import logging
from typing import List, Dict, Optional
from feature_taxonomy import (
    ALL_FEATURES, FeatureCategory, get_feature_definition,
    FEATURE_CATEGORY_MAP
)

logger = logging.getLogger(__name__)

class TaxonomyMapper:
    """Maps detected features to complete 60+ prismatic taxonomy"""
    
    def __init__(self):
        self.feature_mapping = self._build_feature_mapping()
    
    def _build_feature_mapping(self) -> Dict:
        """
        Build mapping from current detected types to taxonomy types
        Current system detects ~10-15 types, taxonomy has 60+
        """
        return {
            # Current hole types → Enhanced taxonomy
            'through': 'through_hole',
            'blind': 'blind_hole',
            'counterbore': 'counterbore',
            'countersink': 'countersink',
            'tapped': 'tapped_hole',
            'hole': 'through_hole',  # Default
            
            # Current pocket types → Enhanced taxonomy
            'rectangular': 'closed_rectangular_pocket',
            'circular': 'circular_pocket',
            'rounded': 'obround_pocket',
            'pocket': 'closed_rectangular_pocket',  # Default
            'general_pocket': 'irregular_pocket',
            
            # Current slot types → Enhanced taxonomy
            'through_slot': 'through_slot_rectangular',
            'blind_slot': 'blind_slot_rectangular',
            't_slot': 't_slot',
            'keyway': 'keyway_slot',
            'slot': 'blind_slot_rectangular',  # Default
            
            # Turning/boss features
            'base_cylinder': 'cylindrical_boss',
            'step': 'simple_step',
            'boss': 'cylindrical_boss',
            
            # Fillets and chamfers
            'constant_radius': 'constant_radius_fillet',
            'variable_radius': 'variable_radius_fillet',
            'fillet': 'constant_radius_fillet',
            '45_degree': 'equal_distance_chamfer',
            'angled': 'distance_angle_chamfer',
            'chamfer': 'equal_distance_chamfer',
            
            # Grooves
            'groove': 'external_groove',
            'o_ring': 'o_ring_groove',
        }
    
    def enhance_feature(self, feature: Dict) -> Dict:
        """
        Enhance detected feature with taxonomy information
        
        Args:
            feature: Raw feature from production recognizer
            
        Returns:
            Enhanced feature with taxonomy metadata
        """
        feature_type = feature.get('type', '')
        subtype = feature.get('subtype', '')
        
        # Map to taxonomy type
        enhanced_type = None
        
        # Try mapping subtype first (more specific)
        if subtype in self.feature_mapping:
            enhanced_type = self.feature_mapping[subtype]
        # Fall back to feature type
        elif feature_type in self.feature_mapping:
            enhanced_type = self.feature_mapping[feature_type]
        # Use original if no mapping
        else:
            enhanced_type = subtype or feature_type
        
        # Get taxonomy definition
        definition = get_feature_definition(enhanced_type)
        
        # Build enhanced feature
        enhanced = {
            **feature,  # Keep all original fields
            'taxonomy_type': enhanced_type,
            'feature_hierarchy': definition.category.value if definition else feature_type,
            'boundary_condition': definition.boundary.value if definition and definition.boundary else None,
            'profile_type': definition.profile.value if definition and definition.profile else None,
            'ap224_code': definition.ap224_code if definition else None,
            'detection_priority': definition.detection_priority if definition else 999,
        }
        
        return enhanced
    
    def enhance_feature_list(self, features: List[Dict]) -> List[Dict]:
        """Enhance all features in list"""
        return [self.enhance_feature(f) for f in features]
    
    def generate_taxonomy_summary(self, features: List[Dict]) -> Dict:
        """Generate summary by taxonomy categories"""
        summary = {
            'by_category': {},
            'by_type': {},
            'total': len(features)
        }
        
        for feature in features:
            # Count by category
            category = feature.get('feature_hierarchy', 'other')
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Count by specific type
            taxonomy_type = feature.get('taxonomy_type', feature.get('type', 'unknown'))
            summary['by_type'][taxonomy_type] = summary['by_type'].get(taxonomy_type, 0) + 1
        
        return summary


class EnhancedRecognizerWrapper:
    """
    Wraps existing production recognizers with enhanced taxonomy support
    
    Usage:
        wrapper = EnhancedRecognizerWrapper(base_recognizer)
        result = wrapper.recognize_features(step_file_path)
    """
    
    def __init__(self, base_recognizer):
        self.base_recognizer = base_recognizer
        self.mapper = TaxonomyMapper()
    
    def recognize_features(self, step_file_path: str) -> Dict:
        """
        Run feature recognition with taxonomy enhancement
        
        Returns enhanced result with:
        - instances: Enhanced feature instances with taxonomy metadata
        - taxonomy_summary: Summary by category and type
        - feature_summary: Original summary (for compatibility)
        """
        # Run base recognition
        result = self.base_recognizer.recognize_features(step_file_path)
        
        if result.get('status') != 'success':
            return result
        
        # Enhance instances
        original_instances = result.get('instances', [])
        enhanced_instances = self.mapper.enhance_feature_list(original_instances)
        
        # Generate taxonomy summary
        taxonomy_summary = self.mapper.generate_taxonomy_summary(enhanced_instances)
        
        # Update result
        result['instances'] = enhanced_instances
        result['taxonomy_summary'] = taxonomy_summary
        result['recognition_method'] = 'rule_based_enhanced'
        
        logger.info(f"Enhanced {len(enhanced_instances)} features with taxonomy")
        logger.info(f"Taxonomy summary: {taxonomy_summary['by_category']}")
        
        return result
