"""
Recognizer Output Utilities
===========================

Helper functions for standardizing recognizer output formats.

This module provides utilities to ensure all recognizers output both:
- face_ids (snake_case) - Our internal format
- faceIds (camelCase) - Analysis Situs compatible format

Usage in recognizers:
    from .recognizer_utils import standardize_feature_output
    
    feature = {
        'type': 'blind_hole',
        'face_ids': [1, 2, 3],
        'diameter': 10.0,
        ...
    }
    
    return standardize_feature_output(feature)
"""

from typing import Dict, List, Any


def standardize_feature_output(feature: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize feature output to include both snake_case and camelCase formats.
    
    Ensures compatibility with both our internal systems and Analysis Situs format.
    
    Args:
        feature: Feature dict from recognizer
        
    Returns:
        Standardized feature dict with both face_ids and faceIds
    """
    # Ensure both formats exist
    if 'face_ids' in feature and 'faceIds' not in feature:
        feature['faceIds'] = feature['face_ids']
    elif 'faceIds' in feature and 'face_ids' not in feature:
        feature['face_ids'] = feature['faceIds']
        
    # Ensure lists (not None)
    if 'face_ids' in feature and feature['face_ids'] is None:
        feature['face_ids'] = []
    if 'faceIds' in feature and feature['faceIds'] is None:
        feature['faceIds'] = []
        
    return feature


def standardize_features_list(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardize a list of features.
    
    Args:
        features: List of feature dicts
        
    Returns:
        List of standardized feature dicts
    """
    return [standardize_feature_output(f) for f in features]


def validate_feature_output(feature: Dict[str, Any]) -> bool:
    """
    Validate that feature has required fields.
    
    Args:
        feature: Feature dict to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['type']
    
    for field in required_fields:
        if field not in feature:
            return False
            
    # face_ids or faceIds should exist
    if 'face_ids' not in feature and 'faceIds' not in feature:
        return False
        
    return True


def add_analysis_situs_metadata(feature: Dict[str, Any], aag_graph=None) -> Dict[str, Any]:
    """
    Add Analysis Situs compatible metadata fields.
    
    Adds fields like:
    - fullyRecognized (bool)
    - confidence (float)
    - semanticCodes (dict) - for DFM warnings
    
    Args:
        feature: Feature dict
        aag_graph: Optional AAG graph for advanced analysis
        
    Returns:
        Feature with added metadata
    """
    # Add fullyRecognized flag if not present
    if 'fullyRecognized' not in feature and 'fully_recognized' not in feature:
        # Default to True if feature was successfully recognized
        feature['fullyRecognized'] = True
        
    # Add confidence score if not present
    if 'confidence' not in feature:
        # Default confidence based on feature type
        feature_type = feature.get('type', '')
        if 'hole' in feature_type or 'pocket' in feature_type:
            feature['confidence'] = 0.85
        else:
            feature['confidence'] = 0.75
            
    return feature


def convert_to_analysis_situs_format(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert feature list to Analysis Situs JSON format.
    
    Analysis Situs format:
    {
        "holes": [...],
        "pockets": [...],
        "slots": [...],
        "prismaticMilling": [...]
    }
    
    Args:
        features: List of feature dicts
        
    Returns:
        Dict in Analysis Situs format
    """
    output = {
        'holes': [],
        'pockets': [],
        'slots': [],
        'steps': [],
        'bosses': [],
        'fillets': [],
        'chamfers': [],
        'other': []
    }
    
    for feature in features:
        feature_type = feature.get('type', '')
        
        # Standardize output
        feature = standardize_feature_output(feature)
        feature = add_analysis_situs_metadata(feature)
        
        # Route to appropriate category
        if 'hole' in feature_type:
            output['holes'].append(feature)
        elif 'pocket' in feature_type:
            output['pockets'].append(feature)
        elif 'slot' in feature_type:
            output['slots'].append(feature)
        elif 'step' in feature_type or 'island' in feature_type:
            output['steps'].append(feature)
        elif 'boss' in feature_type:
            output['bosses'].append(feature)
        elif 'fillet' in feature_type:
            output['fillets'].append(feature)
        elif 'chamfer' in feature_type:
            output['chamfers'].append(feature)
        else:
            output['other'].append(feature)
            
    # Remove empty categories
    output = {k: v for k, v in output.items() if v}
    
    return output


def merge_duplicate_face_ids(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect and merge features with overlapping face_ids.
    
    This handles cases where multiple recognizers detect the same feature
    or where features share faces (e.g., counterbore = hole + counterbore).
    
    Args:
        features: List of feature dicts
        
    Returns:
        Deduplicated feature list
    """
    if not features:
        return []
        
    # Build face_id to feature mapping
    face_to_features = {}
    
    for i, feature in enumerate(features):
        face_ids = feature.get('face_ids', []) or feature.get('faceIds', [])
        for fid in face_ids:
            if fid not in face_to_features:
                face_to_features[fid] = []
            face_to_features[fid].append(i)
            
    # Find overlapping features
    merged_indices = set()
    output_features = []
    
    for i, feature in enumerate(features):
        if i in merged_indices:
            continue
            
        face_ids = set(feature.get('face_ids', []) or feature.get('faceIds', []))
        
        # Check for significant overlap with other features
        overlapping = []
        for j, other_feature in enumerate(features):
            if i >= j or j in merged_indices:
                continue
                
            other_face_ids = set(other_feature.get('face_ids', []) or other_feature.get('faceIds', []))
            overlap = face_ids & other_face_ids
            
            # If > 50% overlap, consider merging
            if overlap and len(overlap) / max(len(face_ids), len(other_face_ids)) > 0.5:
                overlapping.append(j)
                
        if overlapping:
            # Merge features - prefer more complex type
            merged = _merge_two_features(feature, features[overlapping[0]])
            output_features.append(merged)
            merged_indices.add(i)
            merged_indices.update(overlapping)
        else:
            output_features.append(feature)
            merged_indices.add(i)
            
    return output_features


def _merge_two_features(f1: Dict, f2: Dict) -> Dict:
    """
    Merge two overlapping features.
    
    Prefers the more complex/specific feature type.
    """
    # Type hierarchy (more complex first)
    type_priority = {
        'counter_drilled_hole': 10,
        'stepped_hole': 9,
        'threaded_hole': 8,
        'blind_hole': 7,
        'through_hole': 6,
        'pocket': 5,
        'slot': 4,
        'step': 3,
        'boss': 2,
        'unknown': 1
    }
    
    t1 = f1.get('type', 'unknown')
    t2 = f2.get('type', 'unknown')
    
    # Choose feature with higher priority type
    if type_priority.get(t1, 0) >= type_priority.get(t2, 0):
        primary = f1.copy()
        secondary = f2
    else:
        primary = f2.copy()
        secondary = f1
        
    # Merge face_ids
    face_ids1 = set(primary.get('face_ids', []) or primary.get('faceIds', []))
    face_ids2 = set(secondary.get('face_ids', []) or secondary.get('faceIds', []))
    merged_face_ids = sorted(list(face_ids1 | face_ids2))
    
    primary['face_ids'] = merged_face_ids
    primary['faceIds'] = merged_face_ids
    
    # Adjust confidence (merged features slightly less certain)
    if 'confidence' in primary:
        primary['confidence'] *= 0.95
        
    return primary
