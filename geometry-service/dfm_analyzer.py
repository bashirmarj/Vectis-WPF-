"""
DFM (Design for Manufacturing) Warning System
=============================================

Detects manufacturability issues matching Analysis Situs warnings.

This module analyzes recognized manufacturing features and generates DFM warnings
for issues that would cause problems during CNC machining:
- Impossible corners (tight internal radii)
- Deep narrow pockets (high aspect ratios)
- Thin walls (machining instability)
- Inaccessible features (tool collision)

Warning Codes (Analysis Situs Compatible):
- 2201: Impossible corner (tight internal radius)
- 2202: Deep narrow feature (aspect ratio > 5:1)
- 2203: Thin wall (< 1mm thickness)
- 2204: Inaccessible feature (tool can't reach)
"""

import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DFMAnalyzer:
    """
    Analyzes manufacturing features for DFM issues.
    
    Usage:
        analyzer = DFMAnalyzer(aag_graph)
        warnings = analyzer.analyze_features(features)
        # warnings = [{'code': 2201, 'label': '...', 'faceIds': [...], ...}, ...]
    """
    
    def __init__(self, aag_graph=None):
        """
        Args:
            aag_graph: Optional AAGGraph object for detailed topology analysis
        """
        self.aag = aag_graph
        self.warnings = []
        
    def analyze_features(self, features: List[Dict]) -> List[Dict]:
        """
        Analyze features for DFM issues.
        
        Args:
            features: List of recognized features from pattern matcher
            
        Returns:
            List of warning dicts: [{
                'code': int,                    # DFM warning code
                'label': str,                   # Warning label (Analysis Situs format)
                'faceIds': List[int],           # Affected face IDs
                'face_ids': List[int],          # Duplicate for compatibility
                'message': str,                 # Human-readable description
                'severity': str                 # 'warning', 'error', 'info'
            }]
        """
        logger.info("🔍 Running DFM analysis...")
        
        self.warnings = []
        
        for feature in features:
            # Check for impossible corners
            self._check_impossible_corners(feature)
            
            # Check for deep/narrow features
            self._check_deep_narrow_features(feature)
            
            # Check for thin walls
            self._check_thin_walls(feature)
            
        logger.info(f"  ⚠️  Found {len(self.warnings)} DFM warnings")
        
        return self.warnings
        
    def _check_impossible_corners(self, feature: Dict):
        """
        Detect impossible corners (internal radii < tool radius).
        
        Analysis Situs detects these by checking concave edges with
        tight radii that can't be reached by standard end mills.
        
        Heuristic:
        - Pockets and slots with sharp corners
        - Minimum radius < 0.5mm (too small for standard tools)
        - Any rectangular pockets (90° corners)
        """
        feature_type = feature.get('type', '')
        face_ids = feature.get('face_ids', []) or feature.get('faceIds', [])
        
        if not face_ids:
            return
            
        # Check for features likely to have sharp corners
        if feature_type in ['pocket', 'slot', 'step']:
            # Check for corner radius
            corner_radius = feature.get('corner_radius', 0.0)
            min_radius = feature.get('min_radius', 0.0)
            
            # Heuristic: if no radius specified or very small radius
            if corner_radius < 0.5 or min_radius < 0.5:
                self.warnings.append({
                    'code': 2201,
                    'label': 'cncCodePartBodyWarnImpossibleCorner',
                    'faceIds': face_ids,
                    'face_ids': face_ids,  # Duplicate for compatibility
                    'message': f"Tight corners in {feature_type} (radius < 0.5mm) - consider adding fillets",
                    'severity': 'warning',
                    'feature_type': feature_type
                })
                
        # Check for rectangular pockets (90° corners are impossible without tool deflection)
        if feature_type == 'pocket':
            width = feature.get('width', 0)
            length = feature.get('length', 0)
            
            # If width and length are specified (rectangular pocket), flag corner warning
            if width > 0 and length > 0:
                aspect_ratio = max(width, length) / min(width, length) if min(width, length) > 0 else 1.0
                
                # Only warn for non-circular pockets
                if aspect_ratio > 1.2:  # Not circular
                    self.warnings.append({
                        'code': 2201,
                        'label': 'cncCodePartBodyWarnImpossibleCorner',
                        'faceIds': face_ids,
                        'face_ids': face_ids,
                        'message': f"Rectangular pocket corners cannot be perfectly sharp - use corner radius ≥ 0.5mm",
                        'severity': 'warning',
                        'feature_type': feature_type
                    })
                    
    def _check_deep_narrow_features(self, feature: Dict):
        """
        Check for deep/narrow pockets (high aspect ratio).
        
        Standard machining rule: depth should be < 5× diameter
        Exceeding this causes:
        - Tool deflection
        - Poor surface finish
        - Potential tool breakage
        """
        feature_type = feature.get('type', '')
        
        if feature_type not in ['pocket', 'hole', 'slot']:
            return
            
        depth = feature.get('depth', 0.0) or feature.get('total_depth', 0.0)
        
        # Determine feature width/diameter
        diameter = feature.get('diameter', 0.0)
        width = feature.get('width', 0.0)
        
        # Use appropriate dimension
        min_dimension = diameter or width
        
        if min_dimension > 0 and depth > 0:
            aspect_ratio = depth / min_dimension
            
            # Standard rule: aspect ratio > 5:1 is problematic
            if aspect_ratio > 5.0:
                face_ids = feature.get('face_ids', []) or feature.get('faceIds', [])
                
                self.warnings.append({
                    'code': 2202,
                    'label': 'cncCodePartBodyWarnDeepNarrowFeature',
                    'faceIds': face_ids,
                    'face_ids': face_ids,
                    'message': f"Deep {feature_type}: aspect ratio {aspect_ratio:.1f}:1 exceeds recommended 5:1 (depth={depth:.1f}mm, width={min_dimension:.1f}mm)",
                    'severity': 'warning',
                    'feature_type': feature_type,
                    'aspect_ratio': aspect_ratio
                })
                
    def _check_thin_walls(self, feature: Dict):
        """
        Check for thin walls (< 1mm thickness).
        
        Thin walls are prone to:
        - Vibration during machining
        - Deflection under tool pressure
        - Potential breakage
        
        Typical minimum: 1-2mm for aluminum, 2-3mm for softer materials
        """
        feature_type = feature.get('type', '')
        
        # Wall thickness is most relevant for pockets and slots
        if feature_type not in ['pocket', 'slot']:
            return
            
        # Check if we have wall thickness information
        wall_thickness = feature.get('wall_thickness', None)
        min_wall = feature.get('min_wall', None)
        
        thickness = wall_thickness or min_wall
        
        if thickness and thickness < 1.0:  # < 1mm is risky
            face_ids = feature.get('face_ids', []) or feature.get('faceIds', [])
            
            self.warnings.append({
                'code': 2203,
                'label': 'cncCodePartBodyWarnThinWall',
                'faceIds': face_ids,
                'face_ids': face_ids,
                'message': f"Thin wall detected: {thickness:.2f}mm (minimum recommended: 1.0mm)",
                'severity': 'warning',
                'feature_type': feature_type,
                'thickness': thickness
            })
            

def generate_dfm_summary(warnings: List[Dict]) -> Dict:
    """
    Generate summary statistics for DFM warnings.
    
    Args:
        warnings: List of warning dicts from DFMAnalyzer
        
    Returns:
        Summary dict: {
            'total_warnings': int,
            'by_code': Dict[int, int],
            'by_severity': Dict[str, int],
            'critical_count': int
        }
    """
    if not warnings:
        return {
            'total_warnings': 0,
            'by_code': {},
            'by_severity': {},
            'critical_count': 0
        }
        
    by_code = {}
    by_severity = {}
    
    for warning in warnings:
        # Count by code
        code = warning.get('code', 0)
        by_code[code] = by_code.get(code, 0) + 1
        
        # Count by severity
        severity = warning.get('severity', 'warning')
        by_severity[severity] = by_severity.get(severity, 0) + 1
        
    critical_count = by_severity.get('error', 0)
    
    return {
        'total_warnings': len(warnings),
        'by_code': by_code,
        'by_severity': by_severity,
        'critical_count': critical_count
    }


def format_dfm_warnings_for_output(warnings: List[Dict]) -> Dict:
    """
    Format DFM warnings in Analysis Situs compatible format.
    
    Args:
        warnings: List of warning dicts
        
    Returns:
        Formatted warnings dict matching Analysis Situs structure:
        {
            'semanticCodes': {
                'warnings': [...]
            }
        }
    """
    if not warnings:
        return {
            'semanticCodes': {
                'warnings': []
            }
        }
        
    # Convert to Analysis Situs format
    formatted_warnings = []
    
    for warning in warnings:
        formatted_warnings.append({
            'code': warning['code'],
            'label': warning['label'],
            'faceIds': warning.get('faceIds', []),
            'message': warning.get('message', ''),
            # Additional fields for our system
            'severity': warning.get('severity', 'warning'),
            'feature_type': warning.get('feature_type', 'unknown')
        })
        
    return {
        'semanticCodes': {
            'warnings': formatted_warnings
        }
    }
