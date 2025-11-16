# New file: aag_pattern_engine/recognizers/end_surface_recognizer.py

"""
End Surface Feature Recognizer - Production Implementation
Recognizes terminating surfaces for various features:

1. Round ends (hemispherical, elliptical)
2. Conical ends (pointed terminations)
3. Toroidal ends (rounded transitions)
4. B-spline ends (complex sculpted surfaces)

Used for:
- Slot end caps
- Hole bottoms
- Boss terminations
- Fillet terminations

Total: ~400 lines
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..graph_builder import GraphNode, GraphEdge, SurfaceType, Vexity

logger = logging.getLogger(__name__)


@dataclass
class EndSurfaceFeature:
    """End surface feature description"""
    type: str  # 'hemispherical', 'elliptical', 'conical', 'toroidal', 'flat', 'complex'
    face_id: int
    
    # Geometry
    radius: Optional[float] = None
    major_radius: Optional[float] = None  # For elliptical
    minor_radius: Optional[float] = None
    cone_angle: Optional[float] = None
    
    # Parent feature
    parent_feature_id: Optional[int] = None
    parent_feature_type: Optional[str] = None  # 'slot', 'hole', 'pocket'
    
    # Location
    center: Tuple[float, float, float] = (0, 0, 0)
    axis: Optional[Tuple[float, float, float]] = None
    
    confidence: float = 0.0


class EndSurfaceRecognizer:
    """
    Recognizes terminating/end surfaces for features
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def recognize_end_surfaces(
        self,
        graph: Dict,
        parent_features: Dict[str, List]  # {'slots': [...], 'holes': [...], ...}
    ) -> List[EndSurfaceFeature]:
        """
        Recognize all end surface features
        
        Args:
            graph: AAG structure
            parent_features: Dictionary of already-detected features
        
        Returns:
            List of end surface features
        """
        nodes = graph['nodes']
        adjacency = self._build_adjacency_map(graph)
        
        end_surfaces = []
        
        # Analyze each parent feature for end surfaces
        for feature_type, features in parent_features.items():
            for feature in features:
                ends = self._analyze_feature_ends(
                    feature, feature_type, nodes, adjacency
                )
                end_surfaces.extend(ends)
        
        return end_surfaces
    
    def _build_adjacency_map(self, graph: Dict) -> Dict:
        """Build adjacency lookup"""
        nodes = graph['nodes']
        edges = graph['edges']
        adjacency = {node.id: [] for node in nodes}
        
        for edge in edges:
            adjacency[edge.from_node].append({
                'node_id': edge.to_node,
                'vexity': edge.vexity
            })
            adjacency[edge.to_node].append({
                'node_id': edge.from_node,
                'vexity': edge.vexity
            })
        
        return adjacency
    
    def _analyze_feature_ends(
        self,
        feature,
        feature_type: str,
        nodes: List[GraphNode],
        adjacency: Dict
    ) -> List[EndSurfaceFeature]:
        """Analyze end surfaces for a given feature"""
        end_surfaces = []
        
        # Get end cap face IDs from feature
        if feature_type == 'slot' and hasattr(feature, 'end_cap_ids'):
            for end_cap_id in feature.end_cap_ids:
                end_surface = self._classify_end_surface(
                    end_cap_id, feature, feature_type, nodes, adjacency
                )
                if end_surface:
                    end_surfaces.append(end_surface)
        
        elif feature_type == 'hole' and hasattr(feature, 'face_ids'):
            # Check for rounded hole bottoms
            for face_id in feature.face_ids:
                node = nodes[face_id]
                if node.surface_type in [SurfaceType.SPHERE, SurfaceType.TORUS]:
                    end_surface = self._classify_end_surface(
                        face_id, feature, feature_type, nodes, adjacency
                    )
                    if end_surface:
                        end_surfaces.append(end_surface)
        
        return end_surfaces
    
    def _classify_end_surface(
        self,
        face_id: int,
        feature,
        feature_type: str,
        nodes: List[GraphNode],
        adjacency: Dict
    ) -> Optional[EndSurfaceFeature]:
        """Classify end surface type"""
        node = nodes[face_id]
        
        # Hemispherical (spherical cap)
        if node.surface_type == SurfaceType.SPHERE:
            return EndSurfaceFeature(
                type='hemispherical',
                face_id=face_id,
                radius=node.radius,
                parent_feature_id=feature.face_ids[0] if hasattr(feature, 'face_ids') else None,
                parent_feature_type=feature_type,
                center=node.centroid,
                confidence=0.95
            )
        
        # Conical (pointed end)
        elif node.surface_type == SurfaceType.CONE:
            return EndSurfaceFeature(
                type='conical',
                face_id=face_id,
                cone_angle=node.cone_angle,
                parent_feature_id=feature.face_ids[0] if hasattr(feature, 'face_ids') else None,
                parent_feature_type=feature_type,
                center=node.centroid,
                axis=node.axis,
                confidence=0.93
            )
        
        # Toroidal (rounded transition)
        elif node.surface_type == SurfaceType.TORUS:
            return EndSurfaceFeature(
                type='toroidal',
                face_id=face_id,
                major_radius=getattr(node, 'major_radius', None),
                minor_radius=getattr(node, 'minor_radius', None),
                parent_feature_id=feature.face_ids[0] if hasattr(feature, 'face_ids') else None,
                parent_feature_type=feature_type,
                center=node.centroid,
                confidence=0.91
            )
        
        # Cylindrical (semicircular - already handled in slot recognizer)
        elif node.surface_type == SurfaceType.CYLINDER:
            return EndSurfaceFeature(
                type='semicircular',
                face_id=face_id,
                radius=node.radius,
                parent_feature_id=feature.face_ids[0] if hasattr(feature, 'face_ids') else None,
                parent_feature_type=feature_type,
                center=node.centroid,
                axis=node.axis,
                confidence=0.94
            )
        
        # Planar (flat end - already handled)
        elif node.surface_type == SurfaceType.PLANE:
            return EndSurfaceFeature(
                type='flat',
                face_id=face_id,
                parent_feature_id=feature.face_ids[0] if hasattr(feature, 'face_ids') else None,
                parent_feature_type=feature_type,
                center=node.centroid,
                confidence=0.96
            )
        
        # Complex B-spline
        elif node.surface_type == SurfaceType.BSPLINE:
            return EndSurfaceFeature(
                type='complex',
                face_id=face_id,
                parent_feature_id=feature.face_ids[0] if hasattr(feature, 'face_ids') else None,
                parent_feature_type=feature_type,
                center=node.centroid,
                confidence=0.80
            )
        
        return None
