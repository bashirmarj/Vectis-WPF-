"""
Tool Accessibility Analyzer
============================

Determines which machining axes can reach each face.

Analysis Situs provides:
"accessibleSideMillingAxes": [[0, 0, 1], [-0, -0, -1]]

This tells CAM systems:
- Which tool orientations are feasible
- Which faces need special setups
- Which features are impossible to machine
"""

import logging
import numpy as np
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class ToolAccessibilityAnalyzer:
    """
    Analyzes tool accessibility for each face.
    
    Determines:
    - End milling axes (perpendicular to face)
    - Side milling axes (parallel to face)
    - Collision-free axes
    """
    
    def __init__(self, aag_graph):
        self.aag = aag_graph
        
        # Primary machining axes
        self.primary_axes = [
            (1, 0, 0),   # X+
            (-1, 0, 0),  # X-
            (0, 1, 0),   # Y+
            (0, -1, 0),  # Y-
            (0, 0, 1),   # Z+ (most common)
            (0, 0, -1),  # Z-
        ]
        
    def analyze_all_faces(self) -> Dict[int, Dict]:
        """
        Analyze accessibility for all faces.
        
        Returns:
            Dict: {face_id: {
                'end_milling_axes': [...],
                'side_milling_axes': [...],
                'inaccessible_reason': str or None
            }}
        """
        logger.info("Analyzing tool accessibility for all faces...")
        
        results = {}
        
        for face_id, face_data in self.aag.nodes.items():
            results[face_id] = self._analyze_face(face_id, face_data)
            
        # Statistics
        accessible_count = sum(1 for r in results.values() 
                              if r['end_milling_axes'] or r['side_milling_axes'])
        
        logger.info(f"  Accessible faces: {accessible_count}/{len(results)}")
        logger.info(f"  Inaccessible faces: {len(results) - accessible_count}")
        
        return results
        
    def _analyze_face(self, face_id: int, face_data: Dict) -> Dict:
        """
        Analyze single face accessibility.
        
        Args:
            face_id: Face ID
            face_data: Face attributes
            
        Returns:
            Accessibility dict
        """
        surf_type = face_data.get('surface_type')
        
        if surf_type == 'plane':
            return self._analyze_planar_face(face_id, face_data)
        elif surf_type == 'cylinder':
            return self._analyze_cylindrical_face(face_id, face_data)
        else:
            return {
                'end_milling_axes': [],
                'side_milling_axes': [],
                'inaccessible_reason': 'complex_surface'
            }
            
    def _analyze_planar_face(self, face_id: int, face_data: Dict) -> Dict:
        """
        Analyze planar face accessibility.
        
        End milling:
        - Tool axis perpendicular to face (along normal)
        
        Side milling:
        - Tool axis parallel to face (perpendicular to normal)
        - No collision with adjacent faces
        
        Args:
            face_id: Face ID
            face_data: Face attributes
            
        Returns:
            Accessibility dict
        """
        normal = np.array(face_data.get('normal', [0, 0, 1]))
        
        end_milling = []
        side_milling = []
        
        # Check each axis
        for axis in self.primary_axes:
            axis_np = np.array(axis)
            dot = abs(np.dot(normal, axis_np))
            
            # End milling: axis aligned with normal
            if dot > 0.95:  # Within ~18 degrees
                if not self._has_end_milling_collision(face_id, axis_np):
                    end_milling.append(axis)
                    
            # Side milling: axis perpendicular to normal
            elif dot < 0.2:  # Within ~78 degrees from perpendicular
                if not self._has_side_milling_collision(face_id, axis_np):
                    side_milling.append(axis)
                    
        return {
            'end_milling_axes': end_milling,
            'side_milling_axes': side_milling,
            'inaccessible_reason': None if (end_milling or side_milling) else 'no_clear_axis'
        }
        
    def _analyze_cylindrical_face(self, face_id: int, face_data: Dict) -> Dict:
        """
        Analyze cylindrical face accessibility.
        
        Cylinders are typically side-milled along axis.
        
        Args:
            face_id: Face ID
            face_data: Face attributes
            
        Returns:
            Accessibility dict
        """
        axis = np.array(face_data.get('axis', [0, 0, 1]))
        
        end_milling = []
        side_milling = []
        
        # Check each machining axis
        for tool_axis in self.primary_axes:
            tool_axis_np = np.array(tool_axis)
            dot = abs(np.dot(axis, tool_axis_np))
            
            # Side milling: tool parallel to cylinder axis
            if dot > 0.95:
                if not self._has_side_milling_collision(face_id, tool_axis_np):
                    side_milling.append(tool_axis)
                    
        return {
            'end_milling_axes': end_milling,
            'side_milling_axes': side_milling,
            'inaccessible_reason': None if side_milling else 'no_clear_axis'
        }
        
    def _has_end_milling_collision(self, face_id: int, tool_axis: np.ndarray) -> bool:
        """
        Check if end mill along axis would collide with adjacent faces.
        
        Simple heuristic: Check if any adjacent face blocks the tool path.
        
        Args:
            face_id: Face being machined
            tool_axis: Tool approach direction
            
        Returns:
            True if collision detected
        """
        # Get adjacent faces
        adjacent = self.aag.get_adjacent_faces(face_id)
        
        face_center = np.array(self.aag.nodes[face_id].get('center', [0, 0, 0]))
        
        for adj_id in adjacent:
            adj_face = self.aag.nodes[adj_id]
            adj_center = np.array(adj_face.get('center', [0, 0, 0]))
            
            # Vector from face to adjacent
            to_adj = adj_center - face_center
            
            # If adjacent is in tool path direction, collision likely
            dot = np.dot(to_adj, tool_axis)
            
            if dot > 0.01:  # Adjacent in approach direction
                # Check if it blocks (perpendicular)
                adj_normal = np.array(adj_face.get('normal', [0, 0, 1]))
                normal_dot = abs(np.dot(adj_normal, tool_axis))
                
                if normal_dot > 0.7:  # Face blocks tool
                    return True
                    
        return False
        
    def _has_side_milling_collision(self, face_id: int, tool_axis: np.ndarray) -> bool:
        """
        Check if side mill along axis would collide.
        
        Args:
            face_id: Face being machined
            tool_axis: Tool axis direction
            
        Returns:
            True if collision detected
        """
        # Simplified: Assume side milling is feasible if face is accessible
        # More sophisticated version would check tool diameter vs. clearance
        
        return False  # Placeholder - full implementation would check geometry
        
    def annotate_features_with_accessibility(self, features: List[Dict]) -> List[Dict]:
        """
        Add accessibility info to recognized features.
        
        Args:
            features: List of feature dicts
            
        Returns:
            Features with accessibility annotations
        """
        accessibility = self.analyze_all_faces()
        
        for feature in features:
            face_ids = feature.get('face_ids', [])
            
            # Aggregate accessibility for all faces in feature
            all_end_axes = set()
            all_side_axes = set()
            
            for face_id in face_ids:
                if face_id in accessibility:
                    acc = accessibility[face_id]
                    
                    for axis in acc['end_milling_axes']:
                        all_end_axes.add(tuple(axis))
                        
                    for axis in acc['side_milling_axes']:
                        all_side_axes.add(tuple(axis))
                        
            feature['accessible_end_milling_axes'] = [list(a) for a in all_end_axes]
            feature['accessible_side_milling_axes'] = [list(a) for a in all_side_axes]
            
            # Flag inaccessible features
            if not all_end_axes and not all_side_axes:
                feature['manufacturing_warning'] = 'inaccessible_feature'
                
        return features


def analyze_tool_accessibility(aag_graph):
    """Convenience function for accessibility analysis."""
    analyzer = ToolAccessibilityAnalyzer(aag_graph)
    return analyzer.analyze_all_faces()
    

def annotate_features(aag_graph, features: List[Dict]) -> List[Dict]:
    """Add accessibility annotations to features."""
    analyzer = ToolAccessibilityAnalyzer(aag_graph)
    return analyzer.annotate_features_with_accessibility(features)
