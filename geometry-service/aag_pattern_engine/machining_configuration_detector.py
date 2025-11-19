"""
Machining Configuration Detector
=================================

Detects multiple machining setups (axis/depth combinations) on a single removal volume.
This replaces the old "split volumes" approach.

Analysis Situs Concept:
- One solid can have multiple "prismaticMilling" configurations
- Each configuration has: axis, depth, wall faces, bottom faces
- Features are detected WITHIN each configuration

Example from Analysis Situs:
{
  "prismaticMilling": [
    {
      "axis": [0, 0, 1],
      "depth": 29,
      "faceIds": [38, 39, 40, ...],  # 48 wall faces
      "bottomFaces": [{"faceId": 103, ...}]
    },
    {
      "axis": [0, 0, 1],
      "depth": 29,
      "faceIds": [33, 54, 56],  # Different pocket
      "bottomFaces": [{"faceId": 33, ...}]
    }
  ]
}
"""

import logging
import numpy as np
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class MachiningConfigurationDetector:
    """
    Detects machining configurations (setups) on a removal volume.
    
    A "configuration" is:
    - Primary machining axis (Z, Y, X, etc.)
    - Depth of cut from stock surface
    - Set of faces machined in this setup
    """
    
    def __init__(self, aag_graph):
        """
        Args:
            aag_graph: AAGGraph dict for the removal volume
        """
        self.aag = aag_graph
        
        # Primary machining axes (most common for prismatic)
        self.primary_axes = [
            (0, 0, 1),   # Z-up (top-down machining)
            (0, 0, -1),  # Z-down (bottom-up)
            (0, 1, 0),   # Y+ (side)
            (0, -1, 0),  # Y- (side)
            (1, 0, 0),   # X+ (side)
            (-1, 0, 0),  # X- (side)
        ]
        
    def detect_configurations(self) -> List[Dict]:
        """
        Detect all machining configurations in the removal volume.
        
        Returns:
            List of configuration dicts:
            [{
                'axis': (x, y, z),
                'depth': float (mm),
                'bottom_faces': [face_ids],
                'wall_faces': [face_ids],
                'clearance': float (mm),
                'type': 'primary' or 'secondary'
            }]
        """
        logger.info("Detecting machining configurations...")
        
        configurations = []
        
        # Analyze each primary axis
        for axis in self.primary_axes:
            config = self._analyze_axis_configuration(axis)
            if config is not None:
                configurations.append(config)
                
        logger.info(f"✓ Found {len(configurations)} machining configuration(s)")
        
        # Sort by importance (primary first, largest depth first)
        configurations.sort(key=lambda c: (
            0 if c['type'] == 'primary' else 1,
            -c['depth']
        ))
        
        return configurations
        
    def _analyze_axis_configuration(self, axis: Tuple[float, float, float]) -> Dict:
        """
        Analyze if this axis represents a valid machining configuration.
        
        Process:
        1. Find all planar faces perpendicular to axis (potential bottoms)
        2. For each bottom candidate, grow wall ring
        3. Validate as removal feature
        4. Compute depth and clearance
        
        Args:
            axis: (x, y, z) unit vector
            
        Returns:
            Configuration dict or None if no valid config
        """
        axis_np = np.array(axis)
        
        # Find bottom face candidates (perpendicular to axis)
        bottom_candidates = self._find_bottom_candidates(axis_np)
        
        if not bottom_candidates:
            return None
            
        # Find walls for each bottom
        valid_pockets = []
        
        for bottom_id in bottom_candidates:
            pocket = self._analyze_pocket_from_bottom(bottom_id, axis_np)
            if pocket is not None:
                valid_pockets.append(pocket)
                
        if not valid_pockets:
            return None
            
        # Merge all pockets into single configuration
        all_bottom_faces = []
        all_wall_faces = []
        max_depth = 0.0
        min_clearance = float('inf')
        
        for pocket in valid_pockets:
            all_bottom_faces.extend(pocket['bottom_faces'])
            all_wall_faces.extend(pocket['wall_faces'])
            max_depth = max(max_depth, pocket['depth'])
            min_clearance = min(min_clearance, pocket.get('clearance', float('inf')))
            
        # Determine if primary configuration
        is_primary = len(all_wall_faces) > 10 or max_depth > 10.0  # mm
        
        return {
            'axis': axis,
            'depth': max_depth,
            'bottom_faces': list(set(all_bottom_faces)),  # Remove duplicates
            'wall_faces': list(set(all_wall_faces)),
            'clearance': min_clearance if min_clearance != float('inf') else None,
            'type': 'primary' if is_primary else 'secondary',
            'pocket_count': len(valid_pockets)
        }
        
    def _find_bottom_candidates(self, axis: np.ndarray) -> List[int]:
        """
        Find faces that could be pocket bottoms (perpendicular to machining axis).
        
        Args:
            axis: Machining axis unit vector
            
        Returns:
            List of face IDs
        """
        candidates = []
        nodes = self.aag.get('nodes', {})
        
        for face_id, face_data in nodes.items():
            # Must be planar
            if face_data.get('surface_type') != 'plane':
                continue
                
            # Normal must be parallel to axis (perpendicular to cutting direction)
            normal = np.array(face_data.get('normal', [0, 0, 1]))
            dot = abs(np.dot(normal, axis))
            
            if dot > 0.98:  # Nearly parallel (within ~11 degrees)
                candidates.append(face_id)
                
        return candidates
        
    def _analyze_pocket_from_bottom(self, bottom_id: int, axis: np.ndarray) -> Dict:
        """
        Analyze pocket starting from bottom face.
        
        Process (Analysis Situs approach):
        1. Start from bottom face
        2. Find adjacent faces (wall candidates)
        3. Filter walls: must be vertical relative to axis
        4. Compute depth from bottom to top surface
        5. Validate as removal feature (walls should be concave)
        
        Args:
            bottom_id: Face ID of pocket bottom
            axis: Machining axis
            
        Returns:
            Pocket dict or None
        """
        nodes = self.aag.get('nodes', {})
        adjacency = self.aag.get('adjacency', {})
        
        bottom_face = nodes.get(bottom_id)
        if not bottom_face:
            return None
        
        # Get adjacent faces
        adjacent = adjacency.get(bottom_id, [])
        
        if not adjacent:
            return None
            
        # Filter for vertical walls
        wall_faces = []
        
        for adj_data in adjacent:
            adj_id = adj_data.get('neighbor')
            if adj_id is None:
                continue
                
            adj_face = nodes.get(adj_id)
            if not adj_face:
                continue
            
            # Wall must be perpendicular to axis (parallel to cutting direction)
            adj_normal = np.array(adj_face.get('normal', [0, 0, 1]))
            dot = abs(np.dot(adj_normal, axis))
            
            if dot < 0.2:  # Nearly perpendicular (within ~11 degrees)
                # Check if this edge is concave (indicates material removal)
                vexity = adj_data.get('vexity', 'smooth')
                if vexity == 'concave' or vexity == 'smooth':
                    wall_faces.append(adj_id)
                    
        if not wall_faces:
            return None
            
        # Compute depth (distance from bottom to furthest wall top)
        bottom_center = np.array(bottom_face.get('center', [0, 0, 0]))
        max_depth = 0.0
        
        for wall_id in wall_faces:
            wall_face = nodes.get(wall_id)
            if wall_face:
                wall_center = np.array(wall_face.get('center', [0, 0, 0]))
                depth = abs(np.dot(wall_center - bottom_center, axis))
                max_depth = max(max_depth, depth)
                
        if max_depth < 1.0:  # Less than 1mm - not a significant pocket
            return None
            
        return {
            'bottom_faces': [bottom_id],
            'wall_faces': wall_faces,
            'depth': max_depth,
            'clearance': None  # TODO: Compute tool clearance
        }
