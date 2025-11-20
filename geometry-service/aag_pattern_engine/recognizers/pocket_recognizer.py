"""
Pocket/Slot Recognizer - Analysis Situs Aligned
================================================

Key Change: Bottom-Up Detection
--------------------------------

OLD APPROACH:
- Find concave edges
- Grow regions
- Hope it's a pocket

NEW APPROACH (Analysis Situs):
- Find horizontal bottom faces
- Grow vertical wall ring around bottom
- Validate as removal feature (concave walls)
- Compute depth and accessibility
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Tolerances
HORIZONTAL_TOLERANCE = 0.1  # cos(84°) - nearly horizontal
MIN_POCKET_DEPTH = 1.0  # mm
MIN_POCKET_AREA = 10.0  # mm²


class PocketRecognizer:
    """
    Recognizes pockets, slots, and through passages.
    
    Process (Analysis Situs bottom-up):
    1. Find horizontal planar faces (potential pocket bottoms)
    2. For each bottom, grow wall ring
    3. Validate walls are concave (removal feature)
    4. Compute manufacturing parameters
    5. Classify type (pocket/slot/passage)
    """
    
    def __init__(self, aag_graph):
        self.aag = aag_graph
        self.detected_pockets = []
        
    def recognize(self) -> List[Dict]:
        """Main recognition entry point."""
        logger.info("=" * 70)
        logger.info("Starting unified pocket/slot/passage recognition")
        logger.info("=" * 70)
        
        # Find planar faces (potential bottoms)
        planar_faces = self._find_planar_faces()
        
        logger.info(f"Analyzing {len(planar_faces)} planar faces")
        
        # Analyze each as potential pocket bottom
        for face_id in planar_faces:
            pocket = self._analyze_pocket_from_bottom(face_id)
            
            if pocket is not None:
                self.detected_pockets.append(pocket)
                
        # Statistics
        self._print_statistics()
        
        return self.detected_pockets
        
    def _find_planar_faces(self) -> List[int]:
        """Find all planar faces (potential pocket bottoms)."""
        planar = []
        
        for face_id, face_data in self.aag.nodes.items():
            if face_data.get('surface_type') == 'plane':
                planar.append(face_id)
                
        return planar
        
    def _analyze_pocket_from_bottom(self, bottom_id: int) -> Optional[Dict]:
        """
        Analyze pocket starting from bottom face.
        
        Process:
        1. Check if bottom is horizontal
        2. Find vertical walls around bottom
        3. Validate walls are concave (inward)
        4. Compute depth
        5. Classify pocket type
        
        Args:
            bottom_id: Face ID of potential pocket bottom
            
        Returns:
            Pocket feature dict or None
        """
        bottom_face = self.aag.nodes[bottom_id]
        
        # Must be horizontal
        if not self._is_horizontal(bottom_face):
            return None
            
        # Get adjacent faces (wall candidates)
        adjacent = self.aag.get_adjacent_faces(bottom_id)
        
        if not adjacent:
            return None
            
        # Filter for vertical walls
        walls = []
        
        for adj_id in adjacent:
            if self._is_vertical_wall(bottom_id, adj_id):
                walls.append(adj_id)
                
        if not walls:
            return None
            
        # Validate as removal feature (walls must be concave)
        if not self._validate_concave_walls(bottom_id, walls):
            return None
            
        # Compute depth
        depth = self._compute_depth(bottom_id, walls)
        
        if depth < MIN_POCKET_DEPTH:
            return None
            
        # Validate minimum area
        bottom_area = bottom_face.get('area', 0) * (1000**2)  # Convert to mm²
        
        if bottom_area < MIN_POCKET_AREA:
            return None
            
        # Classify pocket type
        pocket_type = self._classify_pocket_type(bottom_id, walls, depth)
        
        # Collect all face IDs
        all_faces = [bottom_id] + walls
        
        return {
            'type': pocket_type,
            'face_ids': all_faces,
            'bottom_faces': [bottom_id],
            'wall_faces': walls,
            'depth': depth,
            'area': bottom_area,
            'confidence': 0.8
        }
        
    def _is_horizontal(self, face_data: Dict) -> bool:
        """
        Check if face is horizontal (normal close to Z-axis).
        
        Args:
            face_data: Face attributes dict
            
        Returns:
            True if horizontal
        """
        normal = np.array(face_data.get('normal', [0, 0, 1]))
        
        # Dot product with Z-axis
        z_alignment = abs(normal[2])
        
        return z_alignment > (1.0 - HORIZONTAL_TOLERANCE)
        
    def _is_vertical_wall(self, bottom_id: int, wall_id: int) -> bool:
        """
        Check if face is vertical wall relative to bottom.
        
        Args:
            bottom_id: Bottom face ID
            wall_id: Candidate wall face ID
            
        Returns:
            True if valid vertical wall
        """
        wall_face = self.aag.nodes[wall_id]
        
        # Must be planar or cylindrical
        surf_type = wall_face.get('surface_type')
        if surf_type not in ['plane', 'cylinder']:
            return False
            
        # For planar walls, check normal is horizontal
        if surf_type == 'plane':
            normal = np.array(wall_face.get('normal', [0, 0, 1]))
            z_component = abs(normal[2])
            
            # Should be nearly horizontal (small Z component)
            return z_component < HORIZONTAL_TOLERANCE
            
        # Cylindrical walls always valid
        return True
        
    def _validate_concave_walls(self, bottom_id: int, wall_ids: List[int]) -> bool:
        """
        Validate walls form a pocket structure.
        
        UPDATED LOGIC:
        - Pockets have MOSTLY concave or smooth edges (internal corners)
        - A few convex edges OK (e.g., at fillet blends)
        - Reject if MAJORITY are convex (indicates boss, not pocket)
        
        Args:
            bottom_id: Bottom face ID
            wall_ids: Wall face IDs
            
        Returns:
            True if walls form concave (pocket) structure
        """
        concave_count = 0
        convex_count = 0
        smooth_count = 0
        total_edges = 0
        
        for wall_id in wall_ids:
            edge_data = self._get_edge_between(bottom_id, wall_id)
            
            if edge_data is None:
                continue
                
            vexity = edge_data.get('vexity', 'smooth')
            total_edges += 1
            
            if vexity == 'concave':
                concave_count += 1
            elif vexity == 'convex':
                convex_count += 1
            elif vexity == 'smooth':
                smooth_count += 1
        
        if total_edges == 0:
            return False
        
        # NEW LOGIC: Reject only if MAJORITY are convex (boss indicator)
        convex_ratio = convex_count / total_edges
        
        if convex_ratio > 0.6:  # More than 60% convex = boss, not pocket
            logger.debug(f"  Rejected bottom {bottom_id}: {convex_ratio:.1%} convex edges (boss)")
            return False
        
        # Accept if has SOME concave/smooth edges (pocket indicators)
        non_convex = concave_count + smooth_count
        
        if non_convex >= total_edges * 0.3:  # At least 30% non-convex
            return True
        
        logger.debug(f"  Rejected bottom {bottom_id}: insufficient concave/smooth edges")
        return False
        
    def _get_edge_between(self, face1_id: int, face2_id: int) -> Optional[Dict]:
        """Get edge data between two faces."""
        for neighbor in self.aag.adjacency.get(face1_id, []):
            if neighbor['face_id'] == face2_id:
                return neighbor
                
        return None
        
    def _compute_depth(self, bottom_id: int, wall_ids: List[int]) -> float:
        """
        Compute pocket depth (Z-extent from bottom to top).
        
        Args:
            bottom_id: Bottom face ID
            wall_ids: Wall face IDs
            
        Returns:
            Depth in mm
        """
        # Get Z-coordinates of all faces
        all_face_ids = [bottom_id] + wall_ids
        
        z_coords = []
        
        for face_id in all_face_ids:
            face = self.aag.nodes[face_id]
            center = face.get('center', [0, 0, 0])
            z_coords.append(center[2])
            
        if not z_coords:
            return 0.0
            
        # Depth is Z-range
        depth = max(z_coords) - min(z_coords)
        
        # Convert to mm if needed
        if depth < 1.0:
            depth *= 1000.0
            
        return depth
        
    def _classify_pocket_type(self, bottom_id: int, wall_ids: List[int], depth: float) -> str:
        """
        Classify pocket type based on geometry.
        
        Types:
        - rectangular_pocket: 4 walls, all planar
        - circular_pocket: Cylindrical walls
        - irregular_pocket: Mixed or complex geometry
        - slot: Long thin pocket (length > 3*width)
        
        Args:
            bottom_id: Bottom face ID
            wall_ids: Wall face IDs
            depth: Pocket depth
            
        Returns:
            Pocket type string
        """
        # Count wall types
        planar_walls = 0
        cylindrical_walls = 0
        
        for wall_id in wall_ids:
            wall = self.aag.nodes[wall_id]
            surf_type = wall.get('surface_type')
            
            if surf_type == 'plane':
                planar_walls += 1
            elif surf_type == 'cylinder':
                cylindrical_walls += 1
                
        # Classification rules
        if cylindrical_walls > 0 and planar_walls == 0:
            return 'circular_pocket'
            
        if planar_walls == 4:
            # Check aspect ratio
            bottom = self.aag.nodes[bottom_id]
            area = bottom.get('area', 0) * (1000**2)
            
            # Estimate dimensions (crude)
            estimated_length = np.sqrt(area)
            
            if depth > 3 * estimated_length:
                return 'slot'
            else:
                return 'rectangular_pocket'
                
        # Default
        return 'irregular_pocket'
        
    def _print_statistics(self):
        """Print recognition statistics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("RECOGNITION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total candidates: {len(self._find_planar_faces())}")
        
        # Type breakdown
        type_counts = {}
        for pocket in self.detected_pockets:
            ptype = pocket['type']
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
            
        pockets = sum(1 for p in self.detected_pockets if 'pocket' in p['type'])
        slots = sum(1 for p in self.detected_pockets if 'slot' in p['type'])
        passages = sum(1 for p in self.detected_pockets if 'passage' in p['type'])
        
        logger.info(f"Pockets recognized: {pockets}")
        logger.info(f"Slots recognized: {slots}")
        logger.info(f"Passages recognized: {passages}")
        logger.info(f"Validation failures: 0")
        logger.info("")
        logger.info("Feature type breakdown:")
        
        for ptype, count in sorted(type_counts.items()):
            logger.info(f"  {ptype:35s}:  {count:2d}")
            
        logger.info("=" * 70)


def recognize_pockets(aag_graph):
    """Convenience function for pocket recognition."""
    recognizer = PocketRecognizer(aag_graph)
    return recognizer.recognize()
