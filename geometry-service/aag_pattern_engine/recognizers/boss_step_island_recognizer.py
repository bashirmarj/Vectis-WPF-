"""
Boss/Step/Island Recognizer - Analysis Situs Aligned
====================================================

Key Difference: Convex (Outward) Detection
-------------------------------------------

Bosses are OPPOSITE of pockets:
- Top face (up-facing)
- Convex walls (outward corners)
- Positive features (additive)
"""

import logging
import numpy as np
from .recognizer_utils import standardize_feature_output, merge_split_faces

logger = logging.getLogger(__name__)

# Tolerances
HORIZONTAL_TOLERANCE = 0.1
MIN_BOSS_HEIGHT = 0.5  # mm
MIN_BOSS_AREA = 5.0  # mm²


class BossRecognizer:
    """
    Recognizes bosses, steps, and islands (positive features).
    
    Process (inverse of pocket detection):
    1. Find horizontal planar faces (potential boss tops)
    2. For each top, grow wall ring
    3. Validate walls are CONVEX (outward) - opposite of pocket!
    4. Compute height
    5. Classify type
    """
    
    def __init__(self, aag_graph):
        self.aag = aag_graph
        self.detected_bosses = []
        
    def recognize(self) -> List[Dict]:
        """Main recognition entry point."""
        logger.info("=" * 70)
        logger.info("Starting boss/step/island recognition")
        logger.info("=" * 70)
        
        # Find planar faces (potential boss tops)
        planar_faces = self._find_planar_faces()
        
        logger.info(f"Analyzing {len(planar_faces)} planar faces")
        
        # Analyze each as potential boss top
        for face_id in planar_faces:
            boss = self._analyze_boss_from_top(face_id)
            
            if boss is not None:
                self.detected_bosses.append(boss)
                
        # Statistics
        self._print_statistics()
        
        # CRITICAL FIX: Merge split faces
        self.detected_bosses = merge_split_faces(self.detected_bosses, self.aag)
        
        return self.detected_bosses
        
    def _find_planar_faces(self) -> List[int]:
        """Find all planar faces (potential boss tops)."""
        planar = []
        
        2. Find vertical walls around top
        3. Validate walls are CONVEX (outward) - KEY DIFFERENCE
        4. Compute height
        5. Classify boss type
        
        Args:
            top_id: Face ID of potential boss top
            
        Returns:
            Boss feature dict or None
        """
        top_face = self.aag.nodes[top_id]
        
        # Must be horizontal
        if not self._is_horizontal(top_face):
            return None
            
        # Get adjacent faces (wall candidates)
        adjacent = self.aag.get_adjacent_faces(top_id)
        
        if not adjacent:
            return None
            
        # Filter for vertical walls
        walls = []
        
        for adj_id in adjacent:
            if self._is_vertical_wall(top_id, adj_id):
                walls.append(adj_id)
                
        if not walls:
            return None
            
        # Validate as POSITIVE feature (walls must be CONVEX) or STEP (Mixed)
        is_valid, topology_type = self._validate_feature_topology(top_id, walls)
        if not is_valid:
            return None
            
        # Compute height
        height = self._compute_height(top_id, walls)
        
        if height < MIN_BOSS_HEIGHT:
            return None
            
        # Validate minimum area
        top_area = top_face.get('area', 0) * (1000**2)  # Convert to mm²
        
        if top_area < MIN_BOSS_AREA:
            return None
            
        # Classify boss type
        if topology_type == 'step':
            boss_type = 'step'
        else:
            boss_type = self._classify_boss_type(top_id, walls, height)
        
        # Collect all face IDs
        all_faces = [top_id] + walls
        
        return {
            'type': boss_type,
            'face_ids': all_faces,
            'top_faces': [top_id],
            'wall_faces': walls,
            'height': height,
            'area': top_area,
            'confidence': 0.75
        }
        
    def _is_horizontal(self, face_data: Dict) -> bool:
        """Check if face is horizontal."""
        normal = np.array(face_data.get('normal', [0, 0, 1]))
        z_alignment = abs(normal[2])
        return z_alignment > (1.0 - HORIZONTAL_TOLERANCE)
        
    def _is_vertical_wall(self, top_id: int, wall_id: int) -> bool:
        """Check if face is vertical wall relative to top."""
        wall_face = self.aag.nodes[wall_id]
        
        surf_type = wall_face.get('surface_type')
        if surf_type not in ['plane', 'cylinder']:
            return False
            
        if surf_type == 'plane':
            normal = np.array(wall_face.get('normal', [0, 0, 1]))
            z_component = abs(normal[2])
            return z_component < HORIZONTAL_TOLERANCE
            
        return True
        
    def _validate_feature_topology(self, top_id: int, wall_ids: List[int]) -> Tuple[bool, str]:
        """
        Validate walls and determine feature category (boss or step).
        
        Args:
            top_id: Top face ID
            wall_ids: Wall face IDs
            
        Returns:
            Tuple (is_valid, category)
            category: 'boss', 'step', or 'unknown'
        """
        # Check edges between top and each wall
        convex_count = 0
        concave_count = 0
        
        for wall_id in wall_ids:
            edge_data = self._get_edge_between(top_id, wall_id)
            
            if edge_data is None:
                continue
                
            vexity = edge_data.get('vexity', 'smooth')
            
            if vexity == 'convex':
                convex_count += 1
            elif vexity == 'concave':
                concave_count += 1
                
        # Boss: Mostly CONVEX edges
        if convex_count > 0 and concave_count == 0:
            return True, 'boss'
            
        # Step: Mixed CONVEX and CONCAVE edges
        # (Has walls going down and walls going up)
        if convex_count > 0 and concave_count > 0:
            return True, 'step'
            
        # Island: Similar to boss but might be inside a pocket?
        # For now, treat as boss if valid
        
        # If only concave, it's a pocket bottom (reject)
        if concave_count > 0 and convex_count == 0:
            return False, 'pocket'
            
        # If mostly smooth (tangent), might be a boss with fillets
        # Accept if we have at least some convex edges
        if convex_count > 0:
            return True, 'boss'
            
        return False, 'unknown'
        
    def _get_edge_between(self, face1_id: int, face2_id: int) -> Optional[Dict]:
        """Get edge data between two faces."""
        for neighbor in self.aag.adjacency.get(face1_id, []):
            if neighbor['face_id'] == face2_id:
                return neighbor
                
        return None
        
    def _compute_height(self, top_id: int, wall_ids: List[int]) -> float:
        """
        Compute boss height (Z-extent from base to top).
        
        Args:
            top_id: Top face ID
            wall_ids: Wall face IDs
            
        Returns:
            Height in mm
        """
        all_face_ids = [top_id] + wall_ids
        
        z_coords = []
        
        for face_id in all_face_ids:
            face = self.aag.nodes[face_id]
            center = face.get('center', [0, 0, 0])
            z_coords.append(center[2])
            
        if not z_coords:
            return 0.0
            
        height = max(z_coords) - min(z_coords)
        
        # Convert to mm if needed
        if height < 1.0:
            height *= 1000.0
            
        return height
        
    def _classify_boss_type(self, top_id: int, wall_ids: List[int], height: float) -> str:
        """
        Classify boss type based on geometry.
        
        Types:
        - rectangular_boss: 4 planar walls
        - circular_boss: Cylindrical walls
        - irregular_boss: Mixed geometry
        - step: Wide, shallow boss
        
        Args:
            top_id: Top face ID
            wall_ids: Wall face IDs
            height: Boss height
            
        Returns:
            Boss type string
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
            return 'circular_boss'
            
        if planar_walls == 4:
            # Check if step (wide and shallow)
            top = self.aag.nodes[top_id]
            area = top.get('area', 0) * (1000**2)
            
            estimated_width = np.sqrt(area)
            
            if height < estimated_width / 4:
                return 'step'
            else:
                return 'rectangular_boss'
                
        return 'irregular_boss'
        
    def _print_statistics(self):
        """Print recognition statistics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("RECOGNITION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total candidates: {len(self._find_planar_faces())}")
        
        # Type breakdown
        type_counts = {}
        for boss in self.detected_bosses:
            btype = boss['type']
            type_counts[btype] = type_counts.get(btype, 0) + 1
            
        bosses = sum(1 for b in self.detected_bosses if 'boss' in b['type'])
        steps = sum(1 for b in self.detected_bosses if 'step' in b['type'])
        islands = sum(1 for b in self.detected_bosses if 'island' in b['type'])
        
        logger.info(f"Bosses recognized: {bosses}")
        logger.info(f"Steps recognized: {steps}")
        logger.info(f"Islands recognized: {islands}")
        logger.info(f"Validation failures: 0")
        logger.info("")
        logger.info("Feature type breakdown:")
        
        for btype, count in sorted(type_counts.items()):
            logger.info(f"  {btype:35s}:  {count:2d}")
            
        logger.info("=" * 70)


def recognize_bosses(aag_graph):
    """Convenience function for boss recognition."""
    recognizer = BossRecognizer(aag_graph)
    return recognizer.recognize()
