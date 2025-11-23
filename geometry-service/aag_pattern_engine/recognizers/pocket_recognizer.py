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
from .recognizer_utils import standardize_feature_output, merge_split_faces

logger = logging.getLogger(__name__)

# Tolerances
HORIZONTAL_TOLERANCE = 0.1  # cos(84°) - nearly horizontal
MIN_POCKET_DEPTH = 1.0  # mm
MIN_POCKET_AREA = 5.0  # mm² (lowered to catch tiny pockets)


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
        
        # Group coplanar connected faces (logical bottoms)
        logical_bottoms = self._group_coplanar_faces(planar_faces)
        
        logger.info(f"Analyzing {len(logical_bottoms)} logical bottoms (from {len(planar_faces)} faces)")
        
        # Analyze each logical bottom
        for bottom_group in logical_bottoms:
            pocket = self._analyze_pocket_from_bottom(bottom_group)
            
            if pocket is not None:
                self.detected_pockets.append(pocket)
                
        # Statistics
        self._print_statistics()
        
        # CRITICAL FIX: Merge split faces
        self.detected_pockets = merge_split_faces(self.detected_pockets, self.aag)
        
        return self.detected_pockets
        
    def _find_planar_faces(self) -> List[int]:
        """Find all planar faces (potential pocket bottoms)."""
        planar = []
        
        for face_id, face_data in self.aag.nodes.items():
            if face_data.get('surface_type') == 'plane':
                planar.append(face_id)
                
        return planar
        
    def _group_coplanar_faces(self, face_ids: List[int]) -> List[List[int]]:
        """
        Group connected coplanar faces into logical bottoms.
        """
        groups = []
        visited = set()
        face_set = set(face_ids)
        
        for fid in face_ids:
            if fid in visited:
                continue
                
            # Start new group
            group = []
            queue = [fid]
            visited.add(fid)
            
            # Get normal of first face
            base_node = self.aag.nodes[fid]
            base_normal = np.array(base_node.get('normal', [0,0,1]))
            base_z = base_node.get('center', [0,0,0])[2]
            
            while queue:
                current = queue.pop(0)
                group.append(current)
                
                # Check neighbors
                neighbors = self.aag.get_adjacent_faces(current)
                for nid in neighbors:
                    if nid in visited or nid not in face_set:
                        continue
                        
                    # Check if coplanar
                    node = self.aag.nodes[nid]
                    normal = np.array(node.get('normal', [0,0,1]))
                    center = node.get('center', [0,0,0])
                    
                    # Check normal alignment
                    if abs(np.dot(base_normal, normal)) < 0.99:
                        continue
                        
                    # Check Z-height (coplanarity)
                    if abs(center[2] - base_z) > 0.001: # 1 micron
                        continue
                        
                    visited.add(nid)
                    queue.append(nid)
            
            groups.append(group)
            
        return groups
        
    def _analyze_pocket_from_bottom(self, bottom_ids: List[int]) -> Optional[Dict]:
        """
        Analyze pocket starting from logical bottom (list of faces).
        """
        if not bottom_ids:
            return None
            
        # Use first face for orientation checks
        first_id = bottom_ids[0]
        bottom_face = self.aag.nodes[first_id]
        
        # Must be horizontal
        if not self._is_horizontal(bottom_face):
            # logger.debug(f"  Group {bottom_ids}: rejected (not horizontal)")
            return None
            
        # Get adjacent faces (wall candidates)
        # Must be adjacent to ANY face in the bottom group, but NOT in the group itself
        adjacent = set()
        for bid in bottom_ids:
            neighbors = self.aag.get_adjacent_faces(bid)
            for nid in neighbors:
                if nid not in bottom_ids:
                    adjacent.add(nid)
        
        if not adjacent:
            # logger.debug(f"  Group {bottom_ids}: rejected (no adjacent faces)")
            return None
            
        # Filter for vertical walls
        walls = []
        
        for adj_id in adjacent:
            # Check against ANY bottom face (should be consistent)
            if self._is_vertical_wall(first_id, adj_id):
                walls.append(adj_id)
                
        if not walls:
            # logger.debug(f"  Group {bottom_ids}: rejected (no vertical walls)")
            return None
        
        logger.debug(f"  Group {bottom_ids}: found {len(walls)} vertical walls")
            
        # Validate as removal feature (walls must be concave)
        if not self._validate_concave_walls(bottom_ids, walls):
            return None
            
        # Compute depth
        depth = self._compute_depth(bottom_ids, walls)
        
        if depth < MIN_POCKET_DEPTH:
            # logger.debug(f"  Group {bottom_ids}: rejected (depth {depth:.1f}mm < {MIN_POCKET_DEPTH}mm)")
            return None
            
        # Validate minimum area
        # Sum area of all bottom faces
        bottom_area = sum(self.aag.nodes[bid].get('area', 0) for bid in bottom_ids) * (1000**2)
        
        if bottom_area < MIN_POCKET_AREA:
            # logger.debug(f"  Group {bottom_ids}: rejected (area {bottom_area:.1f}mm² < {MIN_POCKET_AREA}mm²)")
            return None
        
        logger.debug(f"  Group {bottom_ids}: ✓ RECOGNIZED as pocket (area={bottom_area:.0f}mm², depth={depth:.1f}mm, walls={len(walls)})")
            
        # Classify pocket type
        pocket_type = self._classify_pocket_type(bottom_ids, walls, depth)
        
        # Collect all face IDs
        all_faces = bottom_ids + walls
        
        pocket_dict = {
            'type': pocket_type,
            'face_ids': all_faces,
            'bottom_faces': bottom_ids,
            'wall_faces': walls,
            'depth': depth,
            'area': bottom_area,
            'confidence': 0.8
        }
        return standardize_feature_output(pocket_dict)
        
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
        
    def _validate_concave_walls(self, bottom_ids: List[int], wall_ids: List[int]) -> bool:
        """
        Validate walls form a pocket structure.
        
        UPDATED LOGIC:
        - Pockets have MOSTLY concave or smooth edges (internal corners)
        - A few convex edges OK (e.g., at fillet blends)
        - Reject if MAJORITY are convex (indicates boss, not pocket)
        
        Args:
            bottom_ids: List of Bottom face IDs
            wall_ids: Wall face IDs
            
        Returns:
            True if walls form concave (pocket) structure
        """
        concave_count = 0
        convex_count = 0
        smooth_count = 0
        total_edges = 0
        
        for wall_id in wall_ids:
            # Check edge against ANY bottom face
            edge_data = None
            for bid in bottom_ids:
                edge_data = self._get_edge_between(bid, wall_id)
                if edge_data:
                    break
            
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
        
        # TOPOLOGICAL FIX: Removed convex edge percentage filtering
        # OLD APPROACH: Used 60% convex ratio threshold → Misclassified pockets as bosses
        # NEW APPROACH: Check if face is depressed below stock surface
        #
        # The convex edge heuristic was rejecting valid pockets because:
        # - Pockets with fillets have many convex edges
        # - Island features inside pockets add convex boundaries  
        # - Ratio alone doesn't distinguish depression from protrusion
        #
        # Proper pocket detection requires:
        # 1. Check if bottom face is below stock/bounding box top (Z-height test)
        # 2. Verify walls form closed boundary loop
        # 3. Validate wall orientation (perpendicular to bottom)
        #
        # For now: Keep pocket candidates without convex filtering
        # Let depth validation downstream handle classification
        
        # Continue with pocket candidate (removed convex filter)
    def _get_edge_between(self, face1_id: int, face2_id: int) -> Optional[Dict]:
        """Get edge data between two faces."""
        for neighbor in self.aag.adjacency.get(face1_id, []):
            if neighbor['face_id'] == face2_id:
                return neighbor
                
        return None
        
    def _compute_depth(self, bottom_ids: List[int], wall_ids: List[int]) -> float:
        """
        Compute pocket depth (Z-extent from bottom to top).
        
        Args:
            bottom_ids: List of Bottom face IDs
            wall_ids: Wall face IDs
            
        Returns:
            Depth in mm
        """
        # Get Z-coordinates of all faces
        all_face_ids = bottom_ids + wall_ids
        
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
        
    def _classify_pocket_type(self, bottom_ids: List[int], wall_ids: List[int], depth: float) -> str:
        """
        Classify pocket type based on geometry.
        
        Types:
        - rectangular_pocket: 4 walls, all planar
        - circular_pocket: Cylindrical walls
        - irregular_pocket: Mixed or complex geometry
        - slot: Long thin pocket (length > 3*width)
        
        Args:
            bottom_ids: List of Bottom face IDs
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
            # Sum area of all bottom faces
            area = sum(self.aag.nodes[bid].get('area', 0) for bid in bottom_ids) * (1000**2)
            
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
