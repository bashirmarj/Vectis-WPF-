"""
Slot Feature Recognizer - Production Implementation
Recognizes 4 slot types with comprehensive analysis:
1. Through slots (open on both ends)
2. Blind slots (closed on one end)
3. T-slots (with undercut)
4. Dovetail slots (angled walls)

Total: ~900 lines
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from ..graph_builder import GraphNode, GraphEdge, SurfaceType, Vexity
from ..utils.vexity_helpers import is_vertical_wall_transition

logger = logging.getLogger(__name__)


@dataclass
class SlotEndCapAnalysis:
    """Analysis of slot end cap geometry"""
    type: str  # 'semicircular', 'flat', 'angled', 'complex'
    face_id: int
    radius: Optional[float] = None
    diameter: Optional[float] = None
    location: Tuple[float, float, float] = (0, 0, 0)


@dataclass
class SlotWallAnalysis:
    """Detailed slot wall analysis"""
    face_id: int
    side: str  # 'left', 'right', 'bottom'
    is_parallel_to_opposite: bool
    angle_from_vertical: float = 0.0  # For dovetail detection
    width: Optional[float] = None
    length: float = 0.0


@dataclass
class SlotFeature:
    """Complete slot feature description"""
    type: str  # 'through_slot', 'blind_slot', 't_slot', 'dovetail_slot'
    face_ids: List[int]
    
    # Core geometry
    bottom_face_id: int
    wall_face_ids: List[int]  # [left_wall, right_wall]
    end_cap_ids: List[int]  # End caps (0, 1, or 2)
    
    # Dimensions
    width: float
    length: float
    depth: float
    
    # Slot-specific features
    end_cap_analysis: Optional[List[SlotEndCapAnalysis]] = None
    wall_analysis: Optional[List[SlotWallAnalysis]] = None
    
    # T-slot specific
    undercut_depth: Optional[float] = None
    undercut_width: Optional[float] = None
    neck_width: Optional[float] = None
    
    # Dovetail specific
    dovetail_angle: Optional[float] = None
    top_width: Optional[float] = None
    bottom_width: Optional[float] = None
    
    # Advanced properties
    volume: Optional[float] = None
    centerline_axis: Optional[Tuple[float, float, float]] = None
    is_curved: bool = False
    curve_radius: Optional[float] = None
    
    # Sub-features
    sub_features: Optional[List[int]] = None
    
    # Quality metrics
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)


class SlotRecognizer:
    """
    Production-grade slot recognizer using AAG pattern matching
    
    Recognition strategy:
    1. Find elongated planar bottoms (potential slot floors)
    2. Identify parallel vertical walls on sides
    3. Analyze end caps (semicircular, flat, or missing)
    4. Classify slot type based on wall configuration
    5. Detect special features (undercuts, dovetails)
    6. Compute precise dimensions
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.min_slot_width = 0.001  # 1mm minimum
        self.max_slot_width = 0.100  # 100mm maximum
        self.min_slot_length = 0.003  # 3mm minimum (must be longer than wide)
        self.min_slot_depth = 0.001  # 1mm minimum
        self.max_slot_depth = 0.200  # 200mm maximum
        self.parallel_tolerance = 0.1  # cos(angle) threshold
        self.dovetail_angle_min = 5.0  # degrees
        self.dovetail_angle_max = 30.0  # degrees
        
    def recognize_slots(self, graph: Dict) -> List[SlotFeature]:
        """
        Recognize all slot features in graph
        
        Args:
            graph: AAG with 'nodes' and 'edges'
            
        Returns:
        """
        # Get detected orientation from graph metadata
        self._up_axis = np.array(graph['metadata'].get('up_axis', [0.0, 0.0, 1.0]))
            List of detected slot features
        """
        logger.info("Starting slot recognition...")
        
        nodes = graph['nodes']
        edges = graph['edges']
        
        # Get pre-built adjacency from graph (performance optimization)
        adjacency = graph.get('adjacency')
        if adjacency is None:
            logger.warning("Adjacency not in graph - rebuilding (performance hit)")
            adjacency = self._build_adjacency_map(nodes, edges)
        
        # Find planar bottom candidates (elongated faces)
        planar_nodes = [n for n in nodes if n.surface_type == SurfaceType.PLANE]
        logger.info(f"Analyzing {len(planar_nodes)} planar faces for slots")
        
        slots = []
        processed_nodes = set()
        
        for bottom_candidate in planar_nodes:
            if bottom_candidate.id in processed_nodes:
                continue
            
            # Check if this could be a slot bottom
            if not self._is_potential_slot_bottom(bottom_candidate, adjacency, nodes):
                continue
            
            # Find slot walls (should be 2 parallel vertical walls)
            walls = self._find_slot_walls(bottom_candidate, adjacency, nodes)
            
            if len(walls) != 2:
                continue
            
            # Validate walls are parallel and opposite
            if not self._validate_parallel_walls(walls, nodes):
                continue
            
            # Find end caps
            end_caps = self._find_slot_end_caps(bottom_candidate, walls, adjacency, nodes)
            
            # Analyze wall geometry
            wall_analyses = self._analyze_slot_walls(
                walls, bottom_candidate, adjacency, nodes
            )
            
            # Analyze end caps
            end_cap_analyses = self._analyze_end_caps(end_caps, nodes)
            
            # Classify slot type based on configuration
            slot = None
            
            # 1. T-slot (has undercut)
            t_slot = self._recognize_t_slot(
                bottom_candidate, walls, end_caps,
                wall_analyses, end_cap_analyses,
                adjacency, nodes
            )
            if t_slot:
                logger.debug(f"Detected T-slot: {t_slot.width:.2f}×{t_slot.length:.2f}×{t_slot.depth:.2f}")
                slots.append(t_slot)
                processed_nodes.update(t_slot.face_ids)
                continue
            
            # 2. Dovetail slot (angled walls)
            dovetail = self._recognize_dovetail_slot(
                bottom_candidate, walls, end_caps,
                wall_analyses, end_cap_analyses,
                adjacency, nodes
            )
            if dovetail:
                logger.debug(f"Detected dovetail slot: {dovetail.width:.2f}×{dovetail.length:.2f}")
                slots.append(dovetail)
                processed_nodes.update(dovetail.face_ids)
                continue
            
            # 3. Through slot (no end caps or 2 end caps)
            if len(end_caps) == 0 or len(end_caps) == 2:
                through_slot = self._recognize_through_slot(
                    bottom_candidate, walls, end_caps,
                    wall_analyses, end_cap_analyses,
                    adjacency, nodes
                )
                if through_slot:
                    logger.debug(f"Detected through slot: {through_slot.width:.2f}×{through_slot.length:.2f}")
                    slots.append(through_slot)
                    processed_nodes.update(through_slot.face_ids)
                    continue
            
            # 4. Blind slot (has 1 end cap)
            if len(end_caps) == 1:
                blind_slot = self._recognize_blind_slot(
                    bottom_candidate, walls, end_caps,
                    wall_analyses, end_cap_analyses,
                    adjacency, nodes
                )
                if blind_slot:
                    logger.debug(f"Detected blind slot: {blind_slot.width:.2f}×{blind_slot.length:.2f}")
                    slots.append(blind_slot)
                    processed_nodes.update(blind_slot.face_ids)
                    continue
        
        logger.info(f"✅ Recognized {len(slots)} slot features")
        return slots
    
    def _build_adjacency_map(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> Dict[int, List[Dict]]:
        """Build adjacency lookup"""
        adjacency = {node.id: [] for node in nodes}
        
        for edge in edges:
            adjacency[edge.from_node].append({
                'node_id': edge.to_node,
                'vexity': edge.vexity,
                'angle': edge.dihedral_angle,
                'edge_length': edge.shared_edge_length
            })
            adjacency[edge.to_node].append({
                'node_id': edge.from_node,
                'vexity': edge.vexity,
                'angle': edge.dihedral_angle,
                'edge_length': edge.shared_edge_length
            })
        
        return adjacency
    
    def _is_potential_slot_bottom(
        self,
        node: GraphNode,
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> bool:
        """
        Check if planar face could be a slot bottom
        
        Criteria:
        1. Horizontal or near-horizontal
        2. Elongated shape (length >> width)
        3. Has adjacent vertical walls
        4. Appropriate size range
        """
        # Must be horizontal (bottom of slot)
        if not self._is_horizontal_face(node):
            return False
        
        # Must have sufficient adjacent faces
        adjacent = adjacency[node.id]
        concave_adjacent = [
            adj for adj in adjacent
            if is_vertical_wall_transition(adj['vexity'])
        ]
        
        if len(concave_adjacent) < 2:
            return False
        
        # Check for vertical walls
        vertical_walls = 0
        for adj in concave_adjacent:
            adj_node = nodes[adj['node_id']]
            if self._is_vertical_wall(adj_node):
                vertical_walls += 1
        
        if vertical_walls < 2:
            return False
        
        # Check elongation (must be longer than wide)
        # Estimate from area and shape
        if not self._is_elongated_shape(node):
            return False
        
        return True
    
    def _is_horizontal_face(self, node: GraphNode) -> bool:
        """Check if face is horizontal"""
        normal = np.array(node.normal)
        up = self._up_axis
        dot = abs(np.dot(normal, up))
        return dot > 0.9  # Within ~25° of horizontal
    
    def _is_vertical_wall(self, node: GraphNode) -> bool:
        """Check if face is vertical (slot wall)"""
        if node.surface_type == SurfaceType.PLANE:
            normal = np.array(node.normal)
            up = self._up_axis
            dot = abs(np.dot(normal, up))
            return dot < 0.2  # Within ~80° of horizontal normal = vertical face
        
        elif node.surface_type == SurfaceType.CYLINDER:
            if not node.axis:
                return False
            axis = np.array(node.axis)
            up = self._up_axis
            dot = abs(np.dot(axis, up))
            return dot > 0.9  # Cylinder axis vertical
        
        return False
    
    def _is_elongated_shape(self, node: GraphNode) -> bool:
        """
        Check if face is elongated (slot-like)
        
        Estimates aspect ratio from face area and assumes rectangular shape
        """
        # This is approximate - in production would analyze actual boundary
        area = node.area
        
        # Assume slot, estimate dimensions
        # For now, accept any planar face and validate later with walls
        return area > self.min_slot_width * self.min_slot_length
    
    def _find_slot_walls(
        self,
        bottom: GraphNode,
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> List[int]:
        """
        Find the two parallel walls of a slot
        
        Returns:
            List of 2 wall face IDs (left and right walls)
        """
        adjacent = adjacency[bottom.id]
        
        # Find all vertical walls with concave transition
        wall_candidates = []
        for adj in adjacent:
            if not is_vertical_wall_transition(adj['vexity']):
                continue
            
            wall_node = nodes[adj['node_id']]
            
            # Must be planar (slot walls are flat)
            if wall_node.surface_type != SurfaceType.PLANE:
                continue
            
            # Must be vertical
            if not self._is_vertical_wall(wall_node):
                continue
            
            wall_candidates.append(wall_node.id)
        
        # Should find exactly 2 walls (left and right)
        if len(wall_candidates) == 2:
            return wall_candidates
        
        # If more than 2, find the two most parallel
        if len(wall_candidates) > 2:
            return self._select_most_parallel_pair(wall_candidates, nodes)
        
        return []
    
    def _select_most_parallel_pair(
        self,
        candidates: List[int],
        nodes: List[GraphNode]
    ) -> List[int]:
        """Select the two most parallel walls from candidates"""
        best_pair = []
        best_parallelism = -1
        
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                wall1 = nodes[candidates[i]]
                wall2 = nodes[candidates[j]]
                
                # Check parallelism
                normal1 = np.array(wall1.normal)
                normal2 = np.array(wall2.normal)
                
                # Parallel walls have opposite normals (facing each other)
                dot = np.dot(normal1, normal2)
                parallelism = abs(dot + 1.0)  # Close to -1 means opposite
                
                if parallelism < best_parallelism or best_parallelism < 0:
                    best_parallelism = parallelism
                    best_pair = [candidates[i], candidates[j]]
        
        return best_pair
    
    def _validate_parallel_walls(
        self,
        walls: List[int],
        nodes: List[GraphNode]
    ) -> bool:
        """Validate that two walls are parallel and opposite"""
        if len(walls) != 2:
            return False
        
        wall1 = nodes[walls[0]]
        wall2 = nodes[walls[1]]
        
        normal1 = np.array(wall1.normal)
        normal2 = np.array(wall2.normal)
        
        # Should be opposite (dot ≈ -1)
        dot = np.dot(normal1, normal2)
        
        return dot < -0.9  # Nearly opposite
    
    def _find_slot_end_caps(
        self,
        bottom: GraphNode,
        walls: List[int],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> List[int]:
        """
        Find end cap faces (semicircular or flat ends)
        
        Returns:
            List of end cap face IDs (0, 1, or 2)
        """
        adjacent = adjacency[bottom.id]
        wall_set = set(walls)
        
        end_caps = []
        for adj in adjacent:
            adj_node = nodes[adj['node_id']]
            
            # Skip walls
            if adj_node.id in wall_set:
                continue
            
            # Skip non-vertical wall transitions
            if not is_vertical_wall_transition(adj['vexity']):
                continue
            
            # End caps are typically:
            # - Cylindrical (semicircular end)
            # - Planar (flat end)
            if adj_node.surface_type in [SurfaceType.CYLINDER, SurfaceType.PLANE]:
                # Verify it's perpendicular to slot direction
                if self._is_slot_end_cap(adj_node, bottom, walls, nodes):
                    end_caps.append(adj_node.id)
        
        return end_caps
    
    def _is_slot_end_cap(
        self,
        candidate: GraphNode,
        bottom: GraphNode,
        walls: List[int],
        nodes: List[GraphNode]
    ) -> bool:
        """Check if face is a valid slot end cap"""
        # End cap should be perpendicular to slot direction
        # Slot direction is perpendicular to wall normals
        
        wall1 = nodes[walls[0]]
        wall_normal = np.array(wall1.normal)
        
        # Slot direction (along length)
        slot_direction = np.array([wall_normal[1], -wall_normal[0], 0])
        slot_direction = slot_direction / np.linalg.norm(slot_direction)
        
        if candidate.surface_type == SurfaceType.CYLINDER:
            # Cylindrical end cap: axis should align with slot direction
            if not candidate.axis:
                return False
            cap_axis = np.array(candidate.axis)
            dot = abs(np.dot(cap_axis, slot_direction))
            return dot > 0.9
        
        elif candidate.surface_type == SurfaceType.PLANE:
            # Planar end cap: normal should align with slot direction
            cap_normal = np.array(candidate.normal)
            dot = abs(np.dot(cap_normal, slot_direction))
            return dot > 0.9
        
        return False
    
    def _analyze_slot_walls(
        self,
        walls: List[int],
        bottom: GraphNode,
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> List[SlotWallAnalysis]:
        """Detailed analysis of slot walls"""
        analyses = []
        
        for i, wall_id in enumerate(walls):
            wall = nodes[wall_id]
            
            # Determine side (left or right)
            side = 'left' if i == 0 else 'right'
            
            # Check parallelism
            is_parallel = True  # Already validated in find_walls
            
            # Compute angle from vertical (for dovetail detection)
            angle = self._compute_wall_angle_from_vertical(wall)
            
            # Estimate wall dimensions
            wall_length = self._estimate_wall_length(wall, bottom)
            wall_width = self._estimate_distance_between_walls(walls, nodes)
            
            analysis = SlotWallAnalysis(
                face_id=wall_id,
                side=side,
                is_parallel_to_opposite=is_parallel,
                angle_from_vertical=angle,
                width=wall_width,
                length=wall_length
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def _compute_wall_angle_from_vertical(self, wall: GraphNode) -> float:
        """Compute angle of wall from vertical (for dovetail detection)"""
        normal = np.array(wall.normal)
        horizontal = np.array([normal[0], normal[1], 0])
        
        if np.linalg.norm(horizontal) < 1e-6:
            return 0.0
        
        horizontal = horizontal / np.linalg.norm(horizontal)
        
        # Angle between normal and horizontal plane
        dot = abs(np.dot(normal, horizontal))
        angle = 90.0 - np.degrees(np.arccos(np.clip(dot, 0, 1)))
        
        return angle
    
    def _estimate_wall_length(self, wall: GraphNode, bottom: GraphNode) -> float:
        """Estimate wall length (slot length)"""
        # Approximate from wall area and estimated height
        wall_z = wall.centroid[2]
        bottom_z = bottom.centroid[2]
        height = abs(wall_z - bottom_z)
        
        if height < 1e-6:
            return 0.0
        
        length = wall.area / height
        return length
    
    def _estimate_distance_between_walls(
        self,
        walls: List[int],
        nodes: List[GraphNode]
    ) -> float:
        """Estimate distance between two parallel walls (slot width)"""
        wall1 = nodes[walls[0]]
        wall2 = nodes[walls[1]]
        
        # Distance between centroids projected onto normal direction
        center1 = np.array(wall1.centroid)
        center2 = np.array(wall2.centroid)
        normal = np.array(wall1.normal)
        
        # Project distance onto normal
        diff = center2 - center1
        distance = abs(np.dot(diff, normal))
        
        return distance
    
    def _analyze_end_caps(
        self,
        end_caps: List[int],
        nodes: List[GraphNode]
    ) -> List[SlotEndCapAnalysis]:
        """Analyze end cap geometry"""
        analyses = []
        
        for cap_id in end_caps:
            cap = nodes[cap_id]
            
            if cap.surface_type == SurfaceType.CYLINDER:
                # Semicircular end
                analysis = SlotEndCapAnalysis(
                    type='semicircular',
                    face_id=cap_id,
                    radius=cap.radius,
                    diameter=cap.radius * 2,
                    location=cap.centroid
                )
            elif cap.surface_type == SurfaceType.PLANE:
                # Flat end
                analysis = SlotEndCapAnalysis(
                    type='flat',
                    face_id=cap_id,
                    location=cap.centroid
                )
            else:
                # Complex/unknown
                analysis = SlotEndCapAnalysis(
                    type='complex',
                    face_id=cap_id,
                    location=cap.centroid
                )
            
            analyses.append(analysis)
        
        return analyses
    
    def _recognize_through_slot(
        self,
        bottom: GraphNode,
        walls: List[int],
        end_caps: List[int],
        wall_analyses: List[SlotWallAnalysis],
        end_cap_analyses: List[SlotEndCapAnalysis],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> Optional[SlotFeature]:
        """
        Recognize through slot (open on both ends or no end caps)
        """
        # Compute dimensions
        width = wall_analyses[0].width if wall_analyses else 0.0
        length = wall_analyses[0].length if wall_analyses else 0.0
        depth = self._compute_slot_depth(bottom, walls, nodes)
        
        # Validate dimensions
        if not self._validate_slot_dimensions(width, length, depth):
            return None
        
        # Calculate volume
        volume = width * length * depth
        
        # Compute centerline axis
        centerline = self._compute_slot_centerline(bottom, walls, nodes)
        
        # Check if slot is curved
        is_curved = self._detect_curved_slot(bottom, walls, adjacency, nodes)
        
        # Confidence assessment
        confidence = 0.92
        if len(end_caps) == 2:
            confidence -= 0.02  # Slightly less certain if has end caps
        
        warnings = []
        if is_curved:
            warnings.append('Curved slot detected')
        if len(end_caps) > 0:
            warnings.append(f'{len(end_caps)} end cap(s) detected')
        
        return SlotFeature(
            type='through_slot',
            face_ids=[bottom.id] + walls + end_caps,
            bottom_face_id=bottom.id,
            wall_face_ids=walls,
            end_cap_ids=end_caps,
            width=width,
            length=length,
            depth=depth,
            end_cap_analysis=end_cap_analyses if end_cap_analyses else None,
            wall_analysis=wall_analyses,
            volume=volume,
            centerline_axis=centerline,
            is_curved=is_curved,
            confidence=confidence,
            warnings=warnings
        )
    
    def _recognize_blind_slot(
        self,
        bottom: GraphNode,
        walls: List[int],
        end_caps: List[int],
        wall_analyses: List[SlotWallAnalysis],
        end_cap_analyses: List[SlotEndCapAnalysis],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> Optional[SlotFeature]:
        """
        Recognize blind slot (closed on one end)
        """
        if len(end_caps) != 1:
            return None
        
        # Compute dimensions
        width = wall_analyses[0].width if wall_analyses else 0.0
        length = wall_analyses[0].length if wall_analyses else 0.0
        depth = self._compute_slot_depth(bottom, walls, nodes)
        
        if not self._validate_slot_dimensions(width, length, depth):
            return None
        
        # Volume
        volume = width * length * depth
        
        # Centerline
        centerline = self._compute_slot_centerline(bottom, walls, nodes)
        
        # Curved check
        is_curved = self._detect_curved_slot(bottom, walls, adjacency, nodes)
        
        confidence = 0.90
        
        warnings = []
        if is_curved:
            warnings.append('Curved blind slot')
        if end_cap_analyses[0].type == 'semicircular':
            warnings.append(f'Semicircular end cap, R={end_cap_analyses[0].radius:.3f}')
        
        return SlotFeature(
            type='blind_slot',
            face_ids=[bottom.id] + walls + end_caps,
            bottom_face_id=bottom.id,
            wall_face_ids=walls,
            end_cap_ids=end_caps,
            width=width,
            length=length,
            depth=depth,
            end_cap_analysis=end_cap_analyses,
            wall_analysis=wall_analyses,
            volume=volume,
            centerline_axis=centerline,
            is_curved=is_curved,
            confidence=confidence,
            warnings=warnings
        )
    
    def _recognize_t_slot(
        self,
        bottom: GraphNode,
        walls: List[int],
        end_caps: List[int],
        wall_analyses: List[SlotWallAnalysis],
        end_cap_analyses: List[SlotEndCapAnalysis],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> Optional[SlotFeature]:
        """
        Recognize T-slot (with undercut)
        
        T-slot pattern:
        - Narrow neck
        - Wider undercut below
        - Typically for bolt heads
        """
        # Look for undercut features
        undercut = self._detect_undercut(bottom, walls, adjacency, nodes)
        
        if not undercut:
            return None
        
        # Compute dimensions
        neck_width = wall_analyses[0].width if wall_analyses else 0.0
        slot_length = wall_analyses[0].length if wall_analyses else 0.0
        slot_depth = self._compute_slot_depth(bottom, walls, nodes)
        
        undercut_width = undercut['width']
        undercut_depth = undercut['depth']
        
        if not self._validate_slot_dimensions(neck_width, slot_length, slot_depth):
            return None
        
        # Volume calculation (neck + undercut)
        neck_volume = neck_width * slot_length * (slot_depth - undercut_depth)
        undercut_volume = undercut_width * slot_length * undercut_depth
        total_volume = neck_volume + undercut_volume
        
        centerline = self._compute_slot_centerline(bottom, walls, nodes)
        
        confidence = 0.88
        
        warnings = [
            f'T-slot with undercut: neck {neck_width:.1f}mm, undercut {undercut_width:.1f}mm'
        ]
        
        return SlotFeature(
            type='t_slot',
            face_ids=[bottom.id] + walls + end_caps + undercut['face_ids'],
            bottom_face_id=bottom.id,
            wall_face_ids=walls,
            end_cap_ids=end_caps,
            width=neck_width,
            length=slot_length,
            depth=slot_depth,
            undercut_depth=undercut_depth,
            undercut_width=undercut_width,
            neck_width=neck_width,
            end_cap_analysis=end_cap_analyses if end_cap_analyses else None,
            wall_analysis=wall_analyses,
            volume=total_volume,
            centerline_axis=centerline,
            confidence=confidence,
            warnings=warnings
        )
    
    def _recognize_dovetail_slot(
        self,
        bottom: GraphNode,
        walls: List[int],
        end_caps: List[int],
        wall_analyses: List[SlotWallAnalysis],
        end_cap_analyses: List[SlotEndCapAnalysis],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> Optional[SlotFeature]:
        """
        Recognize dovetail slot (angled walls)
        
        Dovetail pattern:
        - Walls angled inward or outward
        - Angle typically 5-30° from vertical
        - Used for sliding joints
        """
        # Check if walls have dovetail angles
        angles = [w.angle_from_vertical for w in wall_analyses]
        
        # Both walls should have similar angles
        if not all(self.dovetail_angle_min <= a <= self.dovetail_angle_max for a in angles):
            return None
        
        avg_angle = np.mean(angles)
        
        # Compute dimensions
        bottom_width = wall_analyses[0].width if wall_analyses else 0.0
        slot_length = wall_analyses[0].length if wall_analyses else 0.0
        slot_depth = self._compute_slot_depth(bottom, walls, nodes)
        
        # Top width (wider or narrower depending on dovetail direction)
        top_width = self._compute_dovetail_top_width(
            bottom_width, slot_depth, avg_angle
        )
        
        if not self._validate_slot_dimensions(bottom_width, slot_length, slot_depth):
            return None
        
        # Volume (trapezoidal cross-section)
        avg_width = (bottom_width + top_width) / 2
        volume = avg_width * slot_length * slot_depth
        
        centerline = self._compute_slot_centerline(bottom, walls, nodes)
        
        confidence = 0.86
        
        warnings = [
            f'Dovetail slot: {avg_angle:.1f}° angle',
            f'Bottom width: {bottom_width:.1f}mm, Top width: {top_width:.1f}mm'
        ]
        
        return SlotFeature(
            type='dovetail_slot',
            face_ids=[bottom.id] + walls + end_caps,
            bottom_face_id=bottom.id,
            wall_face_ids=walls,
            end_cap_ids=end_caps,
            width=bottom_width,
            length=slot_length,
            depth=slot_depth,
            dovetail_angle=avg_angle,
            top_width=top_width,
            bottom_width=bottom_width,
            end_cap_analysis=end_cap_analyses if end_cap_analyses else None,
            wall_analysis=wall_analyses,
            volume=volume,
            centerline_axis=centerline,
            confidence=confidence,
            warnings=warnings
        )
    
    def _compute_slot_depth(
        self,
        bottom: GraphNode,
        walls: List[int],
        nodes: List[GraphNode]
    ) -> float:
        """Compute slot depth"""
        bottom_z = bottom.centroid[2]
        
        # Average wall top Z
        wall_z_values = [nodes[w].centroid[2] for w in walls]
        avg_wall_z = np.mean(wall_z_values)
        
        depth = abs(avg_wall_z - bottom_z)
        return depth
    
    def _validate_slot_dimensions(
        self,
        width: float,
        length: float,
        depth: float
    ) -> bool:
        """Validate slot dimensions are in acceptable range"""
        # Width check
        if not (self.min_slot_width <= width <= self.max_slot_width):
            return False
        
        # Length check (must be longer than width)
        if length < self.min_slot_length or length < width:
            return False
        
        # Depth check
        if not (self.min_slot_depth <= depth <= self.max_slot_depth):
            return False
        
        return True
    
    def _compute_slot_centerline(
        self,
        bottom: GraphNode,
        walls: List[int],
        nodes: List[GraphNode]
    ) -> Tuple[float, float, float]:
        """Compute slot centerline direction"""
        wall1 = nodes[walls[0]]
        wall2 = nodes[walls[1]]
        
        # Centerline is perpendicular to wall normals
        normal1 = np.array(wall1.normal)
        
        # Slot runs perpendicular to wall normal
        centerline = np.array([-normal1[1], normal1[0], 0])
        
        if np.linalg.norm(centerline) > 1e-6:
            centerline = centerline / np.linalg.norm(centerline)
        
        return tuple(centerline)
    
    def _detect_curved_slot(
        self,
        bottom: GraphNode,
        walls: List[int],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> bool:
        """Detect if slot follows a curved path"""
        # Check if bottom face or walls are curved surfaces
        if bottom.surface_type != SurfaceType.PLANE:
            return True
        
        for wall_id in walls:
            wall = nodes[wall_id]
            if wall.surface_type != SurfaceType.PLANE:
                return True
        
        return False
    
    def _detect_undercut(
        self,
        bottom: GraphNode,
        walls: List[int],
        adjacency: Dict,
        nodes: List[GraphNode]
    ) -> Optional[Dict]:
        """
        Detect T-slot undercut feature
        
        Returns:
            Dictionary with undercut geometry or None
        """
        # Look for additional planar faces below slot bottom
        # with wider spacing than neck
        
        bottom_adjacent = adjacency[bottom.id]
        
        for adj in bottom_adjacent:
            adj_node = nodes[adj['node_id']]
            
            # Look for planar face below current bottom
            if adj_node.surface_type == SurfaceType.PLANE:
                if adj_node.centroid[2] < bottom.centroid[2] - self.tolerance:
                    # Found lower level
                    # Check if it has wider walls
                    lower_walls = self._find_slot_walls(adj_node, adjacency, nodes)
                    
                    if len(lower_walls) == 2:
                        lower_width = self._estimate_distance_between_walls(
                            lower_walls, nodes
                        )
                        upper_width = self._estimate_distance_between_walls(
                            walls, nodes
                        )
                        
                        if lower_width > upper_width * 1.2:  # 20% wider
                            # This is an undercut
                            undercut_depth = bottom.centroid[2] - adj_node.centroid[2]
                            
                            return {
                                'width': lower_width,
                                'depth': undercut_depth,
                                'face_ids': [adj_node.id] + lower_walls
                            }
        
        return None
    
    def _compute_dovetail_top_width(
        self,
        bottom_width: float,
        depth: float,
        angle: float
    ) -> float:
        """
        Compute top width of dovetail based on bottom width, depth, and angle
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Width change = 2 * depth * tan(angle)
        # (factor of 2 because both walls angle)
        width_change = 2 * depth * np.tan(angle_rad)
        
        # Dovetail can taper in or out
        # Assume taper out (wider at top)
        top_width = bottom_width + width_change
        
        return top_width
