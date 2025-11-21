"""
Recognizer API Adapters
========================

Provides standardized API wrappers for Slot/Fillet/Chamfer recognizers.

CRITICAL FIX (2025-11-21 v2):
- Removed unnecessary edge_lookup that was causing AttributeError
- Edges are dicts with 'from_node'/'to_node' keys, not attributes
- Directly iterates edges to build adjacency
"""

import logging
from typing import List, Dict, Any, Union
from dataclasses import asdict
from ..graph_builder import GraphNode, GraphEdge, SurfaceType, Vexity

logger = logging.getLogger(__name__)


def _build_enriched_adjacency(nodes: List[GraphNode], edges: List[Union[GraphEdge, Dict]]) -> Dict[int, List[Dict]]:
    """
    Build adjacency map enriched with vexity and angle data from edges.
    
    CRITICAL FIX: Uses face_id from nodes to match the GraphNode.face_id values
    
    Args:
        nodes: List of GraphNode objects with face_id attribute
        edges: List of GraphEdge objects OR edge dicts with vexity and dihedral_angle
        
    Returns:
        Adjacency dict: {face_id: [{node_id: X, face_id: X, vexity: 'convex', angle: 180.0}, ...]}
    """
    # Initialize adjacency using the ACTUAL face_id from nodes
    adjacency = {node.face_id: [] for node in nodes}
    
    logger.debug(f"_build_enriched_adjacency: Initialized adjacency for {len(nodes)} nodes")
    logger.debug(f"  Node face_id range: {min(n.face_id for n in nodes)} - {max(n.face_id for n in nodes)}")
    logger.debug(f"  Total edges to process: {len(edges)}")
    
    # Populate adjacency with vexity and angle from edges
    edges_processed = 0
    edges_skipped = 0
    
    for edge in edges:
        # Handle both GraphEdge objects and dicts
        if isinstance(edge, dict):
            from_node = edge.get('from_node')
            to_node = edge.get('to_node')
            vexity = edge.get('vexity', 'smooth')
            dihedral_angle = edge.get('dihedral_angle', 180.0)
        else:
            # GraphEdge object
            from_node = edge.from_node
            to_node = edge.to_node
            vexity = edge.vexity.value if hasattr(edge.vexity, 'value') else str(edge.vexity)
            dihedral_angle = edge.dihedral_angle
        
        # Convert vexity enum to string if needed
        if hasattr(vexity, 'value'):
            vexity_str = vexity.value
        else:
            vexity_str = str(vexity)
        
        # Skip if nodes don't exist in our adjacency map
        if from_node not in adjacency or to_node not in adjacency:
            edges_skipped += 1
            continue
        
        # Add forward edge
        adjacency[from_node].append({
            'node_id': to_node,
            'face_id': to_node,
            'vexity': vexity_str,
            'angle': dihedral_angle
        })
        
        # Add reverse edge (adjacency is bidirectional)
        adjacency[to_node].append({
            'node_id': from_node,
            'face_id': from_node,
            'vexity': vexity_str,
            'angle': dihedral_angle
        })
        
        edges_processed += 1
    
    # Log statistics
    logger.debug(f"  Edges processed: {edges_processed}")
    logger.debug(f"  Edges skipped (no matching nodes): {edges_skipped}")
    
    # Sample check
    non_empty = [fid for fid, neighbors in adjacency.items() if len(neighbors) > 0]
    logger.debug(f"  Non-empty adjacency entries: {len(non_empty)}/{len(adjacency)}")
    if non_empty:
        sample_id = non_empty[0]
        logger.debug(f"  Sample adjacency[{sample_id}] has {len(adjacency[sample_id])} neighbors")
        if adjacency[sample_id]:
            logger.debug(f"    First neighbor: {adjacency[sample_id][0]}")
    
    return adjacency


class StandardizedSlotRecognizer:
    """
    Standardized API wrapper for SlotRecognizer.
    
    Provides consistent API matching HoleRecognizer, PocketRecognizer, etc.
    """
    
    def __init__(self, aag_graph):
        """
        Args:
            aag_graph: AAGGraphBuilder instance
        """
        self.aag = aag_graph
        
        # Import the original recognizer
        try:
            from .slot_recognizer import SlotRecognizer
            self._recognizer = SlotRecognizer(tolerance=1e-6)
            self._available = True
        except ImportError as e:
            logger.warning(f"SlotRecognizer not available: {e}")
            self._available = False
            
    def recognize(self) -> List[Dict]:
        """
        Recognize slot features with standardized output.
        
        Returns:
            List of slot feature dicts with standardized format
        """
        if not self._available:
            logger.warning("SlotRecognizer not available - returning empty list")
            return []
            
        try:
            # Build graph dict from AAG builder
            graph = self._build_graph_dict()
            
            # Call original recognizer
            slot_features = self._recognizer.recognize_slots(graph)
            
            # Convert to standardized format
            standardized = []
            for slot in slot_features:
                std_slot = self._standardize_slot(slot)
                standardized.append(std_slot)
                
            logger.info(f"SlotRecognizer: {len(standardized)} slots recognized")
            return standardized
            
        except Exception as e:
            logger.error(f"Slot recognition failed: {e}", exc_info=True)
            return []
            
    def _build_graph_dict(self) -> Dict:
        """
        Build graph dict from AAGGraphBuilder instance.
        
        Returns:
            Graph dict compatible with SlotRecognizer.recognize_slots()
        """
        nodes = []
        edges = []
        
        # Get nodes - convert dict to GraphNode objects
        if hasattr(self.aag, 'nodes'):
            for face_id, node_data in self.aag.nodes.items():
                # Convert dict to typed GraphNode object
                graph_node = GraphNode(
                    face_id=node_data.get('face_id', face_id),
                    surface_type=SurfaceType(node_data['surface_type']),  # Convert string → enum
                    area=node_data['area'],
                    normal=tuple(node_data.get('normal', [0.0, 0.0, 1.0])),
                    center=tuple(node_data['center']) if 'center' in node_data else None,
                    radius=node_data.get('radius')
                )
                nodes.append(graph_node)
                
        # Get edges
        if hasattr(self.aag, 'edges'):
            edges = list(self.aag.edges)
        
        # CRITICAL FIX: Build enriched adjacency with vexity and angle data
        adjacency = _build_enriched_adjacency(nodes, edges)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'adjacency': adjacency,
            'metadata': {
                'up_axis': [0.0, 0.0, 1.0]
            }
        }
        
    def _standardize_slot(self, slot) -> Dict:
        """
        Convert SlotFeature to standardized dict format.
        
        Args:
            slot: SlotFeature dataclass or dict
            
        Returns:
            Standardized feature dict
        """
        # Convert dataclass to dict if needed
        if hasattr(slot, '__dataclass_fields__'):
            slot_dict = asdict(slot)
        elif isinstance(slot, dict):
            slot_dict = slot
        else:
            logger.warning(f"Unknown slot type: {type(slot)}")
            return {}
            
        # Build standardized output
        face_ids = slot_dict.get('face_ids', [])
        
        standardized = {
            'type': slot_dict.get('type', 'slot'),
            'face_ids': face_ids,
            'faceIds': face_ids,  # Dual format for Analysis Situs
            
            # Dimensions
            'width': slot_dict.get('width', 0.0),
            'length': slot_dict.get('length', 0.0),
            'depth': slot_dict.get('depth', 0.0),
            
            # Components
            'bottom_face_id': slot_dict.get('bottom_face_id', -1),
            'wall_face_ids': slot_dict.get('wall_face_ids', []),
            'end_cap_ids': slot_dict.get('end_cap_ids', []),
            
            # Metadata
            'confidence': slot_dict.get('confidence', 0.75),
            'fullyRecognized': True,
            'warnings': slot_dict.get('warnings', [])
        }
        
        # Add type-specific attributes
        if slot_dict.get('type') == 't_slot':
            standardized.update({
                'undercut_depth': slot_dict.get('undercut_depth'),
                'undercut_width': slot_dict.get('undercut_width'),
                'neck_width': slot_dict.get('neck_width')
            })
        elif slot_dict.get('type') == 'dovetail_slot':
            standardized.update({
                'dovetail_angle': slot_dict.get('dovetail_angle'),
                'top_width': slot_dict.get('top_width'),
                'bottom_width': slot_dict.get('bottom_width')
            })
            
        return standardized


class StandardizedFilletRecognizer:
    """
    Standardized API wrapper for FilletRecognizer.
    """
    
    def __init__(self, aag_graph):
        """
        Args:
            aag_graph: AAGGraphBuilder instance
        """
        self.aag = aag_graph
        
        # Import the original recognizer
        try:
            from .fillet_chamfer_recognizer import FilletRecognizer
            self._recognizer = FilletRecognizer(tolerance=1e-6)
            self._available = True
        except ImportError as e:
            logger.warning(f"FilletRecognizer not available: {e}")
            self._available = False
            
    def recognize(self) -> List[Dict]:
        """
        Recognize fillet features with standardized output.
        
        Returns:
            List of fillet feature dicts
        """
        if not self._available:
            return []
            
        try:
            # Build graph dict
            graph = self._build_graph_dict()
            
            # Check if recognizer has correct API
            if hasattr(self._recognizer, 'recognize_fillets'):
                fillet_features = self._recognizer.recognize_fillets(graph)
            elif hasattr(self._recognizer, 'recognize'):
                fillet_features = self._recognizer.recognize()
            else:
                logger.error("FilletRecognizer has no recognize method")
                return []
                
            # Convert to standardized format
            standardized = []
            for fillet in fillet_features:
                std_fillet = self._standardize_fillet(fillet)
                standardized.append(std_fillet)
                
            logger.info(f"FilletRecognizer: {len(standardized)} fillets recognized")
            return standardized
            
        except Exception as e:
            logger.error(f"Fillet recognition failed: {e}", exc_info=True)
            return []
            
    def _build_graph_dict(self) -> Dict:
        """
        Build graph dict from AAG builder with enriched adjacency.
        
        CRITICAL FIX: Adjacency now includes vexity and angle data from edges.
        """
        nodes = []
        edges = []
        
        if hasattr(self.aag, 'nodes'):
            for face_id, node_data in self.aag.nodes.items():
                # Convert dict to typed GraphNode object
                graph_node = GraphNode(
                    face_id=node_data.get('face_id', face_id),
                    surface_type=SurfaceType(node_data['surface_type']),
                    area=node_data['area'],
                    normal=tuple(node_data.get('normal', [0.0, 0.0, 1.0])),
                    center=tuple(node_data['center']) if 'center' in node_data else None,
                    radius=node_data.get('radius')
                )
                nodes.append(graph_node)
                
        if hasattr(self.aag, 'edges'):
            edges = list(self.aag.edges)
        
        # CRITICAL FIX: Build enriched adjacency with vexity and angle data
        adjacency = _build_enriched_adjacency(nodes, edges)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'adjacency': adjacency,
            'metadata': {}
        }
        
    def _standardize_fillet(self, fillet) -> Dict:
        """Convert FilletFeature to standardized dict."""
        # Convert dataclass to dict
        if hasattr(fillet, '__dataclass_fields__'):
            fillet_dict = asdict(fillet)
        elif isinstance(fillet, dict):
            fillet_dict = fillet
        else:
            return {}
            
        face_ids = fillet_dict.get('face_ids', [])
        
        standardized = {
            'type': str(fillet_dict.get('type', 'fillet')),
            'face_ids': face_ids,
            'faceIds': face_ids,
            
            # Geometry
            'radius': fillet_dict.get('radius'),
            'min_radius': fillet_dict.get('min_radius'),
            'max_radius': fillet_dict.get('max_radius'),
            
            # Metadata
            'confidence': fillet_dict.get('confidence', 0.70),
            'fullyRecognized': fillet_dict.get('is_continuous', True),
            'connected_faces': fillet_dict.get('connected_faces', []),
            'blend_count': fillet_dict.get('blend_count', 2),
            'warnings': fillet_dict.get('warnings', [])
        }
        
        return standardized


class StandardizedChamferRecognizer:
    """
    Standardized API wrapper for ChamferRecognizer.
    """
    
    def __init__(self, aag_graph):
        """
        Args:
            aag_graph: AAGGraphBuilder instance
        """
        self.aag = aag_graph
        
        # Import the original recognizer
        try:
            from .fillet_chamfer_recognizer import ChamferRecognizer
            self._recognizer = ChamferRecognizer(tolerance=1e-6)
            self._available = True
        except ImportError as e:
            logger.warning(f"ChamferRecognizer not available: {e}")
            self._available = False
            
    def recognize(self) -> List[Dict]:
        """
        Recognize chamfer features with standardized output.
        
        Returns:
            List of chamfer feature dicts
        """
        if not self._available:
            return []
            
        try:
            # Build graph dict
            graph = self._build_graph_dict()
            
            # Check API
            if hasattr(self._recognizer, 'recognize_chamfers'):
                chamfer_features = self._recognizer.recognize_chamfers(graph)
            elif hasattr(self._recognizer, 'recognize'):
                chamfer_features = self._recognizer.recognize()
            else:
                logger.error("ChamferRecognizer has no recognize method")
                return []
                
            # Convert to standardized format
            standardized = []
            for chamfer in chamfer_features:
                std_chamfer = self._standardize_chamfer(chamfer)
                standardized.append(std_chamfer)
                
            logger.info(f"ChamferRecognizer: {len(standardized)} chamfers recognized")
            return standardized
            
        except Exception as e:
            logger.error(f"Chamfer recognition failed: {e}", exc_info=True)
            return []
            
    def _build_graph_dict(self) -> Dict:
        """
        Build graph dict from AAG builder with enriched adjacency.
        """
        nodes = []
        edges = []
        
        if hasattr(self.aag, 'nodes'):
            for face_id, node_data in self.aag.nodes.items():
                # Convert dict to typed GraphNode object
                graph_node = GraphNode(
                    face_id=node_data.get('face_id', face_id),
                    surface_type=SurfaceType(node_data['surface_type']),
                    area=node_data['area'],
                    normal=tuple(node_data.get('normal', [0.0, 0.0, 1.0])),
                    center=tuple(node_data['center']) if 'center' in node_data else None,
                    radius=node_data.get('radius')
                )
                nodes.append(graph_node)
                
        if hasattr(self.aag, 'edges'):
            edges = list(self.aag.edges)
        
        # CRITICAL FIX: Build enriched adjacency with vexity and angle data
        adjacency = _build_enriched_adjacency(nodes, edges)
            
        return {
            'nodes': nodes,
            'edges': edges,
            'adjacency': adjacency,
            'metadata': {}
        }
        
    def _standardize_chamfer(self, chamfer) -> Dict:
        """Convert ChamferFeature to standardized dict."""
        # Convert dataclass to dict
        if hasattr(chamfer, '__dataclass_fields__'):
            chamfer_dict = asdict(chamfer)
        elif isinstance(chamfer, dict):
            chamfer_dict = chamfer
        else:
            return {}
            
        face_ids = chamfer_dict.get('face_ids', [])
        
        standardized = {
            'type': str(chamfer_dict.get('type', 'chamfer')),
            'face_ids': face_ids,
            'faceIds': face_ids,
            
            # Geometry
            'angle': chamfer_dict.get('angle', 45.0),
            'distance': chamfer_dict.get('distance'),
            'width': chamfer_dict.get('width'),
            
            # Metadata
            'confidence': chamfer_dict.get('confidence', 0.75),
            'fullyRecognized': True,
            'connected_faces': chamfer_dict.get('connected_faces', []),
            'warnings': chamfer_dict.get('warnings', [])
        }
        
        return standardized
