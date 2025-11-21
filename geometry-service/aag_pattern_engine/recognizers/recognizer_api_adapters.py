"""
Recognizer API Adapters - FINAL FIX
====================================

CRITICAL FIX (2025-11-21 v3):
- AAG already builds adjacency with vexity - use it directly!
- Don't rebuild from edges (face ID mismatch after decomposition)
- Only check if adjacency exists and use it
"""

import logging
from typing import List, Dict, Any
from dataclasses import asdict
from ..graph_builder import GraphNode, SurfaceType

logger = logging.getLogger(__name__)


class StandardizedSlotRecognizer:
    def __init__(self, aag_graph):
        self.aag = aag_graph
        try:
            from .slot_recognizer import SlotRecognizer
            self._recognizer = SlotRecognizer(tolerance=1e-6)
            self._available = True
        except ImportError as e:
            logger.warning(f"SlotRecognizer not available: {e}")
            self._available = False
            
    def recognize(self) -> List[Dict]:
        if not self._available:
            return []
        try:
            graph = self._build_graph_dict()
            slot_features = self._recognizer.recognize_slots(graph)
            standardized = [self._standardize_slot(s) for s in slot_features]
            logger.info(f"SlotRecognizer: {len(standardized)} slots recognized")
            return standardized
        except Exception as e:
            logger.error(f"Slot recognition failed: {e}", exc_info=True)
            return []
            
    def _build_graph_dict(self) -> Dict:
        nodes = []
        
        # Build GraphNode objects
        if hasattr(self.aag, 'nodes'):
            for face_id, node_data in self.aag.nodes.items():
                graph_node = GraphNode(
                    face_id=node_data.get('face_id', face_id),
                    surface_type=SurfaceType(node_data['surface_type']),
                    area=node_data['area'],
                    normal=tuple(node_data.get('normal', [0.0, 0.0, 1.0])),
                    center=tuple(node_data['center']) if 'center' in node_data else None,
                    radius=node_data.get('radius')
                )
                nodes.append(graph_node)
        
        # Use pre-built adjacency from AAG (already has vexity!)
        adjacency = dict(self.aag.adjacency) if hasattr(self.aag, 'adjacency') else {}
        
        return {
            'nodes': nodes,
            'edges': list(self.aag.edges) if hasattr(self.aag, 'edges') else [],
            'adjacency': adjacency,
            'metadata': {'up_axis': [0.0, 0.0, 1.0]}
        }
        
    def _standardize_slot(self, slot) -> Dict:
        if hasattr(slot, '__dataclass_fields__'):
            slot_dict = asdict(slot)
        elif isinstance(slot, dict):
            slot_dict = slot
        else:
            return {}
            
        face_ids = slot_dict.get('face_ids', [])
        return {
            'type': slot_dict.get('type', 'slot'),
            'face_ids': face_ids,
            'faceIds': face_ids,
            'width': slot_dict.get('width', 0.0),
            'length': slot_dict.get('length', 0.0),
            'depth': slot_dict.get('depth', 0.0),
            'bottom_face_id': slot_dict.get('bottom_face_id', -1),
            'wall_face_ids': slot_dict.get('wall_face_ids', []),
            'end_cap_ids': slot_dict.get('end_cap_ids', []),
            'confidence': slot_dict.get('confidence', 0.75),
            'fullyRecognized': True,
            'warnings': slot_dict.get('warnings', [])
        }


class StandardizedFilletRecognizer:
    def __init__(self, aag_graph):
        self.aag = aag_graph
        try:
            from .fillet_chamfer_recognizer import FilletRecognizer
            self._recognizer = FilletRecognizer(tolerance=1e-6)
            self._available = True
        except ImportError as e:
            logger.warning(f"FilletRecognizer not available: {e}")
            self._available = False
            
    def recognize(self) -> List[Dict]:
        if not self._available:
            return []
        try:
            graph = self._build_graph_dict()
            
            if hasattr(self._recognizer, 'recognize_fillets'):
                fillet_features = self._recognizer.recognize_fillets(graph)
            elif hasattr(self._recognizer, 'recognize'):
                fillet_features = self._recognizer.recognize()
            else:
                logger.error("FilletRecognizer has no recognize method")
                return []
                
            standardized = [self._standardize_fillet(f) for f in fillet_features]
            logger.info(f"FilletRecognizer: {len(standardized)} fillets recognized")
            return standardized
        except Exception as e:
            logger.error(f"Fillet recognition failed: {e}", exc_info=True)
            return []
            
    def _build_graph_dict(self) -> Dict:
        nodes = []
        
        if hasattr(self.aag, 'nodes'):
            for face_id, node_data in self.aag.nodes.items():
                graph_node = GraphNode(
                    face_id=node_data.get('face_id', face_id),
                    surface_type=SurfaceType(node_data['surface_type']),
                    area=node_data['area'],
                    normal=tuple(node_data.get('normal', [0.0, 0.0, 1.0])),
                    center=tuple(node_data['center']) if 'center' in node_data else None,
                    radius=node_data.get('radius')
                )
                nodes.append(graph_node)
        
        # Use pre-built adjacency (CRITICAL: already has vexity!)
        adjacency = dict(self.aag.adjacency) if hasattr(self.aag, 'adjacency') else {}
        
        return {
            'nodes': nodes,
            'edges': list(self.aag.edges) if hasattr(self.aag, 'edges') else [],
            'adjacency': adjacency,
            'metadata': {}
        }
        
    def _standardize_fillet(self, fillet) -> Dict:
        if hasattr(fillet, '__dataclass_fields__'):
            fillet_dict = asdict(fillet)
        elif isinstance(fillet, dict):
            fillet_dict = fillet
        else:
            return {}
            
        face_ids = fillet_dict.get('face_ids', [])
        return {
            'type': str(fillet_dict.get('type', 'fillet')),
            'face_ids': face_ids,
            'faceIds': face_ids,
            'radius': fillet_dict.get('radius'),
            'min_radius': fillet_dict.get('min_radius'),
            'max_radius': fillet_dict.get('max_radius'),
            'confidence': fillet_dict.get('confidence', 0.70),
            'fullyRecognized': fillet_dict.get('is_continuous', True),
            'connected_faces': fillet_dict.get('connected_faces', []),
            'blend_count': fillet_dict.get('blend_count', 2),
            'warnings': fillet_dict.get('warnings', [])
        }


class StandardizedChamferRecognizer:
    def __init__(self, aag_graph):
        self.aag = aag_graph
        try:
            from .fillet_chamfer_recognizer import ChamferRecognizer
            self._recognizer = ChamferRecognizer(tolerance=1e-6)
            self._available = True
        except ImportError as e:
            logger.warning(f"ChamferRecognizer not available: {e}")
            self._available = False
            
    def recognize(self) -> List[Dict]:
        if not self._available:
            return []
        try:
            graph = self._build_graph_dict()
            
            if hasattr(self._recognizer, 'recognize_chamfers'):
                chamfer_features = self._recognizer.recognize_chamfers(graph)
            elif hasattr(self._recognizer, 'recognize'):
                chamfer_features = self._recognizer.recognize()
            else:
                logger.error("ChamferRecognizer has no recognize method")
                return []
                
            standardized = [self._standardize_chamfer(c) for c in chamfer_features]
            logger.info(f"ChamferRecognizer: {len(standardized)} chamfers recognized")
            return standardized
        except Exception as e:
            logger.error(f"Chamfer recognition failed: {e}", exc_info=True)
            return []
            
    def _build_graph_dict(self) -> Dict:
        nodes = []
        
        if hasattr(self.aag, 'nodes'):
            for face_id, node_data in self.aag.nodes.items():
                graph_node = GraphNode(
                    face_id=node_data.get('face_id', face_id),
                    surface_type=SurfaceType(node_data['surface_type']),
                    area=node_data['area'],
                    normal=tuple(node_data.get('normal', [0.0, 0.0, 1.0])),
                    center=tuple(node_data['center']) if 'center' in node_data else None,
                    radius=node_data.get('radius')
                )
                nodes.append(graph_node)
        
        # Use pre-built adjacency
        adjacency = dict(self.aag.adjacency) if hasattr(self.aag, 'adjacency') else {}
            
        return {
            'nodes': nodes,
            'edges': list(self.aag.edges) if hasattr(self.aag, 'edges') else [],
            'adjacency': adjacency,
            'metadata': {}
        }
        
    def _standardize_chamfer(self, chamfer) -> Dict:
        if hasattr(chamfer, '__dataclass_fields__'):
            chamfer_dict = asdict(chamfer)
        elif isinstance(chamfer, dict):
            chamfer_dict = chamfer
        else:
            return {}
            
        face_ids = chamfer_dict.get('face_ids', [])
        return {
            'type': str(chamfer_dict.get('type', 'chamfer')),
            'face_ids': face_ids,
            'faceIds': face_ids,
            'angle': chamfer_dict.get('angle', 45.0),
            'distance': chamfer_dict.get('distance'),
            'width': chamfer_dict.get('width'),
            'confidence': chamfer_dict.get('confidence', 0.75),
            'fullyRecognized': True,
            'connected_faces': chamfer_dict.get('connected_faces', []),
            'warnings': chamfer_dict.get('warnings', [])
        }
