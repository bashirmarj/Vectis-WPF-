# COMPLETE feature_grouping.py - FULL PRODUCTION CODE
# Ready to copy and paste - no modifications needed

import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FeatureInstance:
    """Represents a single detected feature instance (e.g., one hole)"""
    instance_id: int
    feature_type: str
    face_ids: List[int]
    confidence: float
    geometric_primitive: str = None
    parameters: Dict = None
    bounding_box: Dict = None
    
    def to_dict(self):
        return asdict(self)

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_CLASSES = [
    'plane', 'cylinder', 'cone', 'sphere', 'torus',
    'hole', 'boss', 'pocket', 'slot', 'chamfer',
    'fillet', 'groove', 'step', 'blind_hole', 'through_hole',
    'boss_with_taper'
]

# Define which classes are manufacturing features vs primitives
MACHINING_FEATURES = {'hole', 'boss', 'pocket', 'slot', 'chamfer', 'fillet', 'groove', 'step', 'blind_hole', 'through_hole', 'boss_with_taper'}
GEOMETRIC_PRIMITIVES = {'plane', 'cylinder', 'cone', 'sphere', 'torus'}

# Compatibility rules: which face types can belong to same feature
COMPATIBLE_PAIRS = {
    ('cylinder', 'plane'): True,        # Hole: cylinder bore + bottom plane
    ('cylinder', 'cylinder'): True,     # Counterbore: inner + outer cylinders
    ('plane', 'plane'): True,           # Pocket: multiple planes
    ('cone', 'plane'): True,            # Chamfered hole
    ('torus', 'plane'): True,           # Filleted edge
    ('torus', 'cylinder'): True,        # Fillet on cylindrical bore
}

def are_compatible(type1: str, type2: str) -> bool:
    """Check if two face types can belong to same feature"""
    return COMPATIBLE_PAIRS.get((type1, type2), False) or COMPATIBLE_PAIRS.get((type2, type1), False)

# ============================================================================
# FEATURE GROUPING ENGINE
# ============================================================================

class FeatureGrouper:
    """
    Clusters adjacent faces with similar predictions into feature instances.
    Uses BFS graph traversal with geometric compatibility rules.
    """
    
    def __init__(self, face_predictions: List[Dict], face_adjacency_graph: nx.Graph):
        """
        Args:
            face_predictions: List of {face_id, predicted_class, confidence}
            face_adjacency_graph: NetworkX graph of face adjacencies
        """
        self.face_predictions = face_predictions
        self.face_graph = face_adjacency_graph
        self.visited = set()
        self.feature_instances = []
        self.instance_counter = 0
    
    def _cluster_feature(self, seed_face_id: int, feature_type: str, visited: Set[int]) -> 'FeatureInstance':
        """
        BFS-based clustering: start from seed face and group all adjacent faces
        with same/compatible predicted class.
        """
        
        # Initialize cluster
        cluster_faces = []
        queue = [seed_face_id]
        local_visited = {seed_face_id}
        
        # BFS expansion
        while queue:
            current_face = queue.pop(0)
            cluster_faces.append(current_face)
            
            # Get current face prediction
            current_pred = None
            for pred in self.face_predictions:
                if pred['face_id'] == current_face:
                    current_pred = pred
                    break
            
            if current_pred is None:
                continue
            
            # Check neighbors
            if current_face in self.face_graph:
                neighbors = list(self.face_graph.neighbors(current_face))
                
                for neighbor in neighbors:
                    if neighbor in visited or neighbor in local_visited:
                        continue
                    
                    # Get neighbor prediction
                    neighbor_pred = None
                    for pred in self.face_predictions:
                        if pred['face_id'] == neighbor:
                            neighbor_pred = pred
                            break
                    
                    if neighbor_pred is None:
                        continue
                    
                    # Check compatibility
                    neighbor_class = neighbor_pred['predicted_class']
                    current_class = current_pred['predicted_class']
                    
                    # Accept if same class OR compatible types
                    if (neighbor_class == current_class or 
                        are_compatible(current_class, neighbor_class)):
                        
                        local_visited.add(neighbor)
                        queue.append(neighbor)
        
        # Create feature instance
        confidences = []
        for face_id in cluster_faces:
            for pred in self.face_predictions:
                if pred['face_id'] == face_id:
                    confidences.append(pred['confidence'])
                    break
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine geometric primitive
        primitives = set()
        for face_id in cluster_faces:
            for pred in self.face_predictions:
                if pred['face_id'] == face_id:
                    pred_class = pred['predicted_class']
                    if pred_class in GEOMETRIC_PRIMITIVES:
                        primitives.add(pred_class)
                    break
        
        geometric_primitive = primitives.pop() if len(primitives) == 1 else None
        
        feature = FeatureInstance(
            instance_id=self.instance_counter,
            feature_type=feature_type,
            face_ids=sorted(cluster_faces),
            confidence=float(avg_confidence),
            geometric_primitive=geometric_primitive,
            parameters={},
            bounding_box=None
        )
        
        self.instance_counter += 1
        
        # Mark all faces as visited
        for face_id in cluster_faces:
            visited.add(face_id)
        
        return feature
    
    def group_features(self) -> Tuple[List[FeatureInstance], Dict]:
        """
        Main clustering algorithm:
        1. Prioritize machining features
        2. BFS cluster from each unvisited face
        3. Build feature summary
        """
        
        logger.info("🔄 Grouping faces into feature instances...")
        
        visited = set()
        feature_instances = []
        
        # PHASE 1: Prioritize machining features
        logger.info("  Phase 1: Processing machining features...")
        
        for pred in self.face_predictions:
            face_id = pred['face_id']
            feature_type = pred['predicted_class']
            
            if face_id in visited or feature_type not in MACHINING_FEATURES:
                continue
            
            feature = self._cluster_feature(face_id, feature_type, visited)
            feature_instances.append(feature)
            logger.debug(f"   Instance {feature.instance_id}: {feature_type} with {len(feature.face_ids)} faces (conf={feature.confidence:.2f})")
        
        # PHASE 2: Process remaining geometric primitives
        logger.info("  Phase 2: Processing geometric primitives...")
        
        for pred in self.face_predictions:
            face_id = pred['face_id']
            feature_type = pred['predicted_class']
            
            if face_id in visited:
                continue
            
            feature = self._cluster_feature(face_id, feature_type, visited)
            feature_instances.append(feature)
            logger.debug(f"   Instance {feature.instance_id}: {feature_type} with {len(feature.face_ids)} faces (conf={feature.confidence:.2f})")
        
        # Build summary
        feature_summary = {}
        for feature in feature_instances:
            ftype = feature.feature_type
            feature_summary[ftype] = feature_summary.get(ftype, 0) + 1
        
        feature_summary['total_features'] = len(feature_instances)
        
        logger.info(f"✅ Grouping complete: {len(feature_instances)} feature instances")
        logger.info(f"   Summary: {feature_summary}")
        
        return feature_instances, feature_summary

# ============================================================================
# PUBLIC API
# ============================================================================

def group_faces_to_features(face_predictions: List[Dict], face_adjacency_graph: nx.Graph) -> Dict:
    """
    Main entry point: Convert face-level predictions to feature instances.
    
    Args:
        face_predictions: List of {face_id, predicted_class, confidence}
        face_adjacency_graph: NetworkX graph of face adjacencies
    
    Returns:
        {
            'feature_instances': [FeatureInstance, ...],
            'feature_summary': {feature_type: count, ...},
            'num_features': int
        }
    """
    
    if not face_predictions or not face_adjacency_graph:
        logger.warning("Empty input to feature grouping")
        return {
            'feature_instances': [],
            'feature_summary': {},
            'num_features': 0
        }
    
    try:
        grouper = FeatureGrouper(face_predictions, face_adjacency_graph)
        instances, summary = grouper.group_features()
        
        return {
            'feature_instances': [inst.to_dict() for inst in instances],
            'feature_summary': summary,
            'num_features': len(instances)
        }
    
    except Exception as e:
        logger.error(f"❌ Feature grouping failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'feature_instances': [],
            'feature_summary': {},
            'num_features': 0,
            'error': str(e)
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_feature_parameters(feature_instance: Dict, shape) -> Dict:
    """
    Extract geometric parameters for a feature (diameter, depth, angle, etc.).
    """
    parameters = {}
    
    feature_type = feature_instance['feature_type']
    face_ids = feature_instance['face_ids']
    
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        import math
        
        # Get all faces
        all_faces = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            all_faces.append(topods.Face(face_explorer.Current()))
            face_explorer.Next()
        
        # Feature-specific parameter extraction
        if feature_type in ['hole', 'boss']:
            # Extract cylinder parameters (diameter, depth)
            for face_id in face_ids:
                if face_id >= len(all_faces):
                    continue
                
                try:
                    face = all_faces[face_id]
                    surf = BRepAdaptor_Surface(face)
                    
                    if surf.GetType() == GeomAbs_Cylinder:
                        cyl = surf.Cylinder()
                        diameter = cyl.Radius() * 2
                        
                        parameters['diameter_mm'] = round(diameter, 2)
                        
                        # Calculate depth from face area
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(face, props)
                        area = props.Mass()
                        
                        if area > 0:
                            depth = area / (math.pi * cyl.Radius())
                            parameters['depth_mm'] = round(depth, 2)
                except:
                    pass
        
        elif feature_type == 'chamfer':
            # Extract chamfer angle
            for face_id in face_ids:
                if face_id >= len(all_faces):
                    continue
                
                try:
                    face = all_faces[face_id]
                    surf = BRepAdaptor_Surface(face)
                    
                    if surf.GetType() == GeomAbs_Cone:
                        cone = surf.Cone()
                        angle_rad = cone.SemiAngle()
                        angle_deg = math.degrees(angle_rad)
                        
                        parameters['chamfer_angle_deg'] = round(angle_deg, 1)
                except:
                    pass
        
        return parameters
    
    except Exception as e:
        logger.debug(f"Could not extract parameters: {e}")
        return parameters
