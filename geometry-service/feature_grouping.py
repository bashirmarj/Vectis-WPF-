# Feature Grouping Module
# Solves the face-to-feature clustering problem
# Converts face-level predictions to feature instances

import numpy as np
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)

# 16 MFCAD feature classes
FEATURE_CLASSES = [
    "hole", "boss", "pocket", "slot", "chamfer", "fillet",
    "groove", "step", "plane", "cylinder", "cone", "sphere",
    "torus", "bspline", "revolution", "extrusion"
]

# Geometric primitive types (typically indicate surface type)
GEOMETRIC_PRIMITIVES = {"plane", "cylinder", "cone", "sphere", "torus", "bspline"}

# Machining features (compound features made of multiple faces)
MACHINING_FEATURES = {"hole", "boss", "pocket", "slot", "chamfer", "fillet", "groove", "step", "revolution", "extrusion"}


class FeatureInstance:
    """Represents a single detected feature instance (hole, pocket, etc.)"""
    
    def __init__(self, instance_id: int, feature_type: str, face_ids: List[int], 
                 confidence: float, probabilities: Dict[str, float]):
        self.instance_id = instance_id
        self.feature_type = feature_type
        self.face_ids = face_ids  # Grouped faces that form this feature
        self.confidence = confidence  # Average confidence
        self.probabilities = probabilities
        self.geometric_primitive = None  # e.g., "cylinder" for a hole
        self.parameters = {}  # diameter, depth, etc.
        self.bounding_box = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable format"""
        return {
            "instance_id": self.instance_id,
            "feature_type": self.feature_type,
            "face_ids": self.face_ids,
            "confidence": round(self.confidence, 3),
            "geometric_primitive": self.geometric_primitive,
            "parameters": self.parameters,
            "bounding_box": self.bounding_box
        }


class FeatureGrouper:
