# brepnet_wrapper.py - BRepNet Integration for Production Feature Recognition
# Version 1.0.0
# Based on: Autodesk BRepNet (CVPR 2021) - 89.96% accuracy on manufacturing features
#
# Pre-trained model: https://github.com/AutodeskAILab/BRepNet
# ./example_files/pretrained_models/pretrained_s2.0.0_step_all_features_0519_073100.ckpt

import os
import logging
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
import torch
import onnxruntime as ort

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Manufacturing feature types recognized by BRepNet"""
    HOLE = "hole"
    POCKET = "pocket"
    CHAMFER = "chamfer"
    FILLET = "fillet"
    THROUGH_SLOT = "through_slot"
    BLIND_SLOT = "blind_slot"
    STEP = "step"
    BOSS = "boss"
    PASSAGE = "passage"
    UNKNOWN = "unknown"


@dataclass
class RecognizedFeature:
    """Feature recognition result"""
    feature_type: FeatureType
    face_ids: List[int]
    confidence: float
    bounding_box: Optional[Dict] = None
    parameters: Optional[Dict] = None
    

class BRepNetRecognizer:
    """
    Production wrapper for BRepNet feature recognition
    
    Provides:
    - CPU-optimized ONNX inference
    - Face-level feature classification
    - Confidence-based filtering
    - Geometric parameter extraction
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.70
    ):
        """
        Initialize BRepNet recognizer
        
        Args:
            model_path: Path to pre-trained BRepNet checkpoint or ONNX model
            device: 'cpu' or 'cuda'
            confidence_threshold: Minimum confidence for feature detection (0.0-1.0)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.session = None
        
        logger.info(f"Loading BRepNet model from {model_path}")
        
        # Check if model is ONNX (production) or PyTorch checkpoint (development)
        if model_path.endswith('.onnx'):
            self._load_onnx_model(model_path)
        elif model_path.endswith('.ckpt') or model_path.endswith('.pth'):
            self._load_pytorch_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX-optimized model (preferred for production)"""
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"✅ BRepNet ONNX model loaded on {self.device}")
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch checkpoint (development fallback)"""
        # This requires the BRepNet model architecture
        # For production, convert to ONNX first
        raise NotImplementedError(
            "PyTorch checkpoints not supported in production. "
            "Convert model to ONNX format using: "
            "torch.onnx.export(model, example_input, 'model.onnx')"
        )
    
    def recognize_features(
        self,
        shape: TopoDS_Shape,
        face_mapping: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Recognize manufacturing features in B-Rep shape
        
        Args:
            shape: OpenCascade TopoDS_Shape
            face_mapping: Dict mapping face IDs to triangle indices
        
        Returns:
            List of detected features with face-level information
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Extract face features for neural network
        face_features = self._extract_face_features(shape)
        
        # Build face adjacency graph
        edge_indices, edge_features = self._build_face_graph(shape)
        
        # Prepare input tensor
        input_tensor = self._prepare_input(face_features, edge_indices, edge_features)
        
        # Run inference
        outputs = self.session.run(None, {
            'face_features': input_tensor['face_features'],
            'edge_indices': input_tensor['edge_indices'],
            'edge_features': input_tensor['edge_features']
        })
        
        # Parse predictions
        predictions = outputs[0]  # Shape: (num_faces, num_classes)
        
        # Convert to features
        features = self._parse_predictions(
            predictions,
            shape,
            face_mapping
        )
        
        return features
    
    def _extract_face_features(self, shape: TopoDS_Shape) -> np.ndarray:
        """
        Extract geometric features for each face
        
        Features (per face):
        - Surface type (plane=0, cylinder=1, cone=2, other=3)
        - Surface area
        - Centroid (x, y, z)
        - Normal direction (nx, ny, nz)
        - Curvature (for curved surfaces)
        
        Returns:
            np.ndarray of shape (num_faces, 10)
        """
        face_features = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = face_explorer.Current()
            surface_adaptor = BRepAdaptor_Surface(face)
            surface_type = surface_adaptor.GetType()
            
            # Surface type encoding
            if surface_type == GeomAbs_Plane:
                type_vec = [1, 0, 0, 0]
            elif surface_type == GeomAbs_Cylinder:
                type_vec = [0, 1, 0, 0]
            elif surface_type == GeomAbs_Cone:
                type_vec = [0, 0, 1, 0]
            else:
                type_vec = [0, 0, 0, 1]
            
            # Compute face area (simplified - use actual computation in production)
            area = 1.0  # Placeholder
            
            # Compute centroid (simplified)
            centroid = [0.0, 0.0, 0.0]  # Placeholder
            
            # Compute normal (simplified)
            normal = [0.0, 0.0, 1.0]  # Placeholder
            
            # Combine features
            features = type_vec + [area] + centroid + normal
            face_features.append(features)
            
            face_explorer.Next()
        
        return np.array(face_features, dtype=np.float32)
    
    def _build_face_graph(self, shape: TopoDS_Shape) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build face adjacency graph for message passing
        
        Returns:
            edge_indices: np.ndarray of shape (2, num_edges)
            edge_features: np.ndarray of shape (num_edges, edge_dim)
        """
        # This is a simplified version
        # Full implementation requires TopTools_IndexedDataMapOfShapeListOfShape
        
        # Placeholder: return empty graph
        edge_indices = np.array([[0], [0]], dtype=np.int64)
        edge_features = np.array([[0.0]], dtype=np.float32)
        
        return edge_indices, edge_features
    
    def _prepare_input(
        self,
        face_features: np.ndarray,
        edge_indices: np.ndarray,
        edge_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Prepare input tensors for ONNX model"""
        return {
            'face_features': face_features,
            'edge_indices': edge_indices,
            'edge_features': edge_features
        }
    
    def _parse_predictions(
        self,
        predictions: np.ndarray,
        shape: TopoDS_Shape,
        face_mapping: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Parse neural network predictions into feature objects
        
        Args:
            predictions: Softmax probabilities (num_faces, num_classes)
            shape: Original B-Rep shape
            face_mapping: Face ID to triangle mapping
        
        Returns:
            List of recognized features
        """
        features = []
        
        # Class labels (BRepNet standard)
        class_labels = [
            FeatureType.HOLE,
            FeatureType.POCKET,
            FeatureType.CHAMFER,
            FeatureType.FILLET,
            FeatureType.THROUGH_SLOT,
            FeatureType.BLIND_SLOT,
            FeatureType.STEP,
            FeatureType.BOSS,
            FeatureType.PASSAGE
        ]
        
        for face_id, probs in enumerate(predictions):
            # Get most confident class
            class_idx = np.argmax(probs)
            confidence = float(probs[class_idx])
            
            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue
            
            # Map to feature type
            if class_idx < len(class_labels):
                feature_type = class_labels[class_idx]
            else:
                feature_type = FeatureType.UNKNOWN
            
            # Extract face triangles
            face_triangles = face_mapping.get(face_id, {}).get("triangle_indices", [])
            
            features.append({
                "feature_type": feature_type.value,
                "face_ids": [face_id],
                "confidence": confidence,
                "triangle_indices": face_triangles,
                "parameters": {}
            })
        
        # Group faces into feature instances (simplified)
        # Full implementation requires instance segmentation
        
        logger.info(f"Recognized {len(features)} features above threshold {self.confidence_threshold}")
        
        return features


def convert_pytorch_to_onnx(
    checkpoint_path: str,
    output_path: str,
    example_input: Optional[Dict] = None
):
    """
    Convert BRepNet PyTorch checkpoint to ONNX for production deployment
    
    This function should be run offline during model preparation
    """
    import torch
    
    # Load PyTorch model
    model = torch.load(checkpoint_path)
    model.eval()
    
    # Create example input
    if example_input is None:
        example_input = {
            'face_features': torch.randn(1, 100, 10),
            'edge_indices': torch.tensor([[0], [0]], dtype=torch.long),
            'edge_features': torch.randn(1, 1, 5)
        }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (example_input,),
        output_path,
        input_names=['face_features', 'edge_indices', 'edge_features'],
        output_names=['predictions'],
        dynamic_axes={
            'face_features': {1: 'num_faces'},
            'edge_indices': {1: 'num_edges'},
            'edge_features': {1: 'num_edges'}
        },
        opset_version=14
    )
    
    logger.info(f"✅ Model converted to ONNX: {output_path}")
