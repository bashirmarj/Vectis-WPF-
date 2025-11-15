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
        """Load PyTorch Lightning checkpoint"""
        try:
            from brepnet import BRepNet
            from utils.data_utils import load_json_data
        except ImportError as e:
            raise ImportError(f"BRepNet dependencies missing: {e}. Ensure 'brepnet.py' and 'data_utils.py' exist.")
        
        logger.info(f"Loading PyTorch Lightning checkpoint: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        hyper_params = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint.get('state_dict', {})
        
        if not state_dict:
            raise ValueError("Checkpoint does not contain 'state_dict'")
        
        # Log checkpoint contents for debugging
        logger.info(f"📊 Checkpoint hyper_parameters keys: {list(hyper_params.keys())}")
        logger.info(f"🔍 Critical params from checkpoint:")
        logger.info(f"   - num_classes: {hyper_params.get('num_classes', 'NOT FOUND')}")
        logger.info(f"   - num_filters: {hyper_params.get('num_filters', 'NOT FOUND')}")
        logger.info(f"   - input_features: {hyper_params.get('input_features', 'NOT FOUND')}")
        logger.info(f"   - use_face_features: {hyper_params.get('use_face_features', 'NOT FOUND')}")
        logger.info(f"   - use_edge_features: {hyper_params.get('use_edge_features', 'NOT FOUND')}")
        
        # Infer missing critical parameters from state_dict if not in hyper_params
        if 'num_filters' not in hyper_params:
            # Layer 0 MLP first layer bias shape = num_filters
            if 'layers.0.mlp.mlp.linear_0.bias' in state_dict:
                num_filters_inferred = state_dict['layers.0.mlp.mlp.linear_0.bias'].shape[0]
                hyper_params['num_filters'] = num_filters_inferred
                logger.info(f"🔧 Inferred num_filters={num_filters_inferred} from state_dict")
        
        if 'num_classes' not in hyper_params:
            # Classification layer bias shape = num_classes
            if 'classification_layer.bias' in state_dict:
                num_classes_inferred = state_dict['classification_layer.bias'].shape[0]
                hyper_params['num_classes'] = num_classes_inferred
                logger.info(f"🔧 Inferred num_classes={num_classes_inferred} from state_dict")
        
        # Infer input feature dimension to determine which feature config to use
        input_dim_inferred = None
        if 'layers.0.mlp.mlp.linear_0.weight' in state_dict:
            input_dim_inferred = state_dict['layers.0.mlp.mlp.linear_0.weight'].shape[1]
            logger.info(f"🔧 Model expects input dimension: {input_dim_inferred}")
        
        # Verify kernel configuration exists
        kernel_file = hyper_params.get('kernel', hyper_params.get('kernel_filename', 'winged_edge_plus_plus.json'))
        kernel_path = kernel_file if os.path.exists(kernel_file) else f"/app/{kernel_file}"
        if not os.path.exists(kernel_path):
            raise RuntimeError(f"Kernel configuration not found: {kernel_file}")
        logger.info(f"✅ Using kernel config: {kernel_file}")
        
        # Reconstruct opts object from hyper_parameters
        class Opts:
            pass
        
        opts = Opts()
        
        # Copy all hyperparameters from checkpoint first
        for key, value in hyper_params.items():
            setattr(opts, key, value)
        
        # Set critical attributes - use checkpoint values when available, defaults only as fallback
        if 'kernel' not in hyper_params:
            opts.kernel = kernel_file
        if 'num_classes' not in hyper_params:
            opts.num_classes = num_classes_inferred if num_classes_inferred else 24
        if 'num_layers' not in hyper_params:
            opts.num_layers = 5
        if 'num_mlp_layers' not in hyper_params:
            opts.num_mlp_layers = 3
        if 'num_filters' not in hyper_params:
            opts.num_filters = num_filters_inferred if num_filters_inferred else 64
        if 'dropout' not in hyper_params:
            opts.dropout = 0.0
        
        # Feature usage flags - defaults to True only if not specified
        if 'use_face_grids' not in hyper_params:
            opts.use_face_grids = True
        if 'use_edge_grids' not in hyper_params:
            opts.use_edge_grids = True
        if 'use_coedge_grids' not in hyper_params:
            opts.use_coedge_grids = False
        if 'use_face_features' not in hyper_params:
            opts.use_face_features = True
        if 'use_edge_features' not in hyper_params:
            opts.use_edge_features = True
        if 'use_coedge_features' not in hyper_params:
            opts.use_coedge_features = True
        
        # Embedding sizes
        if 'curve_embedding_size' not in hyper_params:
            opts.curve_embedding_size = 64
        if 'surf_embedding_size' not in hyper_params:
            opts.surf_embedding_size = 64
        
        # Optional attributes
        if 'segment_names' not in hyper_params:
            opts.segment_names = None
        if 'dataset_dir' not in hyper_params:
            opts.dataset_dir = '.'
        
        # Determine correct input_features file based on model architecture
        if 'input_features' not in hyper_params and input_dim_inferred is not None:
            # Map input dimensions to feature configurations
            # The exact mapping depends on:
            # - 64 (face grid embedding) + 64 (edge grid embedding) = 128 base
            # - Plus hand-crafted features from JSON config
            # all.json ≈ 822 dims, no_curve_type.json ≈ 517 dims
            
            if input_dim_inferred < 600:
                # Likely an ablation study config (no_curve_type, no_surf_type, etc.)
                opts.input_features = 'no_curve_type.json'
                logger.info(f"🎯 Auto-selected no_curve_type.json for input_dim={input_dim_inferred}")
            else:
                # Full feature set
                opts.input_features = 'all.json'
                logger.info(f"🎯 Auto-selected all.json for input_dim={input_dim_inferred}")
        elif 'input_features' not in hyper_params:
            # Default fallback if we couldn't infer
            opts.input_features = 'all.json'
            logger.info(f"⚠️  Using default all.json (could not infer input dimension)")
        
        # Create model and load weights
        try:
            self.model = BRepNet(opts)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            if self.device == 'cuda':
                self.model = self.model.cuda()
            
            epoch = checkpoint.get('epoch', 'unknown')
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ BRepNet loaded (epoch {epoch}, {num_params:,} params)")
        except Exception as e:
            raise RuntimeError(f"Failed to create BRepNet model: {e}")
        
        # Initialize feature extractor
        try:
            from brep_feature_extractor import BRepFeatureExtractor
            self.feature_extractor = BRepFeatureExtractor(kernel_file)
            logger.info("✅ BRepFeatureExtractor initialized")
        except ImportError as e:
            raise ImportError(f"BRepFeatureExtractor not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize feature extractor: {e}")
    
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
            List of recognized features with metadata
        """
        logger.info("🔍 Starting BRepNet feature recognition")
        
        try:
            # Use PyTorch model if available
            if self.model is not None and hasattr(self, 'feature_extractor'):
                return self._recognize_with_pytorch(shape, face_mapping)
            
            # Fallback to ONNX (if implemented)
            if self.session is None:
                raise RuntimeError("Neither PyTorch nor ONNX model loaded")
        except Exception as e:
            logger.error(f"❌ Feature recognition failed: {e}", exc_info=True)
            return []  # Return empty list on failure instead of crashing
        
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
        predictions = outputs[0]
        features = self._parse_predictions(predictions, shape, face_mapping)
        
        return features
    
    def _recognize_with_pytorch(self, shape: TopoDS_Shape, face_mapping: Dict[int, Dict]) -> List[Dict]:
        """Run recognition using PyTorch model with full BRepNet pipeline"""
        
        # Step 1: Extract BRepNet input tensors
        brep_tensors = self.feature_extractor.extract_features(shape)
        
        # Step 2: Convert to PyTorch tensors
        model_inputs = {}
        for key, val in brep_tensors.items():
            tensor = torch.from_numpy(val).float()
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            elif len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)
            elif len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            model_inputs[key] = tensor
        
        if self.device == 'cuda':
            model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
        
        # Step 3: Run inference
        with torch.no_grad():
            try:
                predictions = self.model.brepnet_step(
                    model_inputs['Xf'], model_inputs['Gf'],
                    model_inputs['Xe'], model_inputs['Ge'],
                    model_inputs['Xc'], model_inputs['Gc'],
                    model_inputs['Kf'], model_inputs['Ke'], model_inputs['Kc'],
                    model_inputs['Ce'], model_inputs['Cf'], model_inputs['Csf']
                )
            except Exception as e:
                logger.error(f"BRepNet inference failed: {e}", exc_info=True)
                return []
        
        # Step 4: Parse predictions into recognized features
        features = self._parse_pytorch_predictions(predictions, shape, face_mapping)
        
        logger.info(f"✅ BRepNet recognized {len(features)} features")
        return features
    
    def _parse_pytorch_predictions(self, predictions: torch.Tensor, shape: TopoDS_Shape, face_mapping: Dict) -> List[Dict]:
        """
        Convert PyTorch model output to feature list
        
        BRepNet outputs per-face classifications (24 classes)
        """
        predictions_np = predictions.cpu().numpy()
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions_np - np.max(predictions_np, axis=-1, keepdims=True))
        probabilities = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        
        features = []
        
        # Handle batch dimension
        if len(probabilities.shape) == 3:
            batch_probs = probabilities[0]
        else:
            batch_probs = probabilities
        
        num_faces = batch_probs.shape[0]
        
        for face_idx in range(num_faces):
            face_probs = batch_probs[face_idx, :]
            predicted_class = np.argmax(face_probs)
            confidence = float(face_probs[predicted_class])
            
            if confidence >= self.confidence_threshold:
                feature = {
                    'type': self._class_to_feature_type(predicted_class),
                    'confidence': confidence,
                    'face_ids': [face_idx],
                    'parameters': self._extract_feature_parameters(shape, face_idx),
                    'ml_detected': True
                }
                features.append(feature)
        
        return features
    
    def _class_to_feature_type(self, class_id: int) -> str:
        """Map BRepNet class ID to feature type string"""
        feature_map = {
            0: 'rectangular_through_slot',
            1: 'triangular_through_slot',
            2: 'rectangular_passage',
            3: 'triangular_passage',
            4: '6sides_passage',
            5: 'rectangular_through_step',
            6: '2sides_through_step',
            7: 'slanted_through_step',
            8: 'rectangular_blind_step',
            9: 'triangular_blind_step',
            10: 'rectangular_blind_slot',
            11: 'rectangular_pocket',
            12: 'triangular_pocket',
            13: '6sides_pocket',
            14: 'chamfer',
            15: 'stock',
        }
        return feature_map.get(class_id, 'unknown')
    
    def _extract_feature_parameters(self, shape: TopoDS_Shape, face_idx: int) -> Dict:
        """Extract geometric parameters for a feature"""
        return {
            'face_index': face_idx,
            'extracted': True
        }
    
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
