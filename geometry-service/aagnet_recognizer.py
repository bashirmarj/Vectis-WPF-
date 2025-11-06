# -*- coding: utf-8 -*-
"""
AAGNet Feature Recognition Service for Vectis Machining
Integrates AAGNet graph neural network for multi-task machining feature recognition
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import numpy as np
import torch
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

# Add AAGNet modules to path
AAGNET_PATH = Path(__file__).parent / 'AAGNet-main'
sys.path.insert(0, str(AAGNET_PATH))

from dataset.AAGExtractor import AAGExtractor
from dataset.topologyCheker import TopologyChecker
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureInstance:
    """Container for recognized machining feature instance"""
    def __init__(self, name: str, faces: list, bottoms: list, confidence: float = 0.0):
        self.name = name
        self.faces = faces
        self.bottoms = bottoms
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            'type': self.name,
            'face_indices': self.faces,
            'bottom_faces': self.bottoms,
            'confidence': self.confidence
        }


class AAGNetRecognizer:
    """
    AAGNet-based machining feature recognizer with multi-task learning
    Supports semantic segmentation, instance segmentation, and bottom face detection
    """
    
    # 24 machining feature classes + stock
    FEATURE_NAMES = [
        'chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', 
        '6sides_passage', 'triangular_through_slot', 'rectangular_through_slot', 
        'circular_through_slot', 'rectangular_through_step', '2sides_through_step', 
        'slanted_through_step', 'Oring', 'blind_hole', 'triangular_pocket', 
        'rectangular_pocket', '6sides_pocket', 'circular_end_pocket', 
        'rectangular_blind_slot', 'v_circular_end_blind_slot', 
        'h_circular_end_blind_slot', 'triangular_blind_step', 'circular_blind_step', 
        'rectangular_blind_step', 'round', 'stock'
    ]
    
    def __init__(self, device: str = 'cpu', model_weights_path: Optional[str] = None):
        """
        Initialize AAGNet recognizer
        
        Args:
            device: 'cpu' or 'cuda' for GPU inference
            model_weights_path: Path to pretrained model weights
        """
        self.device = device
        self.eps = 1e-6
        self.inst_thres = 0.5  # Instance segmentation threshold
        self.bottom_thres = 0.5  # Bottom face classification threshold
        self.center_and_scale = False
        self.normalize = True
        
        # Model configuration (optimized for production)
        self.config = {
            "edge_attr_dim": 12,
            "node_attr_dim": 10,
            "edge_attr_emb": 64,
            "node_attr_emb": 64,
            "edge_grid_dim": 0,
            "node_grid_dim": 7,
            "edge_grid_emb": 0,
            "node_grid_emb": 64,
            "num_layers": 3,
            "delta": 2,
            "mlp_ratio": 2,
            "drop": 0.25,
            "drop_path": 0.25,
            "head_hidden_dim": 64,
            "conv_on_edge": False,
            "use_uv_gird": True,
            "use_edge_attr": True,
            "use_face_attr": True,
            "architecture": "AAGNetGraphEncoder",
        }
        
        self.n_classes = 25  # 24 features + stock
        self.topoChecker = TopologyChecker()
        
        # Load attribute schema and statistics
        self.weight_path = model_weights_path or str(AAGNET_PATH / 'weights' / 'weight_on_MFInstseg.pth')
        self.attribute_schema = load_json_or_pkl(str(AAGNET_PATH / 'feature_lists' / 'all.json'))
        self.stat = load_statistics(str(AAGNET_PATH / 'weights' / 'attr_stat.json'))
        
        # Initialize model
        self._init_model()
        logger.info(f"AAGNet recognizer initialized on {device}")

    def _init_model(self):
        """Initialize AAGNet model architecture"""
        self.recognizer = AAGNetSegmentor(
            num_classes=self.n_classes,
            arch=self.config['architecture'],
            edge_attr_dim=self.config['edge_attr_dim'],
            node_attr_dim=self.config['node_attr_dim'],
            edge_attr_emb=self.config['edge_attr_emb'],
            node_attr_emb=self.config['node_attr_emb'],
            edge_grid_dim=self.config['edge_grid_dim'],
            node_grid_dim=self.config['node_grid_dim'],
            edge_grid_emb=self.config['edge_grid_emb'],
            node_grid_emb=self.config['node_grid_emb'],
            num_layers=self.config['num_layers'],
            delta=self.config['delta'],
            mlp_ratio=self.config['mlp_ratio'],
            drop=self.config['drop'],
            drop_path=self.config['drop_path'],
            head_hidden_dim=self.config['head_hidden_dim'],
            conv_on_edge=self.config['conv_on_edge'],
            use_uv_gird=self.config['use_uv_gird'],
            use_edge_attr=self.config['use_edge_attr'],
            use_face_attr=self.config['use_face_attr'],
        )
        
        # Load pretrained weights
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path, map_location=self.device)
            self.recognizer.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded model weights from {self.weight_path}")
        else:
            logger.warning(f"Model weights not found at {self.weight_path}, using random initialization")
        
        self.recognizer = self.recognizer.to(self.device)
        self.recognizer.eval()

    def recognize_features(self, step_file_path: str) -> Dict:
        """
        Recognize machining features from STEP file
        
        Args:
            step_file_path: Path to STEP file
            
        Returns:
            Dictionary containing:
            - instances: List of feature instances with face assignments
            - semantic_labels: Per-face feature type predictions
            - extended_attributes: Face/edge geometric attributes
            - success: Boolean indicating success
            - error: Error message if failed
        """
        start_time = time.time()
        correlation_id = f"aagnet_{int(time.time() * 1000)}"
        
        logger.info(f"[{correlation_id}] Starting AAGNet feature recognition for {step_file_path}")
        
        try:
            # Step 1: Extract gAAG (geometric Attributed Adjacency Graph)
            logger.info(f"[{correlation_id}] Extracting gAAG from STEP file")
            aag_extractor = AAGExtractor(step_file_path, self.attribute_schema)
            aag_data = aag_extractor.process()
            
            # Step 2: Convert to DGL graph tensor
            logger.info(f"[{correlation_id}] Converting gAAG to tensor format")
            sample = load_one_graph(step_file_path, aag_data)
            
            if self.normalize:
                sample = standardization(sample, self.stat)
            
            graph = sample["graph"].to(self.device)
            
            # Step 3: Run inference
            logger.info(f"[{correlation_id}] Running AAGNet inference")
            with torch.no_grad():
                seg_out, inst_out, bottom_out = self.recognizer(graph)
            
            # Step 4: Post-process semantic segmentation
            face_logits = seg_out.cpu().numpy()
            semantic_labels = np.argmax(face_logits, axis=1).tolist()
            
            # Step 5: Post-process instance segmentation
            inst_out = inst_out[0]  # Extract first element
            inst_out = inst_out.sigmoid()
            adj_matrix = (inst_out > self.inst_thres).cpu().numpy().astype('int32')
            
            # Step 6: Post-process bottom face classification
            bottom_out = bottom_out.sigmoid()
            bottom_logits = (bottom_out > self.bottom_thres).cpu().numpy()
            
            # Step 7: Extract feature instances using connected components
            instances = self._extract_instances(
                adj_matrix, 
                face_logits, 
                bottom_logits
            )
            
            # Step 8: Extract extended attributes
            extended_attrs = self._extract_extended_attributes(aag_data)
            
            elapsed_time = time.time() - start_time
            logger.info(f"[{correlation_id}] Recognition complete in {elapsed_time:.2f}s, found {len(instances)} features")
            
            return {
                'success': True,
                'correlation_id': correlation_id,
                'instances': [inst.to_dict() for inst in instances],
                'semantic_labels': semantic_labels,
                'extended_attributes': extended_attrs,
                'num_faces': len(semantic_labels),
                'num_instances': len(instances),
                'processing_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"[{correlation_id}] Feature recognition failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'correlation_id': correlation_id,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def _extract_instances(
        self, 
        adj_matrix: np.ndarray, 
        face_logits: np.ndarray, 
        bottom_logits: np.ndarray
    ) -> List[FeatureInstance]:
        """
        Extract feature instances from adjacency matrix using connected components
        
        Args:
            adj_matrix: Instance adjacency matrix (n_faces x n_faces)
            face_logits: Semantic segmentation logits per face
            bottom_logits: Bottom face classification per face
            
        Returns:
            List of FeatureInstance objects
        """
        instances = []
        used_flags = np.zeros(adj_matrix.shape[0], dtype=np.bool_)
        
        for row_idx, row in enumerate(adj_matrix):
            if used_flags[row_idx]:
                continue
            
            if np.sum(row) <= self.eps:
                # Stock face or isolated face
                continue
            
            # Extract connected component
            proposal = set()
            for col_idx, item in enumerate(row):
                if used_flags[col_idx]:
                    continue
                if item:
                    proposal.add(col_idx)
                    used_flags[col_idx] = True
            
            if len(proposal) == 0:
                continue
            
            # Determine feature type by majority voting
            face_list = list(proposal)
            sum_logits = np.sum([face_logits[f] for f in face_list], axis=0)
            feature_class = np.argmax(sum_logits)
            
            # Skip stock features
            if feature_class == 24:
                continue
            
            # Calculate confidence
            confidence = float(np.max(sum_logits) / len(face_list))
            
            # Extract bottom faces
            bottom_faces = [f for f in face_list if bottom_logits[f]]
            
            # Create instance
            instance = FeatureInstance(
                name=self.FEATURE_NAMES[feature_class],
                faces=face_list,
                bottoms=bottom_faces,
                confidence=confidence
            )
            instances.append(instance)
        
        return instances

    def _extract_extended_attributes(self, aag_data: Dict) -> Dict:
        """
        Extract extended geometric attributes from gAAG data
        
        Args:
            aag_data: Raw gAAG data from extractor
            
        Returns:
            Dictionary with face and edge attributes
        """
        return {
            'face_attributes': aag_data.get('graph_face_attr', []),
            'edge_attributes': aag_data.get('graph_edge_attr', []),
            'face_uv_grids': len(aag_data.get('graph_face_grid', [])),
            'edge_curve_samples': len(aag_data.get('graph_edge_grid', []))
        }


def validate_step_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate STEP file can be loaded
    
    Args:
        file_path: Path to STEP file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        
        if status != IFSelect_RetDone:
            return False, "Failed to read STEP file"
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        if shape.IsNull():
            return False, "STEP file contains null shape"
        
        return True, None
        
    except Exception as e:
        return False, f"STEP validation error: {str(e)}"


# Flask endpoint integration
def create_flask_endpoint(app, recognizer: AAGNetRecognizer):
    """
    Create Flask endpoint for AAGNet feature recognition
    
    Args:
        app: Flask application instance
        recognizer: Initialized AAGNetRecognizer instance
    """
    from flask import request, jsonify
    
    @app.route('/api/aagnet/recognize', methods=['POST'])
    def recognize_features():
        """
        Recognize machining features from uploaded STEP file
        
        Request:
            file: STEP file upload
            
        Response:
            JSON with feature recognition results
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Validate STEP file
            is_valid, error_msg = validate_step_file(tmp_path)
            if not is_valid:
                return jsonify({'error': error_msg}), 400
            
            # Run recognition
            result = recognizer.recognize_features(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 500
                
        except Exception as e:
            logger.error(f"Endpoint error: {str(e)}", exc_info=True)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Test the recognizer
    recognizer = AAGNetRecognizer(device='cpu')
    
    # Test with example file
    test_file = str(AAGNET_PATH / 'examples' / 'partA.step')
    if os.path.exists(test_file):
        result = recognizer.recognize_features(test_file)
        print(json.dumps(result, indent=2))
    else:
        print(f"Test file not found: {test_file}")
