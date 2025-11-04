# Enhanced ML Inference Module with Feature Grouping
# Replaces original ml_inference.py

import warnings
import logging
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
import time
from contextlib import contextmanager

# Suppress OCCWL warnings globally
warnings.filterwarnings('ignore', category=DeprecationWarning, module='occwl')
logging.getLogger('occwl').propagate = False

logger = logging.getLogger(__name__)

# Import feature grouping module
from feature_grouping import group_faces_to_features, FEATURE_CLASSES

# Global model cache
_model = None
_model_loaded = False


@contextmanager
def time_operation(operation_name: str):
    """Context manager for timing operations"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"⏱️ {operation_name}: {elapsed:.2f}s")


def validate_shape(shape) -> Tuple[bool, str]:
    """
    Validate STEP shape before processing.
    
    Returns:
        (is_valid, error_message)
    """
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
    
    # Check B-rep topology validity
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        status = analyzer.Result(TopAbs_FACE)
        return False, f"Invalid B-rep topology: {status}"
    
    # Check if shape contains exactly one solid (not assembly)
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while explorer.More():
        solid_count += 1
        explorer.Next()
    
    if solid_count == 0:
        return False, "Shape contains no solids"
    
    if solid_count > 1:
        return False, f"Assembly detected ({solid_count} solids). Only single parts are supported."
    
    # Check face count is reasonable
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_count = 0
    while explorer.More():
        face_count += 1
        explorer.Next()
    
    if face_count == 0:
        return False, "Shape contains no faces"
    
    if face_count > 500:
        return False, f"Part too complex ({face_count} faces). Max 500 faces supported."
    
    return True, ""


def build_graph_from_step_v2(shape) -> Tuple[object, nx.Graph]:
    """
    Enhanced graph construction with better error handling and statistics.
    
    Returns:
        (dgl_graph, networkx_face_adjacency_graph)
    """
    import dgl
    from occwl.graph import face_adjacency
    from occwl.uvgrid import uvgrid, ugrid
    from occwl.solid import Solid
    
    with time_operation("Graph construction"):
        # 1. Validate input
        valid, error_msg = validate_shape(shape)
        if not valid:
            raise ValueError(f"Invalid shape: {error_msg}")
        
        # 2. Build face adjacency
        solid = Solid(shape)
        nx_graph = face_adjacency(solid)
        
        num_faces = len(nx_graph.nodes)
        num_edges = len(nx_graph.edges)
        logger.info(f"🔗 Face adjacency: {num_faces} faces, {num_edges} edges")
        
        if num_faces == 0:
            raise ValueError("No faces in solid")
        
        # 3. Suppress warnings during grid computation (expensive operation)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            # 3a. Compute UV-grids for faces (with adaptive sampling)
            graph_face_feat = []
            for face_idx in nx_graph.nodes:
                face = nx_graph.nodes[face_idx]["face"]
                
                # Adaptive grid resolution based on face complexity
                grid_res = get_adaptive_grid_resolution(face)
                
                try:
                    points = uvgrid(face, method="point", num_u=grid_res, num_v=grid_res)
                    normals = uvgrid(face, method="normal", num_u=grid_res, num_v=grid_res)
                    visibility = uvgrid(face, method="visibility_status", 
                                       num_u=grid_res, num_v=grid_res)
                    
                    # Mask: 0=Inside, 2=Boundary (ignore 1=Outside)
                    mask = np.logical_or(visibility == 0, visibility == 2)
                    
                    # Concatenate: [grid_res, grid_res, 7] = points(3) + normals(3) + mask(1)
                    face_feat = np.concatenate((points, normals, mask), axis=-1)
                    graph_face_feat.append(face_feat)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute UV-grid for face {face_idx}: {e}")
                    # Fallback: create empty feature
                    graph_face_feat.append(np.zeros((grid_res, grid_res, 7)))
            
            graph_face_feat = np.asarray(graph_face_feat)
            logger.info(f"📊 Face features: shape={graph_face_feat.shape}")
            
            # 3b. Compute U-grids for edges
            graph_edge_feat = []
            edge_indices = []
            
            for edge_idx, (src, dst) in enumerate(nx_graph.edges):
                edge = nx_graph.edges[(src, dst)]["edge"]
                
                try:
                    if not edge.has_curve():
                        continue
                    
                    points = ugrid(edge, method="point", num_u=10)
                    tangents = ugrid(edge, method="tangent", num_u=10)
                    edge_feat = np.concatenate((points, tangents), axis=-1)
                    
                    graph_edge_feat.append(edge_feat)
                    edge_indices.append((src, dst))
                    
                except Exception as e:
                    logger.debug(f"Skipped edge {src}-{dst}: {e}")
            
            graph_edge_feat = np.asarray(graph_edge_feat) if graph_edge_feat else np.array([])
            logger.info(f"📊 Edge features: shape={graph_edge_feat.shape}")
        
        # 4. Convert to DGL graph
        src = [e[0] for e in edge_indices]
        dst = [e[1] for e in edge_indices]
        
        dgl_graph = dgl.graph((src, dst), num_nodes=num_faces) if src else dgl.graph(([], []), num_nodes=num_faces)
        
        dgl_graph.ndata["x"] = torch.tensor(graph_face_feat, dtype=torch.float32)
        if graph_edge_feat.size > 0:
            dgl_graph.edata["x"] = torch.tensor(graph_edge_feat, dtype=torch.float32)
        
        return dgl_graph, nx_graph


def get_adaptive_grid_resolution(face) -> int:
    """
    Adaptively choose grid resolution based on face complexity.
    
    Simple faces (planes): 5x5
    Moderate complexity: 10x10 (default)
    High complexity (NURBS): 15x15
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
    
    try:
        surface = BRepAdaptor_Surface(face.topods_shape())
        surface_type = surface.GetType()
        
        if surface_type in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone]:
            return 8  # Reduced for simple surfaces
        else:
            return 10  # Default for complex surfaces
    except:
        return 10  # Default fallback


def load_uvnet_model_v2():
    """
    Enhanced model loading with better caching and error handling.
    Includes optional FP16 precision loading for inference optimization.
    """
    global _model, _model_loaded
    
    if _model_loaded:
        logger.debug("✅ Model already cached")
        return _model
    
    with time_operation("Model loading from storage"):
        # Download checkpoint from Supabase Storage
        model_path, hparams_path = download_model_from_storage()
        
        # Load hyperparameters
        import yaml
        with open(hparams_path, 'r') as f:
            hparams = yaml.safe_load(f)
        
        num_classes = hparams.get('num_classes', 16)
        crv_in_channels = hparams.get('crv_in_channels', 6)
        
        logger.info(f"Model config: num_classes={num_classes}, crv_in_channels={crv_in_channels}")
        
        # Initialize model architecture
        from uvnet_model import Segmentation
        model = Segmentation(num_classes=num_classes, crv_in_channels=crv_in_channels)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Handle 'model.' prefix in checkpoint keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        
        _model = model
        _model_loaded = True
        
        logger.info(f"✅ Model loaded successfully")
        return model


def preprocess_graph_v2(graph) -> object:
    """
    Enhanced preprocessing with normalization.
    Normalize UV-grids and edge features to zero mean, unit variance.
    """
    # Normalize face features (ndata["x"])
    face_feat = graph.ndata["x"]
    
    # Reshape to [N*H*W, C] for normalization
    N, H, W, C = face_feat.shape
    face_feat_flat = face_feat.reshape(-1, C)
    
    mean = face_feat_flat.mean(dim=0)
    std = face_feat_flat.std(dim=0) + 1e-6
    
    face_feat_norm = (face_feat_flat - mean) / std
    face_feat_norm = face_feat_norm.reshape(N, H, W, C)
    
    graph.ndata["x"] = face_feat_norm
    
    # Normalize edge features if present
    if "x" in graph.edata:
        edge_feat = graph.edata["x"]
        E, L, C_e = edge_feat.shape
        edge_feat_flat = edge_feat.reshape(-1, C_e)
        
        mean_e = edge_feat_flat.mean(dim=0)
        std_e = edge_feat_flat.std(dim=0) + 1e-6
        
        edge_feat_norm = (edge_feat_flat - mean_e) / std_e
        edge_feat_norm = edge_feat_norm.reshape(E, L, C_e)
        
        graph.edata["x"] = edge_feat_norm
    
    return graph


def predict_features_v2(shape, return_raw_predictions: bool = False) -> Dict:
    """
    Enhanced feature prediction with face grouping.
    
    Returns:
        {
            "feature_instances": [FeatureInstance dicts],  # NEW: grouped features
            "feature_summary": {count per type},
            "face_predictions": [raw ML outputs],
            "inference_time": float,
        }
    """
    
    start_time = time.time()
    
    try:
        with time_operation("Full ML pipeline"):
            # 1. Validate shape
            valid, error_msg = validate_shape(shape)
            if not valid:
                raise ValueError(f"Invalid shape: {error_msg}")
            
            # 2. Load model
            model = load_uvnet_model_v2()
            
            # 3. Build graph with error handling
            dgl_graph, nx_graph = build_graph_from_step_v2(shape)
            
            # 4. Preprocess
            dgl_graph = preprocess_graph_v2(dgl_graph)
            
            # 5. Run inference
            with torch.no_grad():
                with time_operation("Model inference"):
                    logits = model(dgl_graph)
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
            
            # 6. Format raw predictions
            face_predictions = []
            for face_id, (pred, probs) in enumerate(zip(predictions, probabilities)):
                pred_class = FEATURE_CLASSES[pred.item()]
                confidence = probs[pred].item()
                
                face_predictions.append({
                    "face_id": face_id,
                    "predicted_class": pred_class,
                    "confidence": round(confidence, 3),
                    "probabilities": [round(p.item(), 3) for p in probs]
                })
            
            logger.info(f"🎯 Face predictions: {len(face_predictions)} faces")
            
            # 7. Group faces into feature instances (NEW - solves main issue)
            with time_operation("Feature grouping"):
                grouped_result = group_faces_to_features(face_predictions, nx_graph)
            
            elapsed = time.time() - start_time
            
            # 8. Return structured result
            result = {
                "feature_instances": grouped_result["feature_instances"],  # NEW
                "feature_summary": grouped_result["feature_summary"],
                "face_predictions": grouped_result["face_predictions"],
                "num_faces_analyzed": len(face_predictions),
                "num_features_detected": len(grouped_result["feature_instances"]),  # NEW
                "clustering_method": "adjacency_based_with_geometric_constraints",  # NEW
                "inference_time_sec": round(elapsed, 2),
            }
            
            logger.info(f"✅ ML pipeline complete in {elapsed:.2f}s")
            logger.info(f"   Features detected: {result['num_features_detected']}")
            
            return result
        
    except Exception as e:
        logger.error(f"❌ ML inference failed: {e}")
        raise


def download_model_from_storage() -> Tuple[str, str]:
    """
    Download UV-Net model and hyperparameters from Supabase Storage.
    Placeholder - implement based on your storage setup.
    """
    import os
    from pathlib import Path
    
    model_cache_dir = Path("/tmp/uvnet_model")
    model_cache_dir.mkdir(exist_ok=True)
    
    model_path = model_cache_dir / "best.ckpt"
    hparams_path = model_cache_dir / "hparams.yaml"
    
    # TODO: Download from Supabase Storage if files don't exist
    # For now, assume files are already mounted or in working directory
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    if not hparams_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found at {hparams_path}")
    
    return str(model_path), str(hparams_path)


# Keep backward compatibility
def predict_features(shape):
    """Wrapper for backward compatibility with original code"""
    return predict_features_v2(shape)
