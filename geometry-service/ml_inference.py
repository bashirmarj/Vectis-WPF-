"""
ML Inference Module for UV-Net Feature Recognition
Integrates pre-trained UV-Net model for advanced CAD feature detection
"""
import os
import logging
import numpy as np
import tempfile
from pathlib import Path

logger = logging.getLogger("ml_inference")

# Feature class mappings (16 classes from MFCAD dataset)
FEATURE_CLASSES = [
    "hole", "boss", "pocket", "slot", "chamfer", "fillet", "groove", "step",
    "plane", "cylinder", "cone", "sphere", "torus", "bspline", "revolution", "extrusion"
]

# Lazy imports for ML libraries
_torch = None
_dgl = None
_model = None
_model_loaded = False

def _import_ml_libs():
    """Lazy import of ML libraries to avoid startup overhead"""
    global _torch, _dgl
    if _torch is None:
        import torch
        _torch = torch
    if _dgl is None:
        import dgl
        _dgl = dgl
    return _torch, _dgl


def download_model_from_storage():
    """Download UV-Net model checkpoint from Supabase Storage"""
    try:
        from supabase import create_client
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("❌ Supabase credentials not configured")
            return None, None
            
        supabase = create_client(supabase_url, supabase_key)
        
        # Create local cache directory
        cache_dir = Path("/tmp/ml_models")
        cache_dir.mkdir(exist_ok=True)
        
        model_path = cache_dir / "best.ckpt"
        hparams_path = cache_dir / "hparams.yaml"
        
        # Download if not cached
        if not model_path.exists():
            logger.info("📥 Downloading model checkpoint from Supabase Storage...")
            model_data = supabase.storage.from_("trained-models").download("best.ckpt")
            model_path.write_bytes(model_data)
            logger.info(f"✅ Model downloaded to {model_path}")
        
        if not hparams_path.exists():
            logger.info("📥 Downloading hparams.yaml...")
            hparams_data = supabase.storage.from_("trained-models").download("hparams.yaml")
            hparams_path.write_bytes(hparams_data)
            logger.info(f"✅ Hparams downloaded to {hparams_path}")
        
        return str(model_path), str(hparams_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return None, None


def load_uvnet_model():
    """Load UV-Net model from checkpoint"""
    global _model, _model_loaded
    
    if _model_loaded and _model is not None:
        return _model
    
    try:
        torch, _ = _import_ml_libs()
        
        # Download model files
        model_path, hparams_path = download_model_from_storage()
        if not model_path or not hparams_path:
            logger.error("❌ Model files not available")
            return None
        
        # Import model architecture
        from uvnet_model import Segmentation
        
        # Load hyperparameters
        import yaml
        with open(hparams_path, 'r') as f:
            hparams = yaml.safe_load(f)
        
        logger.info(f"📋 Loaded hparams: {hparams}")
        
        # Initialize model
        num_classes = hparams.get('num_classes', 16)
        crv_in_channels = hparams.get('crv_in_channels', 6)
        
        logger.info(f"🔄 Loading UV-Net model from {model_path}")
        
        # Load checkpoint
        model = Segmentation(num_classes=num_classes, crv_in_channels=crv_in_channels)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle state_dict with 'model.' prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Remove 'model.' prefix from checkpoint keys
            if key.startswith('model.'):
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the modified state dict into the inner model
        model.model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        
        _model = model
        _model_loaded = True
        
        logger.info("✅ UV-Net model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load UV-Net model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def build_graph_from_step(shape):
    """
    Convert OpenCASCADE shape to DGL graph format
    Ported from References/cad-feature-detection-master/backend/app/build_graph.py
    """
    try:
        _, dgl = _import_ml_libs()
        from occwl_stub import face_adjacency, uvgrid, ugrid, Solid
        
        # Build face adjacency graph
        graph = face_adjacency(Solid(shape))
        
        # Compute UV-grids for faces (10x10 sampling)
        graph_face_feat = []
        for face_idx in graph.nodes:
            face = graph.nodes[face_idx]["face"]
            
            # Sample points, normals, visibility
            points = uvgrid(face, method="point", num_u=10, num_v=10)
            normals = uvgrid(face, method="normal", num_u=10, num_v=10)
            visibility = uvgrid(face, method="visibility_status", num_u=10, num_v=10)
            
            # Mask: 0=Inside, 2=Boundary
            mask = np.logical_or(visibility == 0, visibility == 2)
            
            # Concatenate features
            face_feat = np.concatenate((points, normals, mask), axis=-1)
            graph_face_feat.append(face_feat)
        
        graph_face_feat = np.asarray(graph_face_feat)
        
        # Compute U-grids for edges (10 samples)
        graph_edge_feat = []
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            if not edge.has_curve():
                continue
            
            points = ugrid(edge, method="point", num_u=10)
            tangents = ugrid(edge, method="tangent", num_u=10)
            edge_feat = np.concatenate((points, tangents), axis=-1)
            graph_edge_feat.append(edge_feat)
        
        graph_edge_feat = np.asarray(graph_edge_feat)
        
        # Convert to DGL graph
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        
        dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
        
        # Add features to graph
        torch, _ = _import_ml_libs()
        dgl_graph.ndata["x"] = torch.tensor(graph_face_feat, dtype=torch.float32)
        dgl_graph.edata["x"] = torch.tensor(graph_edge_feat, dtype=torch.float32)
        
        logger.info(f"📊 Built DGL graph: {len(graph.nodes)} faces, {len(edges)} edges")
        return dgl_graph
        
    except Exception as e:
        logger.error(f"❌ Failed to build graph: {e}")
        return None


def preprocess_graph(graph):
    """
    Preprocess graph features (centering and scaling)
    Based on References/cad-feature-detection-master/feature_detector/model/code/inference.py
    """
    try:
        torch, _ = _import_ml_libs()
        
        # Center node features
        node_features = graph.ndata["x"]
        node_mean = node_features.mean(dim=0, keepdim=True)
        node_features = node_features - node_mean
        
        # Scale by bounding box diagonal
        bbox_min = node_features.min(dim=0)[0]
        bbox_max = node_features.max(dim=0)[0]
        diagonal = torch.norm(bbox_max - bbox_min)
        node_features = node_features / diagonal
        
        graph.ndata["x"] = node_features
        
        # Center edge features
        if graph.edata["x"].shape[0] > 0:
            edge_features = graph.edata["x"]
            edge_mean = edge_features.mean(dim=0, keepdim=True)
            edge_features = edge_features - edge_mean
            edge_features = edge_features / diagonal
            graph.edata["x"] = edge_features
        
        return graph
        
    except Exception as e:
        logger.error(f"❌ Failed to preprocess graph: {e}")
        return graph


def predict_features(shape):
    """
    Run ML inference on STEP shape to predict feature classes per face
    
    Returns:
        dict with:
        - face_predictions: List of {face_id, predicted_class, confidence, probabilities}
        - feature_summary: Counts per feature type
    """
    try:
        torch, _ = _import_ml_libs()
        
        # Load model
        model = load_uvnet_model()
        if model is None:
            logger.warning("⚠️ Model not available, skipping ML inference")
            return None
        
        # Build graph
        graph = build_graph_from_step(shape)
        if graph is None:
            return None
        
        # Preprocess
        graph = preprocess_graph(graph)
        
        # Permute features to match model input format
        # Node features: [N, H, W, C] -> [N, C, H, W]
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        # Edge features: [E, L, C] -> [E, C, L]
        if graph.edata["x"].shape[0] > 0:
            graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        
        # Run inference
        with torch.no_grad():
            logits = model(graph)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # Format results
        face_predictions = []
        feature_counts = {cls: 0 for cls in FEATURE_CLASSES}
        
        for face_id, (pred, probs) in enumerate(zip(predictions, probabilities)):
            pred_class = FEATURE_CLASSES[pred.item()]
            confidence = probs[pred].item()
            
            face_predictions.append({
                "face_id": face_id,
                "predicted_class": pred_class,
                "confidence": round(confidence, 3),
                "probabilities": [round(p.item(), 3) for p in probs]
            })
            
            feature_counts[pred_class] += 1
        
        logger.info(f"✅ ML inference complete: {len(face_predictions)} faces analyzed")
        
        return {
            "face_predictions": face_predictions,
            "feature_summary": {
                "total_faces": len(face_predictions),
                **feature_counts
            }
        }
        
    except Exception as e:
        logger.error(f"❌ ML inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
