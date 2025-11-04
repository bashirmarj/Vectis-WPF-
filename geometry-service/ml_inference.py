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
    Build graph from STEP shape using EXACT training logic
    EXACT copy from References/cad-feature-detection-master/backend/app/build_graph.py
    """
    import warnings
    
    # Suppress OCCWL deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        try:
            torch, dgl = _import_ml_libs()
            from occwl.graph import face_adjacency
            from occwl.uvgrid import uvgrid, ugrid
            from occwl.solid import Solid
            from OCC.Extend.TopologyUtils import TopologyExplorer
            
            # Extract the first solid from the shape (handles compounds)
            t = TopologyExplorer(shape)
            solids = list(t.solids())
            if not solids:
                logger.error("❌ No solids found in shape")
                return None
            
            # Use the first solid
            solid_shape = solids[0]
            
            logger.info(f"🔗 Building face adjacency graph...")
            # Build face adjacency graph with B-rep entities
            solid = Solid(solid_shape)
            graph = face_adjacency(solid)
            num_faces = len(graph.nodes)
            logger.info(f"   Graph: {num_faces} faces, {len(graph.edges)} edges")
        
            # Compute UV-grids for faces (10x10 sampling)
            logger.info(f"🔄 Computing UV-grids for {num_faces} faces...")
            graph_face_feat = []
            for i, face_idx in enumerate(graph.nodes):
                face = graph.nodes[face_idx]["face"]
                
                # Sample points, normals, visibility
                points = uvgrid(face, method="point", num_u=10, num_v=10)
                normals = uvgrid(face, method="normal", num_u=10, num_v=10)
                visibility_status = uvgrid(face, method="visibility_status", num_u=10, num_v=10)
                
                # Mask: 0=Inside, 2=Boundary (ignore 1=Outside)
                mask = np.logical_or(visibility_status == 0, visibility_status == 2)
                
                # Concatenate channel-wise: [H, W, 7] = [points(3) + normals(3) + mask(1)]
                face_feat = np.concatenate((points, normals, mask), axis=-1)
                graph_face_feat.append(face_feat)
                
                # Log progress every 20 faces
                if (i + 1) % 20 == 0:
                    logger.info(f"   Processed {i+1}/{num_faces} faces")
            
            logger.info(f"✅ UV-grids computed for all {num_faces} faces")
            graph_face_feat = np.asarray(graph_face_feat)
        
            # Compute U-grids for edges (10 samples)
            logger.info(f"🔄 Computing U-grids for {len(graph.edges)} edges...")
            graph_edge_feat = []
            for edge_idx in graph.edges:
                edge = graph.edges[edge_idx]["edge"]
                
                # Ignore degenerate edges (e.g., at apex of cone)
                if not edge.has_curve():
                    continue
                
                # Sample points and tangents
                points = ugrid(edge, method="point", num_u=10)
                tangents = ugrid(edge, method="tangent", num_u=10)
                
                # Concatenate: [L, 6] = [points(3) + tangents(3)]
                edge_feat = np.concatenate((points, tangents), axis=-1)
                graph_edge_feat.append(edge_feat)
            
            logger.info(f"✅ U-grids computed for {len(graph_edge_feat)} valid edges")
            graph_edge_feat = np.asarray(graph_edge_feat)
        
            # Convert to DGL graph
            edges = list(graph.edges)
            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            n_nodes = len(graph.nodes)
            
            dgl_graph = dgl.graph((src, dst), num_nodes=n_nodes)
            dgl_graph.ndata["x"] = torch.tensor(graph_face_feat, dtype=torch.float32)
            dgl_graph.edata["x"] = torch.tensor(graph_edge_feat, dtype=torch.float32)
            
            logger.info(f"✅ DGL graph built: {n_nodes} faces, {len(edges)} edges")
            return dgl_graph
            
        except Exception as e:
            logger.error(f"❌ Failed to build graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def preprocess_graph(graph):
    """
    Preprocess graph using EXACT training logic
    EXACT copy from References/cad-feature-detection-master/feature_detector/model/code/inference.py
    """
    try:
        torch, _ = _import_ml_libs()
        import util
        
        # Center and scale UV-grids using training preprocessing
        graph.ndata["x"], center, scale = util.center_and_scale_uvgrid(
            graph.ndata["x"], return_center_scale=True
        )
        
        # Apply same transform to edge features (points only, first 3 channels)
        graph.edata["x"][..., :3] -= center
        graph.edata["x"][..., :3] *= scale
        
        # Permute to match model input format
        # Node features: [N, H, W, C] -> [N, C, H, W]
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2).type(torch.FloatTensor)
        
        # Edge features: [E, L, C] -> [E, C, L]
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1).type(torch.FloatTensor)
        
        return graph
        
    except Exception as e:
        logger.error(f"❌ Failed to preprocess graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return graph


def predict_features(shape):
    """
    Run ML inference on STEP shape to predict feature classes per face
    
    Returns:
        dict with:
        - face_predictions: List of {face_id, predicted_class, confidence, probabilities}
        - feature_summary: Counts per feature type
    """
    import warnings
    
    # Suppress OCCWL deprecation warnings during entire inference
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        try:
            torch, _ = _import_ml_libs()
            
            # Load model
            logger.info("🔍 Loading UV-Net model...")
            model = load_uvnet_model()
            if model is None:
                logger.warning("⚠️ Model not available, skipping ML inference")
                return None
            logger.info("✅ UV-Net model loaded successfully")
            
            # Build graph
            graph = build_graph_from_step(shape)
            if graph is None:
                logger.error("❌ Graph construction returned None")
                return None
        
            # Preprocess (includes permutation to [N, C, H, W] and [E, C, L])
            graph = preprocess_graph(graph)
            
            # Run inference
            logger.info("🔮 Running inference on graph...")
            with torch.no_grad():
                logits = model(graph)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            logger.info("✅ Inference complete")
            
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
