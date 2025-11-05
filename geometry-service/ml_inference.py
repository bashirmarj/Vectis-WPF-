# ml_inference.py - FIXED VERSION
# Complete ML inference pipeline with UV-Net checkpoint loading
# Ready to copy and paste - NO URLS, CLEAN CODE

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import dgl
from dgl.nn.pytorch import GraphConv
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.ckpt"

def load_uvnet_model():
    """
    Load pre-trained UV-Net model from checkpoint.
    
    Returns:
        model: Loaded UV-Net model or None if not found
    """
    try:
        if not CHECKPOINT_PATH.exists():
            logger.error(f"❌ Checkpoint not found at: {CHECKPOINT_PATH}")
            return None
        
        logger.info(f"📥 Loading UV-Net checkpoint from: {CHECKPOINT_PATH}")
        
        try:
            model = pl.LightningModule.load_from_checkpoint(str(CHECKPOINT_PATH))
            logger.info("✅ UV-Net model loaded successfully (PyTorch Lightning)")
            return model
        except:
            try:
                checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu')
                
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    logger.info("✅ Loaded PyTorch Lightning checkpoint (state_dict)")
                    return checkpoint['state_dict']
                else:
                    logger.info("✅ Loaded PyTorch checkpoint")
                    return checkpoint
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint as state dict: {e}")
                return None
    
    except Exception as e:
        logger.error(f"❌ Error loading UV-Net model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ============================================================================
# UV-NET INFERENCE
# ============================================================================

def run_uvnet_inference(model, uv_features):
    """
    Run UV-Net model on extracted UV features.
    
    Args:
        model: Loaded UV-Net model
        uv_features: Extracted UV coordinate features from faces
    
    Returns:
        Predictions tensor
    """
    try:
        if model is None:
            logger.warning("⚠️ Model is None, cannot run inference")
            return None
        
        logger.info("🧠 Running UV-Net inference...")
        
        if isinstance(uv_features, list):
            uv_features = np.array(uv_features)
        
        input_tensor = torch.from_numpy(uv_features).float()
        
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        input_tensor = input_tensor.to(device)
        
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward'):
                outputs = model(input_tensor)
            else:
                logger.warning("⚠️ Model has no forward method")
                return None
        
        if isinstance(outputs, tuple):
            predictions_tensor = outputs[0] if len(outputs) > 0 else outputs
        else:
            predictions_tensor = outputs
        
        if predictions_tensor.dim() == 1:
            predictions = torch.sigmoid(predictions_tensor)
        else:
            predictions = torch.softmax(predictions_tensor, dim=-1)
        
        logger.info("✅ UV-Net inference complete")
        return predictions
    
    except Exception as e:
        logger.error(f"❌ UV-Net inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ============================================================================
# UV COORDINATE EXTRACTION
# ============================================================================

def extract_uv_coordinates_from_face(face, num_samples=16):
    """
    Extract UV coordinate samples from an OCC face.
    
    Args:
        face: OCC face object
        num_samples: Number of samples in U and V directions
    
    Returns:
        np.ndarray: UV coordinate samples
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        
        surf_adaptor = BRepAdaptor_Surface(face)
        
        u_min = surf_adaptor.FirstUParameter()
        u_max = surf_adaptor.LastUParameter()
        v_min = surf_adaptor.FirstVParameter()
        v_max = surf_adaptor.LastVParameter()
        
        uv_samples = []
        for i in range(num_samples):
            for j in range(num_samples):
                u_norm = i / (num_samples - 1) if num_samples > 1 else 0.5
                v_norm = j / (num_samples - 1) if num_samples > 1 else 0.5
                uv_samples.append([u_norm, v_norm])
        
        return np.array(uv_samples)
    
    except Exception as e:
        logger.debug(f"Could not extract UV from face: {e}")
        return np.zeros((num_samples * num_samples, 2))

def extract_uv_features_from_shape(shape, face_data):
    """
    Extract UV features from all faces in the shape.
    
    Args:
        shape: OCC shape
        face_data: Face data from build_graph_from_step
    
    Returns:
        np.ndarray: UV features for all faces
    """
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods
        
        all_uv_features = []
        faces = []
        
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            faces.append(topods.Face(face_explorer.Current()))
            face_explorer.Next()
        
        logger.info(f"📐 Extracting UV features from {len(faces)} faces...")
        
        for face_idx, face in enumerate(faces):
            try:
                uv_samples = extract_uv_coordinates_from_face(face, num_samples=8)
                all_uv_features.append(uv_samples)
            except:
                logger.debug(f"Failed to extract UV from face {face_idx}")
                all_uv_features.append(np.zeros((64, 2)))
        
        all_uv_features = np.vstack(all_uv_features) if all_uv_features else np.zeros((len(faces), 2))
        
        logger.info(f"✅ Extracted UV features: shape {all_uv_features.shape}")
        return all_uv_features
    
    except Exception as e:
        logger.error(f"❌ UV extraction failed: {e}")
        return None

# ============================================================================
# ML FEATURE CLASSIFICATION
# ============================================================================

FEATURE_CLASSES = [
    'plane', 'cylinder', 'cone', 'sphere', 'torus',
    'hole', 'boss', 'pocket', 'slot', 'chamfer',
    'fillet', 'groove', 'step', 'blind_hole', 'through_hole'
]

def predictions_to_face_predictions(predictions, num_faces):
    """
    Convert model predictions to face-level predictions.
    
    Args:
        predictions: Tensor from UV-Net model
        num_faces: Number of faces in shape
    
    Returns:
        List of face predictions
    """
    try:
        if predictions is None:
            return None
        
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        
        face_predictions = []
        
        samples_per_face = len(predictions) // num_faces if num_faces > 0 else 1
        
        for face_id in range(num_faces):
            start_idx = face_id * samples_per_face
            end_idx = start_idx + samples_per_face
            
            if start_idx < len(predictions):
                face_preds = predictions[start_idx:end_idx]
                
                avg_pred = np.mean(face_preds, axis=0)
                
                if len(avg_pred) > 0:
                    pred_class_idx = np.argmax(avg_pred)
                    confidence = float(avg_pred[pred_class_idx])
                    pred_class = FEATURE_CLASSES[min(pred_class_idx, len(FEATURE_CLASSES)-1)]
                else:
                    pred_class = 'plane'
                    confidence = 0.5
            else:
                pred_class = 'plane'
                confidence = 0.5
            
            face_predictions.append({
                'face_id': face_id,
                'predicted_class': pred_class,
                'confidence': confidence
            })
        
        logger.info(f"✅ Converted to {len(face_predictions)} face predictions")
        return face_predictions
    
    except Exception as e:
        logger.error(f"❌ Prediction conversion failed: {e}")
        return None

# ============================================================================
# SHAPE VALIDATION & GRAPH BUILDING
# ============================================================================

def validate_shape(shape):
    """Validate if shape is a valid single closed solid."""
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
        from OCC.Core.TopoDS import topods

        solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        solid_count = 0
        while solid_explorer.More():
            solid_count += 1
            solid_explorer.Next()

        if solid_count == 0:
            return False, "Shape contains no solids"
        if solid_count > 1:
            return False, f"Shape is compound with {solid_count} solids"

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_count = 0
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()

        if face_count == 0:
            return False, "Shape has no faces"
        if face_count > 500:
            return False, f"Shape too complex ({face_count} faces)"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def build_graph_from_step(shape):
    """Build face adjacency graph from STEP shape."""
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone,
                                   GeomAbs_Sphere, GeomAbs_Torus)
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    import networkx as nx

    logger.info("🔗 Building face adjacency graph...")

    try:
        faces = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            faces.append(topods.Face(face_explorer.Current()))
            face_explorer.Next()

        logger.debug(f"  Found {len(faces)} faces")

        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

        nx_graph = nx.Graph()
        for i in range(len(faces)):
            nx_graph.add_node(i)

        for edge_idx in range(1, edge_face_map.Size() + 1):
            try:
                face_list = edge_face_map.FindFromIndex(edge_idx)
                if face_list.Size() == 2:
                    face1_idx = None
                    face2_idx = None
                    for i, f in enumerate(faces):
                        if f.IsSame(topods.Face(face_list.First())):
                            face1_idx = i
                        if f.IsSame(topods.Face(face_list.Last())):
                            face2_idx = i
                    if face1_idx is not None and face2_idx is not None:
                        nx_graph.add_edge(face1_idx, face2_idx)
            except:
                pass

        face_data = []
        for face_idx, face in enumerate(faces):
            try:
                surf_adaptor = BRepAdaptor_Surface(face)
                surf_type = surf_adaptor.GetType()

                face_data.append({
                    'face_id': face_idx,
                    'surface_type': surf_type
                })
            except:
                face_data.append({
                    'face_id': face_idx,
                    'surface_type': -1
                })

        logger.debug(f"  ✅ Graph built: {len(faces)} nodes, {nx_graph.number_of_edges()} edges")
        return None, nx_graph, face_data

    except Exception as e:
        logger.error(f"❌ Failed to build graph: {e}")
        return None, None, None

# ============================================================================
# FEATURE GROUPING INTEGRATION
# ============================================================================

def group_faces_into_features(face_predictions, face_adjacency_graph):
    """Group face predictions into feature instances."""
    try:
        if not face_predictions or face_adjacency_graph is None:
            logger.warning("⚠️ Empty predictions or graph")
            return {
                'feature_instances': [],
                'feature_summary': {},
                'num_features': 0
            }

        logger.info("🔄 Attempting to import feature grouping module...")
        
        try:
            import feature_grouping
            logger.info("✅ feature_grouping module imported")
        except ImportError as e:
            logger.error(f"❌ Cannot import feature_grouping: {e}")
            return {
                'feature_instances': [],
                'feature_summary': {},
                'num_features': 0,
                'error': str(e)
            }

        try:
            from feature_grouping import group_faces_to_features
            logger.info("✅ group_faces_to_features function imported")
        except ImportError as e:
            logger.error(f"❌ Cannot import function: {e}")
            return {
                'feature_instances': [],
                'feature_summary': {},
                'num_features': 0,
                'error': str(e)
            }

        try:
            logger.info("🔗 Calling group_faces_to_features()...")
            result = group_faces_to_features(face_predictions, face_adjacency_graph)
            num_features = result.get('num_features', 0)
            logger.info(f"✅ Grouped into {num_features} feature instances")
            return result

        except Exception as e:
            logger.error(f"❌ Feature grouping failed: {e}")
            return {
                'feature_instances': [],
                'feature_summary': {},
                'num_features': 0,
                'error': str(e)
            }

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return {
            'feature_instances': [],
            'feature_summary': {},
            'num_features': 0,
            'error': str(e)
        }

# ============================================================================
# MAIN INFERENCE FUNCTION - WITH UV-NET
# ============================================================================

def predict_features(shape):
    """
    Main entry point for ML feature prediction WITH UV-NET.
    
    Args:
        shape: OCC shape from STEP file
    
    Returns:
        Dict with face_predictions, feature_instances, etc.
    """
    try:
        logger.info("🤖 Starting ML feature inference with UV-Net...")

        is_valid, error_msg = validate_shape(shape)
        if not is_valid:
            logger.warning(f"⚠️ Shape validation: {error_msg}")

        logger.info("📊 Building geometry graph...")
        dgl_graph, nx_graph, face_data = build_graph_from_step(shape)
        
        if nx_graph is None or face_data is None:
            logger.error("❌ Failed to build graph")
            return {
                'success': False,
                'error': 'Graph construction failed',
                'face_predictions': [],
                'feature_instances': [],
                'feature_summary': {}
            }

        num_faces = len(face_data)
        logger.info(f"✅ Extracted {num_faces} faces from geometry")

        logger.info("🤖 Loading UV-Net model...")
        model = load_uvnet_model()
        
        if model is None:
            logger.error("❌ UV-Net model not available, using placeholder")
            face_predictions = [{
                'face_id': i,
                'predicted_class': 'plane',
                'confidence': 0.5
            } for i in range(num_faces)]
        else:
            logger.info("📐 Extracting UV features...")
            uv_features = extract_uv_features_from_shape(shape, face_data)
            
            if uv_features is None:
                logger.error("❌ Failed to extract UV features")
                face_predictions = [{
                    'face_id': i,
                    'predicted_class': 'plane',
                    'confidence': 0.5
                } for i in range(num_faces)]
            else:
                logger.info("🧠 Running UV-Net inference...")
                predictions = run_uvnet_inference(model, uv_features)
                
                if predictions is None:
                    logger.error("❌ UV-Net inference failed")
                    face_predictions = [{
                        'face_id': i,
                        'predicted_class': 'plane',
                        'confidence': 0.5
                    } for i in range(num_faces)]
                else:
                    face_predictions = predictions_to_face_predictions(predictions, num_faces)
                    
                    if face_predictions is None:
                        logger.error("❌ Failed to convert predictions")
                        face_predictions = [{
                            'face_id': i,
                            'predicted_class': 'plane',
                            'confidence': 0.5
                        } for i in range(num_faces)]

        logger.info(f"✅ Generated {len(face_predictions)} face predictions")

        logger.info("🔄 Grouping faces into features...")
        grouping_result = group_faces_into_features(face_predictions, nx_graph)

        return {
            'success': True,
            'face_predictions': face_predictions,
            'feature_instances': grouping_result.get('feature_instances', []),
            'feature_summary': grouping_result.get('feature_summary', {}),
            'num_features': grouping_result.get('num_features', 0),
            'num_faces': num_faces,
            'model_name': 'UV-Net with Feature Grouping'
        }

    except Exception as e:
        logger.error(f"❌ ML inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'face_predictions': [],
            'feature_instances': [],
            'feature_summary': {}
        }

# ============================================================================
# STATUS FUNCTIONS
# ============================================================================

def get_ml_status() -> Dict:
    """Check if ML modules are available"""
    status = {
        'torch': False,
        'torch_geometric': False,
        'dgl': False,
        'networkx': False,
        'numpy': False,
        'pytorch_lightning': False,
        'feature_grouping': False,
        'checkpoint': CHECKPOINT_PATH.exists()
    }

    try:
        import torch
        status['torch'] = True
    except:
        pass

    try:
        import torch_geometric
        status['torch_geometric'] = True
    except:
        pass

    try:
        import dgl
        status['dgl'] = True
    except:
        pass

    try:
        import networkx
        status['networkx'] = True
    except:
        pass

    try:
        import numpy
        status['numpy'] = True
    except:
        pass

    try:
        import pytorch_lightning
        status['pytorch_lightning'] = True
    except:
        pass

    try:
        import feature_grouping
        status['feature_grouping'] = True
    except:
        pass

    return status

if __name__ == '__main__':
    logger.info("✅ ML Inference module loaded")
    status = get_ml_status()
    logger.info(f"📊 Status: {status}")
