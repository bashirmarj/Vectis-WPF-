import os
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import json
import traceback
import sys
import numpy as np
import hashlib
from typing import Dict, Any, Optional, List, Tuple
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set environment variables for OpenCascade
os.environ['PYTHONOCC_SHUNT_GUI'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

# Import OpenCascade modules
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Face, TopoDS_Edge
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_AddOptimal
    from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    logger.info("✅ OpenCascade modules loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import OpenCascade modules: {e}")

# Import feature recognition modules
from brepnet_wrapper import BRepNetRecognizer, FeatureType

# Import other modules
from machining_estimator import MachiningTimeEstimator, SetupConfiguration, MachiningOperation, ToolType

app = Flask(__name__)
CORS(app)

# Initialize feature recognizers
brepnet_recognizer = None

def initialize_ml_models():
    """Initialize ML models for feature recognition"""
    global brepnet_recognizer
    
    # Try to load BRepNet model
    try:
        model_path = Path("models/pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt")
        if not model_path.exists():
            logger.warning(f"⚠️ BRepNet model not found at {model_path}")
            return False
            
        logger.info(f"🔄 Loading BRepNet from {model_path}...")
        
        brepnet_recognizer = BRepNetRecognizer(
            model_path=model_path,
            device="cpu",  # Use CPU for production
            confidence_threshold=0.30  # LOWERED FROM 0.70 to capture more features like chamfers and fillets
        )
        logger.info("✅ BRepNet recognizer loaded successfully with confidence threshold=0.30")
    except FileNotFoundError as e:
        logger.error(f"❌ BRepNet model file not found: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to load BRepNet: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True

# Initialize models on startup
ml_models_available = initialize_ml_models()
if ml_models_available:
    logger.info("✅ ML models initialized successfully")
else:
    logger.warning("⚠️ Running without ML models - using geometric analysis only")

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for caching"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def parse_step_file(step_path: str) -> Optional[TopoDS_Shape]:
    """Parse STEP file and return shape"""
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        
        if status != IFSelect_RetDone:
            logger.error(f"Failed to read STEP file: {step_path}")
            return None
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        if shape.IsNull():
            logger.error("Loaded shape is null")
            return None
            
        return shape
    except Exception as e:
        logger.error(f"Error parsing STEP file: {e}")
        logger.error(traceback.format_exc())
        return None

def tessellate_shape(shape: TopoDS_Shape, linear_deflection: float = 0.5, angular_deflection: float = 0.17) -> Dict[str, Any]:
    """Tessellate shape to extract mesh data with vertex-to-face mapping"""
    try:
        start_time = time.time()
        
        # Create mesh with controlled parameters
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()
        
        if not mesh.IsDone():
            logger.error("Mesh generation failed")
            return None
        
        vertices = []
        vertex_set = set()
        vertex_map = {}  # Map unique vertices to indices
        indices = []
        normals = []
        edges = []
        face_groups = []
        vertex_face_ids = []  # Will store face ID for each vertex
        
        # First pass: collect faces
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = []
        while face_explorer.More():
            faces.append(topods.Face(face_explorer.Current()))
            face_explorer.Next()
        
        # Process each face
        face_offset = 0
        for face_id, face in enumerate(faces):
            try:
                # Get triangulation for this face
                loc = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, loc)
                
                if triangulation is None:
                    continue
                
                # Get transformation
                transformation = loc.Transformation()
                
                # Process vertices for this face
                face_vertices = []
                face_vertex_indices = []
                
                for i in range(1, triangulation.NbNodes() + 1):
                    pt = triangulation.Node(i)
                    # Apply transformation
                    pt.Transform(transformation)
                    
                    vertex_key = (round(pt.X(), 6), round(pt.Y(), 6), round(pt.Z(), 6))
                    
                    if vertex_key not in vertex_map:
                        vertex_idx = len(vertices)
                        vertex_map[vertex_key] = vertex_idx
                        vertices.append([pt.X(), pt.Y(), pt.Z()])
                        # Store which face this vertex belongs to
                        vertex_face_ids.append(face_id)
                    
                    face_vertex_indices.append(vertex_map[vertex_key])
                
                # Process triangles for this face
                face_triangles = []
                orientation = face.Orientation()
                
                for i in range(1, triangulation.NbTriangles() + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    
                    idx1 = face_vertex_indices[n1 - 1]
                    idx2 = face_vertex_indices[n2 - 1]
                    idx3 = face_vertex_indices[n3 - 1]
                    
                    # Apply orientation
                    from OCC.Core.TopAbs import TopAbs_REVERSED
                    if orientation == TopAbs_REVERSED:
                        indices.extend([idx1, idx3, idx2])
                    else:
                        indices.extend([idx1, idx2, idx3])
                    
                    face_triangles.append(len(indices) // 3 - 1)
                
                # Store face group info
                if face_triangles:
                    face_groups.append({
                        'face_id': face_id,
                        'start_idx': face_triangles[0] * 3,
                        'count': len(face_triangles) * 3
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process face {face_id}: {e}")
                continue
        
        # Ensure vertex_face_ids has the correct length
        if len(vertex_face_ids) < len(vertices):
            # Pad with -1 for vertices that couldn't be mapped to a face
            vertex_face_ids.extend([-1] * (len(vertices) - len(vertex_face_ids)))
        
        # Generate placeholder normals (can be computed properly if needed)
        normals = [[0, 0, 1]] * len(vertices)
        
        # Process edges
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edge_set = set()
        
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            
            # Get edge vertices
            vertex_exp = TopExp_Explorer(edge, TopAbs_VERTEX)
            edge_verts = []
            while vertex_exp.More() and len(edge_verts) < 2:
                vertex = topods.Vertex(vertex_exp.Current())
                pt = BRep_Tool.Pnt(vertex)
                vertex_key = (round(pt.X(), 6), round(pt.Y(), 6), round(pt.Z(), 6))
                
                if vertex_key in vertex_map:
                    edge_verts.append(vertex_map[vertex_key])
                
                vertex_exp.Next()
            
            if len(edge_verts) == 2:
                edge_key = tuple(sorted(edge_verts))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append(edge_verts)
            
            edge_explorer.Next()
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Tessellated: {len(vertices)} vertices, {len(indices)//3} triangles, {len(faces)} faces in {elapsed:.1f}ms")
        logger.info(f"Created vertex_face_ids: {len(vertex_face_ids)} vertices mapped to faces")
        
        return {
            'vertices': vertices,
            'indices': indices,
            'normals': normals,
            'edges': edges,
            'face_groups': face_groups,
            'vertex_face_ids': vertex_face_ids,  # Added vertex-to-face mapping
            'num_faces': len(faces),
            'num_edges': len(edges),
            'num_vertices': len(vertices),
            'tessellation_time': elapsed
        }
        
    except Exception as e:
        logger.error(f"Error during tessellation: {e}")
        logger.error(traceback.format_exc())
        return None

def compute_bounding_box(shape: TopoDS_Shape) -> Dict[str, float]:
    """Compute bounding box of shape"""
    try:
        bbox = Bnd_Box()
        brepbndlib_AddOptimal(shape, bbox, True, True)
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        return {
            'min': {'x': xmin, 'y': ymin, 'z': zmin},
            'max': {'x': xmax, 'y': ymax, 'z': zmax},
            'dimensions': {
                'x': xmax - xmin,
                'y': ymax - ymin,
                'z': zmax - zmin
            },
            'center': {
                'x': (xmin + xmax) / 2,
                'y': (ymin + ymax) / 2,
                'z': (zmin + zmax) / 2
            }
        }
    except Exception as e:
        logger.error(f"Error computing bounding box: {e}")
        return None

def compute_volume_and_surface_area(shape: TopoDS_Shape) -> Dict[str, float]:
    """Compute volume and surface area"""
    try:
        # Volume
        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        volume = props.Mass()
        
        # Surface area
        props_surf = GProp_GProps()
        brepgprop_SurfaceProperties(shape, props_surf)
        surface_area = props_surf.Mass()
        
        return {
            'volume': volume,
            'surface_area': surface_area
        }
    except Exception as e:
        logger.error(f"Error computing volume and surface area: {e}")
        return {'volume': 0, 'surface_area': 0}

def classify_part_type(shape: TopoDS_Shape, bounding_box: Dict) -> str:
    """
    Classify part as prismatic, turning, or sheet metal
    """
    try:
        dims = bounding_box['dimensions']
        x, y, z = dims['x'], dims['y'], dims['z']
        
        # Sort dimensions
        sorted_dims = sorted([x, y, z])
        min_dim = sorted_dims[0]
        mid_dim = sorted_dims[1]
        max_dim = sorted_dims[2]
        
        # Sheet metal detection (one dimension much smaller)
        thickness_ratio = min_dim / max_dim if max_dim > 0 else 1
        if thickness_ratio < 0.1:
            return "sheet_metal"
        
        # Count cylindrical faces for turning detection
        cylindrical_count = 0
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_Cylinder:
                cylindrical_count += 1
            face_explorer.Next()
        
        # Turning part detection (high cylindrical face ratio)
        total_faces = 0
        face_exp2 = TopExp_Explorer(shape, TopAbs_FACE)
        while face_exp2.More():
            total_faces += 1
            face_exp2.Next()
        
        if total_faces > 0:
            cylinder_ratio = cylindrical_count / total_faces
            if cylinder_ratio > 0.3:  # 30% or more cylindrical faces
                return "turning"
        
        # Default to prismatic
        return "prismatic"
        
    except Exception as e:
        logger.error(f"Error classifying part: {e}")
        return "prismatic"

def run_ml_feature_recognition(shape: TopoDS_Shape, step_file_path: str) -> Optional[Dict[str, Any]]:
    """Run ML-based feature recognition using BRepNet"""
    global brepnet_recognizer
    
    if not brepnet_recognizer:
        logger.warning("BRepNet recognizer not available")
        return None
    
    try:
        logger.info("🔍 Starting BRepNet feature recognition")
        start_time = time.time()
        
        # Run BRepNet recognition
        result = brepnet_recognizer.recognize_features_from_step(
            step_file_path=step_file_path,
            shape=shape
        )
        
        if result:
            elapsed = time.time() - start_time
            logger.info(f"✅ BRepNet recognized {len(result.get('instances', []))} features")
            
            return {
                'instances': result.get('instances', []),
                'num_features_detected': len(result.get('instances', [])),
                'inference_time_sec': elapsed,
                'avg_confidence': result.get('avg_confidence', 0.0),
                'recognition_method': 'brepnet'
            }
        else:
            logger.warning("BRepNet returned no results")
            return None
            
    except Exception as e:
        logger.error(f"ML feature recognition failed: {e}")
        logger.error(traceback.format_exc())
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_models_available': ml_models_available,
        'timestamp': time.time()
    })

@app.route('/analyze', methods=['POST'])
def analyze_cad():
    """Main endpoint for CAD file analysis"""
    request_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"
    logger.info(f"[{request_id}] Starting analysis")
    
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
            
            # Compute file hash and size for logging
            file_hash = compute_file_hash(temp_path)
            file_size = os.path.getsize(temp_path)
            logger.info(f"[{request_id}] File hash: {file_hash}, size: {file_size} bytes")
        
        try:
            # Parse STEP file
            logger.info(f"[{request_id}] Parsing STEP file")
            shape = parse_step_file(temp_path)
            if shape is None:
                return jsonify({'error': 'Failed to parse STEP file'}), 400
            
            # Tessellate mesh
            logger.info(f"[{request_id}] Tessellating mesh")
            mesh_data = tessellate_shape(shape)
            if mesh_data is None:
                return jsonify({'error': 'Failed to generate mesh'}), 500
            
            # Compute bounding box
            logger.info(f"[{request_id}] Computing bounding box")
            bounding_box = compute_bounding_box(shape)
            
            # Compute volume and surface area
            logger.info(f"[{request_id}] Computing volume and surface area")
            volume_data = compute_volume_and_surface_area(shape)
            
            # Classify part type
            logger.info(f"[{request_id}] Classifying part type")
            part_type = classify_part_type(shape, bounding_box)
            logger.info(f"[{request_id}] 📊 Part classified as: {part_type}")
            
            # Run ML feature recognition
            ml_features = None
            if ml_models_available and part_type in ['prismatic', 'turning']:
                logger.info(f"[{request_id}] 🤖 Running BRepNet ML feature recognition")
                ml_features = run_ml_feature_recognition(shape, temp_path)
                if ml_features:
                    logger.info(f"[{request_id}] ✅ BRepNet found {ml_features['num_features_detected']} features in {ml_features['inference_time_sec']*1000:.0f}ms")
            
            # Prepare response
            response = {
                'success': True,
                'request_id': request_id,
                'mesh_data': mesh_data,
                'bounding_box': bounding_box,
                'volume': volume_data['volume'],
                'surface_area': volume_data['surface_area'],
                'part_type': part_type,
                'ml_features': ml_features
            }
            
            # Log completion
            total_time = time.time()
            logger.info(f"[{request_id}] ✅ Analysis complete in {(time.time() - total_time)*1000:.0f}ms")
            
            # Force garbage collection
            gc.collect()
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"[{request_id}] Error during analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting CAD analysis service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
