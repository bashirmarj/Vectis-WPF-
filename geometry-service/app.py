# app.py - Production CAD Feature Recognition Service
# Version 2.0.0 - Clean BRepNet Implementation
# Based on: "Production-Ready CAD Feature Recognition" Research Report
# 
# Architecture:
# - BRepNet for prismatic features (89.96% accuracy)
# - Geometric fallback for turning features
# - Face-level mapping for 3D visualization
# - ONNX-optimized CPU inference
# - 50-100 files/day capacity on 4 vCPU

import os
import io
import time
import hashlib
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client

# OpenCascade imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Cylinder, GeomAbs_Plane,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
    GeomAbs_Hyperbola, GeomAbs_Parabola, GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve, GeomAbs_OffsetCurve, GeomAbs_OtherCurve
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformAbscissa
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopExp import topexp
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.Interface import Interface_Static
import math

# ========== TEMPORARY: Skip feature recognition for faster processing ==========
# Set to False to re-enable feature recognition (AAG, geometric recognition, volume decomposition)
# When True: Only tessellation + basic metrics run (5-15 seconds)
# When False: Full feature recognition runs (60+ seconds)
SKIP_FEATURE_RECOGNITION = True
# ================================================================================

# Local modules - always needed
from volume_decomposer import VolumeDecomposer
from lump_classifier import LumpClassifier
from feature_mapper import FeatureMapper

# Heavy ML modules - conditional import to avoid loading BRepNet when skipped
if not SKIP_FEATURE_RECOGNITION:
    from brepnet_wrapper import BRepNetRecognizer, FeatureType
    from geometric_fallback import TurningFeatureDetector
    from aag_pattern_engine.recognizers.fillet_chamfer_recognizer import FilletRecognizer

# === Configuration ===
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase initialized")
    except Exception as e:
        logger.error(f"❌ Supabase initialization failed: {e}")

# Initialize recognizers
brepnet_recognizer = None
turning_detector = None

if not SKIP_FEATURE_RECOGNITION:
    try:
        # Load BRepNet with pre-trained PyTorch Lightning checkpoint
        model_path = "models/pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt"
        logger.info(f"🔄 Loading BRepNet from {model_path}...")
        
        brepnet_recognizer = BRepNetRecognizer(
            model_path=model_path,
            device="cpu",  # Use CPU for production
            confidence_threshold=0.30  # Lowered from 0.70 to capture more features
        )
        logger.info("✅ BRepNet recognizer loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"❌ BRepNet model file not found: {e}")
        brepnet_recognizer = None
    except ImportError as e:
        logger.error(f"❌ BRepNet dependencies missing: {e}")
        brepnet_recognizer = None
    except Exception as e:
        logger.error(f"❌ BRepNet loading failed: {e}", exc_info=True)
        brepnet_recognizer = None

    try:
        # Geometric fallback for turning features
        turning_detector = TurningFeatureDetector(tolerance=0.001)
        logger.info("✅ Turning feature detector loaded")
    except Exception as e:
        logger.error(f"❌ Turning detector failed: {e}")
else:
    logger.info("⏭️ SKIP_FEATURE_RECOGNITION=True - Skipping BRepNet and ML model loading")


@dataclass
class ProcessingResult:
    """Standard processing result format"""
    success: bool
    correlation_id: str
    processing_time_ms: int
    mesh_data: Optional[Dict] = None
    features: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None


def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracking"""
    return f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA-256 hash for caching"""
    return hashlib.sha256(file_content).hexdigest()


def detect_step_units(step_file_path: str) -> dict:
    """
    Detect the unit system used in a STEP file.
    
    OpenCASCADE reads STEP files and can convert to its internal unit system.
    By default, OpenCASCADE uses millimeters (MM) as internal units.
    
    Args:
        step_file_path: Path to the STEP file
        
    Returns:
        dict with 'source_unit', 'target_unit', and 'scale_to_mm' keys
    """
    # Get current cascade unit setting (default is MM)
    try:
        cascade_unit = Interface_Static.CVal("xstep.cascade.unit")
    except:
        cascade_unit = "MM"
    
    # Common STEP unit codes and their scale to MM
    unit_scales_to_mm = {
        "MM": 1.0,           # Millimeters
        "M": 1000.0,         # Meters -> multiply by 1000 to get MM
        "CM": 10.0,          # Centimeters -> multiply by 10 to get MM
        "IN": 25.4,          # Inches -> multiply by 25.4 to get MM
        "FT": 304.8,         # Feet -> multiply by 304.8 to get MM
        "MICRON": 0.001,     # Micrometers -> multiply by 0.001 to get MM
    }
    
    scale = unit_scales_to_mm.get(cascade_unit.upper(), 1.0)
    
    return {
        "source_unit": cascade_unit,
        "target_unit": "MM",
        "scale_to_mm": scale
    }


def get_unit_scale_to_mm(unit_code: str) -> float:
    """
    Get the scale factor to convert from a given unit to millimeters.
    
    Args:
        unit_code: Unit code string (MM, M, CM, IN, FT, etc.)
        
    Returns:
        Scale factor to multiply values to get millimeters
    """
    unit_scales = {
        "MM": 1.0,
        "M": 1000.0,
        "CM": 10.0,
        "IN": 25.4,
        "FT": 304.8,
        "MICRON": 0.001,
        "UM": 0.001,
    }
    return unit_scales.get(unit_code.upper(), 1.0)

def tessellate_shape(shape, linear_deflection=0.005, angular_deflection=12.0) -> Dict:
    """
    Tessellate STEP shape into mesh with face-to-triangle mapping
    
    Args:
        shape: OpenCascade TopoDS_Shape
        linear_deflection: Linear deflection in meters (0.001 = 1mm)
        angular_deflection: Angular deflection in degrees (12° = professional CAD standard)
    
    Returns:
        Dict with vertices, indices, normals, and face_mapping
    """
    start = time.time()
    
    # Perform tessellation
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    
    if not mesh.IsDone():
        raise ValueError("Tessellation failed")
    
    # Extract mesh data with face mapping
    vertices = []
    indices = []
    normals = []
    face_mapping = {}  # {face_id: [triangle_indices]}
    vertex_to_face = {}  # NEW: Map vertex index to face ID
    
    global_vertex_index = 0
    global_triangle_index = 0
    vertex_map = {}  # {(x,y,z): index} for deduplication
    
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    
    while face_explorer.More():
        face = face_explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is not None:
            face_triangle_start = global_triangle_index
            transformation = location.Transformation()
            
            # Extract vertices for this face
            face_vertex_map = {}
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(transformation)
                coord = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
                
                if coord not in vertex_map:
                    vertex_map[coord] = global_vertex_index
                    vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])
                    vertex_to_face[global_vertex_index] = face_id  # NEW: Map vertex to face
                    global_vertex_index += 1
                
                face_vertex_map[i] = vertex_map[coord]
            
            # Extract triangles
            face_triangles = []
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                
                # Respect face orientation
                if face.Orientation() == 1:  # REVERSED
                    indices.extend([
                        face_vertex_map[n1],
                        face_vertex_map[n3],
                        face_vertex_map[n2]
                    ])
                else:
                    indices.extend([
                        face_vertex_map[n1],
                        face_vertex_map[n2],
                        face_vertex_map[n3]
                    ])
                
                face_triangles.append(global_triangle_index)
                global_triangle_index += 1
            
            # Store face mapping
            face_mapping[face_id] = {
                "triangle_indices": face_triangles,
                "triangle_range": [face_triangle_start, global_triangle_index - 1]
            }
        
        face_id += 1
        face_explorer.Next()
    
    # Compute per-vertex normals (average of adjacent face normals)
    normals = compute_vertex_normals(vertices, indices, len(vertices) // 3)
    
    # NEW: Convert vertex_to_face map to array
    vertex_count = len(vertices) // 3
    vertex_face_ids = [-1] * vertex_count
    for vertex_idx, face_idx in vertex_to_face.items():
        vertex_face_ids[vertex_idx] = face_idx
    
    elapsed = (time.time() - start) * 1000
    logger.info(f"Tessellated: {len(vertices)//3} vertices, {len(indices)//3} triangles, {face_id} faces in {elapsed:.1f}ms")
    logger.info(f"Created vertex_face_ids: {len(vertex_face_ids)} vertices mapped to faces")
    
    return {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "face_mapping": face_mapping,
        "vertex_face_ids": vertex_face_ids,  # NEW: Add to return dict
        "face_count": face_id,
        "triangle_count": len(indices) // 3,
        "vertex_count": len(vertices) // 3
    }


def compute_vertex_normals(vertices: List[float], indices: List[int], vertex_count: int) -> List[float]:
    """Compute smooth vertex normals from triangle data"""
    normals = np.zeros((vertex_count, 3), dtype=np.float32)
    
    # Accumulate face normals at each vertex
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
        
        v0 = np.array(vertices[i0*3:i0*3+3])
        v1 = np.array(vertices[i1*3:i1*3+3])
        v2 = np.array(vertices[i2*3:i2*3+3])
        
        # Compute face normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Accumulate at each vertex
        normals[i0] += normal
        normals[i1] += normal
        normals[i2] += normal
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms
    
    return normals.flatten().tolist()


def extract_bounding_box(shape) -> Dict:
    """Extract axis-aligned bounding box"""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    return {
        "min": [xmin, ymin, zmin],
        "max": [xmax, ymax, zmax],
        "dimensions": [xmax - xmin, ymax - ymin, zmax - zmin],
        "center": [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    }


def extract_volume_and_surface_area(shape) -> Tuple[float, float]:
    """Compute volume and surface area"""
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()
    
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    surface_area = props.Mass()
    
    return volume, surface_area


def classify_part_type(shape) -> str:
    """
    Classify part as prismatic (milling) or turning (lathe)
    Based on dominant surface types and geometry
    """
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    planar_area = 0.0
    cylindrical_area = 0.0
    total_area = 0.0
    
    while face_explorer.More():
        face = face_explorer.Current()
        surface_adaptor = BRepAdaptor_Surface(face)
        surface_type = surface_adaptor.GetType()
        
        # Compute face area
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        total_area += area
        
        if surface_type == GeomAbs_Plane:
            planar_area += area
        elif surface_type == GeomAbs_Cylinder:
            cylindrical_area += area
        
        face_explorer.Next()
    
    if total_area == 0:
        return "unknown"
    
    planar_ratio = planar_area / total_area
    cylindrical_ratio = cylindrical_area / total_area
    
    # Classification heuristic
    if cylindrical_ratio > 0.6:
        return "turning"
    elif planar_ratio > 0.5:
        return "prismatic"
    else:
        return "mixed"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "vectis-geometry-service",
        "version": "2.0.0",
        "brepnet_loaded": brepnet_recognizer is not None,
        "turning_detector_loaded": turning_detector is not None,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/analyze', methods=['POST'])
def analyze_cad():
    """
    Analyze STEP file and extract features
    
    Request:
        - file: STEP file (multipart/form-data)
        - correlation_id: Optional correlation ID for tracking
    
    Response:
        - success: bool
        - correlation_id: str
        - processing_time_ms: int
        - mesh_data: Dict with vertices, indices, normals, face_mapping
        - features: List[Dict] with detected features
        - metadata: Dict with part classification and metrics
    """
    correlation_id = request.form.get('correlation_id') or generate_correlation_id()
    start_time = time.time()
    
    logger.info(f"[{correlation_id}] Starting analysis")
    
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify(ProcessingResult(
                success=False,
                correlation_id=correlation_id,
                processing_time_ms=0,
                error="No file provided"
            ).__dict__), 400
        
        file = request.files['file']
        if not file.filename.lower().endswith(('.step', '.stp')):
            return jsonify(ProcessingResult(
                success=False,
                correlation_id=correlation_id,
                processing_time_ms=0,
                error="Only STEP files supported (.step, .stp)"
            ).__dict__), 400
        
        # Read file content
        file_content = file.read()
        file_hash = compute_file_hash(file_content)
        logger.info(f"[{correlation_id}] File hash: {file_hash}, size: {len(file_content)} bytes")
        
        # Check cache (optional - implement Redis/Supabase caching)
        #     logger.info(f"[{correlation_id}] Cache hit")
        #     return jsonify(cached_result)
        
        # Write to temporary file for OpenCascade
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Parse STEP file
            logger.info(f"[{correlation_id}] Parsing STEP file")
            reader = STEPControl_Reader()
            read_status = reader.ReadFile(tmp_path)
            
            if read_status != 1:  # IFSelect_RetDone
                raise ValueError(f"STEP read failed with status {read_status}")
            
            reader.TransferRoots()
            shape = reader.OneShape()
            
            if shape.IsNull():
                raise ValueError("Failed to extract shape from STEP file")
            
            # Detect units from STEP file
            unit_info = detect_step_units(tmp_path)
            logger.info(f"[{correlation_id}] 📏 Units detected: {unit_info['source_unit']} (scale to mm: {unit_info['scale_to_mm']})")
            
            # Extract geometric data
            logger.info(f"[{correlation_id}] Tessellating mesh")
            mesh_data = tessellate_shape(shape)
            
            # Add unit information to mesh_data
            mesh_data['unit_info'] = unit_info

            # Extract measurement edges for SolidWorks-style measurements
            logger.info(f"[{correlation_id}] 📐 Extracting measurement edges")
            try:
                measurement_edges = extract_measurement_edges(shape)
                mesh_data['tagged_edges'] = measurement_edges
                logger.info(f"[{correlation_id}] ✅ Extracted {len(measurement_edges)} measurement edges")
            except Exception as edge_error:
                logger.warning(f"[{correlation_id}] ⚠️ Measurement edge extraction failed: {edge_error}")
                mesh_data['tagged_edges'] = []

            logger.info(f"[{correlation_id}] Computing bounding box")
            bbox = extract_bounding_box(shape)
            
            logger.info(f"[{correlation_id}] Computing volume and surface area")
            volume, surface_area = extract_volume_and_surface_area(shape)
            
            logger.info(f"[{correlation_id}] Classifying part type")
            part_type = classify_part_type(shape)
            logger.info(f"[{correlation_id}] 📊 Part classified as: {part_type}")
            
            # Feature recognition
            features = []
            recognition_methods = []
            
            if SKIP_FEATURE_RECOGNITION:
                logger.info(f"[{correlation_id}] ⏭️ Feature recognition SKIPPED (SKIP_FEATURE_RECOGNITION=True)")
                recognition_methods.append("Skipped (SKIP_FEATURE_RECOGNITION=True)")
            elif part_type in ["prismatic", "mixed"] and brepnet_recognizer:
                logger.info(f"[{correlation_id}] 🤖 Running BRepNet ML feature recognition")
                try:
                    brepnet_start = time.time()
                    brepnet_features = brepnet_recognizer.recognize_features(
                        shape,
                        mesh_data["face_mapping"]
                    )
                    brepnet_time = int((time.time() - brepnet_start) * 1000)
                    features.extend(brepnet_features)
                    logger.info(f"[{correlation_id}] ✅ BRepNet found {len(brepnet_features)} features in {brepnet_time}ms")
                    recognition_methods.append(f"BRepNet ({len(brepnet_features)} features)")
                except Exception as e:
                    logger.error(f"[{correlation_id}] ❌ BRepNet failed: {e}", exc_info=True)
                    recognition_methods.append("BRepNet (failed)")
            elif part_type in ["prismatic", "mixed"]:
                logger.info(f"[{correlation_id}] ⚠️ BRepNet not available, using Hybrid Geometric Recognition")
                recognition_methods.append("Hybrid (Volume + AAG)")
                
                # === Hybrid Feature Recognition ===
                
                # 1. Build AAG
                from aag_pattern_engine.graph_builder import AAGGraphBuilder
                builder = AAGGraphBuilder(shape)
                aag = builder.build()
                
                # 2. Surface Features (Fillets/Chamfers)
                try:
                    fillet_recognizer = FilletRecognizer(aag)
                    fillet_features = fillet_recognizer.recognize()
                    features.extend(fillet_features)
                except Exception as e:
                    logger.error(f"[{correlation_id}] Surface recognition failed: {e}")


                # 3. Volumetric Features (Holes, Pockets, Steps)
                try:
                    decomposer = VolumeDecomposer()
                    decomposition_results = decomposer.decompose(shape, part_type="prismatic")
                    
                    if decomposition_results:
                        classifier = LumpClassifier()
                        mapper = FeatureMapper(shape, aag)
                        
                        classified_lumps = []
                        for lump in decomposition_results:
                            classification = classifier.classify(lump['shape'], lump['stock_bbox'])
                            lump_data = lump.copy()
                            lump_data.update(classification)
                            classified_lumps.append(lump_data)
                            
                        volumetric_features = mapper.map_features(classified_lumps)
                        features.extend(volumetric_features)
                except Exception as e:
                    logger.error(f"[{correlation_id}] Volume decomposition failed: {e}")
                    
                # Standardize
                from aag_pattern_engine.recognizers.recognizer_utils import standardize_feature_output
                features = [standardize_feature_output(f) for f in features]
            
            if not SKIP_FEATURE_RECOGNITION and part_type in ["turning", "mixed"] and turning_detector:
                logger.info(f"[{correlation_id}] Running geometric turning feature detection")
                try:
                    turning_features = turning_detector.detect_features(shape)
                    features.extend(turning_features)
                    logger.info(f"[{correlation_id}] Found {len(turning_features)} turning features")
                except Exception as e:
                    logger.error(f"[{correlation_id}] Turning detector failed: {e}")
            
            # Compile metadata
            metadata = {
                "file_hash": file_hash,
                "filename": file.filename,
                "part_type": part_type,
                "volume_mm3": volume * 1000,  # Convert to mm³
                "surface_area_mm2": surface_area * 1000,  # Convert to mm²
                "bounding_box": bbox,
                "feature_count": len(features),
                "recognition_methods": recognition_methods,
                "brepnet_available": brepnet_recognizer is not None,
                "turning_detector_available": turning_detector is not None
            }
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            result = ProcessingResult(
                success=True,
                correlation_id=correlation_id,
                processing_time_ms=elapsed_ms,
                mesh_data=mesh_data,
                features=features,
                metadata=metadata
            )
            
            logger.info(f"[{correlation_id}] ✅ Analysis complete in {elapsed_ms}ms")
            
            # Store to cache/database (optional)
            # store_to_cache(file_hash, result)
            
            return jsonify(asdict(result)), 200
        
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{correlation_id}] ❌ Error: {str(e)}", exc_info=True)
        
        return jsonify(ProcessingResult(
            success=False,
            correlation_id=correlation_id,
            processing_time_ms=elapsed_ms,
            error=str(e)
        ).__dict__), 500


@app.route('/analyze-aag', methods=['POST'])
def analyze_aag():
    """
    AAG-based geometric pattern matching feature recognition.
    Returns multi-face features using AAG Pattern Matcher.
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())
    start_time = time.time()
    
    # Capture ALL logs from ALL loggers (root logger)
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.DEBUG)  # Capture all levels
    log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    # Add to root logger to capture logs from all modules
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)  # Ensure we capture everything
    root_logger.addHandler(log_handler)
    
    logger.info(f"[{correlation_id}] ⚙️ AAG recognition request received")
    
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'correlation_id': correlation_id
            }), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({
                'success': False,
                'error': 'Empty filename',
                'correlation_id': correlation_id
            }), 400
        
        # Read file content
        file_content = file.read()
        file_hash = compute_file_hash(file_content)
        logger.info(f"[{correlation_id}] 📁 File hash: {file_hash}")
        
        # Track what succeeded for graceful degradation
        mesh_data = None
        features = []
        metadata = {}
        errors = []
        warnings = []
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # === STEP 1: Load STEP file (CRITICAL) ===
            logger.info(f"[{correlation_id}] 📂 Loading STEP file")
            reader = STEPControl_Reader()
            read_status = reader.ReadFile(tmp_path)
            
            if read_status != 1:
                raise ValueError(f"STEP read failed with status {read_status}")
            
            reader.TransferRoots()
            shape = reader.OneShape()
            
            if shape.IsNull():
                raise ValueError("Failed to extract shape from STEP file")
            
            # === STEP 2: Tessellate with face mapping ===
            logger.info(f"[{correlation_id}] 🔺 Tessellating with face mapping")
            mesh_data = tessellate_shape(shape)

            # Extract measurement edges for SolidWorks-style measurements
            logger.info(f"[{correlation_id}] 📐 Extracting measurement edges")
            try:
                measurement_edges = extract_measurement_edges(shape)
                mesh_data['tagged_edges'] = measurement_edges
                logger.info(f"[{correlation_id}] ✅ Extracted {len(measurement_edges)} measurement edges")
            except Exception as edge_error:
                logger.warning(f"[{correlation_id}] ⚠️ Measurement edge extraction failed: {edge_error}")
                mesh_data['tagged_edges'] = []

            # === STEP 3 & 4: Feature Recognition (conditionally skipped) ===
            if SKIP_FEATURE_RECOGNITION:
                logger.info(f"[{correlation_id}] ⏭️ Feature recognition SKIPPED (SKIP_FEATURE_RECOGNITION=True)")
                # No AAG building, no geometric recognition, no volume decomposition
            else:
                # === STEP 3: Build AAG (Required for Fillets & Mapping) ===
                logger.info(f"[{correlation_id}] 🕸️ Building AAG Graph")
                from aag_pattern_engine.graph_builder import AAGGraphBuilder
                builder = AAGGraphBuilder(shape)
                aag = builder.build()
                
                # === STEP 4: Feature Recognition ===
                
                # Initialize results
                decomposition_results = {}
                
                # BLOCK 1: Geometric Recognition (Holes & Fillets & Countersinks & Tapered Holes)
                try:
                    logger.info(f"[{correlation_id}] 🔍 Block 1: Geometric Recognition")
                    from aag_pattern_engine.geometric_recognizer import recognize_simple_features
                    
                    holes_geo, fillets_geo, countersinks_geo, tapered_geo = recognize_simple_features(shape)
                    
                    # Convert holes to feature format
                    for hole_info in holes_geo:
                        features.append({
                            'type': hole_info['type'],
                            'method': 'geometric',
                            'face_ids': hole_info['face_ids'],
                            'radius': hole_info['radius'],
                            'confidence': 1.0
                        })
                    
                    # Convert fillets to feature format
                    for fillet_info in fillets_geo:
                        features.append({
                            'type': 'fillet',
                            'method': 'geometric',
                            'face_ids': [fillet_info['face_id']],
                            'radius': fillet_info['radius'],
                            'confidence': 1.0
                        })
                    
                    # Convert countersinks to feature format
                    for csink_info in countersinks_geo:
                        features.append({
                            'type': 'countersink',
                            'method': 'geometric',
                            'face_ids': csink_info['face_ids'],
                            'cone_angle': csink_info['cone_angle'],
                            'hole_radius': csink_info['hole_radius'],
                            'confidence': 1.0
                        })
                    
                    # Convert tapered holes to feature format
                    for tapered_info in tapered_geo:
                        features.append({
                            'type': 'tapered_hole',
                            'method': 'geometric',
                            'face_ids': tapered_info['face_ids'],
                            'angle': tapered_info['angle'],
                            'confidence': 1.0
                        })
                    
                    logger.info(f"[{correlation_id}] ✅ Block 1: {len(holes_geo)} holes, {len(fillets_geo)} fillets, {len(countersinks_geo)} countersinks, {len(tapered_geo)} tapered")
                    
                except Exception as e:
                    logger.error(f"[{correlation_id}] ❌ Geometric recognition failed: {e}")
                    errors.append(f"Geometric recognition error: {str(e)}")
                
                # Track consumed face IDs to prevent duplicates
                consumed_face_ids = set()
                for hole_info in holes_geo:
                    consumed_face_ids.update(hole_info['face_ids'])
                for fillet_info in fillets_geo:
                    consumed_face_ids.add(fillet_info['face_id'])
                for csink_info in countersinks_geo:
                    consumed_face_ids.update(csink_info['face_ids'])
                for tapered_info in tapered_geo:
                    consumed_face_ids.update(tapered_info['face_ids'])
                
                logger.info(f"[{correlation_id}] 🔒 Consumed {len(consumed_face_ids)} face IDs from geometric recognizer")
                
                # BLOCK 2: Volume Decomposition (Pockets/Cavities)
                try:
                    from volume_decomposer import VolumeDecomposer
                    from lump_classifier import LumpClassifier
                    from feature_mapper import FeatureMapper
                    
                    decomposer = VolumeDecomposer()
                    decomposition_results = decomposer.decompose(shape, part_type="prismatic")
                    
                    if decomposition_results:
                        classifier = LumpClassifier()
                        mapper = FeatureMapper(shape, aag)
                        
                        classified_lumps = []
                        for lump in decomposition_results:
                            classification = classifier.classify(lump['shape'], lump['stock_bbox'])
                            lump_data = lump.copy()
                            lump_data.update(classification)
                            classified_lumps.append(lump_data)
                            
                        volumetric_features = mapper.map_features(classified_lumps)
                        
                        # FILTER OUT FEATURES WITH CONSUMED FACES
                        filtered_features = []
                        skipped_count = 0
                        for feature in volumetric_features:
                            feature_face_ids = set(feature.get('face_ids', []))
                            if feature_face_ids & consumed_face_ids:
                                skipped_count += 1
                                logger.debug(f"[{correlation_id}] Skipping duplicate feature with face IDs: {feature_face_ids & consumed_face_ids}")
                            else:
                                filtered_features.append(feature)
                        
                        features.extend(filtered_features)
                        
                        logger.info(f"[{correlation_id}] ✅ Found {len(filtered_features)} volumetric features ({skipped_count} duplicates filtered)")
                    else:
                        logger.warning(f"[{correlation_id}] Volume decomposition returned no features")
                        
                except Exception as e:
                    logger.error(f"[{correlation_id}] ❌ Volume decomposition failed: {e}", exc_info=True)
                    errors.append(f"Volume decomposition failed: {str(e)}")

                # Standardize output
                from aag_pattern_engine.recognizers.recognizer_utils import standardize_feature_output
                features = [standardize_feature_output(f) for f in features]
            
            # === STEP 5: Extract metadata ===
            try:
                bbox = extract_bounding_box(shape)
                volume, surface_area = extract_volume_and_surface_area(shape)
                part_type = classify_part_type(shape)
                
                metadata = {
                    "file_hash": file_hash,
                    "filename": file.filename,
                    "part_type": part_type,
                    "volume_mm3": volume * 1000,
                    "surface_area_mm2": surface_area * 1000000,
                    "bounding_box_mm": {
                        "x": bbox["dimensions"][0] * 1000,
                        "y": bbox["dimensions"][1] * 1000,
                        "z": bbox["dimensions"][2] * 1000
                    },
                    "recognition_method": "Hybrid (Volume + AAG)",
                    "total_features": len(features),
                    "multi_face_features": sum(1 for f in features if len(f.get('face_ids', [])) > 1)
                }
            except Exception as e:
                logger.warning(f"[{correlation_id}] ⚠️ Metadata extraction failed: {e}")
                warnings.append(f"Metadata extraction failed: {str(e)}")
                metadata = {
                    "file_hash": file_hash,
                    "filename": file.filename,
                    "recognition_method": "Hybrid (Volume + AAG)",
                    "total_features": len(features)
                }
            
            # === STEP 6: Build response ===
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            success = len(errors) == 0
            status = "success" if success else "partial_success" if mesh_data else "failure"
            
            result = ProcessingResult(
                success=success,
                correlation_id=correlation_id,
                processing_time_ms=processing_time_ms,
                mesh_data=mesh_data,
                features=features,
                metadata=metadata
            )
            
            response_data = asdict(result)
            
            if errors:
                response_data['errors'] = errors
            if warnings:
                response_data['warnings'] = warnings
            
            response_data['status'] = status
            
            logger.info(f"[{correlation_id}] 🎉 AAG analysis complete in {processing_time_ms}ms")
            logger.info(f"[{correlation_id}] 📊 Status: {status}, Features: {len(features)}, Errors: {len(errors)}, Warnings: {len(warnings)}")
            
            if mesh_data:
                logger.info(f"[{correlation_id}] ✅ Mesh data available: {len(mesh_data['vertices'])//3} vertices")
            
            # Capture logs and add to response
            root_logger = logging.getLogger()
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(original_level)  # Restore original level
            captured_logs = log_stream.getvalue()
            response_data['processing_logs'] = captured_logs
            
            return jsonify(response_data), 200 if success else 206
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{correlation_id}] ❌ Unexpected error: {e}", exc_info=True)
        
        # Capture logs even on error
        root_logger = logging.getLogger()
        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)  # Restore original level
        captured_logs = log_stream.getvalue()
        
        response = {
            'success': False,
            'status': 'partial_success' if mesh_data else 'failure',
            'error': str(e),
            'correlation_id': correlation_id,
            'processing_time_ms': processing_time_ms,
            'errors': [str(e)],
            'processing_logs': captured_logs
        }
        
        if mesh_data:
            response['mesh_data'] = mesh_data
            response['features'] = features
            response['metadata'] = metadata or {}
            logger.info(f"[{correlation_id}] ⚠️ Returning mesh despite error")
        
        return jsonify(response), 206 if mesh_data else 500


# =============================================================================
# MEASUREMENT EDGE EXTRACTION - SolidWorks-style analytical edge data
# This function extracts B-Rep edge geometry for precise measurements
# Added: December 2024
# =============================================================================

def extract_measurement_edges(shape, num_discretization_points: int = 24) -> List[Dict]:
    """
    Extract B-Rep edges with analytical geometry data for SolidWorks-style measurements.
    Includes adjacent face normals for visibility analysis.

    This provides:
    - Edge type classification (line, circle, arc, ellipse, etc.)
    - Analytical properties (center, radius, diameter for circles)
    - Discretized points for rendering curved edges
    - Length measurements
    - Adjacent face normals (for visibility culling)

    Coordinates are in METERS to match mesh vertices from tessellate_shape().
    Measurement values (length, radius, diameter) are in MM for display.

    Args:
        shape: OpenCascade TopoDS_Shape
        num_discretization_points: Number of points for curve discretization (default 24)

    Returns:
        List of edge dictionaries with analytical and rendering data
    """
    extraction_start = time.time()
    edges_data = []
    processed_edge_hashes = set()

    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edge_index = 0

    while edge_explorer.More():
        edge = edge_explorer.Current()

        # Deduplicate edges (shared edges between faces)
        edge_hash = edge.HashCode(2147483647)
        if edge_hash in processed_edge_hashes:
            edge_explorer.Next()
            continue
        processed_edge_hashes.add(edge_hash)

        try:
            # Create curve adaptor for edge analysis
            curve_adaptor = BRepAdaptor_Curve(edge)

            # Get curve parameters
            first_param = curve_adaptor.FirstParameter()
            last_param = curve_adaptor.LastParameter()

            # Skip degenerate edges
            if abs(last_param - first_param) < 1e-10:
                edge_explorer.Next()
                continue

            # Get curve type
            curve_type = curve_adaptor.GetType()

            # Get start and end points
            # Note: OpenCASCADE uses millimeters (MM) as default internal units
            start_pnt = curve_adaptor.Value(first_param)
            end_pnt = curve_adaptor.Value(last_param)

            # Compute edge length (OpenCASCADE uses MM internally)
            props = GProp_GProps()
            brepgprop.LinearProperties(edge, props)
            length_mm = props.Mass()  # Already in mm (OpenCASCADE default unit)

            # Determine curve type string
            type_map = {
                GeomAbs_Line: "line",
                GeomAbs_Circle: "circle",
                GeomAbs_Ellipse: "ellipse",
                GeomAbs_Hyperbola: "hyperbola",
                GeomAbs_Parabola: "parabola",
                GeomAbs_BezierCurve: "bezier",
                GeomAbs_BSplineCurve: "bspline",
                GeomAbs_OffsetCurve: "offset",
            }
            type_str = type_map.get(curve_type, "other")

            # Check if edge is closed
            is_closed = start_pnt.Distance(end_pnt) < 1e-6

            # Build edge data (coordinates in METERS to match mesh)
            edge_data = {
                'feature_id': edge_index,
                'type': type_str,
                'length': round(length_mm, 4),  # mm for display
                'is_closed': is_closed,
                'start': [
                    round(start_pnt.X(), 8),  # mm (OpenCASCADE default unit)
                    round(start_pnt.Y(), 8),
                    round(start_pnt.Z(), 8)
                ],
                'end': [
                    round(end_pnt.X(), 8),  # mm (OpenCASCADE default unit)
                    round(end_pnt.Y(), 8),
                    round(end_pnt.Z(), 8)
                ]
            }

            # === ANALYTICAL DATA FOR SPECIFIC CURVE TYPES ===

            if curve_type == GeomAbs_Circle:
                circle = curve_adaptor.Circle()
                center = circle.Location()
                axis = circle.Axis().Direction()
                radius_mm = circle.Radius()  # Already in mm (OpenCASCADE default unit)

                edge_data['center'] = [
                    round(center.X(), 8),  # mm
                    round(center.Y(), 8),
                    round(center.Z(), 8)
                ]
                edge_data['radius'] = round(radius_mm, 4)  # mm
                edge_data['diameter'] = round(radius_mm * 2, 4)  # mm
                edge_data['normal'] = [
                    round(axis.X(), 6),
                    round(axis.Y(), 6),
                    round(axis.Z(), 6)
                ]

                # Check if full circle (2π sweep)
                sweep_angle = abs(last_param - first_param)
                edge_data['is_full_circle'] = sweep_angle >= (2 * math.pi - 0.01)
                edge_data['sweep_angle'] = round(sweep_angle, 6)

                if not edge_data['is_full_circle']:
                    # Arc - compute arc length
                    edge_data['arc_length'] = round(radius_mm * sweep_angle, 4)

            elif curve_type == GeomAbs_Ellipse:
                ellipse = curve_adaptor.Ellipse()
                center = ellipse.Location()

                edge_data['center'] = [
                    round(center.X(), 8),
                    round(center.Y(), 8),
                    round(center.Z(), 8)
                ]
                edge_data['major_radius'] = round(ellipse.MajorRadius(), 4)  # mm (OpenCASCADE default unit)
                edge_data['minor_radius'] = round(ellipse.MinorRadius(), 4)  # mm (OpenCASCADE default unit)

            elif curve_type == GeomAbs_Line:
                line = curve_adaptor.Line()
                direction = line.Direction()

                edge_data['direction'] = [
                    round(direction.X(), 6),
                    round(direction.Y(), 6),
                    round(direction.Z(), 6)
                ]

            # === DISCRETIZED POINTS ===
            # Generate points along the curve for proper visualization

            try:
                # For lines, just use start and end
                if curve_type == GeomAbs_Line:
                    edge_data['snap_points'] = [edge_data['start'], edge_data['end']]

                else:
                    # For curves, discretize with uniform spacing
                    discretizer = GCPnts_QuasiUniformAbscissa(curve_adaptor, num_discretization_points)

                    if discretizer.IsDone() and discretizer.NbPoints() >= 2:
                        snap_points = []
                        
                        for i in range(1, discretizer.NbPoints() + 1):
                            param = discretizer.Parameter(i)
                            pnt = curve_adaptor.Value(param)
                            snap_points.append([
                                round(pnt.X(), 8),
                                round(pnt.Y(), 8),
                                round(pnt.Z(), 8)
                            ])
                        
                        edge_data['snap_points'] = snap_points
                    else:
                        # Fallback to start/end
                        edge_data['snap_points'] = [edge_data['start'], edge_data['end']]
                        
            except Exception as disc_error:
                logger.warning(f"Edge {edge_index} discretization failed: {disc_error}")
                edge_data['snap_points'] = [edge_data['start'], edge_data['end']]

            edges_data.append(edge_data)
            edge_index += 1

        except Exception as edge_error:
            logger.warning(f"Failed to process edge {edge_index}: {edge_error}")

        edge_explorer.Next()

    elapsed_ms = (time.time() - extraction_start) * 1000

    logger.info(f"📐 Extracted {len(edges_data)} measurement edges in {elapsed_ms:.1f}ms")

    return edges_data


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Vectis Geometry Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
