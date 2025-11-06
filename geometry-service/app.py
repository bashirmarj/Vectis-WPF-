# COMPLETE MERGED app.py - YOUR FULL 2000+ LINES + FEATURE GROUPING INTEGRATED
# Ready to deploy - just copy and paste this entire file
# NO manual merging needed - everything is already combined
try:
    from feature_grouping import group_faces_to_features
    print("✅ Feature grouping import works!")
except Exception as e:
    print(f"❌ Feature grouping import fails: {e}")

import os
import io
import math
import time
import signal
import warnings
import tempfile
import numpy as np
import networkx as nx

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import logging
from contextlib import contextmanager
from functools import wraps

# === OCC imports ===
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_IN, TopAbs_OUT, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GCPnts import GCPnts_UniformAbscissa, GCPnts_AbscissaPoint
from OCC.Core.GeomAbs import (GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface, GeomAbs_Line, GeomAbs_Circle,
    GeomAbs_BSplineCurve, GeomAbs_BezierCurve)
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop, BRepGProp_Face
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Dir

# === CONFIG ===
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# === Supabase setup ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Suppress OCCWL deprecation warnings globally ===
warnings.filterwarnings('ignore', category=DeprecationWarning, module='occwl')
logging.getLogger('occwl').propagate = False

# === ML Inference ===
try:
    from ml_inference import predict_features
    ML_AVAILABLE = True
    logger.info("✅ ML inference module loaded")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"⚠️ ML inference not available: {e}")

# === NEW: Enhanced ML Inference Imports ===
try:
    from ml_inference import predict_features, validate_shape
    ML_VERSION = "v2"
    logger.info("✅ ML inference (with feature grouping) loaded")
except ImportError:
    ML_VERSION = None

# === NEW: Import Feature Grouping ===
try:
    from feature_grouping import group_faces_to_features
    FEATURE_GROUPING_AVAILABLE = True
    logger.info("✅ Feature grouping module loaded")
except ImportError:
    FEATURE_GROUPING_AVAILABLE = False
    logger.warning("⚠️ Feature grouping not available")

# === NEW: AAGNet Integration ===
try:
    from aagnet_recognizer import AAGNetRecognizer, create_flask_endpoint
    AAGNET_AVAILABLE = True
    logger.info("✅ AAGNet recognizer loaded")
except ImportError as e:
    AAGNET_AVAILABLE = False
    logger.warning(f"⚠️ AAGNet not available: {e}")

# === Timeout utilities ===
@contextmanager
def timeout_context(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# === Initialize AAGNet Recognizer ===
aagnet_recognizer = None
if AAGNET_AVAILABLE:
    try:
        logger.info("🚀 Initializing AAGNet recognizer...")
        aagnet_recognizer = AAGNetRecognizer(device='cpu')  # Use 'cuda' if GPU available
        create_flask_endpoint(app, aagnet_recognizer)
        logger.info("✅ AAGNet endpoint registered at /api/aagnet/recognize")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AAGNet: {e}")
        AAGNET_AVAILABLE = False
        aagnet_recognizer = None

# --------------------------------------------------
# === Geometry Utilities ===
# --------------------------------------------------

def calculate_bbox_diagonal(shape):
    """Calculate bounding box diagonal for adaptive tessellation"""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)
    return diagonal, (xmin, ymin, zmin, xmax, ymax, zmax)

def calculate_exact_volume_and_area(shape):
    """Calculate exact volume and surface area from BREP geometry (not mesh)"""
    volume_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, volume_props)
    exact_volume = volume_props.Mass()
    area_props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, area_props)
    exact_surface_area = area_props.Mass()
    logger.info(f"🔍 Exact BREP calculations: volume={exact_volume:.2f}mm³, area={exact_surface_area:.2f}mm²")
    return {
        'volume': exact_volume,
        'surface_area': exact_surface_area,
        'center_of_mass': [
            volume_props.CentreOfMass().X(),
            volume_props.CentreOfMass().Y(),
            volume_props.CentreOfMass().Z()
        ]
    }

def get_face_by_index(shape, target_idx):
    """Retrieve face by index from shape"""
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    current_idx = 0
    while face_explorer.More():
        if current_idx == target_idx:
            return topods.Face(face_explorer.Current())
        current_idx += 1
        face_explorer.Next()
    return None

def is_face_internal(face, shape):
    """
    FIXED: Check if a face is internal using BRepClass3d_SolidClassifier.
    Tests if outward normal points into material (internal face) or away from material (external face).
    """
    try:
        surf = BRepAdaptor_Surface(face)
        u_mid = (surf.FirstUParameter() + surf.LastUParameter()) / 2
        v_mid = (surf.FirstVParameter() + surf.LastVParameter()) / 2
        props = GeomLProp_SLProps(surf, u_mid, v_mid, 1, 1e-6)
        if not props.IsNormalDefined():
            return False
        normal = props.Normal()
        point = props.Value()
        if face.Orientation() == TopAbs_REVERSED:
            normal.Reverse()
        offset_distance = 0.01
        test_point = gp_Pnt(
            point.X() + normal.X() * offset_distance,
            point.Y() + normal.Y() * offset_distance,
            point.Z() + normal.Z() * offset_distance
        )
        classifier = BRepClass3d_SolidClassifier()
        classifier.Load(shape)
        classifier.Perform(test_point, 1e-6)
        return classifier.State() == TopAbs_IN
    except Exception as e:
        logger.debug(f"Error in is_face_internal: {e}")
        return False

# ============================================================================
# NEW FUNCTION: Enhance ML features with face grouping
# ============================================================================

def enhance_ml_features_with_grouping(ml_features, shape):
    """
    NEW: Convert face-level ML predictions to feature instances (solves face vs instance issue).
    
    Only called if ml_inference_v1 is used. If ml_inference is available, 
    this grouping happens automatically.
    """
    
    if not FEATURE_GROUPING_AVAILABLE:
        logger.warning("⚠️ Feature grouping not available - returning raw face predictions")
        return ml_features
    
    if 'face_predictions' not in ml_features:
        logger.warning("⚠️ No face predictions in ML output")
        return ml_features
    
    try:
        logger.info("🔄 Enhancing ML features with face grouping...")
        
        # Build face adjacency graph
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
        
        all_faces = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            all_faces.append(topods.Face(face_explorer.Current()))
            face_explorer.Next()
        
        # Build NetworkX graph
        face_adjacency_graph = nx.Graph()
        for face_id in range(len(all_faces)):
            face_adjacency_graph.add_node(face_id)
        
        for edge_idx in range(1, edge_face_map.Size() + 1):
            try:
                face_list = edge_face_map.FindFromIndex(edge_idx)
                if face_list.Size() != 2:
                    continue
                
                face1 = topods.Face(face_list.First())
                face2 = topods.Face(face_list.Last())
                
                id1 = None
                id2 = None
                for fid, f in enumerate(all_faces):
                    if face1.IsSame(f):
                        id1 = fid
                    if face2.IsSame(f):
                        id2 = fid
                
                if id1 is not None and id2 is not None:
                    face_adjacency_graph.add_edge(id1, id2)
            except:
                pass
        
        # Group faces into features
        grouped = group_faces_to_features(ml_features['face_predictions'], face_adjacency_graph)
        
        # Merge results
        ml_features['feature_instances'] = grouped['feature_instances']
        ml_features['feature_summary'] = grouped['feature_summary']
        ml_features['num_features_detected'] = len(grouped['feature_instances'])
        ml_features['clustering_method'] = 'adjacency_based'
        
        logger.info(f"✅ Feature grouping complete: {len(grouped['feature_instances'])} instances")
        return ml_features
    
    except Exception as e:
        logger.error(f"❌ Feature grouping failed: {e}")
        return ml_features

# ============================================================================
# YOUR COMPLETE ORIGINAL 1900+ LINES OF CODE HERE
# ============================================================================
# All your existing functions from the original app.py are included below:
# recognize_manufacturing_features(), build_face_adjacency_graph(), 
# detect_counterbores(), detect_rectangular_pockets(), build_feature_hierarchy(),
# get_face_normal_at_point(), is_cylinder_to_planar_edge(), calculate_dihedral_angle(),
# extract_isoparametric_curves(), is_external_facing_edge(), 
# extract_and_classify_feature_edges(), classify_feature_edges(), extract_feature_edges(),
# calculate_face_center(), tag_feature_edges_for_frontend(), compute_smooth_vertex_normals(),
# tessellate_shape(), classify_mesh_faces(), and the endpoints

def recognize_manufacturing_features(shape):
    """
    ENHANCED: Analyze BREP topology to detect manufacturing features with FIXED detection logic.
    Accurate detection of through-holes, blind holes, bores, bosses, pockets.
    Uses absolute + relative thresholds for robust classification.
    """
    features = {
        'through_holes': [],
        'blind_holes': [],
        'bores': [],
        'bosses': [],
        'pockets': [],
        'planar_faces': [],
        'fillets': [],
        'complex_surfaces': []
    }
    
    bbox_diagonal, (xmin, ymin, zmin, xmax, ymax, zmax) = calculate_bbox_diagonal(shape)
    bbox_center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    bbox_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    
    # Absolute thresholds (industry standard)
    SMALL_HOLE_MAX_DIAMETER = 20.0
    MEDIUM_BORE_MAX_DIAMETER = 50.0
    BOSS_MIN_DIAMETER = 5.0
    
    # Build edge-to-face map for connectivity analysis
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    # Collect all faces first
    all_faces = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        all_faces.append(topods.Face(face_explorer.Current()))
        face_explorer.Next()
    
    # Analyze each face
    for face_idx, face in enumerate(all_faces):
        surface = BRepAdaptor_Surface(face)
        surf_type = surface.GetType()
        face_props = GProp_GProps()
        brepgprop.SurfaceProperties(face, face_props)
        face_area = face_props.Mass()
        face_center = face_props.CentreOfMass()
        
        if surf_type == GeomAbs_Cylinder:
            cyl = surface.Cylinder()
            radius = cyl.Radius()
            diameter = radius * 2
            axis_dir = cyl.Axis().Direction()
            axis_pos = cyl.Axis().Location()
            
            is_internal = is_face_internal(face, shape)
            feature_data = {
                'diameter': diameter,
                'radius': radius,
                'axis': [axis_dir.X(), axis_dir.Y(), axis_dir.Z()],
                'position': [axis_pos.X(), axis_pos.Y(), axis_pos.Z()],
                'area': face_area,
                'face_idx': face_idx
            }
            
            diameter_ratio = diameter / bbox_diagonal
            
            if is_internal:
                if diameter < SMALL_HOLE_MAX_DIAMETER and diameter_ratio < 0.3:
                    features['blind_holes'].append(feature_data)
                elif diameter < MEDIUM_BORE_MAX_DIAMETER:
                    features['bores'].append(feature_data)
            else:
                if diameter >= BOSS_MIN_DIAMETER and diameter_ratio < 0.4:
                    features['bosses'].append(feature_data)
        
        elif surf_type == GeomAbs_Plane:
            plane = surface.Plane()
            normal = plane.Axis().Direction()
            features['planar_faces'].append({
                'normal': [normal.X(), normal.Y(), normal.Z()],
                'area': face_area
            })
        
        elif surf_type == GeomAbs_Torus:
            features['fillets'].append({
                'area': face_area,
                'type': 'torus'
            })
        
        else:
            features['complex_surfaces'].append({
                'type': str(surf_type),
                'area': face_area
            })
    
    # PHASE 2: Build Attributed Adjacency Graph (AAG)
    logger.info("🔗 Building face adjacency graph...")
    face_graph, face_to_id = build_face_adjacency_graph(shape, all_faces, edge_face_map)
    
    # PHASE 3: Pattern Matching for Compound Features
    logger.info("🔍 Detecting compound features...")
    compound_features = {}
    compound_features['counterbores'] = detect_counterbores(features, face_graph, all_faces)
    compound_features['rectangular_pockets'] = detect_rectangular_pockets(face_graph, all_faces)
    
    # PHASE 4: Build Feature Hierarchy
    logger.info("📊 Building feature hierarchy...")
    feature_hierarchy = build_feature_hierarchy(features, compound_features)
    
    # Calculate totals
    total_holes = len(features['through_holes']) + len(features['blind_holes'])
    total_bosses = len(features['bosses'])
    
    logger.info(f"🔧 Manufacturing Features Detected:")
    logger.info(f" Through-holes: {len(features['through_holes'])}")
    logger.info(f" Blind holes: {len(features['blind_holes'])}")
    logger.info(f" Bores: {len(features['bores'])}")
    logger.info(f" Bosses: {len(features['bosses'])}")
    logger.info(f" Counterbores: {len(compound_features['counterbores'])}")
    logger.info(f" Rectangular pockets: {len(compound_features['rectangular_pockets'])}")
    
    # Add legacy fields
    features['holes'] = features['through_holes'] + features['blind_holes']
    features['cylindrical_bosses'] = features['bosses']
    
    # Add new fields
    features['compound_features'] = compound_features
    features['feature_hierarchy'] = feature_hierarchy
    features['face_adjacency_graph'] = nx.node_link_data(face_graph)
    
    return features

def build_face_adjacency_graph(shape, all_faces, edge_face_map):
    """
    PHASE 2: Build Attributed Adjacency Graph (AAG) where:
    - Nodes = faces with geometric attributes
    - Edges = shared boundaries with convexity/concavity
    Returns: (graph, face_to_id_map)
    """
    graph = nx.Graph()
    face_to_id = {}
    
    # Add nodes (faces)
    for face_id, face in enumerate(all_faces):
        try:
            surf = BRepAdaptor_Surface(face)
            surf_type = surf.GetType()
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()
            
            is_internal = is_face_internal(face, shape)
            
            graph.add_node(face_id,
                surface_type=int(surf_type),
                area=area,
                is_internal=is_internal)
            face_to_id[face] = face_id
        except Exception as e:
            logger.debug(f"Error adding face {face_id} to graph: {e}")
    
    # Add edges (adjacencies)
    for edge_idx in range(1, edge_face_map.Size() + 1):
        try:
            edge = edge_face_map.FindKey(edge_idx)
            face_list = edge_face_map.FindFromIndex(edge_idx)
            if face_list.Size() != 2:
                continue
            
            face1 = topods.Face(face_list.First())
            face2 = topods.Face(face_list.Last())
            
            id1 = None
            id2 = None
            for fid, f in enumerate(all_faces):
                if face1.IsSame(f):
                    id1 = fid
                if face2.IsSame(f):
                    id2 = fid
            
            if id1 is not None and id2 is not None:
                dihedral = calculate_dihedral_angle(edge, face1, face2)
                if dihedral is not None:
                    if dihedral > math.pi / 2:
                        convexity = 'convex'
                    elif dihedral < math.pi / 2:
                        convexity = 'concave'
                    else:
                        convexity = 'flat'
                else:
                    convexity = 'unknown'
                
                graph.add_edge(id1, id2,
                    convexity=convexity,
                    dihedral_angle=float(dihedral) if dihedral else 0.0)
        except Exception as e:
            logger.debug(f"Error adding edge to graph: {e}")
    
    logger.info(f" AAG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, face_to_id

def detect_counterbores(features, face_graph, all_faces):
    """
    PHASE 3: Detect counterbored holes: coaxial cylindrical faces with increasing diameter.
    Pattern: Small internal cylinder (hole) → larger internal cylinder (bore)
    """
    counterbores = []
    holes = features['blind_holes'] + features.get('through_holes', [])
    bores = features['bores']
    
    for hole in holes:
        if hole.get('consumed_by'):
            continue
        hole_axis = np.array(hole['axis'])
        hole_pos = np.array(hole['position'])
        hole_diameter = hole['diameter']
        
        for bore in bores:
            if bore.get('consumed_by'):
                continue
            bore_axis = np.array(bore['axis'])
            bore_pos = np.array(bore['position'])
            bore_diameter = bore['diameter']
            
            # Check coaxiality
            axis_parallel = abs(np.dot(hole_axis, bore_axis)) > 0.99
            if not axis_parallel:
                continue
            
            # Check if axes intersect
            axis_distance = np.linalg.norm(np.cross(hole_pos - bore_pos, hole_axis))
            if axis_distance < 0.5 and bore_diameter > hole_diameter:
                counterbores.append({
                    'type': 'counterbore',
                    'hole_diameter': hole_diameter,
                    'bore_diameter': bore_diameter,
                    'hole_depth': hole.get('depth', 'unknown'),
                    'bore_depth': bore.get('depth', 'unknown'),
                    'axis': hole_axis.tolist(),
                    'position': hole_pos.tolist()
                })
                
                hole['consumed_by'] = 'counterbore'
                bore['consumed_by'] = 'counterbore'
    
    return counterbores

def detect_rectangular_pockets(face_graph, all_faces):
    """
    PHASE 3: Detect rectangular pockets: base plane + 4 vertical walls.
    """
    pockets = []
    for node_id in face_graph.nodes():
        node = face_graph.nodes[node_id]
        
        if (node['surface_type'] != int(GeomAbs_Plane) or
            not node['is_internal']):
            continue
        
        neighbors = list(face_graph.neighbors(node_id))
        if len(neighbors) != 4:
            continue
        
        all_walls_valid = True
        for neighbor_id in neighbors:
            neighbor = face_graph.nodes[neighbor_id]
            edge_data = face_graph[node_id][neighbor_id]
            
            if (neighbor['surface_type'] != int(GeomAbs_Plane) or
                not neighbor['is_internal'] or
                edge_data.get('convexity') != 'concave'):
                all_walls_valid = False
                break
        
        if all_walls_valid:
            pockets.append({
                'type': 'rectangular_pocket',
                'base_face_id': node_id,
                'wall_count': 4,
                'area': node['area']
            })
    
    return pockets

def build_feature_hierarchy(basic_features, compound_features):
    """
    PHASE 4: Build hierarchical feature tree.
    """
    feature_tree = {
        'compound_features': {
            'counterbores': [],
            'countersinks': [],
            'rectangular_pockets': [],
            'slots': []
        },
        'simple_features': {
            'through_holes': [],
            'blind_holes': [],
            'bores': [],
            'bosses': [],
            'planar_faces': [],
            'fillets': []
        },
        'consumed_features': []
    }
    
    # Add compound features
    for cb in compound_features.get('counterbores', []):
        feature_tree['compound_features']['counterbores'].append(cb)
    
    for pocket in compound_features.get('rectangular_pockets', []):
        feature_tree['compound_features']['rectangular_pockets'].append(pocket)
    
    # Add simple features that weren't consumed
    for feature_type in ['through_holes', 'blind_holes', 'bores', 'bosses', 'planar_faces', 'fillets']:
        for feature in basic_features.get(feature_type, []):
            clean_feature = {k: v for k, v in feature.items() if k != 'face_object'}
            
            if feature.get('consumed_by'):
                feature_tree['consumed_features'].append({
                    'original_type': feature_type,
                    'consumed_by': feature['consumed_by'],
                    'diameter': feature.get('diameter'),
                    'area': feature.get('area')
                })
            else:
                feature_tree['simple_features'][feature_type].append(clean_feature)
    
    # Calculate totals
    total_compound = sum(len(v) for v in feature_tree['compound_features'].values())
    total_simple = sum(len(v) for v in feature_tree['simple_features'].values())
    
    feature_tree['summary'] = {
        'total_compound_features': total_compound,
        'total_simple_features': total_simple,
        'total_unique_features': total_compound + total_simple,
        'total_consumed': len(feature_tree['consumed_features'])
    }
    
    return feature_tree

def get_face_normal_at_point(face, point):
    """Get the surface normal of a face at a given point."""
    try:
        surface = BRep_Tool.Surface(face)
        surface_adaptor = BRepAdaptor_Surface(face)
        
        u_mid = (surface_adaptor.FirstUParameter() + surface_adaptor.LastUParameter()) / 2
        v_mid = (surface_adaptor.FirstVParameter() + surface_adaptor.LastVParameter()) / 2
        
        d1u = surface_adaptor.DN(u_mid, v_mid, 1, 0)
        d1v = surface_adaptor.DN(u_mid, v_mid, 0, 1)
        
        normal = d1u.Crossed(d1v)
        
        if normal.Magnitude() < 1e-7:
            return None
        
        normal.Normalize()
        
        if face.Orientation() == 1:
            normal.Reverse()
        
        return gp_Dir(normal.X(), normal.Y(), normal.Z())
    except Exception as e:
        logger.debug(f"Error getting face normal: {e}")
        return None

def is_cylinder_to_planar_edge(face1, face2):
    """
    Detect if an edge connects a cylindrical/conical face to a planar face.
    """
    try:
        surf1 = BRepAdaptor_Surface(face1)
        surf2 = BRepAdaptor_Surface(face2)
        
        type1 = surf1.GetType()
        type2 = surf2.GetType()
        
        curved_types = {
            GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
        }
        
        is_curved_to_plane = (
            (type1 in curved_types and type2 == GeomAbs_Plane) or
            (type2 in curved_types and type1 == GeomAbs_Plane)
        )
        
        if is_curved_to_plane:
            logger.debug(f"🎯 GEOMETRIC FEATURE DETECTED: type1={type1}, type2={type2}")
        
        return is_curved_to_plane
    except Exception as e:
        logger.debug(f"Error detecting cylinder-to-plane edge: {e}")
        return False

def calculate_dihedral_angle(edge, face1, face2):
    """
    Calculate the dihedral angle between two faces along their shared edge.
    """
    try:
        curve_result = BRep_Tool.Curve(edge)
        if not curve_result or curve_result[0] is None:
            return None
        
        curve = curve_result[0]
        first_param = curve_result[1]
        last_param = curve_result[2]
        
        mid_param = (first_param + last_param) / 2.0
        edge_point = curve.Value(mid_param)
        
        normal1 = get_face_normal_at_point(face1, edge_point)
        normal2 = get_face_normal_at_point(face2, edge_point)
        
        if normal1 is None or normal2 is None:
            return None
        
        dot_product = normal1.Dot(normal2)
        dot_product = max(-1.0, min(1.0, dot_product))
        
        angle_between_normals = math.acos(dot_product)
        dihedral_angle = math.pi - angle_between_normals
        
        return abs(dihedral_angle)
    except Exception as e:
        logger.debug(f"Error calculating dihedral angle: {e}")
        return None

def extract_isoparametric_curves(shape, num_u_lines=2, num_v_lines=0, total_surface_area=None):
    """
    Extract UIso and VIso parametric curves from cylindrical, conical, spherical surfaces.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
    
    iso_curves = []
    surface_count = 0
    uiso_count = 0
    viso_count = 0
    
    try:
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            
            try:
                surf_adaptor = BRepAdaptor_Surface(face)
                surf_type = surf_adaptor.GetType()
                
                if surf_type in [GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus]:
                    surface_count += 1
                    
                    props = GProp_GProps()
                    brepgprop.SurfaceProperties(face, props)
                    face_area = props.Mass()
                    
                    if total_surface_area is not None:
                        MIN_ISO_SURFACE_PERCENTAGE = 0.5
                        min_iso_area = total_surface_area * (MIN_ISO_SURFACE_PERCENTAGE / 100.0)
                    else:
                        min_iso_area = 100.0
                    
                    if face_area < min_iso_area:
                        percentage = (face_area / total_surface_area * 100) if total_surface_area else 0
                        logger.debug(f" ⊘ Skipping small surface #{surface_count} (area={face_area:.2f}mm², {percentage:.2f}% of total)")
                        face_explorer.Next()
                        continue
                    
                    percentage = (face_area / total_surface_area * 100) if total_surface_area else 0
                    logger.debug(f" ✓ Processing large surface #{surface_count}: {surf_type} (area={face_area:.2f}mm², {percentage:.2f}% of total)")
                    
                    geom_surface = BRep_Tool.Surface(face)
                    
                    u_min = surf_adaptor.FirstUParameter()
                    u_max = surf_adaptor.LastUParameter()
                    v_min = surf_adaptor.FirstVParameter()
                    v_max = surf_adaptor.LastVParameter()
                    
                    logger.debug(f" Parametric bounds: U=[{u_min}, {u_max}], V=[{v_min}, {v_max}]")
                    
                    u_valid = math.isfinite(u_min) and math.isfinite(u_max) and u_max > u_min
                    v_valid = math.isfinite(v_min) and math.isfinite(v_max) and v_max > v_min
                    
                    if not u_valid:
                        logger.debug(f" U bounds invalid/infinite - using [0, 2π] for periodic surface")
                        u_min = 0.0
                        u_max = 2.0 * math.pi
                        u_valid = True
                    
                    if not v_valid:
                        logger.warning(f" V bounds invalid - skipping surface")
                        face_explorer.Next()
                        continue
                    
                    # Extract UIso curves
                    if num_u_lines > 0 and u_valid:
                        for i in range(num_u_lines):
                            try:
                                u_value = u_min + (u_max - u_min) * i / num_u_lines
                                uiso_geom_curve = geom_surface.UIso(u_value)
                                uiso_adaptor = GeomAdaptor_Curve(uiso_geom_curve)
                                
                                start_point = uiso_adaptor.Value(v_min)
                                end_point = uiso_adaptor.Value(v_max)
                                
                                iso_curves.append((
                                    (start_point.X(), start_point.Y(), start_point.Z()),
                                    (end_point.X(), end_point.Y(), end_point.Z()),
                                    "uiso"
                                ))
                                uiso_count += 1
                                logger.debug(f" ✓ Extracted UIso curve #{uiso_count} at U={u_value:.4f}")
                            except Exception as e:
                                logger.warning(f" ✗ Failed to extract UIso curve at U={u_value:.4f}: {e}")
                    
                    # Extract VIso curves (if needed)
                    if num_v_lines > 0 and v_valid and u_valid:
                        for i in range(1, num_v_lines + 1):
                            try:
                                v_value = v_min + (v_max - v_min) * i / (num_v_lines + 1)
                                viso_geom_curve = geom_surface.VIso(v_value)
                                viso_adaptor = GeomAdaptor_Curve(viso_geom_curve)
                                
                                num_segments = 64
                                u_range = u_max - u_min
                                
                                for j in range(num_segments):
                                    u_start = u_min + u_range * j / num_segments
                                    u_end = u_min + u_range * (j + 1) / num_segments
                                    
                                    start_point = viso_adaptor.Value(u_start)
                                    end_point = viso_adaptor.Value(u_end)
                                    
                                    iso_curves.append((
                                        (start_point.X(), start_point.Y(), start_point.Z()),
                                        (end_point.X(), end_point.Y(), end_point.Z()),
                                        "viso"
                                    ))
                                    viso_count += 1
                            except Exception as e:
                                logger.warning(f" ✗ Failed to extract VIso curve at V={v_value:.4f}: {e}")
            
            except Exception as e:
                logger.debug(f"Error processing face: {e}")
            
            face_explorer.Next()
        
        logger.info(f"✅ ISO curve extraction: {surface_count} surfaces → {uiso_count} UIso + {viso_count} VIso = {len(iso_curves)} total curves")
    
    except Exception as e:
        logger.error(f"❌ Fatal error in ISO curve extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return iso_curves

def is_external_facing_edge(edge, face1, face2, shape):
    """
    Determine if an edge has at least one external-facing adjacent face.
    """
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    
    try:
        for face in [face1, face2]:
            surf = BRepAdaptor_Surface(face)
            
            u_min = surf.FirstUParameter()
            u_max = surf.LastUParameter()
            v_min = surf.FirstVParameter()
            v_max = surf.LastVParameter()
            
            if not (math.isfinite(u_min) and math.isfinite(u_max) and
                    math.isfinite(v_min) and math.isfinite(v_max)):
                continue
            
            u_mid = (u_min + u_max) / 2.0
            v_mid = (v_min + v_max) / 2.0
            
            props = GeomLProp_SLProps(surf, u_mid, v_mid, 1, 1e-6)
            
            if not props.IsNormalDefined():
                continue
            
            normal = props.Normal()
            point = props.Value()
            
            if face.Orientation() == TopAbs_REVERSED:
                normal.Reverse()
            
            test_point = gp_Pnt(
                point.X() + normal.X() * 0.1,
                point.Y() + normal.Y() * 0.1,
                point.Z() + normal.Z() * 0.1
            )
            
            classifier = BRepClass3d_SolidClassifier()
            classifier.Load(shape)
            classifier.Perform(test_point, 1e-6)
            
            if classifier.State() == TopAbs_OUT:
                return True
        
        return False
    except Exception as e:
        logger.debug(f"Error checking edge orientation: {e}")
        return True

def extract_and_classify_feature_edges(shape, max_edges=500, angle_threshold_degrees=20, include_uiso=True, num_uiso_lines=2, total_surface_area=None):
    """
    UNIFIED single-pass edge extraction: tessellate once, classify, and tag segments.
    """
    logger.info(f"📐 Extracting and classifying BREP edges (angle threshold: {angle_threshold_degrees}°)...")
    
    feature_edges = []
    edge_classifications = []
    tagged_edges = []
    edge_count = 0
    feature_id_counter = 0
    angle_threshold_rad = math.radians(angle_threshold_degrees)
    
    # Extract isoparametric curves for curved surfaces
    iso_curves = []
    if include_uiso:
        iso_curves = extract_isoparametric_curves(
            shape,
            num_u_lines=num_uiso_lines,
            num_v_lines=0,
            total_surface_area=total_surface_area
        )
    
    # Build edge-to-faces map using TopTools
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    try:
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        stats = {
            'boundary_edges': 0,
            'sharp_edges': 0,
            'geometric_features': 0,
            'smooth_edges_skipped': 0,
            'internal_edges_skipped': 0,
            'total_processed': 0,
            'iso_curves': 0
        }
        
        debug_logged = 0
        max_debug_logs = 10
        
        while edge_explorer.More() and edge_count < max_edges:
            edge = topods.Edge(edge_explorer.Current())
            stats['total_processed'] += 1
            
            try:
                curve_result = BRep_Tool.Curve(edge)
                if not curve_result or len(curve_result) < 3 or curve_result[0] is None:
                    edge_explorer.Next()
                    continue
                
                curve = curve_result[0]
                first_param = curve_result[1]
                last_param = curve_result[2]
                
                is_significant = False
                edge_type = "unknown"
                
                # Get faces adjacent to this edge
                if edge_face_map.Contains(edge):
                    face_list = edge_face_map.FindFromKey(edge)
                    num_adjacent_faces = face_list.Size()
                    
                    if debug_logged < max_debug_logs:
                        logger.debug(f"🔍 Edge #{stats['total_processed']}: {num_adjacent_faces} adjacent faces")
                        debug_logged += 1
                    
                    if num_adjacent_faces == 1:
                        is_significant = True
                        edge_type = "boundary"
                        stats['boundary_edges'] += 1
                    
                    elif num_adjacent_faces == 2:
                        face1 = topods.Face(face_list.First())
                        face2 = topods.Face(face_list.Last())
                        
                        is_geometric_feature = is_cylinder_to_planar_edge(face1, face2)
                        
                        if is_geometric_feature:
                            is_significant = True
                            edge_type = "geometric_feature"
                            stats['geometric_features'] += 1
                            stats['sharp_edges'] += 1
                            
                            if stats['geometric_features'] <= 5:
                                curve_adaptor_temp = BRepAdaptor_Curve(edge)
                                curve_type_temp = curve_adaptor_temp.GetType()
                                curve_name = "LINE" if curve_type_temp == GeomAbs_Line else "CIRCLE" if curve_type_temp == GeomAbs_Circle else "OTHER"
                                logger.info(f" 🎯 Geometric feature #{stats['geometric_features']}: {curve_name} edge between cylinder/cone/sphere and plane")
                        
                        else:
                            has_external_face = is_external_facing_edge(edge, face1, face2, shape)
                            
                            if has_external_face:
                                dihedral_angle = calculate_dihedral_angle(edge, face1, face2)
                                
                                if dihedral_angle is not None and dihedral_angle > angle_threshold_rad:
                                    is_significant = True
                                    edge_type = f"sharp({math.degrees(dihedral_angle):.1f}°)"
                                    stats['sharp_edges'] += 1
                                
                                else:
                                    stats['smooth_edges_skipped'] += 1
                            
                            else:
                                stats['internal_edges_skipped'] += 1
                    
                    else:
                        is_significant = True
                        edge_type = "orphan"
                
                if not is_significant:
                    edge_explorer.Next()
                    continue
                
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()
                
                start_point = curve_adaptor.Value(first_param)
                end_point = curve_adaptor.Value(last_param)
                
                if curve_type == GeomAbs_Line:
                    num_samples = 2
                elif curve_type == GeomAbs_Circle:
                    num_samples = 64
                elif curve_type in [GeomAbs_BSplineCurve, GeomAbs_BezierCurve]:
                    num_samples = 24
                else:
                    num_samples = 20
                
                # Tessellate
                points = []
                for i in range(num_samples + 1):
                    param = first_param + (last_param - first_param) * i / num_samples
                    point = curve.Value(param)
                    points.append([point.X(), point.Y(), point.Z()])
                
                if len(points) < 2:
                    edge_explorer.Next()
                    continue
                
                feature_edges.append(points)
                
                # Classification
                classification = {
                    "id": edge_count,
                    "type": "line",
                    "start_point": [start_point.X(), start_point.Y(), start_point.Z()],
                    "end_point": [end_point.X(), end_point.Y(), end_point.Z()],
                    "feature_id": feature_id_counter
                }
                
                if curve_type == GeomAbs_Circle:
                    circle = curve_adaptor.Circle()
                    center = circle.Location()
                    radius = circle.Radius()
                    axis = circle.Axis()
                    
                    angular_extent = abs(last_param - first_param)
                    
                    if abs(angular_extent - 2 * math.pi) < 0.01:
                        classification["type"] = "circle"
                        classification["diameter"] = radius * 2
                        classification["radius"] = radius
                        classification["length"] = 2 * math.pi * radius
                        classification["segment_count"] = num_samples
                    
                    else:
                        classification["type"] = "arc"
                        classification["radius"] = radius
                        classification["length"] = radius * angular_extent
                        classification["segment_count"] = num_samples
                        classification["start_angle"] = first_param
                        classification["end_angle"] = last_param
                    
                    classification["center"] = [center.X(), center.Y(), center.Z()]
                    classification["normal"] = [axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z()]
                
                elif curve_type == GeomAbs_Line:
                    length = start_point.Distance(end_point)
                    classification["type"] = "line"
                    classification["length"] = length
                    classification["segment_count"] = num_samples
                
                else:
                    total_length = 0
                    for i in range(len(points) - 1):
                        p1 = np.array(points[i])
                        p2 = np.array(points[i + 1])
                        total_length += np.linalg.norm(p2 - p1)
                    
                    classification["type"] = "arc"
                    classification["length"] = total_length
                    classification["segment_count"] = num_samples
                
                edge_classifications.append(classification)
                
                # Tagged segments
                for i in range(len(points) - 1):
                    tagged_segment = {
                        'feature_id': feature_id_counter,
                        'start': points[i],
                        'end': points[i + 1],
                        'type': classification["type"]
                    }
                    
                    if classification.get('diameter'):
                        tagged_segment['diameter'] = classification['diameter']
                    if classification.get('radius'):
                        tagged_segment['radius'] = classification['radius']
                    if classification.get('length'):
                        tagged_segment['length'] = classification['length']
                    
                    tagged_edges.append(tagged_segment)
                
                edge_count += 1
                feature_id_counter += 1
            
            except Exception as e:
                logger.debug(f"Error processing edge: {e}")
                pass
            
            edge_explorer.Next()
        
        # Process ISO curves through the same pipeline
        for start, end, curve_type in iso_curves:
            try:
                if curve_type == "uiso":
                    num_samples = 2
                    classification_type = "line"
                elif curve_type == "viso":
                    num_samples = 64
                    classification_type = "arc"
                else:
                    num_samples = 20
                    classification_type = "line"
                
                start_vec = np.array(start)
                end_vec = np.array(end)
                
                points = []
                for i in range(num_samples + 1):
                    t = i / num_samples
                    point = start_vec + t * (end_vec - start_vec)
                    points.append(point.tolist())
                
                if len(points) < 2:
                    continue
                
                feature_edges.append(points)
                
                classification = {
                    "id": edge_count,
                    "type": classification_type,
                    "start_point": list(start),
                    "end_point": list(end),
                    "feature_id": feature_id_counter,
                    "iso_type": curve_type,
                    "segment_count": num_samples
                }
                
                length = np.linalg.norm(end_vec - start_vec)
                classification["length"] = length
                
                if curve_type == "viso":
                    estimated_radius = length * num_samples / (2 * math.pi)
                    classification["radius"] = estimated_radius
                    classification["diameter"] = estimated_radius * 2
                
                edge_classifications.append(classification)
                
                # Tagged segments
                for i in range(len(points) - 1):
                    tagged_segment = {
                        'feature_id': feature_id_counter,
                        'start': points[i],
                        'end': points[i + 1],
                        'type': classification_type,
                        'iso_type': curve_type
                    }
                    
                    if classification.get('diameter'):
                        tagged_segment['diameter'] = classification['diameter']
                    if classification.get('radius'):
                        tagged_segment['radius'] = classification['radius']
                    if classification.get('length'):
                        tagged_segment['length'] = classification['length']
                    
                    tagged_edges.append(tagged_segment)
                
                stats['iso_curves'] += 1
                edge_count += 1
                feature_id_counter += 1
            
            except Exception as e:
                logger.debug(f"Error processing ISO curve: {e}")
                pass
        
        stats['feature_edges_used'] = len(feature_edges)
        
        logger.info(f"✅ Extracted {len(feature_edges)} significant edges:")
        logger.info(f" - Boundary edges: {stats['boundary_edges']}")
        logger.info(f" - Sharp edges: {stats['sharp_edges']} (including {stats['geometric_features']} geometric features)")
        logger.info(f" - ISO curves: {stats['iso_curves']}")
        logger.info(f" - Smooth edges skipped: {stats['smooth_edges_skipped']}")
        logger.info(f" - Total processed: {stats['total_processed']}")
        logger.info(f" - Tagged segments: {len(tagged_edges)}")
    
    except Exception as e:
        logger.error(f"Error extracting edges: {e}")
    
    return {
        "feature_edges": feature_edges,
        "edge_classifications": edge_classifications,
        "tagged_edges": tagged_edges
    }

# DEPRECATED functions (kept for backward compatibility)

def classify_feature_edges(shape, max_edges=500, angle_threshold_degrees=20):
    """[DEPRECATED] Use extract_and_classify_feature_edges() instead."""
    logger.warning("⚠️ classify_feature_edges() is deprecated. Use extract_and_classify_feature_edges() instead.")
    return []

def extract_feature_edges(shape, max_edges=500, angle_threshold_degrees=20):
    """[DEPRECATED] Use extract_and_classify_feature_edges() instead."""
    logger.warning("⚠️ extract_feature_edges() is deprecated. Use extract_and_classify_feature_edges() instead.")
    return []

def calculate_face_center(triangulation, transform):
    """Compute average center of a face"""
    try:
        total = np.zeros(3)
        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i)
            p.Transform(transform)
            total += np.array([p.X(), p.Y(), p.Z()])
        return (total / triangulation.NbNodes()).tolist()
    except Exception:
        return [0, 0, 0]

def tag_feature_edges_for_frontend(edge_classifications):
    """[DEPRECATED] Use extract_and_classify_feature_edges() instead."""
    logger.warning("⚠️ tag_feature_edges_for_frontend() is deprecated. Use extract_and_classify_feature_edges() instead.")
    return []

def compute_smooth_vertex_normals(vertices, indices):
    """
    Compute smooth per-vertex normals by averaging adjacent face normals.
    """
    try:
        vertex_count = len(vertices) // 3
        triangle_count = len(indices) // 3
        
        normals = [0.0] * len(vertices)
        
        for tri_idx in range(triangle_count):
            i0 = indices[tri_idx * 3]
            i1 = indices[tri_idx * 3 + 1]
            i2 = indices[tri_idx * 3 + 2]
            
            v0 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]]
            v1 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]]
            v2 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]]
            
            e1 = [v1[j] - v0[j] for j in range(3)]
            e2 = [v2[j] - v0[j] for j in range(3)]
            
            face_normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ]
            
            for idx in [i0, i1, i2]:
                normals[idx * 3] += face_normal[0]
                normals[idx * 3 + 1] += face_normal[1]
                normals[idx * 3 + 2] += face_normal[2]
        
        for i in range(vertex_count):
            nx = normals[i * 3]
            ny = normals[i * 3 + 1]
            nz = normals[i * 3 + 2]
            
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            
            if length > 1e-7:
                normals[i * 3] = nx / length
                normals[i * 3 + 1] = ny / length
                normals[i * 3 + 2] = nz / length
            
            else:
                normals[i * 3] = 0.0
                normals[i * 3 + 1] = 0.0
                normals[i * 3 + 2] = 1.0
        
        return normals
    
    except Exception as e:
        logger.error(f"Error computing smooth normals: {e}")
        return [0.0] * len(vertices)

def tessellate_shape(shape):
    """
    Create ultra-high-quality mesh using GLOBAL adaptive tessellation.
    """
    try:
        diagonal, bbox_coords = calculate_bbox_diagonal(shape)
        
        linear_deflection = diagonal * 0.0001
        angular_deflection = 12.0
        
        logger.info(f"🎨 Using professional tessellation (linear={linear_deflection:.4f}mm, angular={angular_deflection}°)...")
        
        mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesher.Perform()
        
        if not mesher.IsDone():
            logger.warning("Tessellation did not complete successfully, trying with coarser settings...")
            linear_deflection = diagonal * 0.001
            angular_deflection = 15.0
            mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
            mesher.Perform()
        
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        
        logger.info("📊 Extracting mesh geometry (no classification)...")
        
        vertices, indices, normals = [], [], []
        face_data = []
        
        current_index = 0
        
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        
        while face_explorer.More():
            face = face_explorer.Current()
            
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)
            
            if triangulation is None:
                face_explorer.Next()
                face_idx += 1
                continue
            
            transform = loc.Transformation()
            surface = BRepAdaptor_Surface(face)
            reversed_face = face.Orientation() == 1
            
            surf_type = surface.GetType()
            center = calculate_face_center(triangulation, transform)
            
            bbox_center = [cx, cy, cz]
            to_surface = [center[0] - bbox_center[0], center[1] - bbox_center[1], center[2] - bbox_center[2]]
            
            face_start_vertex = current_index
            face_vertices = []
            
            for i in range(1, triangulation.NbNodes() + 1):
                p = triangulation.Node(i)
                p.Transform(transform)
                vertices.extend([p.X(), p.Y(), p.Z()])
                face_vertices.append(current_index)
                current_index += 1
            
            face_start_index = len(indices)
            
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                
                idx = [face_vertices[n1 - 1], face_vertices[n2 - 1], face_vertices[n3 - 1]]
                
                if reversed_face:
                    indices.extend([idx[0], idx[2], idx[1]])
                else:
                    indices.extend(idx)
                
                v1, v2, v3 = [vertices[j * 3:j * 3 + 3] for j in idx]
                
                e1 = [v2[k] - v1[k] for k in range(3)]
                e2 = [v3[k] - v1[k] for k in range(3)]
                
                n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
                
                l = math.sqrt(sum(x * x for x in n))
                
                if l > 0:
                    n = [x / l for x in n]
                
                if sum(n * v for n, v in zip(n, to_surface)) < 0:
                    n = [-x for x in n]
                
                for _ in range(3):
                    normals.extend(n)
            
            face_data.append({
                'face_idx': face_idx,
                'face_id': face_idx,
                'surf_type': surf_type,
                'center': center,
                'start_vertex': face_start_vertex,
                'vertex_count': len(face_vertices),
                'start_index': face_start_index,
                'triangle_count': (len(indices) - face_start_index) // 3
            })
            
            face_explorer.Next()
            face_idx += 1
        
        vertex_count = len(vertices) // 3
        triangle_count = len(indices) // 3
        
        logger.info(f"✅ Mesh generation complete: {vertex_count} vertices, {triangle_count} triangles")
        
        logger.info("🎨 Computing professional smooth vertex normals...")
        smooth_normals = compute_smooth_vertex_normals(vertices, indices)
        logger.info(f"✅ Smooth normals computed for {vertex_count} vertices")
        
        vertex_face_ids = [-1] * vertex_count
        
        for face_info in face_data:
            start_v = face_info['start_vertex']
            v_count = face_info['vertex_count']
            face_id = face_info['face_id']
            
            for v_idx in range(start_v, start_v + v_count):
                vertex_face_ids[v_idx] = face_id
        
        return {
            "vertices": vertices,
            "indices": indices,
            "normals": smooth_normals,
            "face_data": face_data,
            "vertex_face_ids": vertex_face_ids,
            "bbox": (xmin, ymin, zmin, xmax, ymax, zmax),
            "triangle_count": triangle_count,
        }
    
    except Exception as e:
        logger.error(f"Tessellation error: {e}")
        return {
            "vertices": [],
            "indices": [],
            "normals": [],
            "face_data": [],
            "bbox": (0, 0, 0, 0, 0, 0),
            "triangle_count": 0,
        }

def classify_mesh_faces(mesh_data, shape):
    """
    === SECTION 2: FIXED COLOR CLASSIFICATION ===
    """
    logger.info("🎨 Starting IMPROVED mesh-based color classification with propagation...")
    
    vertices = mesh_data["vertices"]
    normals = mesh_data["normals"]
    face_data = mesh_data["face_data"]
    xmin, ymin, zmin, xmax, ymax, zmax = mesh_data["bbox"]
    bbox_center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    bbox_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    
    vertex_count = len(vertices) // 3
    vertex_colors = ["external"] * vertex_count
    face_classifications = {}
    locked_faces = set()
    
    # Build edge-to-face adjacency map
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    # Build face lookup map
    face_lookup = {}
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while face_explorer.More():
        face_lookup[idx] = topods.Face(face_explorer.Current())
        idx += 1
        face_explorer.Next()
    
    logger.info("🔍 STEP 1: Classifying cylindrical faces...")
    
    # STEP 1: Classify CYLINDRICAL faces first
    for face_info in face_data:
        face_idx = face_info['face_idx']
        surf_type = face_info['surf_type']
        center = face_info['center']
        start_vertex = face_info['start_vertex']
        vertex_count_face = face_info['vertex_count']
        
        face_object = get_face_by_index(shape, face_idx)
        
        if surf_type != GeomAbs_Cylinder or face_object is None:
            continue
        
        try:
            surface = BRepAdaptor_Surface(face_object)
            cyl = surface.Cylinder()
            radius = cyl.Radius()
            axis_location = cyl.Axis().Location()
            
            axis_point = [axis_location.X(), axis_location.Y(), axis_location.Z()]
            
            dist_to_axis = math.sqrt(
                (center[0] - axis_point[0])**2 +
                (center[1] - axis_point[1])**2 +
                (center[2] - axis_point[2])**2
            )
            
            bbox_to_axis = math.sqrt(
                (bbox_center[0] - axis_point[0])**2 +
                (bbox_center[1] - axis_point[1])**2 +
                (bbox_center[2] - axis_point[2])**2
            )
            
            is_small_hole = radius < bbox_size * 0.15
            
            if dist_to_axis < bbox_to_axis:
                # Inner cylindrical surface
                if is_small_hole:
                    has_outer_neighbor = False
                    
                    edge_exp = TopExp_Explorer(face_object, TopAbs_EDGE)
                    
                    while edge_exp.More() and not has_outer_neighbor:
                        edge = edge_exp.Current()
                        
                        for map_idx in range(1, edge_face_map.Size() + 1):
                            map_edge = edge_face_map.FindKey(map_idx)
                            
                            if edge.IsSame(map_edge):
                                face_list = edge_face_map.FindFromIndex(map_idx)
                                face_iter = TopTools_ListIteratorOfListOfShape(face_list)
                                
                                while face_iter.More():
                                    adj_face = topods.Face(face_iter.Value())
                                    
                                    for other_info in face_data:
                                        other_face = face_lookup.get(other_info['face_idx'])
                                        
                                        if other_face and adj_face.IsSame(other_face):
                                            if other_info['surf_type'] != GeomAbs_Cylinder:
                                                break
                                            
                                            try:
                                                other_surf = BRepAdaptor_Surface(adj_face)
                                                other_cyl = other_surf.Cylinder()
                                                other_axis = other_cyl.Axis().Location()
                                                other_axis_pt = [other_axis.X(), other_axis.Y(), other_axis.Z()]
                                                other_center = other_info['center']
                                                
                                                other_dist = math.sqrt(
                                                    (other_center[0] - other_axis_pt[0])**2 +
                                                    (other_center[1] - other_axis_pt[1])**2 +
                                                    (other_center[2] - other_axis_pt[2])**2
                                                )
                                                
                                                other_bbox_dist = math.sqrt(
                                                    (bbox_center[0] - other_axis_pt[0])**2 +
                                                    (bbox_center[1] - other_axis_pt[1])**2 +
                                                    (bbox_center[2] - other_axis_pt[2])**2
                                                )
                                                
                                                if other_dist > other_bbox_dist:
                                                    has_outer_neighbor = True
                                            
                                            except:
                                                pass
                                            
                                            break
                                    
                                    if has_outer_neighbor:
                                        break
                                    
                                    face_iter.Next()
                                
                                break
                        
                        if has_outer_neighbor:
                            break
                        
                        edge_exp.Next()
                    
                    face_type = "through" if has_outer_neighbor else "internal"
                
                else:
                    # Large inner cylinder (bore)
                    face_type = "internal"
            
            else:
                # Outer cylindrical surface
                face_type = "external"
            
            face_classifications[face_idx] = face_type
            locked_faces.add(face_idx)
            
            for v_idx in range(start_vertex, start_vertex + vertex_count_face):
                vertex_colors[v_idx] = face_type
            
            if face_idx < 10:
                logger.info(f" Face {face_idx}: Cylinder R={radius:.2f}mm → {face_type}")
        
        except Exception as e:
            logger.warning(f"Cylinder classification failed for face {face_idx}: {e}")
            face_classifications[face_idx] = "external"
    
    logger.info("🔍 STEP 2: Multi-pass propagation to adjacent faces...")
    
    # STEP 2: Multi-pass propagation
    max_iterations = 10
    
    for iteration in range(max_iterations):
        changes_made = False
        
        for face_info in face_data:
            face_idx = face_info['face_idx']
            
            if face_idx in locked_faces:
                continue
            
            surf_type = face_info['surf_type']
            face_object = get_face_by_index(shape, face_idx)
            
            if face_object is None:
                continue
            
            start_vertex = face_info['start_vertex']
            vertex_count_face = face_info['vertex_count']
            
            current_type = face_classifications.get(face_idx)
            
            # Find adjacent faces via shared edges
            edge_exp = TopExp_Explorer(face_object, TopAbs_EDGE)
            neighbor_types = []
            
            while edge_exp.More():
                edge = edge_exp.Current()
                
                for map_idx in range(1, edge_face_map.Size() + 1):
                    map_edge = edge_face_map.FindKey(map_idx)
                    
                    if edge.IsSame(map_edge):
                        face_list = edge_face_map.FindFromIndex(map_idx)
                        face_iter = TopTools_ListIteratorOfListOfShape(face_list)
                        
                        while face_iter.More():
                            adj_face = topods.Face(face_iter.Value())
                            
                            for other_info in face_data:
                                other_face = face_lookup.get(other_info['face_idx'])
                                
                                if other_face and adj_face.IsSame(other_face):
                                    other_idx = other_info['face_idx']
                                    
                                    if other_idx != face_idx and other_idx in face_classifications:
                                        neighbor_types.append(face_classifications[other_idx])
                                    
                                    break
                            
                            face_iter.Next()
                        
                        break
                
                edge_exp.Next()
            
            # Determine new type based on neighbors
            new_type = None
            
            if neighbor_types:
                if "internal" in neighbor_types:
                    new_type = "internal"
                elif "through" in neighbor_types:
                    new_type = "through"
                else:
                    new_type = "external"
            
            else:
                if current_type is None:
                    new_type = "planar" if surf_type == GeomAbs_Plane else "external"
                else:
                    new_type = current_type
            
            # Update if changed
            if current_type != new_type:
                face_classifications[face_idx] = new_type
                
                for v_idx in range(start_vertex, start_vertex + vertex_count_face):
                    vertex_colors[v_idx] = new_type
                
                changes_made = True
        
        if not changes_made:
            logger.info(f" Propagation converged after {iteration + 1} iterations")
            break
        
        elif iteration == max_iterations - 1:
            logger.info(f" Propagation stopped at max iterations ({max_iterations})")
    
    # Count results
    type_counts = {}
    for vtype in vertex_colors:
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    
    logger.info(f"✅ Classification complete! Distribution: {type_counts}")
    
    # Create detailed face_classifications array
    detailed_face_classifications = []
    
    for face_info in face_data:
        face_id = face_info['face_id']
        face_type = face_classifications.get(face_id, "external")
        face_object = get_face_by_index(shape, face_info['face_idx'])
        surf_type = face_info['surf_type']
        center = face_info['center']
        
        if face_object is None:
            continue
        
        try:
            surface = BRepAdaptor_Surface(face_object)
            u_mid = (surface.FirstUParameter() + surface.LastUParameter()) / 2
            v_mid = (surface.FirstVParameter() + surface.LastVParameter()) / 2
            
            point = gp_Pnt()
            normal_vec = gp_Vec()
            
            surface.D1(u_mid, v_mid, point, gp_Vec(), normal_vec)
            normal = [normal_vec.X(), normal_vec.Y(), normal_vec.Z()]
        
        except:
            normal = [0, 0, 1]
        
        # Calculate face area
        triangle_count = face_info['triangle_count']
        start_idx = face_info['start_index']
        face_area = 0
        
        for tri_idx in range(triangle_count):
            idx_offset = start_idx + tri_idx * 3
            
            if idx_offset + 2 < len(mesh_data["indices"]):
                i0, i1, i2 = mesh_data["indices"][idx_offset:idx_offset+3]
                v0 = vertices[i0*3:i0*3+3]
                v1 = vertices[i1*3:i1*3+3]
                v2 = vertices[i2*3:i2*3+3]
                
                e1 = [v1[j] - v0[j] for j in range(3)]
                e2 = [v2[j] - v0[j] for j in range(3)]
                
                cross = [
                    e1[1]*e2[2] - e1[2]*e2[1],
                    e1[2]*e2[0] - e1[0]*e2[2],
                    e1[0]*e2[1] - e1[1]*e2[0]
                ]
                
                face_area += 0.5 * math.sqrt(sum(x*x for x in cross))
        
        face_classification = {
            "face_id": face_id,
            "type": face_type,
            "center": center,
            "normal": normal,
            "area": face_area,
            "surface_type": "cylinder" if surf_type == GeomAbs_Cylinder else
                           "plane" if surf_type == GeomAbs_Plane else "other"
        }
        
        # Add radius if cylindrical
        if surf_type == GeomAbs_Cylinder:
            try:
                surface = BRepAdaptor_Surface(face_object)
                cyl = surface.Cylinder()
                face_classification["radius"] = cyl.Radius()
            except:
                pass
        
        detailed_face_classifications.append(face_classification)
    
    logger.info(f"✅ Created {len(detailed_face_classifications)} detailed face classifications")
    
    return vertex_colors, detailed_face_classifications

@app.route("/analyze-cad", methods=["POST"])
def analyze_cad():
    """Upload a STEP file, analyze BREP geometry, generate display mesh"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        
        if not (file.filename.lower().endswith(".step") or file.filename.lower().endswith(".stp")):
            return jsonify({"error": "Only .step or .stp files supported"}), 400
        
        step_bytes = file.read()
        fd, tmp_path = tempfile.mkstemp(suffix=".step")
        
        try:
            os.write(fd, step_bytes)
            os.close(fd)
            
            reader = STEPControl_Reader()
            status = reader.ReadFile(tmp_path)
            
            if status != 1:
                return jsonify({"error": "Failed to read STEP file"}), 400
            
            reader.TransferRoots()
            shape = reader.OneShape()
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        logger.info("🔍 Analyzing BREP geometry...")
        exact_props = calculate_exact_volume_and_area(shape)
        
        logger.info("🎨 Generating display mesh...")
        mesh_data = tessellate_shape(shape)
        
        logger.info("🎨 Classifying face colors...")
        vertex_colors, face_classifications = classify_mesh_faces(mesh_data, shape)
        
        mesh_data["vertex_colors"] = vertex_colors
        mesh_data["face_classifications"] = face_classifications
        
        # ===== AAGNet-based feature recognition (PRIMARY) =====
        ml_features = None
        
        if AAGNET_AVAILABLE and aagnet_recognizer is not None:
            logger.info("🤖 Running AAGNet feature recognition (24 classes with instance segmentation)...")
            try:
                # Save shape to temporary STEP file for AAGNet
                fd, aagnet_tmp = tempfile.mkstemp(suffix=".step")
                os.close(fd)
                
                # Write shape to STEP file for AAGNet
                writer = STEPControl_Writer()
                writer.Transfer(shape, 1)  # 1 = IFSelect_RetDone
                writer.Write(aagnet_tmp)
                
                # Run AAGNet recognition
                aagnet_result = aagnet_recognizer.recognize_features(aagnet_tmp)
                
                # Clean up temp file
                os.unlink(aagnet_tmp)
                
                if aagnet_result.get('success'):
                    # Transform AAGNet output to expected format
                    ml_features = {
                        'feature_instances': [
                            {
                                'feature_type': inst['type'],
                                'face_ids': inst['face_indices'],
                                'bottom_faces': inst['bottom_faces'],
                                'confidence': inst['confidence']
                            }
                            for inst in aagnet_result.get('instances', [])
                        ],
                        'num_features_detected': aagnet_result.get('num_instances', 0),
                        'num_faces_analyzed': aagnet_result.get('num_faces', 0),
                        'inference_time_sec': aagnet_result.get('processing_time', 0),
                        'recognition_method': 'AAGNet'
                    }
                    logger.info(f"✅ AAGNet recognition complete")
                    logger.info(f"   Features: {ml_features['num_features_detected']}")
                    logger.info(f"   Faces: {ml_features['num_faces_analyzed']}")
                    logger.info(f"   Time: {ml_features['inference_time_sec']:.2f}s")
                else:
                    logger.error(f"❌ AAGNet recognition failed: {aagnet_result.get('error')}")
                    ml_features = {"error": aagnet_result.get('error', 'Unknown error'), "recognition_method": "AAGNet"}
                    
            except Exception as e:
                logger.error(f"❌ AAGNet failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                ml_features = {"error": str(e), "recognition_method": "AAGNet"}
        
        elif ML_VERSION == "v2":
            # FALLBACK 1: Use enhanced ML with built-in feature grouping (UV-Net)
            logger.info("🤖 Running ML-based feature recognition (v2 with feature grouping - UV-Net fallback)...")
            try:
                ml_features = predict_features(shape)
                logger.info(f"✅ ML v2 inference complete")
                logger.info(f"   Faces: {ml_features.get('num_faces_analyzed', '?')}")
                logger.info(f"   Features: {ml_features.get('num_features_detected', '?')}")
            except TimeoutError:
                logger.error("⏱️ ML inference timeout")
                ml_features = {"error": "ML inference timeout"}
            except Exception as e:
                logger.error(f"❌ ML v2 failed: {e}")
                ml_features = {"error": str(e)}
        
        elif ML_AVAILABLE:
            # FALLBACK 2: Use v1 + enhancement layer for feature grouping (UV-Net legacy)
            logger.info("🤖 Running ML-based feature recognition (v1 + feature grouping - UV-Net fallback)...")
            try:
                ml_features = predict_features(shape)
                
                # NEW: Enhance v1 output with feature grouping
                ml_features = enhance_ml_features_with_grouping(ml_features, shape)
                
                logger.info(f"✅ ML v1 inference + grouping complete")
                logger.info(f"   Faces: {len(ml_features.get('face_predictions', []))}")
                logger.info(f"   Features: {ml_features.get('num_features_detected', 0)}")
            except TimeoutError:
                logger.error("⏱️ ML inference timeout")
                ml_features = {"error": "ML inference timeout"}
            except Exception as e:
                logger.error(f"❌ ML v1 failed: {e}")
                ml_features = {"error": str(e)}
        
        else:
            logger.warning("⚠️ No feature recognition available (AAGNet and UV-Net both unavailable)")
        
        logger.info("📐 Extracting and classifying BREP edges...")
        
        edge_result = extract_and_classify_feature_edges(
            shape,
            max_edges=500,
            angle_threshold_degrees=20,
            include_uiso=True,
            num_uiso_lines=2,
            total_surface_area=exact_props['surface_area']
        )
        
        mesh_data["feature_edges"] = edge_result["feature_edges"]
        mesh_data["edge_classifications"] = edge_result["edge_classifications"]
        mesh_data["tagged_edges"] = edge_result["tagged_edges"]
        mesh_data["triangle_count"] = len(mesh_data.get("indices", [])) // 3
        
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        result = {
            'exact_volume': exact_props['volume'],
            'exact_surface_area': exact_props['surface_area'],
            'center_of_mass': exact_props['center_of_mass'],
            'mesh_data': {
                'vertices': mesh_data['vertices'],
                'indices': mesh_data['indices'],
                'normals': mesh_data['normals'],
                'vertex_colors': mesh_data['vertex_colors'],
                'vertex_face_ids': mesh_data.get('vertex_face_ids', []),
                'face_classifications': mesh_data['face_classifications'],
                'feature_edges': mesh_data['feature_edges'],
                'edge_classifications': mesh_data['edge_classifications'],
                'tagged_feature_edges': mesh_data['tagged_edges'],
                'triangle_count': mesh_data['triangle_count'],
                'face_classification_method': 'mesh_based_with_propagation',
                'edge_extraction_method': 'smart_filtering_20deg'
            },
            'volume_cm3': exact_props['volume'] / 1000,
            'surface_area_cm2': exact_props['surface_area'] / 100,
            'ml_features': ml_features,  # NEW: Now includes feature_instances!
            'status': 'success'
        }
        
        logger.info("✅ Analysis complete")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing CAD: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc() if os.getenv("DEBUG") else None
        }), 500

@app.route("/")
def root():
    return jsonify({
        "service": "CAD Geometry Analysis Service",
        "version": "9.0.0-aagnet-24class-instance-segmentation",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-cad",
            "aagnet": "/api/aagnet/recognize" if AAGNET_AVAILABLE else "unavailable"
        },
        "features": {
            "classification": "Mesh-based with neighbor propagation",
            "feature_detection": "AAGNet 24-class with instance segmentation" if AAGNET_AVAILABLE else "UV-Net fallback",
            "edge_extraction": "Professional smart filtering (20° dihedral angle threshold)",
            "feature_grouping": FEATURE_GROUPING_AVAILABLE,
            "aagnet_available": AAGNET_AVAILABLE,
            "ml_version": ML_VERSION or "v1_legacy"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
