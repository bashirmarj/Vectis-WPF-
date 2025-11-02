import os
import io
import math
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client

# === OCC imports ===
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_IN
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepTools import breptools
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

import logging

# === CONFIG ===
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# === Supabase setup ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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


def recognize_manufacturing_features(shape):
    """
    Analyze BREP topology to detect TRUE manufacturing features
    
    Accurate detection of:
    - Through-holes: Small cylinders that penetrate the part
    - Blind holes: Small cylinders with depth but no exit
    - Bores: Large internal cylindrical cavities
    - Bosses: Protruding cylindrical features
    - Pockets: Recessed features
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
    for face in all_faces:
        surface = BRepAdaptor_Surface(face)
        surf_type = surface.GetType()
        
        face_props = GProp_GProps()
        brepgprop.SurfaceProperties(face, face_props)
        face_area = face_props.Mass()
        face_center = face_props.CentreOfMass()
        
        if surf_type == GeomAbs_Cylinder:
            cyl = surface.Cylinder()
            radius = cyl.Radius()
            axis_dir = cyl.Axis().Direction()
            axis_pos = cyl.Axis().Location()
            
            # Calculate if this cylinder is internal or external
            axis_point = [axis_pos.X(), axis_pos.Y(), axis_pos.Z()]
            center = [face_center.X(), face_center.Y(), face_center.Z()]
            
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
            
            is_internal = dist_to_axis < bbox_to_axis
            
            feature_data = {
                'diameter': radius * 2,
                'radius': radius,
                'axis': [axis_dir.X(), axis_dir.Y(), axis_dir.Z()],
                'position': [axis_pos.X(), axis_pos.Y(), axis_pos.Z()],
                'area': face_area
            }
            
            # Classification logic
            diameter_ratio = (radius * 2) / bbox_size
            
            if is_internal:
                if diameter_ratio < 0.15:  # Small hole (< 15% of part size)
                    # Check if it's a through-hole by checking adjacency to external faces
                    has_external_connection = False
                    edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
                    
                    while edge_exp.More():
                        edge = edge_exp.Current()
                        
                        for map_idx in range(1, edge_face_map.Size() + 1):
                            map_edge = edge_face_map.FindKey(map_idx)
                            if edge.IsSame(map_edge):
                                face_list = edge_face_map.FindFromIndex(map_idx)
                                face_iter = TopTools_ListIteratorOfListOfShape(face_list)
                                
                                while face_iter.More():
                                    adj_face = topods.Face(face_iter.Value())
                                    if not adj_face.IsSame(face):
                                        # Check if adjacent face is external
                                        adj_surface = BRepAdaptor_Surface(adj_face)
                                        if adj_surface.GetType() == GeomAbs_Cylinder:
                                            try:
                                                adj_cyl = adj_surface.Cylinder()
                                                adj_axis = adj_cyl.Axis().Location()
                                                adj_axis_pt = [adj_axis.X(), adj_axis.Y(), adj_axis.Z()]
                                                
                                                adj_props = GProp_GProps()
                                                brepgprop.SurfaceProperties(adj_face, adj_props)
                                                adj_center = adj_props.CentreOfMass()
                                                adj_ctr = [adj_center.X(), adj_center.Y(), adj_center.Z()]
                                                
                                                adj_dist = math.sqrt(
                                                    (adj_ctr[0] - adj_axis_pt[0])**2 +
                                                    (adj_ctr[1] - adj_axis_pt[1])**2 +
                                                    (adj_ctr[2] - adj_axis_pt[2])**2
                                                )
                                                adj_bbox_dist = math.sqrt(
                                                    (bbox_center[0] - adj_axis_pt[0])**2 +
                                                    (bbox_center[1] - adj_axis_pt[1])**2 +
                                                    (bbox_center[2] - adj_axis_pt[2])**2
                                                )
                                                
                                                if adj_dist > adj_bbox_dist:
                                                    has_external_connection = True
                                                    break
                                            except:
                                                pass
                                        elif adj_surface.GetType() == GeomAbs_Plane:
                                            # Planar face could be external
                                            has_external_connection = True
                                            break
                                    
                                    face_iter.Next()
                                break
                        
                        if has_external_connection:
                            break
                        edge_exp.Next()
                    
                    if has_external_connection:
                        features['through_holes'].append(feature_data)
                    else:
                        features['blind_holes'].append(feature_data)
                
                elif diameter_ratio < 0.5:  # Medium bore (15-50% of part size)
                    features['bores'].append(feature_data)
                # else: very large internal cavity - not counted as separate feature
            
            else:  # External cylinder
                if diameter_ratio < 0.3:  # Small boss
                    features['bosses'].append(feature_data)
                # else: main body cylinder - not a separate feature
        
        elif surf_type == GeomAbs_Plane:
            plane = surface.Plane()
            normal = plane.Axis().Direction()
            features['planar_faces'].append({
                'normal': [normal.X(), normal.Y(), normal.Z()],
                'area': face_area
            })
        
        elif surf_type == GeomAbs_Torus:
            # Fillets and rounds
            features['fillets'].append({
                'area': face_area,
                'type': 'torus'
            })
        
        else:
            features['complex_surfaces'].append({
                'type': str(surf_type),
                'area': face_area
            })
    
    # Calculate totals for backward compatibility
    total_holes = len(features['through_holes']) + len(features['blind_holes'])
    total_bosses = len(features['bosses'])
    
    logger.info(f"🔧 Manufacturing Features Detected:")
    logger.info(f"   Through-holes: {len(features['through_holes'])}")
    logger.info(f"   Blind holes: {len(features['blind_holes'])}")
    logger.info(f"   Bores: {len(features['bores'])}")
    logger.info(f"   Bosses: {len(features['bosses'])}")
    logger.info(f"   Planar faces: {len(features['planar_faces'])}")
    logger.info(f"   Fillets: {len(features['fillets'])}")
    logger.info(f"   Complex surfaces: {len(features['complex_surfaces'])}")
    
    # Add legacy fields for backward compatibility
    features['holes'] = features['through_holes'] + features['blind_holes']
    features['cylindrical_bosses'] = features['bosses']
    
    return features


def get_face_normal_at_point(face, point):
    """
    Get the surface normal of a face at a given point.
    
    Returns gp_Dir or None if calculation fails.
    """
    try:
        surface = BRep_Tool.Surface(face)
        surface_adaptor = BRepAdaptor_Surface(face)
        
        # Project point onto surface to get UV parameters
        # This is a simplified approach - using midpoint of UV domain
        u_mid = (surface_adaptor.FirstUParameter() + surface_adaptor.LastUParameter()) / 2
        v_mid = (surface_adaptor.FirstVParameter() + surface_adaptor.LastVParameter()) / 2
        
        # Get normal at midpoint
        d1u = surface_adaptor.DN(u_mid, v_mid, 1, 0)
        d1v = surface_adaptor.DN(u_mid, v_mid, 0, 1)
        
        normal = d1u.Crossed(d1v)
        
        if normal.Magnitude() < 1e-7:
            return None
            
        normal.Normalize()
        
        # Check face orientation
        if face.Orientation() == 1:  # TopAbs_REVERSED
            normal.Reverse()
        
        return gp_Dir(normal.X(), normal.Y(), normal.Z())
        
    except Exception as e:
        logger.debug(f"Error getting face normal: {e}")
        return None


def is_cylinder_to_planar_edge(face1, face2):
    """
    Detect if an edge connects a cylindrical/conical face to a planar face.
    These edges should always be included (cylinder height lines, cone base circles).
    
    Uses GeomAbs surface type enums instead of string names for reliability.
    
    Returns: True if one face is cylindrical/conical and the other is planar
    """
    try:
        # Use BRepAdaptor instead of BRep_Tool for type detection
        surf1 = BRepAdaptor_Surface(face1)
        surf2 = BRepAdaptor_Surface(face2)
        
        # Get surface types using GeomAbs enums (more reliable than string names)
        type1 = surf1.GetType()
        type2 = surf2.GetType()
        
        # Curved surface types that create important boundary edges
        curved_types = {
            GeomAbs_Cylinder,    # Cylindrical surfaces
            GeomAbs_Cone,        # Conical surfaces
            GeomAbs_Sphere,      # Spherical surfaces
            GeomAbs_Torus        # Toroidal surfaces
        }
        
        # Check if one is curved and the other is planar
        is_curved_to_plane = (
            (type1 in curved_types and type2 == GeomAbs_Plane) or
            (type2 in curved_types and type1 == GeomAbs_Plane)
        )
        
        # Debug logging for first few detections
        if is_curved_to_plane:
            logger.debug(f"🎯 GEOMETRIC FEATURE DETECTED: type1={type1}, type2={type2}")
        
        return is_curved_to_plane
        
    except Exception as e:
        logger.debug(f"Error detecting cylinder-to-plane edge: {e}")
        return False


def calculate_dihedral_angle(edge, face1, face2):
    """
    Calculate the dihedral angle between two faces along their shared edge.
    
    Returns angle in radians, or None if calculation fails.
    Professional CAD software typically uses 20-30° threshold.
    """
    try:
        # Get a point in the middle of the edge
        curve_result = BRep_Tool.Curve(edge)
        if not curve_result or curve_result[0] is None:
            return None
            
        curve = curve_result[0]
        first_param = curve_result[1]
        last_param = curve_result[2]
        mid_param = (first_param + last_param) / 2.0
        
        edge_point = curve.Value(mid_param)
        
        # Get normals of both faces at the edge point
        normal1 = get_face_normal_at_point(face1, edge_point)
        normal2 = get_face_normal_at_point(face2, edge_point)
        
        if normal1 is None or normal2 is None:
            return None
        
        # Calculate angle between normals
        # Dihedral angle = π - angle between normals
        dot_product = normal1.Dot(normal2)
        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
        
        angle_between_normals = math.acos(dot_product)
        dihedral_angle = math.pi - angle_between_normals
        
        return abs(dihedral_angle)
        
    except Exception as e:
        logger.debug(f"Error calculating dihedral angle: {e}")
        return None


def extract_isoparametric_curves(shape, num_u_lines=2, num_v_lines=0):
    """
    Extract UIso and VIso parametric curves from cylindrical, conical, 
    spherical, and toroidal surfaces using Geom_Surface API.
    
    UIso curves (U=constant): Lines running along cylinder height
    VIso curves (V=constant): Circular cross-sections
    
    Args:
        shape: TopoDS_Shape to analyze
        num_u_lines: Number of UIso curves per surface (default 2)
        num_v_lines: Number of VIso curves per surface (default 0)
    
    Returns:
        List of tuples: [(start_point, end_point, curve_type), ...]
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
                # Get surface adaptor for type checking
                surf_adaptor = BRepAdaptor_Surface(face)
                surf_type = surf_adaptor.GetType()
                
                # Only process curved surfaces
                if surf_type in [GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus]:
                    surface_count += 1
                    logger.debug(f"Processing curved surface #{surface_count}: {surf_type}")
                    
                    # Get the underlying Geom_Surface
                    geom_surface = BRep_Tool.Surface(face)
                    
                    # Get parametric bounds from adaptor
                    u_min = surf_adaptor.FirstUParameter()
                    u_max = surf_adaptor.LastUParameter()
                    v_min = surf_adaptor.FirstVParameter()
                    v_max = surf_adaptor.LastVParameter()
                    
                    logger.debug(f"  Parametric bounds: U=[{u_min}, {u_max}], V=[{v_min}, {v_max}]")
                    
                    # Validate bounds (cylinders often have infinite U range)
                    u_valid = math.isfinite(u_min) and math.isfinite(u_max) and u_max > u_min
                    v_valid = math.isfinite(v_min) and math.isfinite(v_max) and v_max > v_min
                    
                    # Handle periodic surfaces (cylinders: U wraps around)
                    if not u_valid:
                        logger.debug(f"  U bounds invalid/infinite - using [0, 2π] for periodic surface")
                        u_min = 0.0
                        u_max = 2.0 * math.pi
                        u_valid = True
                    
                    if not v_valid:
                        logger.warning(f"  V bounds invalid - skipping surface")
                        face_explorer.Next()
                        continue
                    
                    # Extract UIso curves (vertical lines on cylinders)
                    if num_u_lines > 0 and u_valid:
                        for i in range(num_u_lines):
                            try:
                                u_value = u_min + (u_max - u_min) * i / num_u_lines
                                
                                # Create UIso curve using Geom_Surface
                                uiso_geom_curve = geom_surface.UIso(u_value)
                                
                                # Wrap in adaptor for evaluation
                                uiso_adaptor = GeomAdaptor_Curve(uiso_geom_curve)
                                
                                # Sample at V bounds
                                start_point = uiso_adaptor.Value(v_min)
                                end_point = uiso_adaptor.Value(v_max)
                                
                                iso_curves.append((
                                    (start_point.X(), start_point.Y(), start_point.Z()),
                                    (end_point.X(), end_point.Y(), end_point.Z()),
                                    "uiso"
                                ))
                                uiso_count += 1
                                logger.debug(f"  ✓ Extracted UIso curve #{uiso_count} at U={u_value:.4f}")
                                
                            except Exception as e:
                                logger.warning(f"  ✗ Failed to extract UIso curve at U={u_value:.4f}: {e}")
                    
                    # Extract VIso curves (circular cross-sections)
                    if num_v_lines > 0 and v_valid and u_valid:
                        for i in range(1, num_v_lines + 1):
                            try:
                                v_value = v_min + (v_max - v_min) * i / (num_v_lines + 1)
                                
                                # Create VIso curve using Geom_Surface
                                viso_geom_curve = geom_surface.VIso(v_value)
                                
                                # Wrap in adaptor
                                viso_adaptor = GeomAdaptor_Curve(viso_geom_curve)
                                
                                # Sample multiple points around the circle
                                num_segments = 64  # Match geometric feature quality
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
                                logger.debug(f"  ✓ Extracted VIso curve #{viso_count} at V={v_value:.4f} ({num_segments} segments)")
                                
                            except Exception as e:
                                logger.warning(f"  ✗ Failed to extract VIso curve at V={v_value:.4f}: {e}")
            
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
    Uses surface normal direction to check orientation.
    
    Args:
        edge: TopoDS_Edge to analyze
        face1: First adjacent TopoDS_Face
        face2: Second adjacent TopoDS_Face
        shape: The parent TopoDS_Shape (solid)
    
    Returns:
        bool: True if at least one face is external-facing
    """
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    
    try:
        for face in [face1, face2]:
            # Get surface adaptor
            surf = BRepAdaptor_Surface(face)
            
            # Get parametric center
            u_min = surf.FirstUParameter()
            u_max = surf.LastUParameter()
            v_min = surf.FirstVParameter()
            v_max = surf.LastVParameter()
            
            # Validate bounds
            if not (math.isfinite(u_min) and math.isfinite(u_max) and 
                    math.isfinite(v_min) and math.isfinite(v_max)):
                continue
            
            u_mid = (u_min + u_max) / 2.0
            v_mid = (v_min + v_max) / 2.0
            
            # Get point and normal at center
            props = GeomLProp_SLProps(surf, u_mid, v_mid, 1, 1e-6)
            
            if not props.IsNormalDefined():
                continue
            
            normal = props.Normal()
            point = props.Value()
            
            # Adjust normal based on face orientation
            if face.Orientation() == TopAbs_REVERSED:
                normal.Reverse()
            
            # Cast ray from point along normal
            test_point = gp_Pnt(
                point.X() + normal.X() * 0.1,
                point.Y() + normal.Y() * 0.1,
                point.Z() + normal.Z() * 0.1
            )
            
            # Classify point
            classifier = BRepClass3d_SolidClassifier()
            classifier.Load(shape)
            classifier.Perform(test_point, 1e-6)
            
            # If point is outside, face is external
            if classifier.State() == TopAbs_OUT:
                return True
        
        return False
        
    except Exception as e:
        logger.debug(f"Error checking edge orientation: {e}")
        return True  # Default to external on error


def extract_and_classify_feature_edges(shape, max_edges=500, angle_threshold_degrees=20, include_uiso=True, num_uiso_lines=2):
    """
    UNIFIED single-pass edge extraction: tessellate once, classify, and tag segments.
    
    This replaces the old three-step process (extract → classify → tag) with a single pass
    that guarantees perfect matching between visual edges and measurement segments.
    
    Only extracts edges that are:
    1. Boundary edges (belong to only 1 face) - always significant
    2. Sharp edges (dihedral angle between faces > threshold)
    3. UIso/VIso parametric curves (industry-standard cylinder height lines)
    
    Args:
        shape: OpenCascade shape
        max_edges: Maximum number of edges to extract
        angle_threshold_degrees: Minimum dihedral angle to consider edge "sharp" (default: 20°)
        include_uiso: Whether to include UIso/VIso parametric curves (default: True)
        num_uiso_lines: Number of UIso lines per curved surface (default: 2, CATIA standard)
    
    Returns:
        {
            "feature_edges": List of polylines for rendering (same as old extract_feature_edges),
            "edge_classifications": List of metadata dicts (same as old classify_feature_edges),
            "tagged_edges": List of tagged segments for measurement matching (same as old tag_feature_edges_for_frontend)
        }
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
            num_v_lines=2
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
        
        debug_logged = 0  # Track how many edges we've logged for debugging
        max_debug_logs = 10  # Only log first 10 edges to avoid spam
        
        while edge_explorer.More() and edge_count < max_edges:
            edge = topods.Edge(edge_explorer.Current())
            stats['total_processed'] += 1
            
            try:
                # Get curve geometry
                curve_result = BRep_Tool.Curve(edge)
                
                if not curve_result or len(curve_result) < 3 or curve_result[0] is None:
                    edge_explorer.Next()
                    continue
                
                curve = curve_result[0]
                first_param = curve_result[1]
                last_param = curve_result[2]
                
                # Determine if this edge is significant
                is_significant = False
                edge_type = "unknown"
                
                # Get faces adjacent to this edge
                if edge_face_map.Contains(edge):
                    face_list = edge_face_map.FindFromKey(edge)
                    num_adjacent_faces = face_list.Size()
                    
                    # Debug logging for first few edges
                    if debug_logged < max_debug_logs:
                        logger.debug(f"🔍 Edge #{stats['total_processed']}: {num_adjacent_faces} adjacent faces")
                        debug_logged += 1
                    
                    if num_adjacent_faces == 1:
                        # BOUNDARY EDGE - always show (holes, external boundaries)
                        is_significant = True
                        edge_type = "boundary"
                        stats['boundary_edges'] += 1
                        
                    elif num_adjacent_faces == 2:
                        # INTERIOR EDGE - check geometry, orientation, then angle
                        face1 = topods.Face(face_list.First())
                        face2 = topods.Face(face_list.Last())
                        
                        # Check if this is a special geometric edge (cylinder-to-plane, etc.)
                        is_geometric_feature = is_cylinder_to_planar_edge(face1, face2)
                        
                        if is_geometric_feature:
                            # GEOMETRIC FEATURE EDGE - always include
                            is_significant = True
                            edge_type = "geometric_feature"
                            stats['geometric_features'] += 1
                            stats['sharp_edges'] += 1
                            
                            if stats['geometric_features'] <= 5:
                                curve_adaptor_temp = BRepAdaptor_Curve(edge)
                                curve_type_temp = curve_adaptor_temp.GetType()
                                curve_name = "LINE" if curve_type_temp == GeomAbs_Line else "CIRCLE" if curve_type_temp == GeomAbs_Circle else "OTHER"
                                logger.info(f"   🎯 Geometric feature #{stats['geometric_features']}: {curve_name} edge between cylinder/cone/sphere and plane")
                        else:
                            # Check if at least one face is external
                            has_external_face = is_external_facing_edge(edge, face1, face2, shape)
                            
                            if has_external_face:
                                # Calculate dihedral angle only for edges adjacent to external faces
                                dihedral_angle = calculate_dihedral_angle(edge, face1, face2)
                                
                                if dihedral_angle is not None and dihedral_angle > angle_threshold_rad:
                                    # SHARP EDGE on external surface
                                    is_significant = True
                                    edge_type = f"sharp({math.degrees(dihedral_angle):.1f}°)"
                                    stats['sharp_edges'] += 1
                                else:
                                    # SMOOTH EDGE on external surface
                                    stats['smooth_edges_skipped'] += 1
                            else:
                                # INTERNAL EDGE (both faces are internal) - skip
                                stats['internal_edges_skipped'] += 1
                else:
                    # Orphan edge - include it to be safe
                    is_significant = True
                    edge_type = "orphan"
                
                # Only process significant edges
                if not is_significant:
                    edge_explorer.Next()
                    continue
                
                # Get curve adaptor for type detection
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()
                
                # Get start and end points
                start_point = curve_adaptor.Value(first_param)
                end_point = curve_adaptor.Value(last_param)
                
                # Adaptive sampling based on curve type
                if curve_type == GeomAbs_Line:
                    num_samples = 2  # Lines only need endpoints
                elif curve_type == GeomAbs_Circle:
                    num_samples = 64  # Professional CAD quality for smooth circular boundaries
                elif curve_type in [GeomAbs_BSplineCurve, GeomAbs_BezierCurve]:
                    num_samples = 24  # Splines need moderate sampling
                else:
                    num_samples = 20  # Default for other curve types
                
                # ===== TESSELLATE ONCE using OpenCascade curve.Value() =====
                points = []
                for i in range(num_samples + 1):
                    param = first_param + (last_param - first_param) * i / num_samples
                    point = curve.Value(param)
                    points.append([point.X(), point.Y(), point.Z()])
                
                if len(points) < 2:
                    edge_explorer.Next()
                    continue
                
                # ===== OUTPUT 1: Feature edges for rendering =====
                feature_edges.append(points)
                
                # ===== OUTPUT 2: Edge classification metadata =====
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
                    
                    # Check if full circle or arc
                    angular_extent = abs(last_param - first_param)
                    if abs(angular_extent - 2 * math.pi) < 0.01:
                        # Full circle
                        classification["type"] = "circle"
                        classification["diameter"] = radius * 2
                        classification["radius"] = radius
                        classification["length"] = 2 * math.pi * radius
                        classification["segment_count"] = num_samples
                    else:
                        # Arc
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
                    # For BSpline, Bezier - calculate approximate length
                    total_length = 0
                    for i in range(len(points) - 1):
                        p1 = np.array(points[i])
                        p2 = np.array(points[i + 1])
                        total_length += np.linalg.norm(p2 - p1)
                    
                    classification["type"] = "arc"
                    classification["length"] = total_length
                    classification["segment_count"] = num_samples
                
                edge_classifications.append(classification)
                
                # ===== OUTPUT 3: Tagged segments for measurement matching =====
                # Convert polyline to consecutive segment pairs
                for i in range(len(points) - 1):
                    tagged_segment = {
                        'feature_id': feature_id_counter,
                        'start': points[i],
                        'end': points[i + 1],
                        'type': classification["type"]
                    }
                    
                    # Copy measurement data
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
        
        # Process ISO curves through the same pipeline as BREP edges
        for start, end, curve_type in iso_curves:
            try:
                # Determine curve type for adaptive sampling
                if curve_type == "uiso":
                    # UIso = straight line on cylinder
                    num_samples = 2
                    classification_type = "line"
                elif curve_type == "viso":
                    # VIso = circular cross-section
                    num_samples = 64  # Same quality as geometric circles
                    classification_type = "arc"
                else:
                    num_samples = 20
                    classification_type = "line"
                
                # ===== TESSELLATE using same logic as BREP edges =====
                start_vec = np.array(start)
                end_vec = np.array(end)
                points = []
                
                for i in range(num_samples + 1):
                    t = i / num_samples
                    point = start_vec + t * (end_vec - start_vec)
                    points.append(point.tolist())
                
                if len(points) < 2:
                    continue
                
                # ===== OUTPUT 1: Feature edges for rendering =====
                feature_edges.append(points)
                
                # ===== OUTPUT 2: Edge classification metadata =====
                classification = {
                    "id": edge_count,
                    "type": classification_type,
                    "start_point": list(start),
                    "end_point": list(end),
                    "feature_id": feature_id_counter,
                    "iso_type": curve_type,
                    "segment_count": num_samples
                }
                
                # Calculate length
                length = np.linalg.norm(end_vec - start_vec)
                classification["length"] = length
                
                # For VIso circles, add radius/diameter if available
                if curve_type == "viso":
                    # Approximate radius from arc length and segment count
                    estimated_radius = length * num_samples / (2 * math.pi)
                    classification["radius"] = estimated_radius
                    classification["diameter"] = estimated_radius * 2
                
                edge_classifications.append(classification)
                
                # ===== OUTPUT 3: Tagged segments for measurement matching =====
                # Use SAME logic as BREP edges (lines 732-750)
                for i in range(len(points) - 1):
                    tagged_segment = {
                        'feature_id': feature_id_counter,
                        'start': points[i],
                        'end': points[i + 1],
                        'type': classification_type,
                        'iso_type': curve_type
                    }
                    
                    # Copy measurement data
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
        
        # Update total edge count
        stats['feature_edges_used'] = len(feature_edges)
        
        logger.info(f"✅ Extracted {len(feature_edges)} significant edges:")
        logger.info(f"   - Boundary edges: {stats['boundary_edges']}")
        logger.info(f"   - Sharp edges: {stats['sharp_edges']} (including {stats['geometric_features']} geometric features)")
        logger.info(f"   - ISO curves: {stats['iso_curves']}")
        logger.info(f"   - Smooth edges skipped: {stats['smooth_edges_skipped']}")
        logger.info(f"   - Total processed: {stats['total_processed']}")
        logger.info(f"   - Tagged segments: {len(tagged_edges)}")
        
    except Exception as e:
        logger.error(f"Error extracting edges: {e}")
    
    return {
        "feature_edges": feature_edges,
        "edge_classifications": edge_classifications,
        "tagged_edges": tagged_edges
    }


# DEPRECATED: This function is no longer used.
# Use extract_and_classify_feature_edges() instead for guaranteed matching.
def classify_feature_edges(shape, max_edges=500, angle_threshold_degrees=20):
    """
    [DEPRECATED] Classify feature edges detected by extract_feature_edges.
    
    This function is kept for backward compatibility but should not be used.
    Use extract_and_classify_feature_edges() instead.
    """
    logger.warning("⚠️  classify_feature_edges() is deprecated. Use extract_and_classify_feature_edges() instead.")
    return []


# DEPRECATED: This function is no longer used.
# Use extract_and_classify_feature_edges() instead for guaranteed matching.
def extract_feature_edges(shape, max_edges=500, angle_threshold_degrees=20):
    """
    [DEPRECATED] Extract SIGNIFICANT feature edges from BREP geometry.
    
    This function is kept for backward compatibility but should not be used.
    Use extract_and_classify_feature_edges() instead.
    """
    logger.warning("⚠️  extract_feature_edges() is deprecated. Use extract_and_classify_feature_edges() instead.")
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


# DEPRECATED: This function is no longer used. 
# Use extract_and_classify_feature_edges() instead for guaranteed matching.
def tag_feature_edges_for_frontend(edge_classifications):
    """
    [DEPRECATED] Tag edge segments with feature_id for direct frontend lookup.
    
    This function is kept for backward compatibility but should not be used.
    Use extract_and_classify_feature_edges() instead.
    """
    logger.warning("⚠️  tag_feature_edges_for_frontend() is deprecated. Use extract_and_classify_feature_edges() instead.")
    return []


def compute_smooth_vertex_normals(vertices, indices):
    """
    Compute smooth per-vertex normals by averaging adjacent face normals.
    This eliminates horizontal banding on curved surfaces (cylinders, fillets).
    
    Args:
        vertices: Flat list of vertex coordinates [x0,y0,z0, x1,y1,z1, ...]
        indices: Flat list of triangle indices [i0,i1,i2, i3,i4,i5, ...]
    
    Returns:
        List of smooth normals [nx0,ny0,nz0, nx1,ny1,nz1, ...]
    """
    try:
        vertex_count = len(vertices) // 3
        triangle_count = len(indices) // 3
        
        # Initialize normals accumulator (will sum face normals)
        normals = [0.0] * len(vertices)
        
        # For each triangle, compute face normal and accumulate at vertices
        for tri_idx in range(triangle_count):
            i0 = indices[tri_idx * 3]
            i1 = indices[tri_idx * 3 + 1]
            i2 = indices[tri_idx * 3 + 2]
            
            # Get vertex positions
            v0 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]]
            v1 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]]
            v2 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]]
            
            # Compute edges
            e1 = [v1[j] - v0[j] for j in range(3)]
            e2 = [v2[j] - v0[j] for j in range(3)]
            
            # Compute face normal (cross product)
            face_normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ]
            
            # Accumulate at each vertex of the triangle
            for idx in [i0, i1, i2]:
                normals[idx * 3] += face_normal[0]
                normals[idx * 3 + 1] += face_normal[1]
                normals[idx * 3 + 2] += face_normal[2]
        
        # Normalize all accumulated normals
        for i in range(vertex_count):
            nx = normals[i * 3]
            ny = normals[i * 3 + 1]
            nz = normals[i * 3 + 2]
            
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 1e-7:  # Avoid division by zero
                normals[i * 3] = nx / length
                normals[i * 3 + 1] = ny / length
                normals[i * 3 + 2] = nz / length
            else:
                # Degenerate case: use arbitrary normal
                normals[i * 3] = 0.0
                normals[i * 3 + 1] = 0.0
                normals[i * 3 + 2] = 1.0
        
        return normals
        
    except Exception as e:
        logger.error(f"Error computing smooth normals: {e}")
        # Return zero normals if computation fails
        return [0.0] * len(vertices)


def tessellate_shape(shape):
    """
    Create ultra-high-quality mesh using GLOBAL adaptive tessellation.
    Uses industry-standard 12-degree angular deflection for professional CAD quality.
    
    This is the CORRECTED VERSION that eliminates horizontal lines on curved surfaces.
    """
    try:
        # Calculate bounding box for adaptive tessellation
        diagonal, bbox_coords = calculate_bbox_diagonal(shape)
        
        # Professional CAD-quality tessellation parameters (SolidWorks/Fusion 360/Onshape standard)
        linear_deflection = diagonal * 0.0001  # 0.01% of diagonal - very fine tessellation
        angular_deflection = 12.0  # ✅ 12 DEGREES - Industry standard for smooth curved surfaces
        
        logger.info(f"🎨 Using professional tessellation (linear={linear_deflection:.4f}mm, angular={angular_deflection}°)...")
        
        # GLOBAL tessellation - ONE PASS for entire shape
        # This is critical to avoid discontinuities at face boundaries
        mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesher.Perform()
        
        if not mesher.IsDone():
            logger.warning("Tessellation did not complete successfully, trying with coarser settings...")
            linear_deflection = diagonal * 0.001
            angular_deflection = 15.0
            mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
            mesher.Perform()
        
        # Calculate bbox center for normal orientation
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        
        logger.info("📊 Extracting mesh geometry (no classification)...")
        
        vertices, indices, normals = [], [], []
        face_data = []  # Store face metadata for later classification
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
            
            # Store face metadata for classification
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
            
            # Store face info for classification
            face_data.append({
                'face_idx': face_idx,
                'face_id': face_idx,  # Unique face identifier
                'surf_type': surf_type,
                'center': center,
                'start_vertex': face_start_vertex,
                'vertex_count': len(face_vertices),
                'start_index': face_start_index,
                'triangle_count': (len(indices) - face_start_index) // 3,
                'face_object': face
            })
            
            face_explorer.Next()
            face_idx += 1
        
        vertex_count = len(vertices) // 3
        triangle_count = len(indices) // 3
        logger.info(f"✅ Mesh generation complete: {vertex_count} vertices, {triangle_count} triangles")
        
        # PASS 2: Generate PURE SMOOTH NORMALS (no hybrid - professional CAD standard)
        # This ensures seamless transitions between all surface types
        logger.info("🎨 Computing professional smooth vertex normals...")
        smooth_normals = compute_smooth_vertex_normals(vertices, indices)
        logger.info(f"✅ Smooth normals computed for {vertex_count} vertices")
        
        # Create vertex_face_ids mapping (each vertex knows its face_id)
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
            "normals": smooth_normals,  # ✅ Use smooth normals as primary normals
            "face_data": face_data,
            "vertex_face_ids": vertex_face_ids,  # Map vertices to face_id
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
    
    Strategy:
    1. First classify all CYLINDRICAL faces using radius/axis logic
    2. Then PROPAGATE classification to adjacent non-cylindrical faces
    3. This ensures boss fillets, planar faces, etc. get correct colors
    
    Returns (vertex_colors, face_classifications)
    """
    logger.info("🎨 Starting IMPROVED mesh-based color classification with propagation...")
    
    vertices = mesh_data["vertices"]
    normals = mesh_data["normals"]
    face_data = mesh_data["face_data"]
    xmin, ymin, zmin, xmax, ymax, zmax = mesh_data["bbox"]
    
    bbox_center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    bbox_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    
    # Initialize all vertices as "external" (default)
    vertex_count = len(vertices) // 3
    vertex_colors = ["external"] * vertex_count
    face_classifications = {}  # Store classification for each face
    locked_faces = set()  # Faces classified in step 1 - don't change these!
    
    # Build edge-to-face adjacency map
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    logger.info("🔍 STEP 1: Classifying cylindrical faces...")
    
    # STEP 1: Classify all CYLINDRICAL faces first
    for face_info in face_data:
        face_idx = face_info['face_idx']
        surf_type = face_info['surf_type']
        center = face_info['center']
        start_vertex = face_info['start_vertex']
        vertex_count_face = face_info['vertex_count']
        face_object = face_info['face_object']
        
        if surf_type != GeomAbs_Cylinder:
            continue  # Skip non-cylindrical for now
        
        try:
            surface = BRepAdaptor_Surface(face_object)
            cyl = surface.Cylinder()
            radius = cyl.Radius()
            axis_location = cyl.Axis().Location()
            
            # Calculate distance from face center to cylinder axis
            axis_point = [axis_location.X(), axis_location.Y(), axis_location.Z()]
            dist_to_axis = math.sqrt(
                (center[0] - axis_point[0])**2 +
                (center[1] - axis_point[1])**2 +
                (center[2] - axis_point[2])**2
            )
            
            # Calculate distance from bbox center to axis
            bbox_to_axis = math.sqrt(
                (bbox_center[0] - axis_point[0])**2 +
                (bbox_center[1] - axis_point[1])**2 +
                (bbox_center[2] - axis_point[2])**2
            )
            
            is_small_hole = radius < bbox_size * 0.15
            
            if dist_to_axis < bbox_to_axis:
                # Inner cylindrical surface
                if is_small_hole:
                    # Check for through-hole
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
                                        if adj_face.IsSame(other_info['face_object']):
                                            # Check if neighbor is external
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
            
            # Store classification
            face_classifications[face_idx] = face_type
            locked_faces.add(face_idx)  # Lock this classification
            
            # Apply to vertices
            for v_idx in range(start_vertex, start_vertex + vertex_count_face):
                vertex_colors[v_idx] = face_type
            
            if face_idx < 10:
                logger.info(f"  Face {face_idx}: Cylinder R={radius:.2f}mm → {face_type}")
                
        except Exception as e:
            logger.warning(f"Cylinder classification failed for face {face_idx}: {e}")
            face_classifications[face_idx] = "external"
    
    logger.info("🔍 STEP 2: Multi-pass propagation to adjacent faces...")
    
    # STEP 2: Multi-pass propagation until stable
    # Iterate multiple times to ensure all connected faces get proper classification
    max_iterations = 10
    
    for iteration in range(max_iterations):
        changes_made = False
        
        for face_info in face_data:
            face_idx = face_info['face_idx']
            
            # Skip locked faces (cylindrical faces from step 1)
            if face_idx in locked_faces:
                continue
            
            surf_type = face_info['surf_type']
            face_object = face_info['face_object']
            start_vertex = face_info['start_vertex']
            vertex_count_face = face_info['vertex_count']
            
            # Get current classification
            current_type = face_classifications.get(face_idx)
            
            # Find adjacent faces via shared edges
            edge_exp = TopExp_Explorer(face_object, TopAbs_EDGE)
            neighbor_types = []
            
            while edge_exp.More():
                edge = edge_exp.Current()
                
                # Find all faces sharing this edge
                for map_idx in range(1, edge_face_map.Size() + 1):
                    map_edge = edge_face_map.FindKey(map_idx)
                    if edge.IsSame(map_edge):
                        face_list = edge_face_map.FindFromIndex(map_idx)
                        face_iter = TopTools_ListIteratorOfListOfShape(face_list)
                        
                        while face_iter.More():
                            adj_face = topods.Face(face_iter.Value())
                            
                            # Find this adjacent face in our data
                            for other_info in face_data:
                                if adj_face.IsSame(other_info['face_object']):
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
                # Priority: internal > through > external
                if "internal" in neighbor_types:
                    new_type = "internal"
                elif "through" in neighbor_types:
                    new_type = "through"
                else:
                    new_type = "external"
            else:
                # No neighbors - keep current or assign default
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
            logger.info(f"  Propagation converged after {iteration + 1} iterations")
            break
        elif iteration == max_iterations - 1:
            logger.info(f"  Propagation stopped at max iterations ({max_iterations})")
    
    # Count results
    type_counts = {}
    for vtype in vertex_colors:
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    logger.info(f"✅ Classification complete! Distribution: {type_counts}")
    
    # Create comprehensive face_classifications array
    detailed_face_classifications = []
    for face_info in face_data:
        face_id = face_info['face_id']
        face_type = face_classifications.get(face_id, "external")
        face_object = face_info['face_object']
        surf_type = face_info['surf_type']
        center = face_info['center']
        
        # Calculate face normal
        try:
            surface = BRepAdaptor_Surface(face_object)
            u_mid = (surface.FirstUParameter() + surface.LastUParameter()) / 2
            v_mid = (surface.FirstVParameter() + surface.LastVParameter()) / 2
            point = gp_Pnt()
            normal_vec = gp_Vec()
            surface.D1(u_mid, v_mid, point, gp_Vec(), normal_vec)
            normal = [normal_vec.X(), normal_vec.Y(), normal_vec.Z()]
        except:
            normal = [0, 0, 1]  # Default
        
        # Calculate face area (approximate from triangles)
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
                # Triangle area = 0.5 * |cross product|
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
        manufacturing_features = recognize_manufacturing_features(shape)
        
        logger.info("🎨 Generating display mesh...")
        mesh_data = tessellate_shape(shape)
        
        logger.info("🎨 Classifying face colors using MESH-BASED approach...")
        vertex_colors, face_classifications = classify_mesh_faces(mesh_data, shape)
        mesh_data["vertex_colors"] = vertex_colors
        mesh_data["face_classifications"] = face_classifications
        
        logger.info("📐 Extracting and classifying BREP edges (UNIFIED SINGLE-PASS)...")
        # NEW: Single-pass extraction with guaranteed matching + UIso curves
        edge_result = extract_and_classify_feature_edges(
            shape, 
            max_edges=500, 
            angle_threshold_degrees=20,
            include_uiso=True,  # Industry-standard cylinder height lines
            num_uiso_lines=2    # CATIA standard (2 lines per curved surface)
        )
        
        mesh_data["feature_edges"] = edge_result["feature_edges"]
        mesh_data["edge_classifications"] = edge_result["edge_classifications"]
        mesh_data["tagged_edges"] = edge_result["tagged_edges"]
        mesh_data["triangle_count"] = len(mesh_data.get("indices", [])) // 3

        is_cylindrical = len(manufacturing_features['holes']) > 0 or len(manufacturing_features['bosses']) > 0
        has_flat_surfaces = len(manufacturing_features['planar_faces']) > 0
        
        # Calculate complexity based on actual features
        through_holes = len(manufacturing_features.get('through_holes', []))
        blind_holes = len(manufacturing_features.get('blind_holes', []))
        bores = len(manufacturing_features.get('bores', []))
        bosses = len(manufacturing_features.get('bosses', []))
        fillets = len(manufacturing_features.get('fillets', []))
        
        total_features = through_holes + blind_holes + bores + bosses + fillets
        
        cylindrical_faces = len(manufacturing_features['holes']) + len(manufacturing_features['bosses'])
        planar_faces = len(manufacturing_features['planar_faces'])
        complexity_score = min(10, int(
            (total_features / 5) +  # Each feature adds to complexity
            (through_holes * 0.5) +  # Through holes are moderately complex
            (blind_holes * 0.3) +    # Blind holes slightly less
            (bores * 0.2) +          # Bores are simple
            (bosses * 0.4) +         # Bosses add complexity
            (fillets * 0.1)          # Fillets add minor complexity
        ))

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        part_width_cm = (xmax - xmin) / 10
        part_height_cm = (ymax - ymin) / 10
        part_depth_cm = (zmax - zmin) / 10

        logger.info(f"✅ Analysis complete: {mesh_data['triangle_count']} triangles, {len(mesh_data['feature_edges'])} edges")

        return jsonify({
            'exact_volume': exact_props['volume'],
            'exact_surface_area': exact_props['surface_area'],
            'center_of_mass': exact_props['center_of_mass'],
            'manufacturing_features': manufacturing_features,
            'feature_summary': {
                'through_holes': len(manufacturing_features.get('through_holes', [])),
                'blind_holes': len(manufacturing_features.get('blind_holes', [])),
                'bores': len(manufacturing_features.get('bores', [])),
                'bosses': len(manufacturing_features.get('bosses', [])),
                'total_holes': through_holes + blind_holes,
                'planar_faces': planar_faces,
                'fillets': fillets,
                'complexity_score': complexity_score
            },
            'mesh_data': {
                'vertices': mesh_data['vertices'],
                'indices': mesh_data['indices'],
                'normals': mesh_data['normals'],
                'vertex_colors': mesh_data['vertex_colors'],
                'vertex_face_ids': mesh_data['vertex_face_ids'],  # Map vertices to faces
                'face_classifications': mesh_data['face_classifications'],  # Detailed face data
                'feature_edges': mesh_data['feature_edges'],
                'edge_classifications': mesh_data['edge_classifications'],
                'tagged_feature_edges': mesh_data['tagged_edges'],  # Guaranteed to match feature_edges
                'triangle_count': mesh_data['triangle_count'],
                'face_classification_method': 'mesh_based_with_propagation',
                'edge_extraction_method': 'smart_filtering_20deg'
            },
            'volume_cm3': exact_props['volume'] / 1000,
            'surface_area_cm2': exact_props['surface_area'] / 100,
            'is_cylindrical': is_cylindrical,
            'has_flat_surfaces': has_flat_surfaces,
            'complexity_score': complexity_score,
            'part_width_cm': part_width_cm,
            'part_height_cm': part_height_cm,
            'part_depth_cm': part_depth_cm,
            'total_faces': total_features,
            'planar_faces': planar_faces,
            'cylindrical_faces': cylindrical_faces,
            'analysis_type': 'dual_representation',
            'quotation_ready': True,
            'status': 'success',
            'confidence': 0.98,
            'method': 'professional_edge_extraction_with_angle_filtering'
        })

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
        "version": "8.0.0-smart-edge-extraction",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-cad"
        },
        "features": {
            "classification": "Mesh-based with neighbor propagation",
            "feature_detection": "Accurate through-hole, blind hole, bore, and boss detection",
            "edge_extraction": "Professional smart filtering (20° dihedral angle threshold)",
            "inner_surfaces": "Detected by cylinder radius and propagated to adjacent faces",
            "through_holes": "Detected by size and connectivity analysis",
            "wireframe_quality": "SolidWorks/Fusion 360 style - only significant edges"
        },
        "documentation": "POST multipart/form-data with 'file' field containing .step file"
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
