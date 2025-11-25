"""
Geometric Feature Recognizer - Block 1

Simple, modular recognition of holes and fillets using edge closure analysis.
Each function does ONE thing only.
"""
import logging
from typing import List, Tuple, Dict
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Circle
from OCC.Core.TopoDS import topods
from occwl.edge import Edge

logger = logging.getLogger(__name__)


def extract_cylinders(shape) -> List[Dict]:
    """
    Extract all cylindrical faces from a shape.
    
    Simple function - ONLY extracts, doesn't analyze.
    
    Args:
        shape: TopoDS_Shape from OCC
        
    Returns:
        List of dicts with cylinder info
    """
    cylinders = []
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    
    while explorer.More():
        face = topods.Face(explorer.Current())
        surf = BRepAdaptor_Surface(face)
        
        if surf.GetType() == GeomAbs_Cylinder:
            cylinder = surf.Cylinder()
            
            cyl_info = {
                'face': face,
                'face_id': face_id,
                'radius': cylinder.Radius(),
                'axis': cylinder.Axis(),
                'location': cylinder.Location()
            }
            cylinders.append(cyl_info)
        
        explorer.Next()
        face_id += 1
    
    logger.info(f"Extracted {len(cylinders)} cylindrical faces")
    return cylinders


def has_closed_circular_edge(face) -> bool:
    """
    Check if a face has a closed circular edge (360° circle).
    
    Simple geometric check - no topology.
    
    Args:
        face: TopoDS_Face
        
    Returns:
        True if has closed circle, False otherwise
    """
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        edge = Edge(edge_shape)
        
        # Check if edge is circular
        curve = BRepAdaptor_Curve(edge_shape)
        if curve.GetType() == GeomAbs_Circle:
            # Check if it's closed (360°)
            if edge.closed_curve() or edge.closed_edge():
                return True
        
        explorer.Next()
    
    return False


def has_arc_edge(face) -> bool:
    """
    Check if a face has arc edges (< 360°).
    
    Simple geometric check - no topology.
    
    Args:
        face: TopoDS_Face
        
    Returns:
        True if has arc edges, False otherwise
    """
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        edge = Edge(edge_shape)
        
        # Check if edge is circular
        curve = BRepAdaptor_Curve(edge_shape)
        if curve.GetType() == GeomAbs_Circle:
            # Check if it's an arc (NOT closed)
            if not (edge.closed_curve() or edge.closed_edge()):
                return True
        
        explorer.Next()
    
    return False


def classify_cylinder(cyl_info: Dict) -> str:
    """
    Classify a cylinder as hole or fillet based on edge closure.
    
    Simple decision: closed circle = hole, arc = fillet.
    
    Args:
        cyl_info: Dict with 'face' key
        
    Returns:
        'hole', 'fillet', or 'unknown'
    """
    face = cyl_info['face']
    face_id = cyl_info['face_id']
    
    # DIAGNOSTIC: Log edge details for first few and specific candidates
    is_target = face_id == 66 or face_id < 5
    
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    edge_count = 0
    full_circles = 0
    arcs = 0
    
    if is_target:
        logger.info(f"--- Analyzing Cylinder {face_id} ---")
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        edge = Edge(edge_shape)
        edge_count += 1
        
        curve_adaptor = BRepAdaptor_Curve(edge_shape)
        curve_type = curve_adaptor.GetType()
        
        if curve_type == GeomAbs_Circle:
            is_closed_edge = edge.closed_edge()
            is_closed_curve = edge.closed_curve()
            
            first = curve_adaptor.FirstParameter()
            last = curve_adaptor.LastParameter()
            angle = abs(last - first)
            is_full_period = angle > 6.28  # > 2*PI - epsilon
            
            if is_target:
                logger.info(f"  Edge {edge_count}: Circle, Angle={angle:.2f}, ClosedEdge={is_closed_edge}, ClosedCurve={is_closed_curve}")
            
            # CRITICAL FIX: closed_curve() checks underlying geometry (always True for circle)
            # We must use closed_edge() (topology) or angle check
            if is_closed_edge or is_full_period:
                full_circles += 1
            else:
                arcs += 1
        else:
            if is_target:
                logger.info(f"  Edge {edge_count}: Type {curve_type} (Not Circle)")
        
        explorer.Next()
    
    if full_circles > 0:
        if is_target: logger.info(f"  -> HOLE (Found {full_circles} full circles)")
        return 'hole'
    elif arcs > 0:
        if is_target: logger.info(f"  -> FILLET (Found {arcs} arcs, 0 full circles)")
        return 'fillet'
    else:
        if is_target: logger.info(f"  -> UNKNOWN (No circular edges)")
        return 'unknown'


def recognize_simple_features(shape) -> Tuple[List[Dict], List[Dict]]:
    """
    Main entry point - recognize holes and fillets.
    
    Simple orchestration of focused functions.
    
    Args:
        shape: TopoDS_Shape
        
    Returns:
        (holes, fillets) - two separate lists
    """
    logger.info("=" * 70)
    logger.info("GEOMETRIC RECOGNIZER - Simple Holes & Fillets")
    logger.info("=" * 70)
    
    # Step 1: Extract all cylinders (simple function)
    cylinders = extract_cylinders(shape)
    
    # Step 2: Classify each cylinder (simple function)
    holes = []
    fillets = []
    unknown = []
    
    for cyl in cylinders:
        classification = classify_cylinder(cyl)
        
        if classification == 'hole':
            holes.append(cyl)
        elif classification == 'fillet':
            fillets.append(cyl)
        else:
            unknown.append(cyl)
    
    # Log results
    logger.info(f"Recognized {len(holes)} holes")
    logger.info(f"Recognized {len(fillets)} fillets")
    if unknown:
        logger.warning(f"Unclassified: {len(unknown)} cylinders")
    
    return holes, fillets
