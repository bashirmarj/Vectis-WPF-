"""
Geometric Feature Recognizer - Block 1

Simple, modular recognition of holes, fillets, and bosses using geometric and topological analysis.
Distinguishes features based on:
1. Geometry: Cylinder parameters.
2. Topology: Edge closure (360° vs Arc).
3. Orientation: Normal direction (Hole vs Boss/Fillet).
4. Continuity: Edge smoothness (Fillet vs Boss).
"""
import logging
from typing import List, Tuple, Dict

from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Circle, GeomAbs_Line, GeomAbs_C0
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from occwl.edge import Edge

logger = logging.getLogger(__name__)


def extract_cylinders(shape) -> List[Dict]:
    """
    Extract all cylindrical faces from a shape.
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


def build_edge_face_map(shape):
    """
    Build a map of Edge -> List[Face] to query adjacency.
    """
    ef_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ef_map)
    return ef_map


def get_edge_continuity(edge_shape, f1, f2) -> bool:
    """
    Check if the edge connection between two faces is smooth (G1/C1 or higher).
    Returns True if smooth, False if sharp (C0).
    """
    # BRep_Tool.Continuity returns a GeomAbs_Shape enum
    # GeomAbs_C0 = 0 (Sharp)
    # GeomAbs_G1 = 1, GeomAbs_C1 = 2, etc. (Smooth)
    try:
        continuity = BRep_Tool.Continuity(edge_shape, f1, f2)
        return continuity > GeomAbs_C0
    except Exception:
        return False


def classify_cylinder(cyl_info: Dict, ef_map) -> str:
    """
    Classify a cylinder as hole, fillet, or boss based on robust geometric rules.
    
    Definitions:
    - HOLE: Cylinder with Normal pointing IN (Material Outside).
    - BOSS: Cylinder with Normal pointing OUT (Material Inside) AND:
            - Is a closed 360° cylinder, OR
            - Has SHARP linear edges (distinct feature boundary).
    - FILLET: Cylinder with Normal pointing OUT (Material Inside) AND:
            - Is an open arc (< 360°), AND
            - Has SMOOTH linear edges (tangent blend).
    """
    face = cyl_info['face']
    face_id = cyl_info['face_id']
    
    # DIAGNOSTIC: Log details for specific candidates
    is_target = face_id == 66 or face_id == 52 or face_id < 5
    
    if is_target:
        logger.info(f"--- Analyzing Cylinder {face_id} ---")

    # STEP 1: Check if it's a closed 360° cylinder or a partial arc
    # This is the PRIMARY distinction
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    has_sharp_linear_edge = False
    has_smooth_linear_edge = False
    is_closed_cylinder = False
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        edge = Edge(edge_shape)
        
        curve_adaptor = BRepAdaptor_Curve(edge_shape)
        curve_type = curve_adaptor.GetType()
        
        if curve_type == GeomAbs_Circle:
            # Check for 360° closure (Seam edge of a full cylinder)
            if edge.closed_edge():
                is_closed_cylinder = True
                if is_target: logger.info(f"  Found 360° seam edge (closed cylinder)")
            
        elif curve_type == GeomAbs_Line:
            # Linear edge (side boundary of a partial cylinder)
            # Check continuity with neighbors
            if ef_map.Contains(edge_shape):
                faces = ef_map.FindFromKey(edge_shape)
                if faces.Size() == 2:
                    # Use proper iterator for TopTools_ListOfShape
                    face_iter = TopTools_ListIteratorOfListOfShape(faces)
                    f1 = topods.Face(face_iter.Value())
                    face_iter.Next()
                    f2 = topods.Face(face_iter.Value())
                    
                    is_smooth = get_edge_continuity(edge_shape, f1, f2)
                    
                    if is_smooth:
                        has_smooth_linear_edge = True
                        if is_target: logger.info(f"  Linear Edge: SMOOTH (Fillet-like)")
                    else:
                        has_sharp_linear_edge = True
                        if is_target: logger.info(f"  Linear Edge: SHARP (Boss-like)")
        
        explorer.Next()
    
    # STEP 2: Decision Logic - Hierarchy is Critical
    
    # A. Closed 360° Cylinder -> Use orientation to distinguish Hole vs Boss
    if is_closed_cylinder:
        orientation = face.Orientation()
        is_reversed = (orientation == TopAbs_REVERSED)
        
        if is_reversed:
            if is_target: logger.info(f"  -> HOLE (360° closed + Normal IN)")
            return 'hole'
        else:
            if is_target: logger.info(f"  -> BOSS (360° closed + Normal OUT)")
            return 'boss'
    
    # B. Partial Cylinder -> Use edge continuity to distinguish Fillet vs Boss
    # (Orientation is NOT reliable for partial cylinders - fillets can be reversed!)
    
    if has_sharp_linear_edge:
        if is_target: logger.info(f"  -> BOSS (Partial cylinder with sharp linear edges)")
        return 'boss'
    
    if has_smooth_linear_edge:
        if is_target: logger.info(f"  -> FILLET (Partial cylinder with smooth linear edges)")
        return 'fillet'
        
    # D. Fallback (e.g. floating face, bad topology)
    if is_target: logger.info(f"  -> UNKNOWN (Ambiguous topology)")
    return 'unknown'


def recognize_simple_features(shape) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Main entry point - recognize holes, fillets, and bosses.
    
    Args:
        shape: TopoDS_Shape
        
    Returns:
        (holes, fillets, bosses)
    """
    logger.info("=" * 70)
    logger.info("GEOMETRIC RECOGNIZER - Robust Topology Analysis")
    logger.info("=" * 70)
    
    # Step 0: Build Edge-Face Map (Global for shape)
    # Needed for continuity checks
    ef_map = build_edge_face_map(shape)
    
    # Step 1: Extract all cylinders
    cylinders = extract_cylinders(shape)
    
    # Step 2: Classify each cylinder
    holes = []
    fillets = []
    bosses = []
    unknown = []
    
    for cyl in cylinders:
        classification = classify_cylinder(cyl, ef_map)
        
        if classification == 'hole':
            holes.append(cyl)
        elif classification == 'fillet':
            fillets.append(cyl)
        elif classification == 'boss':
            bosses.append(cyl)
        else:
            unknown.append(cyl)
    
    # Log results
    logger.info(f"Recognized {len(holes)} holes")
    logger.info(f"Recognized {len(fillets)} fillets")
    logger.info(f"Recognized {len(bosses)} bosses")
    if unknown:
        logger.warning(f"Unclassified: {len(unknown)} cylinders")
    
    return holes, fillets, bosses
