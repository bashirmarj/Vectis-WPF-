"""
Geometric Feature Recognizer - Block 1

Simple, focused recognition using edge closure analysis:
1. Holes: Cylindrical faces with closed circular edges (360°)
2. Fillets: Cylindrical faces with arc edges (< 360°)
3. Counterbores: Co-axial hole cylinders grouped by radius
"""
import logging
import math
from typing import List, Dict

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Circle
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Ax1
from occwl.edge import Edge

logger = logging.getLogger(__name__)


def extract_cylinders(shape) -> List[Dict]:
    """Extract all cylindrical faces with their geometric properties."""
    cylinders = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    
    while explorer.More():
        face = topods.Face(explorer.Current())
        surf = BRepAdaptor_Surface(face)
        
        if surf.GetType() == GeomAbs_Cylinder:
            cylinder = surf.Cylinder()
            axis = cylinder.Axis()
            
            cyl_info = {
                'face': face,
                'face_id': face_id,
                'radius': cylinder.Radius(),
                'axis': axis,
                'axis_location': axis.Location(),
                'axis_direction': axis.Direction()
            }
            cylinders.append(cyl_info)
        
        explorer.Next()
        face_id += 1
    
    logger.info(f"Extracted {len(cylinders)} cylindrical faces")
    return cylinders


def has_closed_circular_edge(face) -> bool:
    """Check if face has a 360° closed circular edge."""
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        edge = Edge(edge_shape)
        
        curve = BRepAdaptor_Curve(edge_shape)
        if curve.GetType() == GeomAbs_Circle:
            if edge.closed_edge():
                return True
        
        explorer.Next()
    
    return False


def has_arc_edge(face) -> bool:
    """Check if face has an arc edge (< 360°)."""
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        edge = Edge(edge_shape)
        
        curve = BRepAdaptor_Curve(edge_shape)
        if curve.GetType() == GeomAbs_Circle:
            if not edge.closed_edge():
                return True
        
        explorer.Next()
    
    return False


def axes_are_coaxial(axis1: gp_Ax1, axis2: gp_Ax1, tolerance=1e-3) -> bool:
    """Check if two axes are co-axial."""
    # Check parallel directions
    dir1 = axis1.Direction()
    dir2 = axis2.Direction()
    dot = abs(dir1.Dot(dir2))
    if dot < 0.999:
        return False
    
    # Check if on same line
    loc1 = axis1.Location()
    loc2 = axis2.Location()
    
    dx = loc2.X() - loc1.X()
    dy = loc2.Y() - loc1.Y()
    dz = loc2.Z() - loc1.Z()
    
    proj_length = dx * dir1.X() + dy * dir1.Y() + dz * dir1.Z()
    
    perp_x = dx - proj_length * dir1.X()
    perp_y = dy - proj_length * dir1.Y()
    perp_z = dz - proj_length * dir1.Z()
    perp_dist = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
    
    return perp_dist < tolerance


def group_coaxial_cylinders(cylinders: List[Dict]) -> List[List[Dict]]:
    """Group cylinders that are co-axial."""
    groups = []
    used = set()
    
    for i, cyl1 in enumerate(cylinders):
        if i in used:
            continue
        
        group = [cyl1]
        used.add(i)
        
        for j, cyl2 in enumerate(cylinders):
            if j in used:
                continue
            
            if axes_are_coaxial(cyl1['axis'], cyl2['axis']):
                group.append(cyl2)
                used.add(j)
        
        groups.append(group)
    
    return groups


def recognize_simple_features(shape):
    """
    Recognize holes and fillets using simple edge closure analysis.
    
    Returns:
        (holes, fillets)
    """
    logger.info("=" * 70)
    logger.info("GEOMETRIC RECOGNIZER - Edge Closure Analysis")
    logger.info("=" * 70)
    
    # Extract all cylinders
    all_cylinders = extract_cylinders(shape)
    
    # Classify by edge type
    hole_cylinders = []
    fillet_cylinders = []
    
    for cyl in all_cylinders:
        if has_closed_circular_edge(cyl['face']):
            hole_cylinders.append(cyl)
        elif has_arc_edge(cyl['face']):
            fillet_cylinders.append(cyl)
    
    logger.info(f"Found {len(hole_cylinders)} hole cylinders (closed circles)")
    logger.info(f"Found {len(fillet_cylinders)} fillet cylinders (arcs)")
    
    # Group co-axial holes for counterbores
    hole_groups = group_coaxial_cylinders(hole_cylinders)
    
    # Create hole features
    holes = []
    for group in hole_groups:
        sorted_group = sorted(group, key=lambda c: c['radius'])
        
        hole_info = {
            'type': 'counterbore' if len(group) > 1 else 'through_hole',
            'face_ids': [c['face_id'] for c in sorted_group],
            'radius': sorted_group[0]['radius'],  # Smallest radius
            'cylinders': sorted_group
        }
        holes.append(hole_info)
    
    # Create fillet features
    fillets = []
    for cyl in fillet_cylinders:
        fillet_info = {
            'face_id': cyl['face_id'],
            'radius': cyl['radius']
        }
        fillets.append(fillet_info)
    
    # Log results
    through_holes = sum(1 for h in holes if h['type'] == 'through_hole')
    counterbores = sum(1 for h in holes if h['type'] == 'counterbore')
    
    logger.info(f"Recognized {len(holes)} holes ({through_holes} through, {counterbores} counterbored)")
    logger.info(f"Recognized {len(fillets)} fillets")
    
    return holes, fillets
