"""
Geometric Feature Recognizer - Block 1

Simple, focused recognition using edge closure analysis:
1. Holes: Cylindrical faces with closed circular edges (360°) + reversed orientation
2. Fillets: Cylindrical faces with arc edges (< 360°)
3. Counterbores: Co-axial hole cylinders grouped by radius
4. Countersinks: Conical faces with reversed orientation
"""
import logging
import math
from typing import List, Dict, Tuple

from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Circle, GeomAbs_Cone, GeomAbs_Plane
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Ax1
from occwl.edge import Edge

logger = logging.getLogger(__name__)


def extract_cylinders_and_cones(shape) -> Tuple[List[Dict], List[Dict]]:
    """Extract cylindrical and conical faces from a shape."""
    cylinders = []
    cones = []
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
                'axis_direction': axis.Direction(),
                'orientation': face.Orientation()
            }
            cylinders.append(cyl_info)
        
        elif surf.GetType() == GeomAbs_Cone:
            cone = surf.Cone()
            axis = cone.Axis()
            
            cone_info = {
                'face': face,
                'face_id': face_id,
                'semi_angle': cone.SemiAngle(),
                'axis': axis,
                'axis_location': axis.Location(),
                'axis_direction': axis.Direction(),
                'orientation': face.Orientation()
            }
            cones.append(cone_info)
        
        explorer.Next()
        face_id += 1
    
    logger.info(f" Extracted {len(cylinders)} cylindrical faces, {len(cones)} conical faces")
    return cylinders, cones


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


def count_end_caps(face, shape) -> int:
    """
    Count planar/conical end caps on a cylindrical face.
    - 0 caps = through hole
    - 1+ caps = blind hole (flat or conical bottom)
    """
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
    
    ef_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ef_map)
    
    caps = 0
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        curve = BRepAdaptor_Curve(edge_shape)
        
        if curve.GetType() == GeomAbs_Circle:
            if ef_map.Contains(edge_shape):
                faces = ef_map.FindFromKey(edge_shape)
                face_iter = TopTools_ListIteratorOfListOfShape(faces)
                
                while face_iter.More():
                    adj_face = topods.Face(face_iter.Value())
                    adj_surf = BRepAdaptor_Surface(adj_face)
                    
                    # Planar or conical bottom
                    if adj_surf.GetType() in (GeomAbs_Plane, GeomAbs_Cone):
                        caps += 1
                        break
                    
                    face_iter.Next()
        
        explorer.Next()
    
    return caps


def axes_are_coaxial(axis1: gp_Ax1, axis2: gp_Ax1, tolerance=1e-3) -> bool:
    """Check if two axes are co-axial."""
    dir1 = axis1.Direction()
    dir2 = axis2.Direction()
    
    if abs(dir1.Dot(dir2)) < 0.999:
        return False
    
    loc1 = axis1.Location()
    loc2 = axis2.Location()
    
    dx, dy, dz = loc2.X() - loc1.X(), loc2.Y() - loc1.Y(), loc2.Z() - loc1.Z()
    proj = dx * dir1.X() + dy * dir1.Y() + dz * dir1.Z()
    
    perp_dist = math.sqrt(
        (dx - proj * dir1.X())**2 +
        (dy - proj * dir1.Y())**2 +
        (dz - proj * dir1.Z())**2
    )
    
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
            if j not in used and axes_are_coaxial(cyl1['axis'], cyl2['axis']):
                group.append(cyl2)
                used.add(j)
        
        groups.append(group)
    
    return groups


def recognize_simple_features(shape) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Recognize holes, fillets, countersinks, and tapered holes.
    
    Returns:
        (holes, fillets, countersinks, tapered_holes)
        
    Hole Types:
        - Through hole: Cylinder with 0 end caps
        - Blind hole: Cylinder with 1+ end caps (planar or conical bottom)
        - Counterbore: Multiple coaxial cylinders
        - Countersink: Cone + coaxial cylinder below
        - Tapered hole: Cone only (no cylinder below)
    """
    logger.info("=" * 70)
    logger.info("GEOMETRIC RECOGNIZER - Edge Closure Analysis")
    logger.info("=" * 70)
    
    cylinders, cones = extract_cylinders_and_cones(shape)
    
    # Step 1: Identify countersink holes (cone + coaxial cylinder pairs)
    countersinks = []
    consumed_cylinder_ids = set()
    consumed_cone_ids = set()
    
    for cone in cones:
        if cone['orientation'] != TopAbs_REVERSED:
            continue  # Only reversed cones (normal IN) are holes
        
        # Find coaxial cylinder below this cone
        matched_cylinder = None
        for cyl in cylinders:
            if cyl['face_id'] in consumed_cylinder_ids:
                continue
            if cyl['orientation'] != TopAbs_REVERSED:
                continue  # Only reversed cylinders
            
            # Check if coaxial
            if axes_are_coaxial(cone['axis'], cyl['axis']):
                matched_cylinder = cyl
                break
        
        if matched_cylinder:
            # This is a countersink hole (cone + cylinder)
            angle_deg = math.degrees(cone['semi_angle']) * 2
            
            countersinks.append({
                'type': 'countersink',
                'face_ids': [cone['face_id'], matched_cylinder['face_id']],
                'cone_angle': angle_deg,
                'hole_radius': matched_cylinder['radius']
            })
            
            consumed_cone_ids.add(cone['face_id'])
            consumed_cylinder_ids.add(matched_cylinder['face_id'])
    
    logger.info(f"Found {len(countersinks)} countersink holes (cone + cylinder pairs)")
    
    # Step 2: Identify tapered holes (cones without coaxial cylinders)
    tapered_holes = []
    for cone in cones:
        if cone['face_id'] in consumed_cone_ids:
            continue
        if cone['orientation'] != TopAbs_REVERSED:
            continue
        
        angle_deg = math.degrees(cone['semi_angle']) * 2
        tapered_holes.append({
            'type': 'tapered_hole',
            'face_ids': [cone['face_id']],
            'angle': angle_deg
        })
    
    logger.info(f"Found {len(tapered_holes)} tapered holes (cone only)")
    
    # Step 3: Classify remaining cylinders (excluding those in countersinks)
    hole_cylinders = []
    fillet_cylinders = []
    
    for cyl in cylinders:
        if cyl['face_id'] in consumed_cylinder_ids:
            continue  # Skip cylinders that are part of countersinks
        
        if has_closed_circular_edge(cyl['face']):
            # Closed 360° - hole or boss
            if cyl['orientation'] == TopAbs_REVERSED:
                hole_cylinders.append(cyl)
        elif has_arc_edge(cyl['face']):
            fillet_cylinders.append(cyl)
    
    logger.info(f"Found {len(hole_cylinders)} hole cylinders")
    logger.info(f"Found {len(fillet_cylinders)} fillet cylinders")
    
    # Step 4: Group and classify cylindrical holes
    hole_groups = group_coaxial_cylinders(hole_cylinders)
    
    holes = []
    for group in hole_groups:
        sorted_group = sorted(group, key=lambda c: c['radius'])
        primary = sorted_group[0]
        
        cap_count = count_end_caps(primary['face'], shape)
        is_blind = cap_count > 0
        
        hole_type = 'counterbore' if len(group) > 1 else ('blind_hole' if is_blind else 'through_hole')
        
        holes.append({
            'type': hole_type,
            'face_ids': [c['face_id'] for c in sorted_group],
            'radius': primary['radius'],
            'cylinders': sorted_group
        })
    
    # Step 5: Create fillets
    fillets = []
    for cyl in fillet_cylinders:
        fillets.append({
            'face_id': cyl['face_id'],
            'radius': cyl['radius']
        })
    
    # Log results
    through = sum(1 for h in holes if h['type'] == 'through_hole')
    blind = sum(1 for h in holes if h['type'] == 'blind_hole')
    counterbore = sum(1 for h in holes if h['type'] == 'counterbore')
    
    logger.info(f"Recognized {len(holes)} cylindrical holes ({through} through, {blind} blind, {counterbore} counterbored)")
    logger.info(f"Recognized {len(fillets)} fillets")
    logger.info(f"Recognized {len(countersinks)} countersink holes")
    logger.info(f"Recognized {len(tapered_holes)} tapered holes")
    
    return holes, fillets, countersinks, tapered_holes
