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


def count_circular_edges(face) -> int:
    """
    Count circular edges on a cylindrical face.
    - 2 circular edges = through hole (top + bottom openings)
    - 1 circular edge = blind hole (top opening only)
    """
    circular_count = 0
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    
    while explorer.More():
        edge_shape = topods.Edge(explorer.Current())
        curve = BRepAdaptor_Curve(edge_shape)
        
        if curve.GetType() == GeomAbs_Circle:
            circular_count += 1
        
        explorer.Next()
    
    return circular_count


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


def calculate_cylinder_depth(face) -> float:
    """Calculate depth of cylindrical face using bounding box."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    
    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    # Depth is along cylinder axis (typically Z, but calculate max dimension)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    
    return max(dx, dy, dz)


def has_planar_or_conical_cap(face, shape) -> bool:
    """
    Check if cylinder has a planar or conical end cap (indicates blind hole).
    Uses proper topology check, not edge counting.
    """
    from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cone
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
    
    # Build edge-face map
    ef_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ef_map)
    
    # Check circular edges for adjacent planar/conical faces
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
                    
                    # Skip the cylinder itself
                    if adj_face.IsSame(face):
                        face_iter.Next()
                        continue
                    
                    adj_surf = BRepAdaptor_Surface(adj_face)
                    surf_type = adj_surf.GetType()
                    
                    if surf_type in (GeomAbs_Plane, GeomAbs_Cone):
                        return True
                    
                    face_iter.Next()
        
        explorer.Next()
    
    return False


def validate_counterbore(cylinders: List[Dict], shape) -> bool:
    """
    Strict validation for counterbore features.
    
    Rules:
    1. Must have 2-3 steps (not 1, not 4+)
    2. Decreasing diameters monotonically
    3. Each step depth < 2× its diameter
    4. Total depth < 3× outer diameter
    
    Returns: True if valid counterbore, False if should go to volume decomposer
    """
    if len(cylinders) < 2 or len(cylinders) > 3:
        return False  # Must be 2-3 steps
    
    # Sort by radius (largest first)
    sorted_cyls = sorted(cylinders, key=lambda c: c['radius'], reverse=True)
    
    # Check monotonically decreasing diameters
    for i in range(len(sorted_cyls) - 1):
        if sorted_cyls[i]['radius'] <= sorted_cyls[i+1]['radius']:
            return False  # Not strictly decreasing
        
        # Check reasonable diameter ratio (next step should be meaningfully smaller)
        ratio = sorted_cyls[i+1]['radius'] / sorted_cyls[i]['radius']
        if ratio > 0.95:  # Steps too similar
            return False
    
    # Check depth constraints
    total_depth = 0
    for cyl in sorted_cyls:
        depth = calculate_cylinder_depth(cyl['face'])
        
        # Each step depth < 2× its diameter
        if depth > 2.0 * cyl['radius'] * 2:  # diameter = 2 * radius
            return False  # Step too deep for its diameter
        
        total_depth += depth
    
    # Total depth < 3× outer diameter
    outer_diameter = sorted_cyls[0]['radius'] * 2
    if total_depth > 3.0 * outer_diameter:
        return False  # Too deep, likely a pocket
    
    return True


def recognize_simple_features(shape) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Recognize SIMPLE geometric features with strict validation.
    
    Scope: Only simple, unambiguous features
    - Simple through/blind holes
    - Validated 2-3 step counterbores
    - Simple countersinks
    - Fillets
    
    Complex/ambiguous features rejected → Volume decomposer handles them
    
    Returns: (holes, fillets, countersinks, tapered_holes)
    """
    logger.info("=" * 70)
    logger.info("GEOMETRIC RECOGNIZER - Production-Grade Validation")
    logger.info("=" * 70)
    
    cylinders, cones = extract_cylinders_and_cones(shape)
    
    # Step 1: Match cones with coaxial cylinders
    countersinks = []
    blind_holes_with_cone = []
    consumed_cylinder_ids = set()
    consumed_cone_ids = set()
    
    for cone in cones:
        if cone['orientation'] != TopAbs_REVERSED:
            continue
        
        matched_cylinder = None
        for cyl in cylinders:
            if cyl['face_id'] in consumed_cylinder_ids:
                continue
            if cyl['orientation'] != TopAbs_REVERSED:
                continue
            
            if axes_are_coaxial(cone['axis'], cyl['axis']):
                matched_cylinder = cyl
                break
        
        if matched_cylinder:
            # Check if they share an edge
            shares_edge = False
            
            cone_explorer = TopExp_Explorer(cone['face'], TopAbs_EDGE)
            cone_edges = set()
            while cone_explorer.More():
               cone_edges.add(cone_explorer.Current())
                cone_explorer.Next()
            
            cyl_explorer = TopExp_Explorer(matched_cylinder['face'], TopAbs_EDGE)
            while cyl_explorer.More():
                if cyl_explorer.Current() in cone_edges:
                    shares_edge = True
                    break
                cyl_explorer.Next()
            
            if shares_edge:
                # Countersink
                angle_deg = math.degrees(cone['semi_angle']) * 2
                countersinks.append({
                    'type': 'countersink',
                    'face_ids': [cone['face_id'], matched_cylinder['face_id']],
                    'cone_angle': angle_deg,
                    'hole_radius': matched_cylinder['radius']
                })
            else:
                # Blind hole with conical bottom
                blind_holes_with_cone.append({
                    'type': 'blind_hole',
                    'face_ids': [matched_cylinder['face_id'], cone['face_id']],
                    'radius': matched_cylinder['radius'],
                    'cylinders': [matched_cylinder],
                    'has_conical_bottom': True
                })
            
            consumed_cone_ids.add(cone['face_id'])
            consumed_cylinder_ids.add(matched_cylinder['face_id'])
    
    logger.info(f"Found {len(countersinks)} countersinks")
    logger.info(f"Found {len(blind_holes_with_cone)} blind holes with conical bottoms")
    
    # Step 2: Standalone cones → Tapered holes
    tapered_holes = []
    for cone in cones:
        if cone['face_id'] in consumed_cone_ids:
            continue
        if cone['orientation'] != TopAbs_REVERSED:
            continue
        
        tapered_holes.append({
            'type': 'tapered_hole',
            'face_ids': [cone['face_id']],
            'angle': math.degrees(cone['semi_angle']) * 2
        })
    
    logger.info(f"Found {len(tapered_holes)} tapered holes")
    
    # Step 3: Classify remaining cylinders
    hole_cylinders = []
    fillet_cylinders = []
    
    for cyl in cylinders:
        if cyl['face_id'] in consumed_cylinder_ids:
            continue
        
        if has_closed_circular_edge(cyl['face']):
            if cyl['orientation'] == TopAbs_REVERSED:
                hole_cylinders.append(cyl)
            continue  # Skip bosses
        
        if has_arc_edge(cyl['face']):
            fillet_cylinders.append(cyl)
    
    logger.info(f"Found {len(hole_cylinders)} potential hole cylinders")
    
    # Step 4: Group and VALIDATE coaxial holes
    hole_groups = group_coaxial_cylinders(hole_cylinders)
    
    holes = []
    holes.extend(blind_holes_with_cone)  # Already validated
    
    rejected_count = 0
    
    for group in hole_groups:
        sorted_group = sorted(group, key=lambda c: c['radius'], reverse=True)
        
        if len(group) == 1:
            # Single cylinder - through or blind
            cyl = group[0]
            depth = calculate_cylinder_depth(cyl['face'])
            diameter = cyl['radius'] * 2
            
            # Diameter validation
            if diameter < 0.5 or diameter > 100:  # Reasonable hole range in mm
                logger.debug(f"Rejected hole: diameter {diameter:.1f}mm out of range")
                rejected_count += 1
                continue
            
            # Depth/diameter ratio check
            depth_ratio = depth / diameter
            if depth_ratio > 10:  # Unreasonably deep for a hole
                logger.debug(f"Rejected deep hole: depth/diameter ratio {depth_ratio:.1f}")
                rejected_count += 1
                continue
            
            # Check if blind or through
            has_cap = has_planar_or_conical_cap(cyl['face'], shape)
            
            holes.append({
                'type': 'blind_hole' if has_cap else 'through_hole',
                'face_ids': [cyl['face_id']],
                'radius': cyl['radius'],
                'cylinders': [cyl]
            })
        
        else:
            # Multi-cylinder - validate as counterbore
            if validate_counterbore(sorted_group, shape):
                holes.append({
                    'type': 'counterbore',
                    'face_ids': [c['face_id'] for c in sorted_group],
                    'radius': sorted_group[-1]['radius'],  # Smallest (primary hole)
                    'cylinders': sorted_group
                })
            else:
                logger.debug(f"Rejected invalid counterbore: {len(group)} steps failed validation")
                rejected_count += 1
    
    # Fillets
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
    
    logger.info(f"✅ Recognized {len(holes)} holes ({through} through, {blind} blind, {counterbore} counterbored)")
    logger.info(f"✅ Recognized {len(fillets)} fillets")
    logger.info(f"✅ Recognized {len(countersinks)} countersinks")
    logger.info(f"✅ Recognized {len(tapered_holes)} tapered holes")
    logger.info(f"⚠️  Rejected {rejected_count} invalid features (→ volume decomposer)")
    
    return holes, fillets, countersinks, tapered_holes
