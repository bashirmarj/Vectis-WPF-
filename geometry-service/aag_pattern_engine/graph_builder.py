"""
AAG Graph Builder - Analysis Situs Aligned
==========================================

CRITICAL FIX: Vexity classification thresholds

OLD (WRONG):
- Smooth threshold: 5 degrees → 61% smooth edges
- Convex count: 15% (way too low!)

NEW (Analysis Situs):
- Smooth threshold: 1 degree → ~30-40% smooth
- Convex count: Should be ~30-40% for typical parts
- Concave count: ~20-30%
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from OCC.Core.TopoDS import topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopExp import topexp
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Vec, gp_Pnt2d
from OCC.Core.GeomLProp import GeomLProp_SLProps

logger = logging.getLogger(__name__)


# ===== BACKWARD COMPATIBILITY TYPES =====
# These types are needed by slot_recognizer.py, fillet_chamfer_recognizer.py,
# and turning_recognizer.py which use the old typed API

class SurfaceType(Enum):
    """Surface type enumeration for backward compatibility"""
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    BSPLINE = "bspline"
    UNKNOWN = "unknown"


class Vexity(Enum):
    """Edge vexity classification for backward compatibility"""
    CONVEX = "convex"
    CONCAVE = "concave"
    SMOOTH = "smooth"


@dataclass
class GraphNode:
    """Backward-compatible node representation"""
    face_id: int
    surface_type: SurfaceType
    area: float
    normal: Tuple[float, float, float]
    center: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    axis: Optional[Tuple[float, float, float]] = None
    angle_deg: Optional[float] = None
    id: Optional[int] = None  # Alias for face_id for backward compatibility
    
    def __post_init__(self):
        """Ensure id matches face_id for legacy recognizer compatibility"""
        if self.id is None:
            self.id = self.face_id


@dataclass
class GraphEdge:
    """Backward-compatible edge representation"""
    face1: int
    face2: int
    vexity: Vexity
    dihedral_deg: float


# CRITICAL: Analysis Situs thresholds
SMOOTH_ANGLE_THRESHOLD = 1.0  # degrees (was 5.0 - too permissive!)
CONVEX_THRESHOLD = 180.0 + SMOOTH_ANGLE_THRESHOLD  # 181.0°
CONCAVE_THRESHOLD = 180.0 - SMOOTH_ANGLE_THRESHOLD  # 179.0°


class AAGGraphBuilder:
    """
    Builds Attributed Adjacency Graph from removal volume.
    
    Nodes: Faces with attributes (area, normal, surface_type, vexity_stats, etc.)
    Edges: Adjacency with dihedral angles and vexity classification
    """
    
    def __init__(self, shape, tolerance: float = 1e-6):
        """
        Args:
            shape: TopoDS_Shape (removal volume)
            tolerance: Geometric tolerance
        """
        self.shape = shape
        self.tolerance = tolerance
        
        # Graph storage
        self.nodes = {}  # face_id -> attributes
        self.adjacency = defaultdict(list)  # face_id -> [neighbor dicts]
        self.edges = []  # All edges for statistics
        
        # Topology caching
        self._face_cache = []
        self._edge_face_map = {}
        
    def build(self) -> Dict:
        """
        Build complete AAG graph.
        
        Returns:
            Graph dict: {
                'nodes': {face_id: attributes},
                'adjacency': {face_id: [neighbors]},
                'statistics': {...}
            }
        """
        logger.info("=" * 70)
        logger.info("Building Attributed Adjacency Graph (AAG) - PATCHED v1.1.0")
        logger.info("=" * 70)
        
        # Step 1: Cache face list
        logger.info("Caching face list...")
        self._build_face_cache()
        logger.info(f"Cached {len(self._face_cache)} faces")
        
        # Step 2: Build edge-face topology
        logger.info("Building edge-face topology map...")
        self._build_edge_face_map()
        
        # Step 3: Extract face attributes
        logger.info("Extracting and validating faces...")
        valid_count = self._extract_face_attributes()
        logger.info(f"Extracted {valid_count} valid faces")
        logger.info(f"Filtered {len(self._face_cache) - valid_count} degenerate faces")
        
        # Step 4: Build adjacency with dihedral angles
        logger.info("Building edges with dihedral angles...")
        self._build_adjacency()
        logger.info(f"Built {len(self.edges)} graph edges")
        
        # Step 5: Build adjacency map
        logger.info("Building adjacency map...")
        self._build_adjacency_map()
        
        # Step 6: Detect part orientation
        logger.info("Detecting part orientation...")
        self._detect_orientation()
        
        # Step 7: Validate
        logger.info("Validating graph integrity...")
        stats = self._compute_statistics()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("AAG GRAPH STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total faces processed: {len(self._face_cache)}")
        logger.info(f"Valid nodes: {len(self.nodes)}")
        logger.info(f"Degenerate faces filtered: {len(self._face_cache) - len(self.nodes)}")
        logger.info(f"Total edges: {len(self.edges)}")
        logger.info(f"Non-manifold edges: {stats['non_manifold_edges']}")
        logger.info(f"Boundary edges: {stats['boundary_edges']}")
        logger.info("")
        logger.info("Vexity Distribution:")
        logger.info(f"  Convex:    {stats['convex_edges']:3d} ({stats['convex_pct']:5.1f}%)")
        logger.info(f"  Concave:   {stats['concave_edges']:3d} ({stats['concave_pct']:5.1f}%)")
        logger.info(f"  Smooth:   {stats['smooth_edges']:4d} ({stats['smooth_pct']:5.1f}%)")
        
        # CRITICAL: Validate vexity distribution
        if stats['convex_pct'] < 10.0:
            logger.warning("⚠ Low convex percentage - check dihedral angle computation!")
        elif stats['convex_pct'] > 20.0 and stats['convex_pct'] < 50.0:
            logger.info("✓ Convex edge detection looks healthy")
        
        logger.info("=" * 70)
        
        return {
            'nodes': self.nodes,
            'adjacency': dict(self.adjacency),
            'statistics': stats
        }
        
    def _build_face_cache(self):
        """Cache all faces for indexing."""
        exp = TopExp_Explorer(self.shape, TopAbs_FACE)
        
        while exp.More():
            face = topods.Face(exp.Current())
            self._face_cache.append(face)
            exp.Next()
            
    def _build_edge_face_map(self):
        """
        Build map: edge -> [face1, face2, ...]
        
        Used to detect adjacency and non-manifold edges.
        """
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(
            self.shape,
            TopAbs_EDGE,
            TopAbs_FACE,
            edge_face_map
        )
        
        # Convert to Python dict
        non_manifold = 0
        boundary = 0
        
        for i in range(1, edge_face_map.Size() + 1):
            edge = topods.Edge(edge_face_map.FindKey(i))
            face_list = edge_face_map.FindFromIndex(i)
            
            face_count = face_list.Size()
            
            if face_count > 2:
                non_manifold += 1
            elif face_count == 1:
                boundary += 1
                
            # Store faces for this edge
            faces = []
            it = TopTools_ListIteratorOfListOfShape(face_list)
            while it.More():
                faces.append(topods.Face(it.Value()))
                it.Next()
                
            self._edge_face_map[edge] = faces
            
        logger.info(f"Edge-face map: {edge_face_map.Size()} edges")
        logger.info(f"  Non-manifold: {non_manifold}")
        logger.info(f"  Boundary: {boundary}")
        
    def _extract_face_attributes(self) -> int:
        """
        Extract attributes for all faces.
        
        Attributes:
        - surface_type: 'plane', 'cylinder', 'cone', 'sphere', etc.
        - area: Surface area (mm²)
        - normal: Unit normal vector (for planes)
        - center: Face centroid
        - curvature: For curved surfaces
        
        Returns:
            Number of valid faces
        """
        valid_count = 0
        
        for idx, face in enumerate(self._face_cache):
            try:
                # Compute geometric properties
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                
                area = props.Mass()
                
                # Filter degenerate faces
                if area < self.tolerance:
                    logger.debug(f"  Filtered degenerate face {idx} (area={area:.2e})")
                    continue
                    
                center = props.CentreOfMass()
                
                # Get surface type and normal
                surface = BRepAdaptor_Surface(face)
                surf_type = surface.GetType()
                
                face_data = {
                    'face_id': idx,
                    'area': area,
                    'center': [center.X(), center.Y(), center.Z()],
                    'surface_type': self._surface_type_to_string(surf_type),
                }
                
                # Extract normal for planar faces
                if surf_type == GeomAbs_Plane:
                    plane = surface.Plane()
                    normal = plane.Axis().Direction()
                    face_data['normal'] = [normal.X(), normal.Y(), normal.Z()]
                    
                # Extract axis for cylindrical faces
                elif surf_type == GeomAbs_Cylinder:
                    cyl = surface.Cylinder()
                    axis = cyl.Axis().Direction()
                    face_data['axis'] = [axis.X(), axis.Y(), axis.Z()]
                    face_data['radius'] = cyl.Radius()
                    
                    # Calculate angular span (crucial for distinguishing holes vs fillets)
                    u_min = surface.FirstUParameter()
                    u_max = surface.LastUParameter()
                    angle_deg = np.degrees(abs(u_max - u_min))
                    face_data['angle_deg'] = angle_deg
                    
                self.nodes[idx] = face_data
                valid_count += 1
                
            except Exception as e:
                logger.warning(f"  Failed to extract face {idx}: {e}")
                continue
                
        return valid_count
        
    def _build_adjacency(self):
        """
        Build adjacency edges with dihedral angles.
        
        For each edge in edge_face_map:
        - If 2 faces share edge → compute dihedral angle
        - Classify vexity (convex/concave/smooth)
        - Store as graph edge
        """
        for edge, faces in self._edge_face_map.items():
            if len(faces) != 2:
                continue  # Skip boundary and non-manifold
                
            face1, face2 = faces
            
            # Get face IDs
            try:
                idx1 = self._face_cache.index(face1)
                idx2 = self._face_cache.index(face2)
            except ValueError:
                continue
                
            # Skip if either face was filtered
            if idx1 not in self.nodes or idx2 not in self.nodes:
                continue
                
            # Compute dihedral angle
            dihedral = self._compute_dihedral_angle(edge, face1, face2)
            
            if dihedral is None:
                continue
                
            # Classify vexity (CRITICAL FIX HERE)
            vexity = self._classify_vexity(dihedral)
            
            # Create edge
            edge_data = {
                'face1': idx1,
                'face2': idx2,
                'dihedral_deg': dihedral,
                'vexity': vexity
            }
            
            self.edges.append(edge_data)
            
    def _compute_dihedral_angle(self, edge, face1, face2) -> float:
        """
        Compute dihedral angle between two faces along shared edge.
        
        Returns:
            Angle in degrees (0-360), or None if computation fails
        """
        try:
            # Get edge parameter range
            first, last = BRep_Tool.Range(edge)
            mid_param = (first + last) / 2.0
            
            # Get edge curve (3D)
            curve = BRepAdaptor_Curve(edge)
            edge_point = curve.Value(mid_param)
            edge_tangent = curve.DN(mid_param, 1)
            
            # Helper to get normal at edge parameter
            def get_normal_at_param(face, param):
                try:
                    # Get UV point on face from edge parameter
                    # Note: BRep_Tool.CurveOnSurface returns (Curve2d, First, Last)
                    c2d, f, l = BRep_Tool.CurveOnSurface(edge, face)
                    uv = c2d.Value(param)
                    
                    # Get surface properties
                    surf_handle = BRep_Tool.Surface(face)
                    props = GeomLProp_SLProps(surf_handle, uv.X(), uv.Y(), 1, 1e-6)
                    
                    if props.IsNormalDefined():
                        n = props.Normal()
                        return np.array([n.X(), n.Y(), n.Z()])
                        
                    return None
                except Exception:
                    return None

            # Get normals
            normal1 = get_normal_at_param(face1, mid_param)
            normal2 = get_normal_at_param(face2, mid_param)
            
            if normal1 is None or normal2 is None:
                # Fallback for planar faces if UV method fails
                normal1 = normal1 if normal1 is not None else self._get_face_normal_at_point(face1, edge_point)
                normal2 = normal2 if normal2 is not None else self._get_face_normal_at_point(face2, edge_point)
                
                if normal1 is None or normal2 is None:
                    return None
                
            # Compute dihedral angle
            dot = np.dot(normal1, normal2)
            dot = np.clip(dot, -1.0, 1.0)
            
            angle_rad = np.arccos(dot)
            angle_deg = np.degrees(angle_rad)
            
            # Determine if reflex angle (> 180°)
            # Cross product tells us orientation
            cross = np.cross(normal1, normal2)
            edge_dir = np.array([edge_tangent.X(), edge_tangent.Y(), edge_tangent.Z()])
            
            # Check edge orientation relative to face normals
            # For a convex edge, the cross product should align with edge direction
            # (depending on face order and edge orientation)
            
            # Standardize edge direction based on face1 traversal?
            # Simplified check:
            if np.dot(cross, edge_dir) < 0:
                angle_deg = 360.0 - angle_deg
                
            return angle_deg
            
        except Exception as e:
            logger.debug(f"  Dihedral computation failed: {e}")
            return None
            
    def _get_face_normal_at_point(self, face, point: gp_Pnt) -> np.ndarray:
        """
        Get face normal at a point.
        
        Simplified: Uses face surface normal from adaptor.
        
        Returns:
            Normal unit vector or None
        """
        try:
            surface = BRepAdaptor_Surface(face)
            
            # For planar faces, use plane normal
            if surface.GetType() == GeomAbs_Plane:
                plane = surface.Plane()
                normal = plane.Axis().Direction()
                return np.array([normal.X(), normal.Y(), normal.Z()])
                
            # For other surfaces, approximate with face orientation
            # (More sophisticated UV projection would be better)
            face_idx = self._face_cache.index(face)
            if face_idx in self.nodes:
                normal = self.nodes[face_idx].get('normal')
                if normal:
                    return np.array(normal)
                    
            return None
            
        except Exception:
            return None
            
    def _classify_vexity(self, dihedral_deg: float) -> str:
        """
        Classify edge vexity based on dihedral angle.
        
        CRITICAL FIX: Analysis Situs thresholds
        
        Rules:
        - dihedral > 181° → CONVEX (outward corner, boss edge)
        - dihedral < 179° → CONCAVE (inward corner, pocket edge)
        - 179° ≤ dihedral ≤ 181° → SMOOTH (tangent, fillet, blend)
        
        Args:
            dihedral_deg: Dihedral angle in degrees
            
        Returns:
            'convex', 'concave', or 'smooth'
        """
        if dihedral_deg > CONVEX_THRESHOLD:
            return "convex"
        elif dihedral_deg < CONCAVE_THRESHOLD:
            return "concave"
        else:
            return "smooth"
            
    def _build_adjacency_map(self):
        """Build fast adjacency lookup: face_id -> [neighbor data]."""
        for edge in self.edges:
            face1 = edge['face1']
            face2 = edge['face2']
            
            # Bidirectional adjacency
            self.adjacency[face1].append({
                'face_id': face2,
                'dihedral_deg': edge['dihedral_deg'],
                'vexity': edge['vexity']
            })
            
            self.adjacency[face2].append({
                'face_id': face1,
                'dihedral_deg': edge['dihedral_deg'],
                'vexity': edge['vexity']
            })

            # CRITICAL FIX: Analysis Situs method - ensure ALL nodes have adjacency entries
        # Even isolated faces (cylinders without edges) need entries for recognizers
        for node_id in self.nodes.keys():
            if node_id not in self.adjacency:
                # Initialize empty adjacency list for isolated faces
                self.adjacency[node_id] = []
            
    def _detect_orientation(self):
        """
        Detect part orientation (Z-up, Y-up, etc.).
        
        Method: Histogram normals of planar faces.
        """
        # Count faces aligned with each axis
        axis_counts = {'X': 0, 'Y': 0, 'Z': 0}
        total_planar = 0
        
        for face_data in self.nodes.values():
            if face_data['surface_type'] != 'plane':
                continue
                
            normal = np.array(face_data.get('normal', [0, 0, 1]))
            total_planar += 1
            
            # Find dominant axis
            abs_normal = np.abs(normal)
            dominant_idx = np.argmax(abs_normal)
            
            if dominant_idx == 0:
                axis_counts['X'] += 1
            elif dominant_idx == 1:
                axis_counts['Y'] += 1
            else:
                axis_counts['Z'] += 1
                
        if total_planar == 0:
            logger.warning("⚠ No planar faces found for orientation detection")
            return
            
        # Compute percentages
        x_pct = axis_counts['X'] / total_planar * 100
        y_pct = axis_counts['Y'] / total_planar * 100
        z_pct = axis_counts['Z'] / total_planar * 100
        
        logger.info("Orientation detection:")
        logger.info(f"  X: {x_pct:.1f}%, Y: {y_pct:.1f}%, Z: {z_pct:.1f}%")
        
        if z_pct > 50:
            logger.info("✓ Detected: Z-up")
        elif y_pct > 50:
            logger.info("✓ Detected: Y-up")
        elif x_pct > 50:
            logger.info("✓ Detected: X-up")
        else:
            logger.warning("⚠ Part appears rotated from axis alignment")
            
    def _compute_statistics(self) -> Dict:
        """Compute graph statistics."""
        # Vexity distribution
        convex = sum(1 for e in self.edges if e['vexity'] == 'convex')
        concave = sum(1 for e in self.edges if e['vexity'] == 'concave')
        smooth = sum(1 for e in self.edges if e['vexity'] == 'smooth')
        
        total_edges = len(self.edges)
        
        # Non-manifold and boundary
        non_manifold = sum(1 for faces in self._edge_face_map.values() if len(faces) > 2)
        boundary = sum(1 for faces in self._edge_face_map.values() if len(faces) == 1)
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': total_edges,
            'convex_edges': convex,
            'concave_edges': concave,
            'smooth_edges': smooth,
            'convex_pct': convex / total_edges * 100 if total_edges > 0 else 0,
            'concave_pct': concave / total_edges * 100 if total_edges > 0 else 0,
            'smooth_pct': smooth / total_edges * 100 if total_edges > 0 else 0,
            'non_manifold_edges': non_manifold,
            'boundary_edges': boundary
        }
        
    def _surface_type_to_string(self, surf_type) -> str:
        """Convert OCC surface type to string."""
        type_map = {
            GeomAbs_Plane: 'plane',
            GeomAbs_Cylinder: 'cylinder',
            GeomAbs_Cone: 'cone',
            GeomAbs_Sphere: 'sphere',
            GeomAbs_Torus: 'torus',
            GeomAbs_BSplineSurface: 'bspline'
        }
        return type_map.get(surf_type, 'other')
        
    def get_adjacent_faces(self, face_id: int) -> List[int]:
        """Get list of adjacent face IDs."""
        return [n['face_id'] for n in self.adjacency.get(face_id, [])]


def build_aag_graph(shape, tolerance: float = 1e-6):
    """
    Convenience function for AAG graph building.
    
    Args:
        shape: TopoDS_Shape
        tolerance: Geometric tolerance
        
    Returns:
        AAGGraph result dict
    """
    builder = AAGGraphBuilder(shape, tolerance)
    return builder.build()
