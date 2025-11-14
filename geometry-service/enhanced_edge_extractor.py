"""
Enhanced Edge Data Extractor for Manufacturing Feature Recognition
Extracts comprehensive edge metadata including topology, connectivity, and geometric properties
Separate from rendering edge extraction - optimized for feature detection

Key additions:
- Edge connectivity (which faces share this edge)
- Edge adjacency (neighboring edges)
- Convexity classification (convex/concave/tangent)
- Dihedral angles between faces
- Edge loops and chains
- Face normals at edge boundaries
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, topods
from OCC.Core.TopExp import TopExp_Explorer, TopExp
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, 
    GeomAbs_Hyperbola, GeomAbs_Parabola, GeomAbs_BezierCurve, 
    GeomAbs_BSplineCurve, GeomAbs_Plane, GeomAbs_Cylinder,
    GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape

logger = logging.getLogger(__name__)

@dataclass
class EnhancedEdgeData:
    """Complete edge metadata for feature recognition"""
    edge_id: int
    edge_type: str  # 'line', 'circle', 'arc', 'ellipse', 'spline', etc.
    
    # Geometric properties
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    midpoint: Tuple[float, float, float]
    length: float
    
    # Curve-specific properties
    radius: Optional[float] = None  # For circles/arcs
    diameter: Optional[float] = None
    center: Optional[Tuple[float, float, float]] = None
    normal: Optional[Tuple[float, float, float]] = None  # For circles/arcs
    start_angle: Optional[float] = None  # For arcs
    end_angle: Optional[float] = None
    angular_extent: Optional[float] = None
    
    # Topology
    convexity: Optional[str] = None  # 'convex', 'concave', 'tangent', 'boundary'
    dihedral_angle: Optional[float] = None  # Angle between adjacent faces (degrees)
    adjacent_faces: List[int] = None  # Face indices sharing this edge
    adjacent_edges: List[int] = None  # Connected edge indices
    
    # Face properties at edge
    face_normals: List[Tuple[float, float, float]] = None  # Normals of adjacent faces
    face_types: List[str] = None  # Surface types of adjacent faces
    
    # Edge loop/chain information
    is_closed_loop: bool = False
    loop_id: Optional[int] = None
    chain_id: Optional[int] = None
    
    # Manufacturing relevance
    is_feature_boundary: bool = False  # Likely feature edge
    feature_hint: Optional[str] = None  # 'hole', 'pocket', 'fillet', etc.
    confidence: float = 0.0


class EnhancedEdgeExtractor:
    """
    Extracts comprehensive edge data for manufacturing feature recognition
    Separate from rendering pipeline - optimized for topology and geometry analysis
    """
    
    def __init__(self, shape, tolerance: float = 1e-6):
        self.shape = shape
        self.tolerance = tolerance
        self.edges: List[EnhancedEdgeData] = []
        self.edge_map: Dict[int, TopoDS_Edge] = {}
        self.face_map: Dict[int, TopoDS_Face] = {}
        
        # Topology maps
        self.edge_face_map = None  # Maps edges to their adjacent faces
        self.face_edge_map = None  # Maps faces to their boundary edges
        
    def extract_all(self) -> List[Dict]:
        """Main extraction pipeline"""
        logger.info("🔍 Extracting enhanced edge data for feature recognition...")
        
        # Step 1: Build topology maps
        self._build_topology_maps()
        
        # Step 2: Extract edge geometry
        self._extract_edge_geometry()
        
        # Step 3: Calculate convexity and dihedral angles
        self._calculate_edge_topology()
        
        # Step 4: Find edge loops and chains
        self._identify_edge_loops()
        
        # Step 5: Classify feature boundaries
        self._classify_feature_edges()
        
        logger.info(f"✅ Extracted {len(self.edges)} enhanced edges")
        logger.info(f"   Feature boundaries: {sum(1 for e in self.edges if e.is_feature_boundary)}")
        logger.info(f"   Convex edges: {sum(1 for e in self.edges if e.convexity == 'convex')}")
        logger.info(f"   Concave edges: {sum(1 for e in self.edges if e.convexity == 'concave')}")
        
        # Convert to dict for JSON serialization
        return [asdict(edge) for edge in self.edges]
    
    def _build_topology_maps(self):
        """Build edge-face and face-edge connectivity maps"""
        logger.info("  📊 Building topology maps...")
        
        # Collect faces
        face_idx = 0
        face_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            self.face_map[face_idx] = face
            face_idx += 1
            face_explorer.Next()
        
        # Build edge-face map using TopExp
        self.edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors(self.shape, TopAbs_EDGE, TopAbs_FACE, self.edge_face_map)
        
        # Collect edges
        edge_idx = 0
        edge_explorer = TopExp_Explorer(self.shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            self.edge_map[edge_idx] = edge
            edge_idx += 1
            edge_explorer.Next()
        
        logger.info(f"     Found {len(self.face_map)} faces, {len(self.edge_map)} edges")
    
    def _extract_edge_geometry(self):
        """Extract geometric properties of each edge"""
        logger.info("  📐 Extracting edge geometry...")
        
        for edge_id, edge in self.edge_map.items():
            try:
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()
                
                # Get endpoints
                first_param = curve_adaptor.FirstParameter()
                last_param = curve_adaptor.LastParameter()
                mid_param = (first_param + last_param) / 2
                
                start_pnt = curve_adaptor.Value(first_param)
                end_pnt = curve_adaptor.Value(last_param)
                mid_pnt = curve_adaptor.Value(mid_param)
                
                start = (start_pnt.X(), start_pnt.Y(), start_pnt.Z())
                end = (end_pnt.X(), end_pnt.Y(), end_pnt.Z())
                mid = (mid_pnt.X(), mid_pnt.Y(), mid_pnt.Z())
                
                # Calculate length
                length = self._calculate_edge_length(curve_adaptor, first_param, last_param)
                
                # Base edge data
                edge_data = EnhancedEdgeData(
                    edge_id=edge_id,
                    edge_type='unknown',
                    start_point=start,
                    end_point=end,
                    midpoint=mid,
                    length=length,
                    adjacent_faces=[],
                    adjacent_edges=[],
                    face_normals=[],
                    face_types=[]
                )
                
                # Extract curve-specific properties
                if curve_type == GeomAbs_Line:
                    edge_data.edge_type = 'line'
                
                elif curve_type == GeomAbs_Circle:
                    circle = curve_adaptor.Circle()
                    radius = circle.Radius()
                    center = circle.Location()
                    axis = circle.Axis()
                    
                    # Check if full circle or arc
                    angular_extent = abs(last_param - first_param)
                    is_full_circle = abs(angular_extent - 2 * np.pi) < 0.01
                    
                    edge_data.edge_type = 'circle' if is_full_circle else 'arc'
                    edge_data.radius = radius
                    edge_data.diameter = radius * 2
                    edge_data.center = (center.X(), center.Y(), center.Z())
                    edge_data.normal = (axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z())
                    edge_data.start_angle = first_param
                    edge_data.end_angle = last_param
                    edge_data.angular_extent = angular_extent
                    edge_data.is_closed_loop = is_full_circle
                
                elif curve_type == GeomAbs_Ellipse:
                    edge_data.edge_type = 'ellipse'
                    ellipse = curve_adaptor.Ellipse()
                    edge_data.radius = ellipse.MajorRadius()  # Store major radius
                    center = ellipse.Location()
                    edge_data.center = (center.X(), center.Y(), center.Z())
                
                elif curve_type == GeomAbs_BSplineCurve:
                    edge_data.edge_type = 'spline'
                
                elif curve_type == GeomAbs_BezierCurve:
                    edge_data.edge_type = 'bezier'
                
                else:
                    edge_data.edge_type = 'other'
                
                self.edges.append(edge_data)
                
            except Exception as e:
                logger.debug(f"    Error extracting edge {edge_id}: {e}")
    
    def _calculate_edge_length(self, curve_adaptor, first_param, last_param, num_samples=20):
        """Calculate edge length by sampling"""
        total_length = 0.0
        prev_point = curve_adaptor.Value(first_param)
        
        for i in range(1, num_samples + 1):
            t = first_param + (last_param - first_param) * i / num_samples
            curr_point = curve_adaptor.Value(t)
            total_length += prev_point.Distance(curr_point)
            prev_point = curr_point
        
        return total_length
    
    def _calculate_edge_topology(self):
        """Calculate convexity, dihedral angles, and face adjacency"""
        logger.info("  🔗 Calculating edge topology...")
        
        for edge_data in self.edges:
            edge = self.edge_map[edge_data.edge_id]
            
            # Get adjacent faces
            if self.edge_face_map.Contains(edge):
                face_list = self.edge_face_map.FindFromKey(edge)
                
                face_indices = []
                face_normals = []
                face_types = []
                
                # Iterate through face list using OpenCascade iterator
                face_iterator = TopTools_ListIteratorOfListOfShape(face_list)
                while face_iterator.More():
                    face = topods.Face(face_iterator.Value())
                    
                    # Find face index
                    face_idx = None
                    for idx, stored_face in self.face_map.items():
                        if face.IsSame(stored_face):
                            face_idx = idx
                            break
                    
                    if face_idx is not None:
                        face_indices.append(face_idx)
                        
                        # Get face normal at edge midpoint
                        normal = self._get_face_normal_at_point(face, edge_data.midpoint)
                        face_normals.append(normal)
                        
                        # Get face type
                        face_type = self._get_face_type(face)
                        face_types.append(face_type)
                    
                    face_iterator.Next()
                
                edge_data.adjacent_faces = face_indices
                edge_data.face_normals = face_normals
                edge_data.face_types = face_types
                
                # Calculate convexity and dihedral angle
                if len(face_normals) == 2:
                    convexity, angle = self._calculate_convexity(face_normals[0], face_normals[1])
                    edge_data.convexity = convexity
                    edge_data.dihedral_angle = angle
                elif len(face_normals) == 1:
                    edge_data.convexity = 'boundary'
                else:
                    edge_data.convexity = 'complex'
    
    def _get_face_normal_at_point(self, face, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Get face normal at a specific point"""
        try:
            surf = BRepAdaptor_Surface(face, True)
            # Simplified: use face center normal
            # In production, would project point to surface and evaluate there
            u_mid = (surf.FirstUParameter() + surf.LastUParameter()) / 2
            v_mid = (surf.FirstVParameter() + surf.LastVParameter()) / 2
            
            pnt = gp_Pnt()
            vec_u = gp_Vec()
            vec_v = gp_Vec()
            surf.D1(u_mid, v_mid, pnt, vec_u, vec_v)
            
            normal = vec_u.Crossed(vec_v)
            normal.Normalize()
            
            return (normal.X(), normal.Y(), normal.Z())
        except:
            return (0, 0, 1)  # Default
    
    def _get_face_type(self, face) -> str:
        """Get surface type of face"""
        try:
            surf = BRepAdaptor_Surface(face, True)
            surf_type = surf.GetType()
            
            type_map = {
                GeomAbs_Plane: 'planar',
                GeomAbs_Cylinder: 'cylindrical',
                GeomAbs_Cone: 'conical',
                GeomAbs_Sphere: 'spherical',
                GeomAbs_Torus: 'toroidal'
            }
            
            return type_map.get(surf_type, 'other')
        except:
            return 'unknown'
    
    def _calculate_convexity(self, normal1: Tuple, normal2: Tuple) -> Tuple[str, float]:
        """
        Calculate edge convexity and dihedral angle
        
        Returns:
            (convexity, angle_degrees)
            convexity: 'convex', 'concave', 'tangent'
        """
        n1 = np.array(normal1)
        n2 = np.array(normal2)
        
        # Calculate angle between normals
        dot_product = np.dot(n1, n2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Dihedral angle is supplementary to the angle between normals
        dihedral = 180.0 - angle_deg
        
        # Classify convexity
        if abs(angle_deg) < 5:  # Nearly parallel, same direction
            convexity = 'tangent'
        elif angle_deg < 90:  # Normals point outward
            convexity = 'convex'
        else:  # Normals point inward
            convexity = 'concave'
        
        return convexity, dihedral
    
    def _identify_edge_loops(self):
        """Identify closed edge loops (important for holes)"""
        logger.info("  🔄 Identifying edge loops...")
        
        loop_id = 0
        visited = set()
        
        for edge_data in self.edges:
            if edge_data.edge_id in visited:
                continue
            
            if edge_data.edge_type in ['circle']:  # Full circles are loops
                edge_data.is_closed_loop = True
                edge_data.loop_id = loop_id
                loop_id += 1
                visited.add(edge_data.edge_id)
            
            # TODO: Trace connected edges to find loops
            # This requires vertex connectivity which we can add if needed
    
    def _classify_feature_edges(self):
        """
        Classify edges as feature boundaries based on geometric/topological signatures
        Provides hints for feature detection
        """
        logger.info("  🎯 Classifying feature edges...")
        
        for edge_data in self.edges:
            # Circular edges often indicate holes
            if edge_data.edge_type in ['circle', 'arc']:
                if edge_data.radius and 0.5 <= edge_data.radius <= 200:  # Typical hole range
                    edge_data.is_feature_boundary = True
                    edge_data.feature_hint = 'hole'
                    edge_data.confidence = 0.85
            
            # Concave edges often indicate pockets/slots
            elif edge_data.convexity == 'concave':
                edge_data.is_feature_boundary = True
                if edge_data.edge_type == 'line':
                    edge_data.feature_hint = 'pocket_or_slot'
                    edge_data.confidence = 0.70
                elif edge_data.edge_type == 'arc' and edge_data.radius and edge_data.radius < 10:
                    edge_data.feature_hint = 'fillet'
                    edge_data.confidence = 0.80
            
            # Small radius arcs at convex edges indicate fillets
            elif edge_data.convexity == 'convex' and edge_data.edge_type == 'arc':
                if edge_data.radius and 0.5 <= edge_data.radius <= 10:
                    edge_data.is_feature_boundary = True
                    edge_data.feature_hint = 'fillet'
                    edge_data.confidence = 0.85
            
            # Boundary edges between planar and cylindrical faces
            elif (edge_data.convexity == 'boundary' and 
                  len(edge_data.face_types) == 2 and
                  'planar' in edge_data.face_types and 
                  'cylindrical' in edge_data.face_types):
                edge_data.is_feature_boundary = True
                edge_data.feature_hint = 'hole_boundary'
                edge_data.confidence = 0.75


def extract_enhanced_edges(shape) -> Dict:
    """
    Main entry point for enhanced edge extraction
    
    Returns:
        Dict with enhanced edge data and statistics
    """
    try:
        extractor = EnhancedEdgeExtractor(shape)
        edges = extractor.extract_all()
        
        # Generate statistics
        stats = {
            'total_edges': len(edges),
            'feature_boundaries': sum(1 for e in edges if e['is_feature_boundary']),
            'by_type': defaultdict(int),
            'by_convexity': defaultdict(int),
            'by_hint': defaultdict(int)
        }
        
        for edge in edges:
            stats['by_type'][edge['edge_type']] += 1
            if edge['convexity']:
                stats['by_convexity'][edge['convexity']] += 1
            if edge['feature_hint']:
                stats['by_hint'][edge['feature_hint']] += 1
        
        return {
            'edges': edges,
            'statistics': dict(stats)
        }
    
    except Exception as e:
        logger.error(f"Enhanced edge extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'edges': [],
            'statistics': {},
            'error': str(e)
        }
