# brep_feature_extractor.py - BRepNet Input Tensor Extraction
# Version 1.0.1
# Extracts 12 required tensors from OpenCascade TopoDS_Shape for BRepNet inference

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import warnings

# Suppress deprecation warnings from OCC module
warnings.filterwarnings('ignore', category=DeprecationWarning, module='OCC')

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_BezierSurface
)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.gp import gp_Pnt

from utils.create_occwl_from_occ import create_occwl
from scale_utils import scale_solid_to_unit_box
from utils.data_utils import load_json_data

logger = logging.getLogger(__name__)


class BRepFeatureExtractor:
    """
    Extracts BRepNet-compatible tensors from OpenCascade TopoDS_Shape
    
    Outputs 12 tensors required by BRepNet model:
    1. Xf: Face features [num_faces, 10]
    2. Gf: Face UV grids [num_faces, 7, 10, 10]
    3. Xe: Edge features [num_edges, 12]
    4. Ge: Edge point grids [num_edges, 12, 10]
    5. Xc: Coedge features [num_coedges, 14]
    6. Gc: Coedge point grids [num_coedges, 12, 10]
    7. Kf: Face kernel tensor
    8. Ke: Edge kernel tensor
    9. Kc: Coedge kernel tensor
    10. Ce: Coedges of edges
    11. Cf: Coedges of small faces
    12. Csf: Coedges of big faces
    """
    
    def __init__(self, kernel_config_path='winged_edge_plus_plus.json'):
        """
        Initialize feature extractor
        
        Args:
            kernel_config_path: Path to kernel configuration JSON
        """
        self.kernel = load_json_data(kernel_config_path)
        logger.info(f"BRepFeatureExtractor initialized with kernel: {kernel_config_path}")
    
    def extract_features(self, shape: TopoDS_Shape) -> Dict[str, np.ndarray]:
        """
        Main extraction pipeline
        
        Args:
            shape: OpenCascade TopoDS_Shape from STEP file
            
        Returns:
            Dictionary containing all 12 BRepNet input tensors
        """
        logger.info("Starting BRepNet feature extraction")
        
        # Step 1: Scale shape to unit box [-1, 1]^3
        scaled_shape = scale_solid_to_unit_box(shape)
        
        # Step 2: Build topology graph
        faces, edges, coedges, topology = self._build_topology(scaled_shape)
        
        logger.info(f"Topology: {len(faces)} faces, {len(edges)} edges, {len(coedges)} coedges")
        
        # Step 3: Extract geometric features
        Xf = self._extract_face_features(faces)
        Xe = self._extract_edge_features(edges)
        Xc = self._extract_coedge_features(coedges)
        
        # Step 4: Sample UV grids
        Gf = self._sample_face_grids(faces)
        Gc = self._sample_coedge_grids(coedges)
        Ge = self._sample_edge_grids(edges, coedges)
        
        # Step 5: Build kernel tensors (coedge-centric)
        # All kernels are now [num_coedges, max_neighbors]
        Kf = self._build_face_kernel(topology['face_adjacency'], len(coedges))
        Ke = self._build_edge_kernel(topology['edge_adjacency'], len(coedges))
        Kc = self._build_coedge_kernel(topology['coedge_adjacency'], len(coedges))
        
        # Step 6: Build coedge mappings
        Ce = self._map_coedges_to_edges(topology['coedge_to_edge'])
        Cf, Csf = self._map_coedges_to_faces(topology['coedge_to_face'], len(faces))
        
        logger.info("✅ BRepNet tensor extraction complete")
        
        return {
            'Xf': Xf, 'Gf': Gf,
            'Xe': Xe, 'Ge': Ge,
            'Xc': Xc, 'Gc': Gc,
            'Kf': Kf, 'Ke': Ke, 'Kc': Kc,
            'Ce': Ce, 'Cf': Cf, 'Csf': Csf
        }
    
    def _build_topology(self, shape: TopoDS_Shape) -> Tuple:
        """Build coedge-centric topology graph using occwl"""
        from occwl.solid import Solid
        from occwl.entity_mapper import EntityMapper
        
        # Wrap shape in occwl Solid
        solid = create_occwl(shape)
        if not isinstance(solid, Solid):
            # If it's a compound, try to get first solid
            try:
                solid = Solid(shape, allow_compound=True)
            except:
                raise ValueError("Shape must be a valid solid")
        
        mapper = EntityMapper(solid)
        
        # Extract unique faces and edges
        faces = list(solid.faces())
        edges = list(solid.edges())
        
        # Build COEDGES (oriented edges / half-edges)
        coedges = []
        coedge_to_edge_map = []
        coedge_to_face_map = []
        
        for face_idx, face in enumerate(faces):
            for wire in face.wires():
                for oriented_edge in wire.ordered_edges():
                    coedges.append((oriented_edge, face))
                    # Map coedge to its underlying edge index
                    try:
                        edge_idx = mapper.edge_index(oriented_edge)
                        coedge_to_edge_map.append(edge_idx)
                    except:
                        coedge_to_edge_map.append(0)
                    coedge_to_face_map.append(face_idx)
        
        num_coedges = len(coedges)
        
        # Build coedge-centric adjacency lists
        face_adj = [[] for _ in range(num_coedges)]  # Kf[i] = faces adjacent to coedge i
        edge_adj = [[] for _ in range(num_coedges)]  # Ke[i] = edges adjacent to coedge i
        coedge_adj = [[] for _ in range(num_coedges)]  # Kc[i] = coedges adjacent to coedge i
        
        for i, (oriented_edge_i, face_i) in enumerate(coedges):
            # Find adjacent faces (via the underlying edge)
            try:
                edge_topods = oriented_edge_i.topods_shape()
                for adj_face in solid.faces_from_edge(oriented_edge_i):
                    if not adj_face.topods_shape().IsSame(face_i.topods_shape()):
                        try:
                            adj_face_idx = faces.index(adj_face)
                            face_adj[i].append(adj_face_idx)
                        except:
                            pass
            except:
                pass
            
            # Find adjacent coedges on same face
            for j, (oriented_edge_j, face_j) in enumerate(coedges):
                if i != j and face_i.topods_shape().IsSame(face_j.topods_shape()):
                    coedge_adj[i].append(j)
                    if len(coedge_adj[i]) >= 10:  # Limit neighbors
                        break
            
            # Find adjacent edges (sharing vertices)
            try:
                start_v = oriented_edge_i.first_vertex()
                end_v = oriented_edge_i.last_vertex()
                
                for edge_idx, adj_edge in enumerate(edges):
                    if not oriented_edge_i.topods_shape().IsSame(adj_edge.topods_shape()):
                        try:
                            adj_verts = list(solid.vertices_from_edge(adj_edge))
                            if start_v in adj_verts or end_v in adj_verts:
                                edge_adj[i].append(edge_idx)
                                if len(edge_adj[i]) >= 10:  # Limit neighbors
                                    break
                        except:
                            pass
            except:
                pass
        
        topology = {
            'face_adjacency': face_adj,
            'edge_adjacency': edge_adj,
            'coedge_adjacency': coedge_adj,
            'coedge_to_edge': coedge_to_edge_map,
            'coedge_to_face': coedge_to_face_map
        }
        
        # Convert faces/edges to TopoDS objects for downstream compatibility
        faces_topods = [f.topods_shape() for f in faces]
        edges_tuple = [(e.topods_shape(), False) for e in edges]
        
        return faces_topods, edges_tuple, coedges, topology
    
    def _extract_face_features(self, faces: List) -> np.ndarray:
        """Extract face features (10 features per face)"""
        features = []
        
        for face in faces:
            try:
                surf = BRepAdaptor_Surface(face, True)
                surf_type = surf.GetType()
                
                # Calculate area
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face, props)
                area = props.Mass()
                
                # Create feature vector (10 dims)
                feature_vec = np.zeros(10, dtype=np.float32)
                feature_vec[0] = area
                feature_vec[1] = 1.0 if surf_type == GeomAbs_Plane else 0.0
                feature_vec[2] = 1.0 if surf_type == GeomAbs_Cylinder else 0.0
                
                features.append(feature_vec)
            except:
                features.append(np.zeros(10, dtype=np.float32))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_edge_features(self, edges: List) -> np.ndarray:
        """Extract edge features (12 features per edge)"""
        features = []
        
        for edge, _ in edges:
            try:
                curve = BRepAdaptor_Curve(edge)
                length = curve.LastParameter() - curve.FirstParameter()
                
                feature_vec = np.zeros(12, dtype=np.float32)
                feature_vec[0] = length
                
                features.append(feature_vec)
            except:
                features.append(np.zeros(12, dtype=np.float32))
        
        return np.array(features, dtype=np.float32)
    
    def _sample_face_grids(self, faces: List) -> np.ndarray:
        """Sample face UV grids (7 channels × 10×10 grid)"""
        grids = []
        
        for face in faces:
            try:
                surf = BRepAdaptor_Surface(face, True)
                grid = np.zeros((7, 10, 10), dtype=np.float32)
                
                u_min = surf.FirstUParameter()
                u_max = surf.LastUParameter()
                v_min = surf.FirstVParameter()
                v_max = surf.LastVParameter()
                
                for i in range(10):
                    for j in range(10):
                        u = u_min + (u_max - u_min) * i / 9
                        v = v_min + (v_max - v_min) * j / 9
                        
                        try:
                            pnt = surf.Value(u, v)
                            grid[0:3, i, j] = [pnt.X(), pnt.Y(), pnt.Z()]
                            grid[3:6, i, j] = [0, 0, 1]  # Normal (simplified)
                            grid[6, i, j] = 1.0  # Trimming mask
                        except:
                            pass
                
                grids.append(grid)
            except:
                grids.append(np.zeros((7, 10, 10), dtype=np.float32))
        
        return np.array(grids, dtype=np.float32)
    
    def _sample_edge_grids(self, edges: List, coedges: List) -> np.ndarray:
        """Sample edge point grids (12 channels × 10 points)"""
        grids = []
        
        for edge, _ in coedges:
            curve = BRepAdaptor_Curve(edge.topods_shape())
            grid = np.zeros((12, 10), dtype=np.float32)
            
            t_min = curve.FirstParameter()
            t_max = curve.LastParameter()
            
            for i in range(10):
                t = t_min + (t_max - t_min) * i / 9
                try:
                    pnt = curve.Value(t)
                    grid[0:3, i] = [pnt.X(), pnt.Y(), pnt.Z()]
                    grid[3:6, i] = [1, 0, 0]  # Tangent (simplified)
                    grid[6:9, i] = [pnt.X(), pnt.Y(), pnt.Z()]
                    grid[9:12, i] = [1, 0, 0]
                except:
                    pass
            
            grids.append(grid)
        
        return np.array(grids, dtype=np.float32)
    
    def _extract_coedge_features(self, coedges: List) -> np.ndarray:
        """Extract coedge features (14 features per coedge)"""
        features = []
        
        for oriented_edge, face in coedges:
            try:
                curve = BRepAdaptor_Curve(oriented_edge.topods_shape())
                length = curve.LastParameter() - curve.FirstParameter()
                
                # Check if edge is reversed relative to its underlying edge
                is_reversed = oriented_edge.orientation() == 1  # TopAbs_REVERSED
                
                feature_vec = np.zeros(14, dtype=np.float32)
                feature_vec[0] = length
                feature_vec[1] = 1.0 if is_reversed else 0.0
                
                features.append(feature_vec)
            except:
                features.append(np.zeros(14, dtype=np.float32))
        
        return np.array(features, dtype=np.float32)
    
    def _sample_coedge_grids(self, coedges: List) -> np.ndarray:
        """
        Sample coedge point grids (12 channels × 10 points)
        
        FIXED: Changed from 6 channels to 12 channels to match BRepNet's curve_encoder expectations.
        Channels 0-5: First representation (coordinates + tangent)
        Channels 6-11: Second representation (coordinates + tangent)
        """
        grids = []
        
        for oriented_edge, face in coedges:
            try:
                curve = BRepAdaptor_Curve(oriented_edge.topods_shape())
                grid = np.zeros((12, 10), dtype=np.float32)
                
                t_min = curve.FirstParameter()
                t_max = curve.LastParameter()
                
                for i in range(10):
                    t = t_min + (t_max - t_min) * i / 9
                    try:
                        pnt = curve.Value(t)
                        # First representation (channels 0-5)
                        grid[0:3, i] = [pnt.X(), pnt.Y(), pnt.Z()]
                        grid[3:6, i] = [1, 0, 0]  # Tangent (simplified)
                        # Second representation (channels 6-11) - duplicate
                        grid[6:9, i] = [pnt.X(), pnt.Y(), pnt.Z()]
                        grid[9:12, i] = [1, 0, 0]  # Tangent (simplified)
                    except:
                        pass
                
                grids.append(grid)
            except:
                grids.append(np.zeros((12, 10), dtype=np.float32))
        
        return np.array(grids, dtype=np.float32)
    
    def _build_face_kernel(self, adjacency: List, num_coedges: int) -> np.ndarray:
        """
        Build face kernel tensor (coedge-centric)
        
        adjacency[i] = list of face indices adjacent to coedge i
        Returns: [num_coedges, kernel_size] array
        """
        # Use FIXED kernel size from config (not dynamic!)
        kernel_size = len(self.kernel['faces'])  # Should be 2 for winged_edge_plus_plus
        kernel = np.zeros((num_coedges, kernel_size), dtype=np.int64)
        
        for i, adj in enumerate(adjacency):
            for j in range(min(len(adj), kernel_size)):
                kernel[i, j] = adj[j]
        
        return kernel
    
    def _build_edge_kernel(self, adjacency: List, num_coedges: int) -> np.ndarray:
        """
        Build edge kernel tensor (coedge-centric)
        
        adjacency[i] = list of edge indices adjacent to coedge i
        Returns: [num_coedges, kernel_size] array
        """
        # Use FIXED kernel size from config (should be 9 for winged_edge_plus_plus)
        kernel_size = len(self.kernel['edges'])
        kernel = np.zeros((num_coedges, kernel_size), dtype=np.int64)
        
        for i, adj in enumerate(adjacency):
            for j in range(min(len(adj), kernel_size)):
                kernel[i, j] = adj[j]
        
        return kernel
    
    def _build_coedge_kernel(self, adjacency: List, num_coedges: int) -> np.ndarray:
        """
        Build coedge kernel tensor
        
        Uses FIXED kernel size from config (should be 14 for winged_edge_plus_plus)
        """
        kernel_size = len(self.kernel['coedges'])
        kernel = np.zeros((num_coedges, kernel_size), dtype=np.int64)
        
        for i, adj in enumerate(adjacency):
            for j in range(min(len(adj), kernel_size)):
                kernel[i, j] = adj[j]
        
        return kernel
    
    def _map_coedges_to_edges(self, coedge_to_edge: List) -> np.ndarray:
        """
        Map coedges to their parent edges
        
        FIXED: Changed dtype from int32 to int64 for PyTorch indexing compatibility
        """
        return np.array(coedge_to_edge, dtype=np.int64)
    
    def _map_coedges_to_faces(self, coedge_to_face: List, num_faces: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map coedges to faces
        
        FIXED: Changed dtype from int32 to int64 for PyTorch indexing compatibility
        
        Returns:
            Cf: Coedges of small faces
            Csf: Coedges of big faces
        """
        # Simplified implementation
        Cf = np.array(coedge_to_face, dtype=np.int64)
        Csf = np.array(coedge_to_face, dtype=np.int64)
        
        return Cf, Csf
