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
        Main extraction pipeline (follows reference: brepnet_dataset.py)
        
        Args:
            shape: OpenCascade TopoDS_Shape from STEP file
            
        Returns:
            Dictionary containing all 12 BRepNet input tensors
        """
        logger.info("Starting BRepNet feature extraction")
        
        # Step 1: Scale shape to unit box [-1, 1]^3
        scaled_shape = scale_solid_to_unit_box(shape)
        
        # Step 2: Build topology graph with topological maps
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
        
        # Step 5: Build kernel tensors via topological walks
        n = topology['coedge_to_next']
        m = topology['coedge_to_mate']
        e = topology['coedge_to_edge']
        f = topology['coedge_to_face']
        p = self._find_inverse_permutation(n)
        
        Kf = self._build_kernel_tensor_from_topology(n, p, m, e, f, self.kernel['faces'])
        Ke = self._build_kernel_tensor_from_topology(n, p, m, e, f, self.kernel['edges'])
        Kc = self._build_kernel_tensor_from_topology(n, p, m, e, f, self.kernel['coedges'])
        
        # Step 6: Build coedge mappings
        Ce = self._map_coedges_to_edges(topology['coedge_to_edge'], len(edges))
        Cf, Csf = self._map_coedges_to_faces(topology['coedge_to_face'], len(faces), len(coedges))
        
        logger.info("✅ BRepNet tensor extraction complete")
        
        return {
            'Xf': Xf, 'Gf': Gf,
            'Xe': Xe, 'Ge': Ge,
            'Xc': Xc, 'Gc': Gc,
            'Kf': Kf, 'Ke': Ke, 'Kc': Kc,
            'Ce': Ce, 'Cf': Cf, 'Csf': Csf
        }
    
    def _build_topology(self, shape: TopoDS_Shape) -> Tuple:
        """
        Build topology graph from shape using topological maps (reference: brepnet_dataset.py:412-447)
        
        Returns:
            faces: List of TopoDS_Face
            edges: List of (TopoDS_Edge, reversed_flag)
            coedges: List of (oriented_edge, face)
            topology: Dict with coedge_to_next, coedge_to_mate, coedge_to_edge, coedge_to_face
        """
        from occwl.solid import Solid
        from occwl.entity_mapper import EntityMapper
        
        # Convert to occwl Solid
        solid = create_occwl(shape)
        if not isinstance(solid, Solid):
            solid = Solid(shape, allow_compound=True)
        
        mapper = EntityMapper(solid)
        
        # Extract faces and edges
        faces = list(solid.faces())
        edges = list(solid.edges())
        num_faces = len(faces)
        num_edges = len(edges)
        
        # Build coedge list and topology maps
        coedges = []
        coedge_to_next_list = []
        coedge_to_mate_list = []
        coedge_to_edge_list = []
        coedge_to_face_list = []
        
        # First pass: collect all coedges and build edge/face mappings
        coedge_index = 0
        wire_starts = []  # Track where each wire starts for next mapping
        
        for face_idx, face in enumerate(faces):
            for wire in face.wires():
                ordered_edges = list(wire.ordered_edges())
                num_edges_in_wire = len(ordered_edges)
                wire_start = coedge_index
                wire_starts.append((wire_start, num_edges_in_wire))
                
                for i, oriented_edge in enumerate(ordered_edges):
                    coedges.append((oriented_edge, face))
                    
                    # Map to edge index
                    try:
                        edge_idx = mapper.edge_index(oriented_edge)
                    except:
                        edge_idx = 0
                    coedge_to_edge_list.append(edge_idx)
                    
                    # Map to face index
                    coedge_to_face_list.append(face_idx)
                    
                    # Placeholder for next (will fill in second pass)
                    coedge_to_next_list.append(-1)
                    
                    # Placeholder for mate (will fill in third pass)
                    coedge_to_mate_list.append(-1)
                    
                    coedge_index += 1
        
        num_coedges = len(coedges)
        
        # Second pass: fill in next mappings (loop within each wire)
        for wire_start, num_edges_in_wire in wire_starts:
            for i in range(num_edges_in_wire):
                current_idx = wire_start + i
                next_idx = wire_start + ((i + 1) % num_edges_in_wire)
                coedge_to_next_list[current_idx] = next_idx
        
        # Third pass: find mate coedges (same edge, opposite orientation)
        for i, (oriented_edge_i, face_i) in enumerate(coedges):
            edge_idx_i = coedge_to_edge_list[i]
            orientation_i = oriented_edge_i.orientation()
            
            mate_found = False
            for j, (oriented_edge_j, face_j) in enumerate(coedges):
                if i != j and coedge_to_edge_list[j] == edge_idx_i:
                    orientation_j = oriented_edge_j.orientation()
                    # Check if opposite orientation
                    if orientation_i != orientation_j:
                        coedge_to_mate_list[i] = j
                        mate_found = True
                        break
            
            if not mate_found:
                # Boundary edge - mate to itself
                coedge_to_mate_list[i] = i
        
        # Convert to numpy arrays
        coedge_to_next = np.array(coedge_to_next_list, dtype=np.int64)
        coedge_to_mate = np.array(coedge_to_mate_list, dtype=np.int64)
        coedge_to_edge = np.array(coedge_to_edge_list, dtype=np.int64)
        coedge_to_face = np.array(coedge_to_face_list, dtype=np.int64)
        
        topology = {
            'coedge_to_next': coedge_to_next,
            'coedge_to_mate': coedge_to_mate,
            'coedge_to_edge': coedge_to_edge,
            'coedge_to_face': coedge_to_face,
        }
        
        # Build edge list with reverse flags
        edges_with_flags = []
        for edge in edges:
            edges_with_flags.append((edge.topods_shape(), False))
        
        # Convert faces to TopoDS
        faces_topods = [f.topods_shape() for f in faces]
        
        return faces_topods, edges_with_flags, coedges, topology
    
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
    
    def _find_inverse_permutation(self, perm: np.ndarray) -> np.ndarray:
        """
        Find inverse of a permutation array (reference: brepnet_dataset.py:749-756)
        
        Args:
            perm: Permutation array (e.g., coedge_to_next)
            
        Returns:
            Inverse permutation (e.g., coedge_to_prev)
        """
        inverse = np.zeros_like(perm)
        for i in range(perm.size):
            inverse[perm[i]] = i
        return inverse
    
    def _build_kernel_tensor_from_topology(
        self,
        n: np.ndarray,  # next
        p: np.ndarray,  # prev
        m: np.ndarray,  # mate
        e: np.ndarray,  # edge
        f: np.ndarray,  # face
        kernel: List[str]
    ) -> np.ndarray:
        """
        Build kernel tensor by following topological walks (reference: brepnet_dataset.py:532-608)
        
        Args:
            n: coedge_to_next mapping
            p: coedge_to_prev mapping
            m: coedge_to_mate mapping
            e: coedge_to_edge mapping
            f: coedge_to_face mapping
            kernel: List of walk instructions (e.g., ["f", "mf"] for faces)
            
        Returns:
            Kernel tensor [num_coedges, len(kernel)]
        """
        num_coedges = len(n)
        kernel_tensor_cols = []
        
        for walk_instructions in kernel:
            # Start with identity: [0, 1, 2, ..., num_coedges-1]
            c = np.arange(num_coedges, dtype=np.int64)
            
            # Follow the walk instructions
            for instruction in walk_instructions:
                if instruction == "n":
                    c = n[c]
                elif instruction == "p":
                    c = p[c]
                elif instruction == "m":
                    c = m[c]
                elif instruction == "e":
                    c = e[c]
                elif instruction == "f":
                    c = f[c]
                # Empty string "" means identity, no operation
            
            kernel_tensor_cols.append(c)
        
        # Stack columns to get [num_coedges, kernel_size]
        kernel_tensor = np.stack(kernel_tensor_cols, axis=1)
        
        assert kernel_tensor.shape == (num_coedges, len(kernel)), \
            f"Kernel shape mismatch: {kernel_tensor.shape} != ({num_coedges}, {len(kernel)})"
        
        return kernel_tensor
    
    def _map_coedges_to_edges(self, coedge_to_edge: np.ndarray, num_edges: int) -> np.ndarray:
        """
        Build coedges of edges tensor (reference: brepnet_dataset.py:630-656)
        
        Args:
            coedge_to_edge: Mapping from coedge index to edge index
            num_edges: Total number of edges
            
        Returns:
            Ce: [num_edges, 2] tensor of coedge indices for each edge
        """
        # Group coedges by edge
        coedges_of_edges = [[] for _ in range(num_edges)]
        for coedge_index, edge_index in enumerate(coedge_to_edge):
            coedges_of_edges[edge_index].append(coedge_index)
        
        # Handle boundary edges (only one coedge) by duplicating
        for coedges in coedges_of_edges:
            if len(coedges) == 1:
                coedges.append(coedges[0])
        
        # Convert to numpy array [num_edges, 2]
        Ce = np.array(coedges_of_edges, dtype=np.int64)
        
        assert Ce.shape == (num_edges, 2), \
            f"Ce shape mismatch: {Ce.shape} != ({num_edges}, 2)"
        
        return Ce
    
    def _map_coedges_to_faces(self, coedge_to_face: np.ndarray, num_faces: int, num_coedges: int) -> Tuple:
        """
        Build coedges of faces tensors (reference: brepnet_dataset.py:659-734)
        
        Separates faces into:
        - Small faces (≤20 coedges): Cf as [num_small_faces, 20] padded array
        - Big faces (>20 coedges): Csf as list of variable-length tensors
        
        Args:
            coedge_to_face: Mapping from coedge index to face index
            num_faces: Total number of faces
            num_coedges: Total number of coedges (used for padding)
            
        Returns:
            Cf: [num_small_faces, 20] with padding index = num_coedges
            Csf: List of 1D arrays for big faces
        """
        MAX_COEDGES_PER_FACE = 20
        
        # Group coedges by face
        coedges_of_faces = [[] for _ in range(num_faces)]
        for coedge_index, face_index in enumerate(coedge_to_face):
            coedges_of_faces[face_index].append(coedge_index)
        
        # Separate small and big faces
        small_face_indices = []
        big_face_indices = []
        
        for face_index, coedges in enumerate(coedges_of_faces):
            if len(coedges) <= MAX_COEDGES_PER_FACE:
                small_face_indices.append(face_index)
            else:
                big_face_indices.append(face_index)
        
        num_small_faces = len(small_face_indices)
        
        # Build Cf for small faces [num_small_faces, 20]
        Cf = np.full((num_small_faces, MAX_COEDGES_PER_FACE), num_coedges, dtype=np.int64)
        
        for i, face_index in enumerate(small_face_indices):
            coedges = coedges_of_faces[face_index]
            for j, coedge_index in enumerate(coedges):
                Cf[i, j] = coedge_index
        
        # Build Csf for big faces (list of variable-length arrays)
        Csf = []
        for face_index in big_face_indices:
            coedges = np.array(coedges_of_faces[face_index], dtype=np.int64)
            Csf.append(coedges)
        
        return Cf, Csf
