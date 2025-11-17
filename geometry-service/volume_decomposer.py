"""
volume_decomposer.py - Manufacturing Volume Decomposition
==========================================================

CRITICAL MISSING COMPONENT for Analysis Situs-compatible architecture.

This implements TOP-DOWN body decomposition:
  STEP File → Classify Body → Decompose Volumes → AAG per Volume

Instead of:
  STEP File → Build Complete AAG → Search for Features (❌ WRONG)

Author: Vectis Machining
Version: 2.0.1 - FIXED: Volume Splitting + Unit Detection
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound, topods
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeShape,
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeSolid
)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
from OCC.Core.gp import gp_Pnt
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.Interface import Interface_Static

logger = logging.getLogger(__name__)


class VolumeType(Enum):
    """Manufacturing volume classification"""
    STOCK = "stock"                    # Original material envelope
    REMOVAL = "removal"                # Material removed (pocket, hole, slot)
    ADDITION = "addition"              # Material added (boss, rib, pad)
    INTERACTION = "interaction"        # Overlapping features
    UNKNOWN = "unknown"


@dataclass
class ManufacturingVolume:
    """
    Isolated manufacturing volume for AAG analysis
    
    This is what gets passed to AAG, NOT the entire part shape
    """
    geometry: TopoDS_Shape
    type: VolumeType
    volume_mm3: float
    bounding_box: Dict[str, float]
    centroid: Tuple[float, float, float]
    parent_part: Optional[str] = None
    feature_hint: Optional[str] = None  # "cylindrical_depression", "rectangular_pocket"
    complexity_score: float = 0.0
    is_simple: bool = True


@dataclass
class DecompositionResult:
    """Result of volume decomposition"""
    success: bool
    stock_volume: Optional[ManufacturingVolume] = None
    removal_volumes: List[ManufacturingVolume] = None
    addition_volumes: List[ManufacturingVolume] = None
    interaction_volumes: List[ManufacturingVolume] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.removal_volumes is None:
            self.removal_volumes = []
        if self.addition_volumes is None:
            self.addition_volumes = []
        if self.interaction_volumes is None:
            self.interaction_volumes = []


class VolumeDecomposer:
    """
    TOP-DOWN manufacturing volume decomposition
    
    Analysis Situs compatible approach:
    1. Compute stock envelope (convex hull or bounding box)
    2. Boolean difference: stock - part = removed volumes
    3. Boolean difference: part - stock = added volumes (bosses)
    4. Split disconnected volumes into separate features
    5. Each volume → separate AAG analysis
    
    This is the FUNDAMENTAL change needed for production.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize decomposer
        
        Args:
            tolerance: Geometric tolerance for boolean operations
        """
        self.tolerance = tolerance
        self.model_units = None  # Detected during decomposition
        logger.info(f"VolumeDecomposer initialized (tolerance={tolerance})")
    
    def decompose(self, shape: TopoDS_Shape, part_type: str) -> DecompositionResult:
        """
        Main decomposition entry point
        
        Args:
            shape: Part geometry from STEP file
            part_type: "prismatic", "rotational", or "hybrid"
            
        Returns:
            DecompositionResult with isolated volumes
        """
        logger.info(f"Starting volume decomposition (type={part_type})")
        
        try:
            if part_type == "prismatic":
                return self._decompose_prismatic(shape)
            elif part_type == "rotational":
                return self._decompose_rotational(shape)
            elif part_type == "hybrid":
                # Use prismatic decomposition for hybrid parts
                return self._decompose_prismatic(shape)
            else:
                logger.warning(f"Unknown part type '{part_type}', using prismatic decomposition")
                return self._decompose_prismatic(shape)
        
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return DecompositionResult(
                success=False,
                error_message=str(e)
            )
    
    def _decompose_prismatic(self, shape: TopoDS_Shape) -> DecompositionResult:
        """
        Decompose prismatic part into manufacturing volumes
        
        Algorithm:
        1. Compute bounding box → stock envelope
        2. Boolean difference: stock - part = removal volumes
        3. Split disconnected removal volumes
        4. Classify each volume (hole, pocket, slot)
        
        Args:
            shape: Prismatic part geometry
            
        Returns:
            DecompositionResult with removal volumes
        """
        logger.info("Decomposing prismatic part...")
        
        try:
            # Detect model units FIRST before any calculations
            self.model_units = self._detect_model_units(shape)
            logger.info(f"  Detected model units: {self.model_units}")
            
            # Step 1: Compute stock envelope (bounding box)
            bbox = self._compute_bounding_box(shape)
            logger.info(f"  Stock envelope: {bbox['size_x']:.1f} × {bbox['size_y']:.1f} × {bbox['size_z']:.1f} mm")
            
            # Create stock solid from bounding box with padding
            # Use raw values from OCC (in detected units), then convert
            bbox_raw = Bnd_Box()
            brepbndlib.Add(shape, bbox_raw)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox_raw.Get()
            
            # Apply padding in the SAME units as model
            padding = 0.1 if self.model_units == "mm" else 0.0001  # 0.1mm or 0.1mm in meters
            stock_solid = BRepPrimAPI_MakeBox(
                gp_Pnt(xmin - padding, ymin - padding, zmin - padding),
                gp_Pnt(xmax + padding, ymax + padding, zmax + padding)
            ).Shape()
            
            # Step 2: Boolean difference - stock minus part = removed material
            logger.info("  Computing boolean difference (stock - part)...")
            cut_op = BRepAlgoAPI_Cut(stock_solid, shape)
            cut_op.SetFuzzyValue(self.tolerance)  # Set tolerance for robustness
            cut_op.Build()
            
            if not cut_op.IsDone():
                logger.error("  Boolean cut operation failed")
                return DecompositionResult(
                    success=False,
                    error_message="Boolean cut operation failed"
                )
            
            removal_shape = cut_op.Shape()
            
            # Diagnostic logging for boolean result
            logger.info(f"  Boolean result analysis:")
            logger.info(f"    Result type: {removal_shape.ShapeType()}")
            logger.info(f"    Is compound: {removal_shape.ShapeType() == TopAbs_COMPOUND}")
            logger.info(f"    Is solid: {removal_shape.ShapeType() == TopAbs_SOLID}")
            
            # Count faces in result
            face_count = 0
            exp = TopExp_Explorer(removal_shape, TopAbs_FACE)
            while exp.More():
                face_count += 1
                exp.Next()
            logger.info(f"    Total faces in result: {face_count}")
            
            # Step 3: Split into disconnected solids
            logger.info("  Splitting removal volumes...")
            removal_volumes = self._split_solids(removal_shape)
            logger.info(f"  ✅ Found {len(removal_volumes)} disconnected removal volume(s)")
            
            # Validate we're getting multiple volumes
            if len(removal_volumes) == 1:
                logger.warning("  ⚠️ Only 1 volume detected - check if boolean result has merged features")
            
            # Step 4: Create ManufacturingVolume objects
            manufacturing_volumes = []
            for i, vol_solid in enumerate(removal_volumes):
                vol_data = self._analyze_volume(vol_solid, VolumeType.REMOVAL, i)
                manufacturing_volumes.append(vol_data)
                logger.info(f"    Volume {i}: {vol_data.volume_mm3:.1f} mm³, hint={vol_data.feature_hint}")
            
            # Create stock volume
            stock_props = GProp_GProps()
            brepgprop.VolumeProperties(stock_solid, stock_props)
            
            # Apply correct unit conversion
            if self.model_units == "mm":
                stock_volume_mm3 = stock_props.Mass()  # Already in mm³
            elif self.model_units == "m":
                stock_volume_mm3 = stock_props.Mass() * 1e9  # m³ to mm³
            else:
                stock_volume_mm3 = stock_props.Mass()  # Assume mm³
            
            logger.info(f"  Stock volume: {stock_volume_mm3:.2f} mm³ (units={self.model_units})")
            
            stock_vol = ManufacturingVolume(
                geometry=stock_solid,
                type=VolumeType.STOCK,
                volume_mm3=stock_volume_mm3,
                bounding_box=bbox,
                centroid=(0, 0, 0),
                feature_hint="stock_envelope"
            )
            
            return DecompositionResult(
                success=True,
                stock_volume=stock_vol,
                removal_volumes=manufacturing_volumes
            )
        
        except Exception as e:
            logger.error(f"Prismatic decomposition failed: {e}")
            import traceback
            traceback.print_exc()
            return DecompositionResult(
                success=False,
                error_message=str(e)
            )
    
    def _decompose_rotational(self, shape: TopoDS_Shape) -> DecompositionResult:
        """
        Decompose rotational part into manufacturing volumes
        
        For rotational parts, we extract:
        1. Main cylindrical envelope (stock)
        2. Radial features (grooves, threads)
        3. Axial features (center hole, cross-holes)
        
        NOTE: This is a simplified implementation. Full Analysis Situs approach
        requires 2D profile extraction and lathe operation sequencing.
        
        Args:
            shape: Rotational part geometry
            
        Returns:
            DecompositionResult with removal volumes
        """
        logger.info("Decomposing rotational part...")
        logger.warning("  Rotational decomposition not fully implemented - using simplified approach")
        
        try:
            # Fallback: Use prismatic decomposition for now
            # TODO: Implement proper cylindrical envelope + profile extraction
            return self._decompose_prismatic(shape)
        
        except Exception as e:
            logger.error(f"Rotational decomposition failed: {e}")
            return DecompositionResult(
                success=False,
                error_message=str(e)
            )
    
    def _compute_bounding_box(self, shape: TopoDS_Shape) -> Dict[str, float]:
        """
        Compute axis-aligned bounding box
        
        Args:
            shape: Part geometry
            
        Returns:
            Dictionary with xmin, xmax, ymin, ymax, zmin, zmax, size_x, size_y, size_z (all in mm)
        """
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        # Convert to millimeters based on detected units
        if self.model_units == "m":
            scale_factor = 1000.0  # m → mm
        else:
            scale_factor = 1.0  # Already in mm
        
        return {
            'xmin': xmin * scale_factor,
            'xmax': xmax * scale_factor,
            'ymin': ymin * scale_factor,
            'ymax': ymax * scale_factor,
            'zmin': zmin * scale_factor,
            'zmax': zmax * scale_factor,
            'size_x': (xmax - xmin) * scale_factor,
            'size_y': (ymax - ymin) * scale_factor,
            'size_z': (zmax - zmin) * scale_factor
        }
    
    def _split_solids(self, shape: TopoDS_Shape) -> List[TopoDS_Solid]:
        """
        Split compound using hash-based face connectivity analysis
        
        Uses BFS to find disconnected face components and builds individual
        solids from each component.
        
        Args:
            shape: Shape that may contain multiple disconnected solids
            
        Returns:
            List of individual solid volumes (should be 22+ for test part)
        """
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        
        # Build edge-to-face adjacency map
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(
            shape,
            TopAbs_EDGE,
            TopAbs_FACE,
            edge_face_map
        )
        
        # Extract all faces and create hash map
        faces = []
        face_hashes = {}
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        idx = 0
        while exp.More():
            face = topods.Face(exp.Current())
            faces.append(face)
            face_hash = face.HashCode(2147483647)  # Max int32
            face_hashes[face_hash] = idx
            idx += 1
            exp.Next()
        
        logger.info(f"    Found {len(faces)} faces to analyze for connectivity")
        
        # BFS to find connected components
        visited = set()
        components = []
        
        for start_idx in range(len(faces)):
            if start_idx in visited:
                continue
            
            # BFS to find all faces connected to this one
            component_faces = []
            stack = [start_idx]
            visited.add(start_idx)
            
            while stack:
                current_idx = stack.pop()
                current_face = faces[current_idx]
                component_faces.append(current_face)
                
                # Find all edges of this face
                edge_exp = TopExp_Explorer(current_face, TopAbs_EDGE)
                while edge_exp.More():
                    edge = topods.Edge(edge_exp.Current())
                    
                    # Find all faces sharing this edge
                    if edge_face_map.Contains(edge):
                        face_list = edge_face_map.FindFromKey(edge)
                        
                        for j in range(1, face_list.Length() + 1):
                            adj_face = topods.Face(face_list.Value(j))
                            adj_hash = adj_face.HashCode(2147483647)
                            
                            if adj_hash in face_hashes:
                                adj_idx = face_hashes[adj_hash]
                                if adj_idx not in visited:
                                    visited.add(adj_idx)
                                    stack.append(adj_idx)
                    
                    edge_exp.Next()
            
            # Build solid from this component
            if len(component_faces) >= 6:  # Minimum faces for a valid solid
                try:
                    shell = BRepBuilderAPI_MakeShell()
                    for face in component_faces:
                        shell.Add(face)
                    
                    if shell.IsDone():
                        solid_builder = BRepBuilderAPI_MakeSolid(shell.Shell())
                        if solid_builder.IsDone():
                            solid = solid_builder.Solid()
                            
                            # Validate
                            analyzer = BRepCheck_Analyzer(solid)
                            if analyzer.IsValid():
                                components.append(solid)
                                logger.debug(f"      Component {len(components)}: {len(component_faces)} faces")
                except Exception as e:
                    logger.warning(f"      Failed to build solid from {len(component_faces)} faces: {e}")
        
        logger.info(f"    ✅ Split into {len(components)} disconnected component(s)")
        
        return components
    
    def _analyze_volume(
        self, 
        solid: TopoDS_Solid, 
        vol_type: VolumeType,
        volume_id: int
    ) -> ManufacturingVolume:
        """
        Analyze isolated volume and extract properties
        
        Args:
            solid: Single solid volume
            vol_type: Type of volume
            volume_id: Sequential identifier
            
        Returns:
            ManufacturingVolume with computed properties
        """
        # Compute volume properties
        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        raw_volume = props.Mass()
        
        # Apply correct unit conversion based on detected units
        if self.model_units == "mm":
            volume_mm3 = raw_volume  # Already in mm³
            logger.debug(f"      Volume {volume_id} (mm³): {volume_mm3:.2f}")
        elif self.model_units == "m":
            volume_mm3 = raw_volume * 1e9  # m³ to mm³
            logger.debug(f"      Volume {volume_id} (m³→mm³): {raw_volume:.6e} → {volume_mm3:.2f}")
        else:
            volume_mm3 = raw_volume  # Assume mm³
            logger.warning(f"      Unknown units for volume {volume_id}, assuming mm³: {volume_mm3:.2f}")
        
        # Sanity check for absurd volumes
        if volume_mm3 > 1_000_000_000:  # 1 cubic meter in mm³
            logger.error(f"      ❌ ABSURD VOLUME: {volume_mm3:.2e} mm³ - UNIT CONVERSION ERROR!")
            logger.error(f"         Raw OCC value: {raw_volume:.2e}")
            logger.error(f"         Detected units: {self.model_units}")
        elif volume_mm3 < 0.001:
            logger.warning(f"      ⚠️ Extremely small volume: {volume_mm3:.2e} mm³")
        
        centroid = props.CentreOfMass()
        
        # Compute bounding box (already handles unit conversion)
        bbox = self._compute_bounding_box(solid)
        
        # Classify feature type based on geometry
        feature_hint = self._classify_feature_hint(solid, bbox, volume_mm3)
        
        # Compute complexity score (number of faces)
        face_count = 0
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        logger.debug(f"      Volume {volume_id}: {face_count} faces")
        
        complexity_score = face_count / 20.0  # Normalize (20 faces = complex)
        is_simple = face_count <= 10
        
        # Convert centroid to mm
        if self.model_units == "m":
            centroid_mm = (centroid.X() * 1000, centroid.Y() * 1000, centroid.Z() * 1000)
        else:
            centroid_mm = (centroid.X(), centroid.Y(), centroid.Z())
        
        return ManufacturingVolume(
            geometry=solid,
            type=vol_type,
            volume_mm3=volume_mm3,
            bounding_box=bbox,
            centroid=centroid_mm,
            feature_hint=feature_hint,
            complexity_score=complexity_score,
            is_simple=is_simple
        )
    
    def _classify_feature_hint(
        self, 
        solid: TopoDS_Solid, 
        bbox: Dict[str, float],
        volume_mm3: float
    ) -> str:
        """
        Classify feature type based on geometric properties
        
        Simple heuristic classification:
        - Cylindrical if height >> width == depth
        - Rectangular if width ~= depth ~= height
        - Slot if one dimension << others
        
        Args:
            solid: Volume geometry
            bbox: Bounding box (in mm)
            volume_mm3: Volume in mm³
            
        Returns:
            Feature hint string
        """
        size_x = bbox['size_x']
        size_y = bbox['size_y']
        size_z = bbox['size_z']
        
        # Normalize dimensions
        dims = sorted([size_x, size_y, size_z])
        min_dim = dims[0]
        mid_dim = dims[1]
        max_dim = dims[2]
        
        # Avoid division by zero
        if min_dim < 0.01:
            return "thin_feature"
        
        aspect_ratio_1 = max_dim / min_dim
        aspect_ratio_2 = mid_dim / min_dim
        
        # Cylindrical: one dimension much larger (drill hole, through hole)
        if aspect_ratio_1 > 3.0 and aspect_ratio_2 < 2.0:
            return "cylindrical_depression"
        
        # Slot: one dimension much smaller than others
        elif aspect_ratio_1 > 5.0 and aspect_ratio_2 > 2.5:
            return "slot_like"
        
        # Rectangular: all dimensions similar
        elif aspect_ratio_1 < 2.0 and aspect_ratio_2 < 1.5:
            return "rectangular_depression"
        
        # Long rectangular (pocket)
        elif aspect_ratio_1 > 2.0 and aspect_ratio_2 < 2.0:
            return "elongated_depression"
        
        else:
            return "complex_depression"
    
    def _detect_model_units(self, shape: TopoDS_Shape) -> str:
        """
        Detect whether model uses millimeters or meters
        
        Strategy:
        1. Compute bounding box diagonal
        2. If diagonal < 10, likely meters (parts are rarely < 10mm)
        3. If diagonal > 1000, likely millimeters (parts rarely > 1 meter)
        4. Ambiguous range: assume millimeters (CAD standard)
        
        Args:
            shape: Part geometry
            
        Returns:
            "mm" or "m"
        """
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        # Compute bounding box diagonal (raw OCC units)
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        diagonal = np.sqrt(dx**2 + dy**2 + dz**2)
        
        logger.info(f"    Bounding box diagonal (raw OCC): {diagonal:.4f}")
        
        # Heuristic thresholds
        if diagonal < 1.0:
            # Very small in OCC units → likely meters
            # Example: 0.5 m part = 500mm diagonal
            units = "m"
            logger.info(f"    Detected METERS (diagonal {diagonal:.4f} < 1.0)")
        elif diagonal > 100.0:
            # Large in OCC units → likely millimeters
            # Example: 500mm part = 500 diagonal
            units = "mm"
            logger.info(f"    Detected MILLIMETERS (diagonal {diagonal:.1f} > 100)")
        else:
            # Ambiguous range 1-100 → assume millimeters (CAD standard)
            units = "mm"
            logger.info(f"    Ambiguous diagonal {diagonal:.2f}, assuming MILLIMETERS")
        
        return units
