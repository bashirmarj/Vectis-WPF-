"""
volume_decomposer.py - Manufacturing Volume Decomposition
==========================================================

CRITICAL MISSING COMPONENT for Analysis Situs-compatible architecture.

This implements TOP-DOWN body decomposition:
  STEP File → Classify Body → Decompose Volumes → AAG per Volume

Instead of:
  STEP File → Build Complete AAG → Search for Features (❌ WRONG)

Author: Vectis Machining
Version: 2.0.0 - Analysis Situs Compatible
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeShape,
    BRepBuilderAPI_Copy
)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepCheck import BRepCheck_Analyzer

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
            # Step 1: Compute stock envelope (bounding box)
            bbox = self._compute_bounding_box(shape)
            logger.info(f"  Stock envelope: {bbox['size_x']:.1f} × {bbox['size_y']:.1f} × {bbox['size_z']:.1f} mm")
            
            # Create stock solid from bounding box with padding
            padding = 0.1  # 0.1mm padding
            stock_solid = BRepPrimAPI_MakeBox(
                gp_Pnt(bbox['xmin'] - padding, bbox['ymin'] - padding, bbox['zmin'] - padding),
                gp_Pnt(bbox['xmax'] + padding, bbox['ymax'] + padding, bbox['zmax'] + padding)
            ).Shape()
            
            # Step 2: Boolean difference - stock minus part = removed material
            logger.info("  Computing boolean difference (stock - part)...")
            cut_op = BRepAlgoAPI_Cut(stock_solid, shape)
            cut_op.Build()
            
            if not cut_op.IsDone():
                logger.error("  Boolean cut operation failed")
                return DecompositionResult(
                    success=False,
                    error_message="Boolean cut operation failed"
                )
            
            removal_shape = cut_op.Shape()
            
            # Step 3: Split into disconnected solids
            logger.info("  Splitting removal volumes...")
            removal_volumes = self._split_solids(removal_shape)
            logger.info(f"  Found {len(removal_volumes)} removal volume(s)")
            
            # Step 4: Create ManufacturingVolume objects
            manufacturing_volumes = []
            for i, vol_solid in enumerate(removal_volumes):
                vol_data = self._analyze_volume(vol_solid, VolumeType.REMOVAL, i)
                manufacturing_volumes.append(vol_data)
                logger.info(f"    Volume {i}: {vol_data.volume_mm3:.1f} mm³, hint={vol_data.feature_hint}")
            
            # Create stock volume
            stock_props = GProp_GProps()
            brepgprop.VolumeProperties(stock_solid, stock_props)
            stock_volume_mm3 = stock_props.Mass() * 1e9  # m³ to mm³
            
            stock_vol = ManufacturingVolume(
                geometry=stock_solid,
                type=VolumeType.STOCK,
                volume_mm3=stock_volume_mm3,
                bounding_box=bbox,
                centroid=(
                    stock_props.CentreOfMass().X(),
                    stock_props.CentreOfMass().Y(),
                    stock_props.CentreOfMass().Z()
                ),
                feature_hint="stock_envelope"
            )
            
            return DecompositionResult(
                success=True,
                stock_volume=stock_vol,
                removal_volumes=manufacturing_volumes
            )
        
        except Exception as e:
            logger.error(f"Prismatic decomposition failed: {e}")
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
            Dictionary with xmin, xmax, ymin, ymax, zmin, zmax, size_x, size_y, size_z
        """
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        return {
            'xmin': xmin * 1000,  # Convert to mm
            'xmax': xmax * 1000,
            'ymin': ymin * 1000,
            'ymax': ymax * 1000,
            'zmin': zmin * 1000,
            'zmax': zmax * 1000,
            'size_x': (xmax - xmin) * 1000,
            'size_y': (ymax - ymin) * 1000,
            'size_z': (zmax - zmin) * 1000
        }
    
    def _split_solids(self, shape: TopoDS_Shape) -> List[TopoDS_Solid]:
        """
        Split compound/shape into individual disconnected solids
        
        Args:
            shape: Shape that may contain multiple solids
            
        Returns:
            List of individual solid volumes
        """
        solids = []
        solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        
        while solid_explorer.More():
            solid = solid_explorer.Current()
            
            # Verify solid is valid
            analyzer = BRepCheck_Analyzer(solid)
            if analyzer.IsValid():
                solids.append(solid)
            else:
                logger.warning(f"  Skipping invalid solid")
            
            solid_explorer.Next()
        
        return solids
    
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
        volume_mm3 = props.Mass() * 1e9  # m³ to mm³
        centroid = props.CentreOfMass()
        
        # Compute bounding box
        bbox = self._compute_bounding_box(solid)
        
        # Classify feature type based on geometry
        feature_hint = self._classify_feature_hint(solid, bbox, volume_mm3)
        
        # Compute complexity score (number of faces)
        from OCC.Core.TopAbs import TopAbs_FACE
        face_count = 0
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        complexity_score = face_count / 20.0  # Normalize (20 faces = complex)
        is_simple = face_count <= 10
        
        return ManufacturingVolume(
            geometry=solid,
            type=vol_type,
            volume_mm3=volume_mm3,
            bounding_box=bbox,
            centroid=(centroid.X(), centroid.Y(), centroid.Z()),
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
            bbox: Bounding box
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
