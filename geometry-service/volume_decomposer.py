"""
Volume Decomposer - Analysis Situs Aligned
==========================================

CRITICAL CHANGES:
- No longer splits removal volume into disconnected components
- Returns single removal volume representing all material removed
- Feature detection happens LATER via machining configuration analysis
- Matches Analysis Situs: "prismaticMilling" array on same solid
"""

import logging
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopoDS import topods
import numpy as np

logger = logging.getLogger(__name__)


class VolumeDecomposer:
    """
    Analysis Situs-style volume decomposition.
    
    Key Difference from Old Approach:
    - OLD: Boolean cut → Split into N volumes → Recognize features per volume
    - NEW: Boolean cut → Single volume → Detect N machining configs → Recognize features per config
    
    The "decomposition" is now about machining strategy, not geometric splitting.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Geometric tolerance for boolean operations (meters)
        """
        self.tolerance = tolerance
        self.detected_units = None
        
    def decompose(self, part_shape, part_type: str = "prismatic"):
        """
        Decompose part into manufacturing volumes.
        
        Args:
            part_shape: TopoDS_Shape of the part
            part_type: "prismatic", "rotational", or "freeform"
            
        Returns:
            List of dicts: [{'shape': TopoDS_Shape, 'hint': str, 'stock_bbox': dict}]
            
        Note: For prismatic parts, returns single removal volume.
              Multiple features detected later via machining config analysis.
        """
        logger.info(f"VolumeDecomposer initialized (tolerance={self.tolerance})")
        logger.info(f"Starting volume decomposition (type={part_type})")
        
        if part_type == "prismatic":
            return self._decompose_prismatic(part_shape)
        elif part_type == "rotational":
            return self._decompose_rotational(part_shape)
        else:
            logger.warning(f"Unknown part type '{part_type}', treating as prismatic")
            return self._decompose_prismatic(part_shape)
            
    def _decompose_prismatic(self, part_shape):
        """
        Prismatic decomposition - Analysis Situs style.
        
        Process:
        1. Compute oriented bounding box (stock envelope)
        2. Create stock solid
        3. Boolean cut: stock - part = removal_volume
        4. Return single removal volume (no splitting!)
        
        Returns:
            [{'shape': removal_volume, 'hint': 'prismatic', 'stock_bbox': {...}}]
        """
        logger.info("Decomposing prismatic part...")
        
        # 1. Detect units and get bounding box
        stock_bbox = self._compute_stock_envelope(part_shape)
        
        logger.info(f"  Detected model units: {self.detected_units}")
        logger.info(f"  Stock envelope: {stock_bbox['dx']:.1f} × {stock_bbox['dy']:.1f} × {stock_bbox['dz']:.1f} {self.detected_units}")
        
        # 2. Create stock block
        logger.info("  Computing boolean difference (stock - part)...")
        stock_solid = self._create_stock_box(stock_bbox)
        
        # 3. Boolean cut
        removal_volume = self._boolean_cut(stock_solid, part_shape)
        
        if removal_volume is None:
            logger.error("  Boolean operation failed!")
            return []
        
        # 4. Validate removal volume
        volume_mm3 = self._compute_volume(removal_volume)
        logger.info(f"  Removal volume: {volume_mm3:.1f} mm³")
        
        # 5. Return as SINGLE volume (key difference from old approach)
        logger.info("✓ Decomposition successful")
        logger.info(f"  Volumes found: 1 (Analysis Situs style: no splitting)")
        
        return [{
            'shape': removal_volume,
            'hint': 'prismatic',
            'stock_bbox': stock_bbox,
            'volume_mm3': volume_mm3,
            'units': self.detected_units
        }]
        
    def _compute_stock_envelope(self, part_shape):
        """
        Compute minimal bounding box for stock material.
        
        Returns:
            dict: {
                'dx', 'dy', 'dz': dimensions in mm
                'xmin', 'ymin', 'zmin': origin in mm
                'center': (x, y, z) in mm
            }
        """
        # Get raw bounding box from OCC
        bbox = Bnd_Box()
        brepbndlib.Add(part_shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        # Detect units from diagonal length
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        diagonal = np.sqrt(dx**2 + dy**2 + dz**2)
        
        logger.info(f"    Bounding box diagonal (raw OCC): {diagonal:.4f}")
        
        # Unit detection heuristic
        if diagonal < 1.0:
            # Likely meters (0.1m to 1m parts)
            self.detected_units = "m"
            scale_to_mm = 1000.0
            logger.info(f"    Detected METERS (diagonal {diagonal:.4f} < 1.0)")
        elif diagonal < 100.0:
            # Likely already mm or cm
            self.detected_units = "mm"
            scale_to_mm = 1.0
            logger.info(f"    Detected MILLIMETERS (diagonal {diagonal:.4f} < 100)")
        else:
            # Likely mm (most CAD default)
            self.detected_units = "mm"
            scale_to_mm = 1.0
            logger.info(f"    Assumed MILLIMETERS (diagonal {diagonal:.4f})")
        
        # Convert to mm
        return {
            'dx': dx * scale_to_mm,
            'dy': dy * scale_to_mm,
            'dz': dz * scale_to_mm,
            'xmin': xmin * scale_to_mm,
            'ymin': ymin * scale_to_mm,
            'zmin': zmin * scale_to_mm,
            'xmax': xmax * scale_to_mm,
            'ymax': ymax * scale_to_mm,
            'zmax': zmax * scale_to_mm,
            'center': (
                (xmin + xmax) / 2 * scale_to_mm,
                (ymin + ymax) / 2 * scale_to_mm,
                (zmin + zmax) / 2 * scale_to_mm
            ),
            'scale_to_mm': scale_to_mm
        }
        
    def _create_stock_box(self, bbox_dict):
        """
        Create stock block solid from bounding box.
        
        ✅ ENHANCED with comprehensive debugging and fallback methods
        """
        dx = bbox_dict['dx']
        dy = bbox_dict['dy']
        dz = bbox_dict['dz']
        xmin = bbox_dict['xmin']
        ymin = bbox_dict['ymin']
        zmin = bbox_dict['zmin']
        scale = bbox_dict['scale_to_mm']

        # Validate dimensions
        if dx <= 0 or dy <= 0 or dz <= 0:
            logger.error(f"Invalid bbox dimensions: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
            raise ValueError(f"Stock box dimensions must be positive: ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm")

        # Create box in OCC units (convert back from mm)
        origin = gp_Pnt(xmin / scale, ymin / scale, zmin / scale)

        # Log creation parameters for debugging
        logger.info(f"  Creating stock box:")
        logger.info(f"    Origin (OCC units): ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})")
        logger.info(f"    Size (OCC units):   ({dx/scale:.6f}, {dy/scale:.6f}, {dz/scale:.6f})")
        logger.info(f"    Scale factor:       {scale}")

        box_maker = BRepPrimAPI_MakeBox(origin, dx / scale, dy / scale, dz / scale)

        if not box_maker.IsDone():
            # Try alternative method: create box from two corner points
            logger.warning("  ⚠️ Box creation from origin failed, trying corner points method...")
            p1 = gp_Pnt(xmin / scale, ymin / scale, zmin / scale)
            p2 = gp_Pnt((xmin + dx) / scale, (ymin + dy) / scale, (zmin + dz) / scale)

            logger.info(f"  Corner 1: ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})")
            logger.info(f"  Corner 2: ({(xmin+dx)/scale:.6f}, {(ymin+dy)/scale:.6f}, {(zmin+dz)/scale:.6f})")

            box_maker = BRepPrimAPI_MakeBox(p1, p2)

            if not box_maker.IsDone():
                logger.error("  ❌ Both box creation methods failed!")
                raise RuntimeError(
                    f"Failed to create stock box. "
                    f"Dimensions: ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm, "
                    f"Origin: ({xmin:.2f}, {ymin:.2f}, {zmin:.2f}) mm, "
                    f"Scale: {scale}"
                )
            else:
                logger.info("  ✓ Stock box created successfully using corner points")
        else:
            logger.info("  ✓ Stock box created successfully")

        return box_maker.Shape()
        
    def _boolean_cut(self, stock, part):
        """
        Perform boolean subtraction: stock - part.
        
        Returns:
            TopoDS_Shape: Removal volume (single solid)
        """
        try:
            cut_op = BRepAlgoAPI_Cut(stock, part)
            cut_op.SetFuzzyValue(self.tolerance)
            cut_op.Build()
            
            if not cut_op.IsDone():
                logger.error("Boolean cut operation failed")
                return None
                
            result = cut_op.Shape()
            
            # Validate result is a solid
            exp = TopExp_Explorer(result, TopAbs_SOLID)
            solid_count = 0
            removal_solid = None
            
            while exp.More():
                solid_count += 1
                removal_solid = topods.Solid(exp.Current())
                exp.Next()
                
            if solid_count == 0:
                logger.error("Boolean result contains no solids")
                return None
            elif solid_count > 1:
                logger.warning(f"Boolean result has {solid_count} solids (expected 1)")
                logger.warning("This may indicate disconnected features - acceptable for Analysis Situs approach")
                
            return result  # Return complete shape (may contain multiple solids)
            
        except Exception as e:
            logger.error(f"Boolean operation exception: {e}")
            return None
            
    def _compute_volume(self, shape):
        """
        Compute volume of shape in mm³.
        
        Returns:
            float: Volume in mm³
        """
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume_occ = props.Mass()  # In OCC units
        
        # Convert to mm³
        if self.detected_units == "m":
            volume_mm3 = volume_occ * (1000.0 ** 3)  # m³ → mm³
        else:
            volume_mm3 = volume_occ  # Already mm³
            
        return volume_mm3
        
    def _decompose_rotational(self, part_shape):
        """
        Rotational part decomposition (turning operations).
        
        Process:
        1. Detect axis of rotation
        2. Compute cylindrical stock envelope
        3. Boolean cut
        4. Return single volume for turning analysis
        
        TODO: Implement when turning features are needed
        """
        logger.warning("Rotational decomposition not yet implemented")
        logger.info("Falling back to prismatic decomposition")
        return self._decompose_prismatic(part_shape)


def decompose_part(part_shape, part_type: str = "prismatic", tolerance: float = 1e-6):
    """
    Convenience function for volume decomposition.
    
    Args:
        part_shape: TopoDS_Shape
        part_type: "prismatic" or "rotational"
        tolerance: Geometric tolerance
        
    Returns:
        List of volume dicts
    """
    decomposer = VolumeDecomposer(tolerance=tolerance)
    return decomposer.decompose(part_shape, part_type)
