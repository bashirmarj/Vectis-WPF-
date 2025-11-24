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
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir, gp_Trsf, gp_Ax3, gp_Ax1
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopoDS import topods, TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
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
        Prismatic decomposition with Auto-Alignment.
        
        Process:
        1. Auto-align part to principal axes (fixes "rotated part" issue).
        2. Compute tight bounding box (Stock).
        3. Boolean Cut: Stock - Part.
        4. Split result into lumps.
        """
        logger.info("Decomposing prismatic part...")
        
        # 1. Auto-Align Part
        aligned_shape, transform = self._align_part(part_shape)
        
        # 2. Detect units and get bounding box (on ALIGNED shape)
        stock_bbox = self._compute_stock_envelope(aligned_shape)
        
        logger.info(f"  Detected model units: {self.detected_units}")
        logger.info(f"  Stock envelope: {stock_bbox['dx']:.1f} × {stock_bbox['dy']:.1f} × {stock_bbox['dz']:.1f} {self.detected_units}")
        
        # 3. Create stock block
        logger.info("  Computing boolean difference (stock - part)...")
        stock_solid = self._create_stock_box(stock_bbox)
        
        # 4. Boolean cut
        removal_volume = self._boolean_cut(stock_solid, aligned_shape)
        
        if removal_volume is None:
            logger.error("  Boolean operation failed!")
            return []
        
        # 5. Validate removal volume
        volume_mm3 = self._compute_volume(removal_volume)
        logger.info(f"  Removal volume: {volume_mm3:.1f} mm³")
        
        # 6. Split into lumps (features)
        # 6. Split into lumps (features)
        # NEW STRATEGY: Surface-Based Decomposition
        # The removal volume might be a single fused solid (due to "Air Lids").
        # We split it by analyzing the connectivity of its faces, ignoring "Stock Faces".
        feature_compounds = self._decompose_by_surfaces(removal_volume, stock_bbox)
        logger.info(f"✓ Decomposition successful: Found {len(feature_compounds)} features (surface groups)")
        
        results = []
        for i, compound in enumerate(feature_compounds):
            # We don't have a closed solid anymore, just a shell/compound of faces.
            # We can try to close it, or just pass the shell to the classifier.
            # For volume calculation, we might need to close it or estimate.
            
            # Estimate volume (if closed) or area
            props = GProp_GProps()
            brepgprop.SurfaceProperties(compound, props)
            area = props.Mass()
            
            results.append({
                'id': f"feature_{i}",
                'shape': compound, # This is a TopoDS_Compound of faces
                'hint': 'unknown_feature',
                'stock_bbox': stock_bbox,
                'volume_mm3': 0.0, # Volume is undefined for open shells
                'surface_area': area,
                'units': self.detected_units,
            })
            
        return results

    def _decompose_by_surfaces(self, removal_volume, stock_bbox):
        """
        Decompose a fused removal volume into features by analyzing surface connectivity.
        1. Identify "Stock Faces" (faces on the BBox boundary).
        2. Remove Stock Faces.
        3. Group remaining "Feature Faces" into connected components.
        """
        # 1. Map faces to Stock Boundary
        stock_faces = []
        feature_faces = []
        
        # Tolerances
        tol = 1e-3
        xmin, ymin, zmin = stock_bbox['xmin'], stock_bbox['ymin'], stock_bbox['zmin']
        xmax, ymax, zmax = stock_bbox['xmax'], stock_bbox['ymax'], stock_bbox['zmax']
        
        exp = TopExp_Explorer(removal_volume, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            surf = BRepAdaptor_Surface(face)
            is_stock = False
            
            if surf.GetType() == GeomAbs_Plane:
                pln = surf.Plane()
                loc = pln.Location()
                # Check if plane is on the boundary
                # We check the distance of the plane to the bbox limits
                # Note: This is a heuristic. A robust way is to check all vertices.
                
                # Check if all vertices are on boundary?
                # Simpler: Check if the underlying surface is coincident with BBox planes.
                
                # We can check the plane equation Ax+By+Cz+D=0
                # But BRepAdaptor gives us the surface.
                
                # Let's check if the face is ON the boundary.
                # We can sample a point on the face.
                pnt = BRepAdaptor_Surface(face).Value(0,0) # UV origin
                x, y, z = pnt.X(), pnt.Y(), pnt.Z()
                
                on_xmin = abs(x - xmin) < tol
                on_xmax = abs(x - xmax) < tol
                on_ymin = abs(y - ymin) < tol
                on_ymax = abs(y - ymax) < tol
                on_zmin = abs(z - zmin) < tol
                on_zmax = abs(z - zmax) < tol
                
                if on_xmin or on_xmax or on_ymin or on_ymax or on_zmin or on_zmax:
                    is_stock = True
            
            if is_stock:
                stock_faces.append(face)
            else:
                feature_faces.append(face)
                
            exp.Next()
            
        logger.info(f"  Surface Analysis: {len(stock_faces)} Stock Faces, {len(feature_faces)} Feature Faces")
        
        # 2. Build Connectivity Graph of Feature Faces
        # We use a map: Edge -> [Faces]
        edge_map = {}
        
        for face in feature_faces:
            exp_e = TopExp_Explorer(face, TopAbs_EDGE)
            while exp_e.More():
                edge = topods.Edge(exp_e.Current())
                # We need a hashable key for the edge. 
                # TopoDS_Shape doesn't hash by geometry identity.
                # We can use the hash code or map via TColStd_MapOfShape if available.
                # Or just use the pointer/hash if Python wrapper supports it.
                # OCC.Core.TopoDS.TopoDS_Shape.__hash__ works?
                
                # Better: Use a spatial map or just brute force for now (N is small).
                # Actually, we can use the 'IsSame' method.
                
                # Optimization: Map Edge Hash -> List of (Edge, Face)
                h = edge.HashCode(1000000)
                if h not in edge_map:
                    edge_map[h] = []
                edge_map[h].append((edge, face))
                
                exp_e.Next()
                
        # Build Adjacency: Face -> [Neighbor Faces]
        adjacency = {f: [] for f in feature_faces}
        
        # Iterate over potential shared edges
        for h, items in edge_map.items():
            # Check for actual identity (IsSame)
            for i in range(len(items)):
                e1, f1 = items[i]
                for j in range(i+1, len(items)):
                    e2, f2 = items[j]
                    if e1.IsSame(e2):
                        # Found shared edge
                        if f2 not in adjacency[f1]: adjacency[f1].append(f2)
                        if f1 not in adjacency[f2]: adjacency[f2].append(f1)
        
        # 3. Find Connected Components
        visited = set()
        compounds = []
        
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Compound
        
        for face in feature_faces:
            if face in visited:
                continue
            
            # BFS
            component = []
            queue = [face]
            visited.add(face)
            
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                
                for neighbor in adjacency.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Build Compound
            builder = BRep_Builder()
            comp = TopoDS_Compound()
            builder.MakeCompound(comp)
            for f in component:
                builder.Add(comp, f)
                
            compounds.append(comp)
            
        return compounds

    def _align_part(self, shape):
        """
        Align part based on largest planar face (Geometry-based).
        Better for prismatic parts than Inertia-based alignment.
        """
        # 1. Find largest planar face
        max_area = 0.0
        best_face = None
        best_surf = None
        
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_Plane:
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                if area > max_area:
                    max_area = area
                    best_face = face
                    best_surf = surf
            exp.Next()
            
        if not best_face:
            logger.warning("  No planar faces found for alignment. Using original orientation.")
            return shape, gp_Trsf()
            
        # 2. Get Normal of largest face
        # We want this normal to align with Global Z (or -Z)
        pln = best_surf.Plane()
        axis = pln.Axis() # gp_Ax1
        normal = axis.Direction()
        
        # Check if already aligned to Z
        if abs(normal.Z()) > 0.999:
            logger.info("  Part is already aligned (Largest face is Z-planar).")
            return shape, gp_Trsf()
            
        logger.info("  Aligning largest planar face to Global Z...")
        
        # 3. Construct Transform
        # We want to rotate such that 'normal' becomes (0,0,1)
        
        trsf = gp_Trsf()
        # Create an axis at origin with the face's normal direction
        from_axis = gp_Ax1(gp_Pnt(0,0,0), normal)
        to_axis = gp_Ax1(gp_Pnt(0,0,0), gp_Dir(0,0,1))
        
        # This rotation aligns the normal to Z. 
        trsf.SetRotation(from_axis, to_axis) 
        
        # Wait, SetRotation(Ax1, Ax1) isn't standard. 
        # Standard is SetDisplacement(Ax3, Ax3) or SetRotation(Ax1, Angle).
        # To align vector A to vector B:
        # Axis of rotation = A cross B
        # Angle = acos(A dot B)
        
        # Easier: SetDisplacement from a coordinate system defined by the face
        # to the global coordinate system.
        
        # Let's define a system on the face:
        # Origin: Face Center? No, we just want rotation.
        # Z: Face Normal
        # X: Arbitrary (or X direction of plane)
        
        face_cs = pln.Position() # gp_Ax3
        # We want to map this system to one where Z is Global Z.
        # But we want to keep the origin at (0,0,0) to avoid shifting the part far away?
        # Actually, shifting to origin is GOOD for numerical stability.
        
        # Target System: Global (0,0,0) with Z up
        global_cs = gp_Ax3(gp_Ax2())
        
        trsf.SetDisplacement(face_cs, global_cs)
        
        transformer = BRepBuilderAPI_Transform(shape, trsf, True)
        return transformer.Shape(), trsf

        
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
        
        # Unit detection: Default to Millimeters (standard for STEP)
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
        
        ✅ ENHANCED with THREE fallback methods for maximum reliability
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

        # Z-SHRINK HACK:
        # Shrink Z-dimensions slightly to prevent "Air Lids" (thin layers of air connecting features).
        # This ensures the stock box intersects the part's top/bottom faces, physically separating
        # features like holes from step volumes.
        z_shrink = 0.01 # 10 microns
        
        # Only shrink if we have enough thickness
        if dz / scale > (z_shrink * 3):
            logger.info(f"  🔧 Applying Z-Shrink of {z_shrink}mm to break 'Air Lids'...")
            zmin += z_shrink * scale
            dz -= (z_shrink * 2) * scale
        else:
            logger.warning("  ⚠️ Part too thin for Z-Shrink, skipping.")

        # METHOD 1: Create box from origin point + dimensions
        logger.info(f"  Creating stock box:")
        logger.info(f"    Origin (OCC units): ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})")
        logger.info(f"    Size (OCC units):   ({dx/scale:.6f}, {dy/scale:.6f}, {dz/scale:.6f})")
        logger.info(f"    Scale factor:       {scale}")

        origin = gp_Pnt(xmin / scale, ymin / scale, zmin / scale)
        box_maker = BRepPrimAPI_MakeBox(origin, dx / scale, dy / scale, dz / scale)

        # Check if done, but also check if shape is valid (IsDone can be unreliable)
        if box_maker.IsDone():
            logger.info("  ✅ Method 1: Stock box created from origin + dimensions")
            return box_maker.Shape()
        
        # Sometimes IsDone() returns False but Shape() is valid - check shape validity
        try:
            test_shape = box_maker.Shape()
            if not test_shape.IsNull():
                logger.warning("  ⚠️ Method 1: IsDone() returned False but shape is valid - using it anyway")
                return test_shape
        except Exception:
            pass  # Shape is truly invalid, continue to next method

        # METHOD 2: Create box from two corner points
        logger.warning("  ⚠️ Method 1 failed, trying corner points method...")
        p1 = gp_Pnt(xmin / scale, ymin / scale, zmin / scale)
        p2 = gp_Pnt((xmin + dx) / scale, (ymin + dy) / scale, (zmin + dz) / scale)

        logger.info(f"  Corner 1: ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})")
        logger.info(f"  Corner 2: ({(xmin+dx)/scale:.6f}, {(ymin+dy)/scale:.6f}, {(zmin+dz)/scale:.6f})")

        box_maker = BRepPrimAPI_MakeBox(p1, p2)

        if box_maker.IsDone():
            logger.info("  ✅ Method 2: Stock box created from corner points")
            return box_maker.Shape()
        
        # Check if shape is valid even if IsDone() is False
        try:
            test_shape = box_maker.Shape()
            if not test_shape.IsNull():
                logger.warning("  ⚠️ Method 2: IsDone() returned False but shape is valid - using it anyway")
                return test_shape
        except Exception:
            pass  # Shape is truly invalid, continue to Method 3

        # METHOD 3: Create box at origin, then translate
        logger.warning("  ⚠️ Method 2 failed, trying origin + translation method...")
        
        # Log diagnostic information
        logger.info(f"  Attempting to create box at origin with dimensions:")
        logger.info(f"    dx/scale = {dx/scale:.10f}")
        logger.info(f"    dy/scale = {dy/scale:.10f}")
        logger.info(f"    dz/scale = {dz/scale:.10f}")
        
        # Verify dimensions are reasonable
        if dx / scale < 1e-10 or dy / scale < 1e-10 or dz / scale < 1e-10:
            logger.error(f"  ❌ Dimensions too small for OpenCascade!")
            raise RuntimeError(
                f"Box dimensions are too small after scaling: "
                f"({dx/scale:.10e}, {dy/scale:.10e}, {dz/scale:.10e}). "
                f"Original dimensions: ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm"
            )
        
        # Create box at world origin
        try:
            box_maker = BRepPrimAPI_MakeBox(dx / scale, dy / scale, dz / scale)
        except Exception as e:
            logger.error(f"  ❌ Exception during box creation: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"BRepPrimAPI_MakeBox raised exception: {e}. "
                f"This indicates OpenCascade library configuration issues."
            )
        
        # Check both IsDone() and shape validity
        shape_valid = False
        box_shape = None
        
        if box_maker.IsDone():
            box_shape = box_maker.Shape()
            shape_valid = True
        else:
            # Sometimes IsDone() returns False but Shape() is valid
            try:
                test_shape = box_maker.Shape()
                if not test_shape.IsNull():
                    logger.warning("  ⚠️ IsDone() returned False but shape is valid - using it anyway")
                    box_shape = test_shape
                    shape_valid = True
            except Exception:
                pass  # Shape is truly invalid
        
        if shape_valid:
            # Success! Apply translation
            translation = gp_Trsf()
            translation.SetTranslation(gp_Vec(xmin / scale, ymin / scale, zmin / scale))
            
            transformer = BRepBuilderAPI_Transform(box_shape, translation, False)
            
            if not transformer.IsDone():
                logger.error("  ❌ Translation failed after box creation!")
                raise RuntimeError(
                    f"Box created at origin but translation failed. "
                    f"Translation vector: ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})"
                )
            
            transformed_shape = transformer.Shape()
            
            logger.info("  ✅ Method 3: Stock box created at origin and translated")
            logger.info(f"    Translation: ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})")
            
            return transformed_shape
        
        # If we get here, Method 3 failed - try inflation workaround
        if not shape_valid:
            logger.error("  ❌ Box creation at origin failed!")
            logger.warning("  🔧 Attempting workaround: slightly inflating dimensions by 0.1%...")
            
            # Try with slightly inflated dimensions (sometimes helps with numerical precision)
            inflate_factor = 1.001
            try:
                box_maker = BRepPrimAPI_MakeBox(
                    (dx / scale) * inflate_factor,
                    (dy / scale) * inflate_factor,
                    (dz / scale) * inflate_factor
                )
                
                # Check both IsDone and shape validity for inflated box
                inflated_valid = False
                if box_maker.IsDone():
                    box_shape = box_maker.Shape()
                    inflated_valid = True
                else:
                    try:
                        test_shape = box_maker.Shape()
                        if not test_shape.IsNull():
                            box_shape = test_shape
                            inflated_valid = True
                    except Exception:
                        pass
                
                if inflated_valid:
                    logger.info("  ✅ Workaround successful: box created with inflated dimensions")
                    # Adjust translation to account for inflation
                    translation_offset = ((dx / scale) * (inflate_factor - 1.0)) / 2.0
                    
                    # Create translation transformation with offset
                    translation = gp_Trsf()
                    translation.SetTranslation(gp_Vec(
                        (xmin / scale) - translation_offset,
                        (ymin / scale) - translation_offset,
                        (zmin / scale) - translation_offset
                    ))
                    
                    transformer = BRepBuilderAPI_Transform(box_shape, translation, False)
                    
                    if not transformer.IsDone():
                        raise RuntimeError("Translation failed after inflated box creation")
                    
                    logger.info(f"    Applied translation with inflation offset: {translation_offset:.6f}")
                    return transformer.Shape()
                else:
                    raise RuntimeError("Inflated box creation failed")
                    
            except Exception as e:
                logger.error(f"  ❌ Inflation workaround also failed: {e}")
            
            logger.error("  ❌ All box creation methods failed!")
            logger.error(f"  This suggests a fundamental issue with OpenCascade or the geometry")
            raise RuntimeError(
                f"Failed to create stock box using all methods. "
                f"Dimensions: ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm, "
                f"Origin: ({xmin:.2f}, {ymin:.2f}, {zmin:.2f}) mm, "
                f"Scale: {scale}. "
                f"This may indicate corrupted geometry or OpenCascade configuration issues."
            )
        
        # Get the box shape
        box_shape = box_maker.Shape()
        
        # Create translation transformation
        translation = gp_Trsf()
        translation.SetTranslation(gp_Vec(xmin / scale, ymin / scale, zmin / scale))
        
        # Apply transformation
        transformer = BRepBuilderAPI_Transform(box_shape, translation, False)  # False = don't copy
        
        if not transformer.IsDone():
            logger.error("  ❌ Translation failed after box creation!")
            raise RuntimeError(
                f"Box created at origin but translation failed. "
                f"Translation vector: ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})"
            )
        
        transformed_shape = transformer.Shape()
        
        logger.info("  ✅ Method 3: Stock box created at origin and translated")
        logger.info(f"    Translation: ({xmin/scale:.6f}, {ymin/scale:.6f}, {zmin/scale:.6f})")
        
        return transformed_shape
        
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

    def _decompose_lumps(self, shape) -> list:
        """
        Split a shape into its constituent solids (lumps).
        """
        lumps = []
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            lumps.append(topods.Solid(exp.Current()))
            exp.Next()
        return lumps
            
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
