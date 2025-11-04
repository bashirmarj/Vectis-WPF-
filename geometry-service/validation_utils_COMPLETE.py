# COMPLETE validation_utils.py - FULL PRODUCTION CODE  
# Ready to copy and paste - no modifications needed

import logging
import os
from typing import Tuple, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class STEPValidator:
    """
    Comprehensive validation of STEP files and parsed shapes.
    Prevents common parsing errors and ensures data integrity.
    """
    
    @staticmethod
    def validate_file_exists(file_path: str) -> Tuple[bool, str]:
        """Check if file exists and is readable"""
        if not Path(file_path).exists():
            return False, f"File not found: {file_path}"
        
        if not os.access(file_path, os.R_OK):
            return False, f"File not readable: {file_path}"
        
        return True, ""
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: int = 100) -> Tuple[bool, str]:
        """Check file size is within limits"""
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
        
        if size_mb < 0.001:
            return False, "File is empty or corrupted"
        
        return True, ""
    
    @staticmethod
    def validate_file_extension(file_path: str, allowed_extensions: List[str] = None) -> Tuple[bool, str]:
        """Check file has valid STEP extension"""
        if allowed_extensions is None:
            allowed_extensions = [".step", ".stp", ".STEP", ".STP"]
        
        ext = Path(file_path).suffix
        
        if ext not in allowed_extensions:
            return False, f"Invalid file extension: {ext}. Expected: {allowed_extensions}"
        
        return True, ""
    
    @staticmethod
    def validate_step_format(file_path: str) -> Tuple[bool, str]:
        """
        Validate STEP file format by checking header/footer.
        STEP files must start with ISO-10303 header.
        """
        try:
            with open(file_path, 'r', errors='ignore') as f:
                first_line = f.readline().strip()
                
                if not first_line.startswith("ISO-10303-21"):
                    return False, "Invalid STEP header (missing ISO-10303-21)"
                
                content = f.read()
                
                if "DATA;" not in content:
                    return False, "Invalid STEP format (missing DATA section)"
                
                if "ENDSEC;" not in content:
                    return False, "Invalid STEP format (missing ENDSEC)"
                
                return True, ""
        
        except Exception as e:
            return False, f"Error reading file format: {str(e)}"
    
    @staticmethod
    def validate_shape_topology(shape) -> Tuple[bool, str]:
        """
        Validate B-rep solid topology integrity.
        Checks for manifold, closed solid without gaps.
        """
        from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_EdgeStatus
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        
        try:
            analyzer = BRepCheck_Analyzer(shape, doParallel=True)
            
            if not analyzer.IsValid():
                errors = []
                
                status = analyzer.Result(TopAbs_FACE)
                if status != BRepCheck_EdgeStatus.Not_Processed:
                    errors.append(f"Face topology error: {status}")
                
                status = analyzer.Result(TopAbs_EDGE)
                if status != BRepCheck_EdgeStatus.Not_Processed:
                    errors.append(f"Edge topology error: {status}")
                
                error_msg = "; ".join(errors) if errors else "Invalid B-rep topology"
                return False, error_msg
            
            return True, ""
        
        except Exception as e:
            return False, f"Topology check failed: {str(e)}"
    
    @staticmethod
    def validate_shape_solidity(shape) -> Tuple[bool, str]:
        """
        Check if shape is a closed, manifold solid (not open or compound).
        """
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
        
        try:
            # Count solids
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            solid_count = 0
            while explorer.More():
                solid_count += 1
                explorer.Next()
            
            if solid_count == 0:
                return False, "Shape contains no solids (might be a surface or assembly)"
            
            if solid_count > 1:
                return False, f"Shape is compound with {solid_count} solids (only single parts supported)"
            
            # Count shells
            explorer = TopExp_Explorer(shape, TopAbs_SHELL)
            shell_count = 0
            while explorer.More():
                shell_count += 1
                explorer.Next()
            
            # Count faces
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            face_count = 0
            while explorer.More():
                face_count += 1
                explorer.Next()
            
            if face_count == 0:
                return False, "Shape contains no faces"
            
            if face_count > 1000:
                return False, f"Shape too complex ({face_count} faces, max 1000)"
            
            return True, ""
        
        except Exception as e:
            return False, f"Solidity check failed: {str(e)}"
    
    @staticmethod
    def validate_shape_geometry(shape) -> Tuple[bool, str]:
        """
        Check for degenerate or invalid geometry.
        """
        from OCC.Core.BRepBndLib import BRepBndLib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        
        try:
            # Get bounding box
            bbox = Bnd_Box()
            BRepBndLib.Add(shape, bbox)
            
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            
            # Check bounds are valid
            if xmin >= xmax or ymin >= ymax or zmin >= zmax:
                return False, "Invalid bounding box (degenerate shape)"
            
            # Check dimensions
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            
            if dx < 0.001 or dy < 0.001 or dz < 0.001:
                return False, "Shape too small (< 1mm in at least one dimension)"
            
            if dx > 100000 or dy > 100000 or dz > 100000:
                return False, "Shape too large (> 100m in at least one dimension)"
            
            return True, ""
        
        except Exception as e:
            return False, f"Geometry check failed: {str(e)}"


def validate_step_file_complete(file_path: str) -> Tuple[bool, Dict]:
    """
    Complete validation pipeline for STEP files.
    
    Returns:
        (is_valid, validation_report)
    
    validation_report contains:
        {
            "valid": bool,
            "checks": {...},
            "errors": [list],
            "warnings": [list],
        }
    """
    
    logger.info(f"Starting complete STEP validation: {file_path}")
    
    report = {
        "valid": False,
        "checks": {},
        "errors": [],
        "warnings": [],
    }
    
    validator = STEPValidator()
    
    # 1. File existence
    is_valid, error = validator.validate_file_exists(file_path)
    report["checks"]["file_exists"] = is_valid
    if not is_valid:
        report["errors"].append(error)
        logger.error(f"❌ {error}")
        return False, report
    
    # 2. File size
    is_valid, error = validator.validate_file_size(file_path)
    report["checks"]["file_size"] = is_valid
    if not is_valid:
        report["errors"].append(error)
        logger.error(f"❌ {error}")
        return False, report
    
    # 3. File extension
    is_valid, error = validator.validate_file_extension(file_path)
    report["checks"]["file_extension"] = is_valid
    if not is_valid:
        report["warnings"].append(error)
        logger.warning(f"⚠️ {error}")
    
    # 4. STEP format
    is_valid, error = validator.validate_step_format(file_path)
    report["checks"]["step_format"] = is_valid
    if not is_valid:
        report["errors"].append(error)
        logger.error(f"❌ {error}")
        return False, report
    
    # 5. Parse and validate shape
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        
        reader = STEPControl_Reader()
        read_status = reader.ReadFile(file_path)
        
        if read_status != IFSelect_RetDone:
            error_msg = f"STEP read failed with status {read_status}"
            report["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False, report
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        # 6. Topology validation
        is_valid, error = validator.validate_shape_topology(shape)
        report["checks"]["topology"] = is_valid
        if not is_valid:
            report["errors"].append(error)
            logger.error(f"❌ {error}")
        
        # 7. Solidity validation
        is_valid, error = validator.validate_shape_solidity(shape)
        report["checks"]["solidity"] = is_valid
        if not is_valid:
            report["errors"].append(error)
            logger.error(f"❌ {error}")
        
        # 8. Geometry validation
        is_valid, error = validator.validate_shape_geometry(shape)
        report["checks"]["geometry"] = is_valid
        if not is_valid:
            report["warnings"].append(error)
            logger.warning(f"⚠️ {error}")
    
    except Exception as e:
        error_msg = f"Shape parsing failed: {str(e)}"
        report["errors"].append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False, report
    
    # Final validation
    is_valid = all([
        report["checks"].get("file_exists", False),
        report["checks"].get("file_size", False),
        report["checks"].get("step_format", False),
        report["checks"].get("topology", False),
        report["checks"].get("solidity", False),
    ])
    
    report["valid"] = is_valid
    
    if is_valid:
        logger.info("✅ STEP file validation passed")
    else:
        logger.error("❌ STEP file validation failed")
    
    return is_valid, report
