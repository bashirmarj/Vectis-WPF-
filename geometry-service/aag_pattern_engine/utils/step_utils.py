"""STEP file loading and processing utilities"""

import logging
from typing import Tuple, Any
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

logger = logging.getLogger(__name__)

def load_step_file(filepath: str):
    """
    Load STEP file and return OpenCascade shape
    
    Args:
        filepath: Path to STEP file (.step or .stp)
    
    Returns:
        OpenCascade TopoDS_Shape
    
    Raises:
        ValueError: If file cannot be loaded
    """
    try:
        shape = read_step_file(filepath)
        
        if shape.IsNull():
            raise ValueError("Loaded shape is null")
        
        return shape
    
    except Exception as e:
        raise ValueError(f"Failed to load STEP file '{filepath}': {str(e)}")

def load_step_with_metadata(filepath: str) -> Tuple[Any, dict, dict]:
    """
    Load STEP file with names and colors metadata
    
    Args:
        filepath: Path to STEP file
    
    Returns:
        Tuple of (shape, names_dict, colors_dict)
    
    Raises:
        ValueError: If file cannot be loaded
    """
    try:
        shape, names, colors = read_step_file_with_names_colors(filepath)
        
        if shape.IsNull():
            raise ValueError("Loaded shape is null")
        
        return shape, names, colors
    
    except Exception as e:
        raise ValueError(f"Failed to load STEP file with metadata '{filepath}': {str(e)}")

def load_step_with_reader(filepath: str):
    """
    Load STEP file using STEPControl_Reader for more control
    
    Args:
        filepath: Path to STEP file
    
    Returns:
        OpenCascade TopoDS_Shape
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"STEP file read failed with status {status}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    if shape.IsNull():
        raise ValueError("Failed to extract shape from STEP file")
    
    return shape
