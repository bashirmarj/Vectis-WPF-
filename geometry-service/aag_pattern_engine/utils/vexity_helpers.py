"""
Semantic vexity classification helpers for feature recognition

These helpers abstract the low-level vexity enum checks into semantic
queries about geometric intent, making recognizer code more maintainable
and tolerant to real-world CAD variations.
"""

from typing import List
from aag_pattern_engine.graph_builder import Vexity


def is_depression_edge(vexity: Vexity) -> bool:
    """
    Check if edge represents material removal (depressed/recessed geometry)
    
    Used for: pocket bottoms, hole bottoms, slot bottoms
    
    Includes SMOOTH edges because CAD systems often create smooth blends
    at depression boundaries due to:
    - Export precision loss
    - B-spline approximations
    - Kernel differences between CAD systems
    
    Args:
        vexity: Edge vexity classification
        
    Returns:
        True if edge is concave or smooth (typical depression boundary)
    """
    return vexity in (Vexity.CONCAVE, Vexity.SMOOTH)


def is_protrusion_edge(vexity: Vexity) -> bool:
    """
    Check if edge represents material addition (protruding geometry)
    
    Used for: fillet blends, boss edges, chamfer ridges
    
    SMOOTH is excluded - true protrusions must be > 180° dihedral angle.
    SMOOTH edges at fillet boundaries usually indicate G1 continuity,
    not the convex blend itself.
    
    Args:
        vexity: Edge vexity classification
        
    Returns:
        True if edge is convex (material protrudes outward)
    """
    return vexity == Vexity.CONVEX


def is_vertical_wall_transition(vexity: Vexity) -> bool:
    """
    Check if edge could be a vertical wall transition
    
    Used for: pocket walls, slot walls, hole walls, boss walls
    
    Includes SMOOTH because vertical walls in real CAD files often have
    slight curvature or smooth continuity with floor/ceiling faces.
    
    Args:
        vexity: Edge vexity classification
        
    Returns:
        True if edge could represent a vertical feature boundary
    """
    return vexity in (Vexity.CONCAVE, Vexity.SMOOTH)


def is_smooth_blend(vexity: Vexity) -> bool:
    """
    Check if edge represents smooth geometric continuity
    
    Used for: identifying tangent surfaces, G1/G2 continuity
    
    Args:
        vexity: Edge vexity classification
        
    Returns:
        True if edge has smooth classification
    """
    return vexity == Vexity.SMOOTH


def requires_strict_concave(vexity: Vexity) -> bool:
    """
    Check if edge is strictly concave (for special cases)
    
    Used when we need to differentiate true depressions from blends.
    Most recognizers should use is_depression_edge() instead.
    
    Args:
        vexity: Edge vexity classification
        
    Returns:
        True only if edge is concave (excludes SMOOTH)
    """
    return vexity == Vexity.CONCAVE
