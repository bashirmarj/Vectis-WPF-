"""Geometric utility functions for AAG pattern matching"""

import numpy as np
from typing import Tuple, List, Optional

def compute_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two points"""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two vectors in degrees
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Angle in degrees [0, 180]
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-10 or v2_norm < 1e-10:
        return 0.0
    
    dot = np.dot(v1, v2)
    cos_angle = np.clip(dot / (v1_norm * v2_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def is_perpendicular(v1: np.ndarray, v2: np.ndarray, tolerance: float = 5.0) -> bool:
    """
    Check if two vectors are perpendicular within tolerance
    
    Args:
        v1: First vector
        v2: Second vector
        tolerance: Angle tolerance in degrees (default 5°)
    
    Returns:
        True if vectors are perpendicular within tolerance
    """
    angle = compute_angle(v1, v2)
    return abs(angle - 90.0) < tolerance

def is_parallel(v1: np.ndarray, v2: np.ndarray, tolerance: float = 5.0) -> bool:
    """
    Check if two vectors are parallel within tolerance
    
    Args:
        v1: First vector
        v2: Second vector
        tolerance: Angle tolerance in degrees (default 5°)
    
    Returns:
        True if vectors are parallel within tolerance
    """
    angle = compute_angle(v1, v2)
    return angle < tolerance or abs(angle - 180.0) < tolerance

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length
    
    Args:
        v: Input vector
    
    Returns:
        Normalized vector (or original if length is zero)
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def compute_cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute cross product of two 3D vectors"""
    return np.cross(v1, v2)

def compute_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute dot product of two vectors"""
    return float(np.dot(v1, v2))

def compute_centroid(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Compute centroid of a set of points"""
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    return tuple(centroid)

def point_to_line_distance(point: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray) -> float:
    """
    Compute distance from point to line
    
    Args:
        point: Point coordinates
        line_point: A point on the line
        line_direction: Line direction vector (will be normalized)
    
    Returns:
        Perpendicular distance from point to line
    """
    line_dir = normalize_vector(line_direction)
    point_vec = point - line_point
    projection = np.dot(point_vec, line_dir) * line_dir
    perpendicular = point_vec - projection
    return float(np.linalg.norm(perpendicular))
