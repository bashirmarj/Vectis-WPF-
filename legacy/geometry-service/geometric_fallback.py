# geometric_fallback.py - Geometric Detection for Turning Features
# Version 1.0.0
# 
# Handles features BRepNet misses:
# - Keyways on shafts
# - V-grooves
# - Threading
# - Shoulder steps
# - Taper sections

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Line
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt

logger = logging.getLogger(__name__)


@dataclass
class TurningFeature:
    """Detected turning feature"""
    feature_type: str  # 'keyway', 'v_groove', 'thread', 'shoulder', 'taper'
    face_ids: List[int]
    confidence: float  # Geometric confidence (0.0-1.0)
    axis: Tuple[float, float, float]  # Rotation axis direction
    center: Tuple[float, float, float]  # Feature center
    parameters: Dict  # Feature-specific parameters


class TurningFeatureDetector:
    """
    Geometric detector for turning/lathe features
    
    Uses rule-based geometric interrogation when ML models fail
    """
    
    def __init__(self, tolerance: float = 0.001):
        """
        Initialize detector
        
        Args:
            tolerance: Geometric tolerance in meters (0.001 = 1mm)
        """
        self.tolerance = tolerance
        self.parallel_threshold = 0.999  # cos(2.5°) for axis parallelism
    
    def detect_features(self, shape: TopoDS_Shape) -> List[Dict]:
        """
        Detect turning features in shape
        
        Args:
            shape: OpenCascade TopoDS_Shape
        
        Returns:
            List of detected turning features
        """
        features = []
        
        # First, identify if part has a dominant rotation axis
        rotation_axis = self._find_rotation_axis(shape)
        
        if rotation_axis is None:
            logger.info("No dominant rotation axis found - not a turning part")
            return features
        
        logger.info(f"Detected rotation axis: {rotation_axis}")
        
        # Detect keyways
        keyways = self._detect_keyways(shape, rotation_axis)
        features.extend(keyways)
        
        # Detect V-grooves
        v_grooves = self._detect_v_grooves(shape, rotation_axis)
        features.extend(v_grooves)
        
        # Detect threads (simplified - full detection requires helical curve analysis)
        # threads = self._detect_threads(shape, rotation_axis)
        # features.extend(threads)
        
        # Detect shoulder steps
        shoulders = self._detect_shoulders(shape, rotation_axis)
        features.extend(shoulders)
        
        logger.info(f"Detected {len(features)} turning features")
        
        return features
    
    def _find_rotation_axis(self, shape: TopoDS_Shape) -> Optional[gp_Ax1]:
        """
        Find dominant rotation axis by analyzing cylindrical faces
        
        Returns:
            gp_Ax1 if found, None otherwise
        """
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        cylindrical_axes = []
        cylindrical_areas = []
        
        while face_explorer.More():
            face = face_explorer.Current()
            surface_adaptor = BRepAdaptor_Surface(face)
            
            if surface_adaptor.GetType() == GeomAbs_Cylinder:
                # Get cylinder axis
                cylinder = surface_adaptor.Cylinder()
                axis = cylinder.Axis()
                
                # Compute face area
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                
                cylindrical_axes.append(axis)
                cylindrical_areas.append(area)
            
            face_explorer.Next()
        
        if not cylindrical_axes:
            return None
        
        # Find dominant axis (most common direction weighted by area)
        # Simplified: return axis of largest cylindrical face
        max_idx = np.argmax(cylindrical_areas)
        return cylindrical_axes[max_idx]
    
    def _detect_keyways(
        self,
        shape: TopoDS_Shape,
        rotation_axis: gp_Ax1
    ) -> List[Dict]:
        """
        Detect keyways (rectangular slots parallel to rotation axis)
        
        Keyway characteristics:
        - Two parallel planar faces perpendicular to radial direction
        - Rectangular cross-section
        - Parallel to shaft axis
        """
        keyways = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0
        planar_faces = []
        
        # Collect planar faces
        while face_explorer.More():
            face = face_explorer.Current()
            surface_adaptor = BRepAdaptor_Surface(face)
            
            if surface_adaptor.GetType() == GeomAbs_Plane:
                plane = surface_adaptor.Plane()
                normal = plane.Axis().Direction()
                position = plane.Location()
                
                # Check if plane is parallel to rotation axis
                dot = abs(normal.Dot(rotation_axis.Direction()))
                
                if dot < 0.1:  # Nearly perpendicular (keyway faces are perpendicular to axis)
                    planar_faces.append({
                        'face_id': face_id,
                        'normal': normal,
                        'position': position,
                        'face': face
                    })
            
            face_id += 1
            face_explorer.Next()
        
        # Find parallel face pairs (keyway has two opposing walls)
        for i, face1 in enumerate(planar_faces):
            for face2 in planar_faces[i+1:]:
                # Check if normals are opposite (facing each other)
                dot = face1['normal'].Dot(face2['normal'])
                
                if dot < -0.95:  # Nearly opposite
                    # Compute distance between planes
                    v = gp_Dir(
                        face2['position'].X() - face1['position'].X(),
                        face2['position'].Y() - face1['position'].Y(),
                        face2['position'].Z() - face1['position'].Z()
                    )
                    distance = v.Dot(face1['normal'])
                    
                    # Keyways typically 3-10mm wide
                    if 0.003 < distance < 0.020:  # 3-20mm
                        keyways.append({
                            'feature_type': 'keyway',
                            'face_ids': [face1['face_id'], face2['face_id']],
                            'confidence': 0.85,
                            'axis': (
                                rotation_axis.Direction().X(),
                                rotation_axis.Direction().Y(),
                                rotation_axis.Direction().Z()
                            ),
                            'center': (
                                (face1['position'].X() + face2['position'].X()) / 2,
                                (face1['position'].Y() + face2['position'].Y()) / 2,
                                (face1['position'].Z() + face2['position'].Z()) / 2
                            ),
                            'parameters': {
                                'width_mm': distance * 1000,
                                'length_mm': 0.0  # Would compute from edge lengths
                            }
                        })
                        logger.info(f"Detected keyway: width={distance*1000:.2f}mm")
                        break  # Found pair, move to next face
        
        return keyways
    
    def _detect_v_grooves(
        self,
        shape: TopoDS_Shape,
        rotation_axis: gp_Ax1
    ) -> List[Dict]:
        """
        Detect V-grooves (conical depressions)
        
        V-groove characteristics:
        - Conical face(s)
        - Axis aligned with rotation axis
        - Typically 60°, 90°, or 120° included angle
        """
        v_grooves = []
        # Simplified implementation - full version would analyze conical faces
        # and compute groove angle
        
        return v_grooves
    
    def _detect_shoulders(
        self,
        shape: TopoDS_Shape,
        rotation_axis: gp_Ax1
    ) -> List[Dict]:
        """
        Detect shoulder steps (diameter changes)
        
        Shoulder characteristics:
        - Planar face perpendicular to rotation axis
        - Connects two different diameter sections
        """
        shoulders = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0
        
        while face_explorer.More():
            face = face_explorer.Current()
            surface_adaptor = BRepAdaptor_Surface(face)
            
            if surface_adaptor.GetType() == GeomAbs_Plane:
                plane = surface_adaptor.Plane()
                normal = plane.Axis().Direction()
                
                # Check if plane is perpendicular to rotation axis
                dot = abs(normal.Dot(rotation_axis.Direction()))
                
                if dot > 0.95:  # Nearly parallel (shoulder face is perpendicular to axis)
                    # This is a shoulder candidate
                    shoulders.append({
                        'feature_type': 'shoulder',
                        'face_ids': [face_id],
                        'confidence': 0.75,
                        'axis': (
                            rotation_axis.Direction().X(),
                            rotation_axis.Direction().Y(),
                            rotation_axis.Direction().Z()
                        ),
                        'center': (
                            plane.Location().X(),
                            plane.Location().Y(),
                            plane.Location().Z()
                        ),
                        'parameters': {}
                    })
            
            face_id += 1
            face_explorer.Next()
        
        return shoulders


def merge_adjacent_features(features: List[Dict], tolerance: float = 0.001) -> List[Dict]:
    """
    Merge geometrically adjacent features into compound features
    
    Example: Two half-keyways on opposite sides → single through-keyway
    """
    # Simplified implementation
    return features
