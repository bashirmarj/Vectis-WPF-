"""
Lump Classifier
===============
Classifies volumetric lumps (TopoDS_Solid) into machining features.

Classification Logic:
1. Analyze faces of the lump.
2. Check for cylindrical surfaces (Holes).
3. Check for planar surfaces (Pockets).
4. Check orientation relative to principal axes.
"""

import logging
import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus, GeomAbs_Sphere
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

logger = logging.getLogger(__name__)

class LumpClassifier:
    def __init__(self):
        pass

    def classify(self, lump_shape, stock_bbox: dict) -> dict:
        """
        Classify a single lump solid.
        
        Args:
            lump_shape: TopoDS_Solid
            stock_bbox: Dictionary with stock bounds (xmin, xmax, etc.)
            
        Returns:
            dict: {
                'type': 'hole' | 'pocket' | 'slot' | 'step' | 'unknown',
                'parameters': {...},
                'confidence': float
            }
        """
        # 1. Analyze surfaces
        surfaces = self._analyze_surfaces(lump_shape)
        
        # 2. Analyze boundary conditions (how it interacts with stock)
        boundary_info = self._analyze_boundaries(lump_shape, stock_bbox)
        
        # 3. Heuristic Classification
        
        # Case A: Hole (Cylinder + Caps)
        if self._is_hole(surfaces):
            return self._classify_hole(surfaces, boundary_info)
            
        # Case B: Step (Open on side)
        # A step typically touches the stock boundary on the side (not just Z min/max)
        if self._is_step(boundary_info):
            return {
                'type': 'step',
                'subtype': 'open_step',
                'confidence': 0.85,
                'boundary_faces': boundary_info['count']
            }
            
        # Case C: Pocket (Prismatic)
        if self._is_pocket(surfaces):
            return self._classify_pocket(surfaces, boundary_info)
            
        return {'type': 'unknown', 'confidence': 0.0}

    def _analyze_boundaries(self, shape, bbox) -> dict:
        """
        Check which faces of the lump align with the stock bounding box.
        Returns dict with 'top', 'bottom', 'sides' booleans.
        """
        tol = 1e-3
        info = {'top': False, 'bottom': False, 'sides': 0, 'count': 0}
        
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            center = props.CentreOfMass()
            
            # Check alignment with bbox planes
            on_boundary = False
            
            # Z Max (Top)
            if abs(center.Z() - bbox['zmax']) < tol:
                info['top'] = True
                on_boundary = True
            # Z Min (Bottom)
            elif abs(center.Z() - bbox['zmin']) < tol:
                info['bottom'] = True
                on_boundary = True
            # X Min/Max
            elif abs(center.X() - bbox['xmin']) < tol or abs(center.X() - bbox['xmax']) < tol:
                info['sides'] += 1
                on_boundary = True
            # Y Min/Max
            elif abs(center.Y() - bbox['ymin']) < tol or abs(center.Y() - bbox['ymax']) < tol:
                info['sides'] += 1
                on_boundary = True
                
            if on_boundary:
                info['count'] += 1
                
            exp.Next()
        return info

    def _is_step(self, boundary_info) -> bool:
        """
        Is this a step?
        Criteria:
        - Touches Top (usually)
        - Touches at least one Side (X or Y)
        - NOT a through hole (which touches Top + Bottom but no sides usually)
        """
        if boundary_info['sides'] > 0 and boundary_info['top']:
            return True
        return False

    def _is_hole(self, surfaces) -> bool:
        """
        Is this lump a hole?
        Criteria:
        - Has at least one cylinder.
        - No complex/freeform surfaces.
        """
        if surfaces['cylinder'] == 0 and surfaces['cone'] == 0:
            return False
            
        if surfaces['other'] > 0:
            return False
            
        return True

    def _classify_hole(self, surfaces, boundary_info) -> dict:
        """Extract hole parameters."""
        # Get max radius (main bore)
        max_r = 0
        axis = [0,0,1]
        
        for cyl in surfaces['cylinders']:
            if cyl['radius'] > max_r:
                max_r = cyl['radius']
                a = cyl['axis']
                axis = [a.X(), a.Y(), a.Z()]
                
        # Determine type
        radii = set(round(c['radius'], 3) for c in surfaces['cylinders'])
        
        subtype = 'simple_hole'
        if len(radii) > 1:
            subtype = 'counterbore_hole'
            
        # Check through vs blind
        is_through = boundary_info['top'] and boundary_info['bottom']
        if is_through:
            subtype += "_through"
        else:
            subtype += "_blind"
            
        return {
            'type': 'hole',
            'subtype': subtype,
            'diameter': max_r * 2.0,
            'axis': axis,
            'confidence': 0.95
        }

    def _is_pocket(self, surfaces) -> bool:
        """
        Is this lump a pocket?
        Criteria:
        - Mostly planar.
        """
        if surfaces['cylinder'] == 0 and surfaces['cone'] == 0:
            return True
        return True # Allow filleted pockets

    def _classify_pocket(self, surfaces, boundary_info) -> dict:
        subtype = 'closed_pocket'
        if boundary_info['sides'] > 0:
            subtype = 'open_pocket'
            
        return {
            'type': 'pocket',
            'subtype': subtype,
            'confidence': 0.8
        }
