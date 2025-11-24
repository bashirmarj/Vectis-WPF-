"""
Feature Mapper
==============
Maps recognized volumetric lumps back to original part faces.

This is CRITICAL for the viewer. The Volume Decomposition gives us "Delta Solids",
but the viewer needs "Part Face IDs" to highlight.

Strategy:
1. Geometric Proximity: Find part faces that are coincident with lump faces.
2. Orientation Check: Lump faces (removed material) should be coincident with Part faces (walls).
"""

import logging
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder

logger = logging.getLogger(__name__)

class FeatureMapper:
    def __init__(self, part_shape, aag_graph):
        self.part_shape = part_shape
        self.aag = aag_graph # We use AAG just for the face cache/lookup
        
    def map_features(self, classified_lumps: list) -> list:
        """
        Map lumps to face IDs.
        
        Args:
            classified_lumps: List of dicts from LumpClassifier
            
        Returns:
            List of feature dicts with 'face_ids' populated
        """
        mapped_features = []
        
        for lump_data in classified_lumps:
            lump_shape = lump_data.get('shape')
            if not lump_shape:
                continue
                
            # Find matching faces
            face_ids = self._find_matching_faces(lump_shape)
            
            # Create final feature object
            feature = lump_data.copy()
            if 'shape' in feature:
                del feature['shape'] # Remove OCC shape from output
                
            feature['face_ids'] = face_ids
            mapped_features.append(feature)
            
        return mapped_features
        
    def _find_matching_faces(self, lump_shape) -> list:
        """
        Find part faces that match the lump's boundary.
        """
        matched_ids = []
        
        # Iterate over lump faces
        lump_exp = TopExp_Explorer(lump_shape, TopAbs_FACE)
        while lump_exp.More():
            lump_face = lump_exp.Current()
            
            # Get lump face props
            lump_props = GProp_GProps()
            brepgprop.SurfaceProperties(lump_face, lump_props)
            lump_center = lump_props.CentreOfMass()
            lump_area = lump_props.Mass()
            
            # Search in AAG nodes (which are the part faces)
            # Handle both dict and object AAG representations
            nodes = self.aag.nodes if hasattr(self.aag, 'nodes') else self.aag['nodes']
            
            for face_id, node in nodes.items():
                # Fast check: Area
                if abs(node['area'] - lump_area) > 1e-3: # Tolerance
                    continue
                    
                # Check Center distance
                node_center = node['center']
                dist = (lump_center.X() - node_center[0])**2 + \
                       (lump_center.Y() - node_center[1])**2 + \
                       (lump_center.Z() - node_center[2])**2
                       
                if dist < 1e-4: # Coincident
                    matched_ids.append(face_id)
                    
            lump_exp.Next()
            
        return list(set(matched_ids))
