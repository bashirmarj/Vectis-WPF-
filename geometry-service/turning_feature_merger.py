"""
turning_feature_merger.py
==========================

SEMANTIC MERGING for turning features to eliminate duplicates.

Solves:
- V-groove split into 2 conical faces → 2 grooves (FIXED: merge into 1)
- Tapered hole detected multiple times → 3 tapers (FIXED: merge into 1)
- Circular edges split → duplicate steps (FIXED: merge into 1)

Strategy:
1. Group features by type (groove, taper, step)
2. Merge features with:
   - Same axis (coaxial within tolerance)
   - Adjacent axial positions (touching or overlapping)
   - Similar geometric properties (diameter, width, angle)
3. Combine face_indices from merged features
4. Update dimensions based on merged geometry
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import replace
import logging

logger = logging.getLogger(__name__)


class TurningFeatureMerger:
    """
    Semantic merging for turning features to eliminate splits from circular edges.
    """
    
    def __init__(self, 
                 axis_tolerance: float = 0.5,      # mm - coaxial tolerance
                 position_tolerance: float = 2.0,   # mm - adjacent position tolerance
                 diameter_tolerance: float = 1.0,   # mm - similar diameter tolerance
                 angle_tolerance: float = 5.0):     # degrees - similar angle tolerance
        """
        Initialize merger with geometric tolerances.
        
        Args:
            axis_tolerance: Maximum distance between axes to consider coaxial (mm)
            position_tolerance: Maximum axial gap to consider adjacent (mm)
            diameter_tolerance: Maximum diameter difference to consider same feature (mm)
            angle_tolerance: Maximum angle difference for tapers (degrees)
        """
        self.axis_tolerance = axis_tolerance
        self.position_tolerance = position_tolerance
        self.diameter_tolerance = diameter_tolerance
        self.angle_tolerance = angle_tolerance
    
    def merge_turning_features(self, features: List) -> List:
        """
        Main entry point: Merge duplicate turning features.
        
        Args:
            features: List of TurningFeature objects
            
        Returns:
            List of merged TurningFeature objects with duplicates removed
        """
        if not features:
            return features
        
        logger.info(f"\n🔗 Semantic merging: {len(features)} raw features")
        
        # Separate by feature type
        grooves = [f for f in features if f.feature_type.value == 'groove']
        tapers = [f for f in features if f.feature_type.value == 'taper']
        steps = [f for f in features if f.feature_type.value == 'step']
        bases = [f for f in features if f.feature_type.value == 'base_cylinder']
        others = [f for f in features if f.feature_type.value not in ['groove', 'taper', 'step', 'base_cylinder']]
        
        # Merge each type
        merged_grooves = self._merge_grooves(grooves)
        merged_tapers = self._merge_tapers(tapers)
        merged_steps = self._merge_steps(steps)
        
        # Combine all
        merged_features = merged_grooves + merged_tapers + merged_steps + bases + others
        
        removed = len(features) - len(merged_features)
        if removed > 0:
            logger.info(f"   → {len(merged_features)} unique features (-{removed} duplicates)")
        
        return merged_features
    
    def _merge_grooves(self, grooves: List) -> List:
        """
        Merge grooves that are part of the same V-groove or rectangular groove.
        
        V-grooves are often detected as 2 separate conical faces.
        Merge criteria:
        - Coaxial (same rotation axis)
        - Adjacent axially (touching positions)
        - Similar diameter range
        """
        if len(grooves) <= 1:
            return grooves
        
        logger.info(f"   🔗 Merging {len(grooves)} grooves...")
        
        merged = []
        used = set()
        
        for i, g1 in enumerate(grooves):
            if i in used:
                continue
            
            # Start a merge group with g1
            group = [g1]
            axis1 = np.array(g1.axis)
            loc1 = np.array(g1.location)
            dia1 = g1.diameter
            
            # Find grooves that should be merged with g1
            for j in range(i + 1, len(grooves)):
                if j in used:
                    continue
                
                g2 = grooves[j]
                axis2 = np.array(g2.axis)
                loc2 = np.array(g2.location)
                dia2 = g2.diameter
                
                # Check if coaxial
                if not self._are_coaxial(axis1, axis2, loc1, loc2):
                    continue
                
                # Check if adjacent axially (along rotation axis)
                axial_distance = abs(np.dot(loc2 - loc1, axis1))
                if axial_distance > self.position_tolerance:
                    continue
                
                # Check if similar diameter (grooves in same feature have similar dia)
                if abs(dia1 - dia2) > self.diameter_tolerance:
                    continue
                
                # This is part of the same groove!
                group.append(g2)
                used.add(j)
            
            # Merge the group into one feature
            if len(group) > 1:
                logger.info(f"      Merging {len(group)} groove sections → 1 groove")
                merged_feature = self._merge_feature_group(group)
                merged.append(merged_feature)
            else:
                merged.append(g1)
            
            used.add(i)
        
        return merged
    
    def _merge_tapers(self, tapers: List) -> List:
        """
        Merge tapers that are part of the same conical surface.
        
        Tapered holes can be split into multiple detections.
        Merge criteria:
        - Coaxial
        - Adjacent positions
        - Similar taper angle
        """
        if len(tapers) <= 1:
            return tapers
        
        logger.info(f"   🔗 Merging {len(tapers)} tapers...")
        
        merged = []
        used = set()
        
        for i, t1 in enumerate(tapers):
            if i in used:
                continue
            
            group = [t1]
            axis1 = np.array(t1.axis)
            loc1 = np.array(t1.location)
            angle1 = t1.taper_angle or 0.0
            
            for j in range(i + 1, len(tapers)):
                if j in used:
                    continue
                
                t2 = tapers[j]
                axis2 = np.array(t2.axis)
                loc2 = np.array(t2.location)
                angle2 = t2.taper_angle or 0.0
                
                # Check if coaxial
                if not self._are_coaxial(axis1, axis2, loc1, loc2):
                    continue
                
                # Check if adjacent
                axial_distance = abs(np.dot(loc2 - loc1, axis1))
                if axial_distance > self.position_tolerance:
                    continue
                
                # Check if similar angle
                if abs(angle1 - angle2) > self.angle_tolerance:
                    continue
                
                # Same taper!
                group.append(t2)
                used.add(j)
            
            if len(group) > 1:
                logger.info(f"      Merging {len(group)} taper sections → 1 taper")
                merged_feature = self._merge_feature_group(group)
                merged.append(merged_feature)
            else:
                merged.append(t1)
            
            used.add(i)
        
        return merged
    
    def _merge_steps(self, steps: List) -> List:
        """
        Merge steps that are at the same radial position.
        
        Circular edges split can cause duplicate step detection.
        Merge criteria:
        - Coaxial
        - Same diameter (radial position)
        - Adjacent or same axial position
        """
        if len(steps) <= 1:
            return steps
        
        logger.info(f"   🔗 Merging {len(steps)} steps...")
        
        merged = []
        used = set()
        
        for i, s1 in enumerate(steps):
            if i in used:
                continue
            
            group = [s1]
            axis1 = np.array(s1.axis)
            loc1 = np.array(s1.location)
            dia1 = s1.diameter
            
            for j in range(i + 1, len(steps)):
                if j in used:
                    continue
                
                s2 = steps[j]
                axis2 = np.array(s2.axis)
                loc2 = np.array(s2.location)
                dia2 = s2.diameter
                
                # Check if coaxial
                if not self._are_coaxial(axis1, axis2, loc1, loc2):
                    continue
                
                # Check if same diameter (same radial position)
                if abs(dia1 - dia2) > self.diameter_tolerance:
                    continue
                
                # Check if adjacent or overlapping axially
                axial_distance = abs(np.dot(loc2 - loc1, axis1))
                if axial_distance > self.position_tolerance + 5.0:  # More lenient for steps
                    continue
                
                # Same step!
                group.append(s2)
                used.add(j)
            
            if len(group) > 1:
                logger.info(f"      Merging {len(group)} step sections → 1 step")
                merged_feature = self._merge_feature_group(group)
                merged.append(merged_feature)
            else:
                merged.append(s1)
            
            used.add(i)
        
        return merged
    
    def _are_coaxial(self, axis1: np.ndarray, axis2: np.ndarray, 
                     loc1: np.ndarray, loc2: np.ndarray) -> bool:
        """
        Check if two features are coaxial.
        
        Args:
            axis1: Direction vector of first feature
            axis2: Direction vector of second feature
            loc1: Location point on first axis
            loc2: Location point on second axis
            
        Returns:
            True if axes are parallel and close enough to be considered same axis
        """
        # Normalize axes
        axis1 = axis1 / (np.linalg.norm(axis1) + 1e-10)
        axis2 = axis2 / (np.linalg.norm(axis2) + 1e-10)
        
        # Check if parallel (within 1 degree)
        axis_alignment = abs(np.dot(axis1, axis2))
        if axis_alignment < 0.9998:  # cos(1°) ≈ 0.9998
            return False
        
        # Check if axes are close (perpendicular distance)
        loc_vec = loc2 - loc1
        cross = np.cross(axis1, loc_vec)
        distance = np.linalg.norm(cross)
        
        return distance < self.axis_tolerance
    
    def _merge_feature_group(self, group: List):
        """
        Merge a group of features into a single feature.
        
        Strategy:
        - Use average location
        - Use average dimensions where applicable
        - Combine all face_indices
        - Average confidence
        - Keep feature type from first feature
        """
        if len(group) == 1:
            return group[0]
        
        # Calculate averages
        avg_location = tuple(np.mean([f.location for f in group], axis=0))
        avg_axis = tuple(np.mean([f.axis for f in group], axis=0))
        avg_diameter = np.mean([f.diameter for f in group])
        avg_length = np.mean([f.length for f in group])
        avg_confidence = np.mean([f.confidence for f in group])
        
        # Combine face indices
        combined_face_indices = []
        for f in group:
            combined_face_indices.extend(f.face_indices)
        combined_face_indices = sorted(set(combined_face_indices))
        
        # Merge optional attributes
        feature_type = group[0].feature_type
        
        if feature_type.value == 'groove':
            avg_groove_width = np.mean([f.groove_width for f in group if f.groove_width])
            groove_type = group[0].groove_type
            
            merged = replace(
                group[0],
                location=avg_location,
                axis=avg_axis,
                diameter=avg_diameter,
                length=avg_length,
                groove_width=avg_groove_width if avg_groove_width else None,
                groove_type=groove_type,
                confidence=avg_confidence,
                face_indices=combined_face_indices
            )
        
        elif feature_type.value == 'taper':
            avg_taper_angle = np.mean([f.taper_angle for f in group if f.taper_angle])
            avg_start_dia = np.mean([f.start_diameter for f in group if f.start_diameter])
            avg_end_dia = np.mean([f.end_diameter for f in group if f.end_diameter])
            
            merged = replace(
                group[0],
                location=avg_location,
                axis=avg_axis,
                diameter=avg_diameter,
                length=avg_length,
                taper_angle=avg_taper_angle if avg_taper_angle else None,
                start_diameter=avg_start_dia if avg_start_dia else None,
                end_diameter=avg_end_dia if avg_end_dia else None,
                confidence=avg_confidence,
                face_indices=combined_face_indices
            )
        
        elif feature_type.value == 'step':
            avg_step_depth = np.mean([f.step_depth for f in group if f.step_depth])
            
            merged = replace(
                group[0],
                location=avg_location,
                axis=avg_axis,
                diameter=avg_diameter,
                length=avg_length,
                step_depth=avg_step_depth if avg_step_depth else None,
                confidence=avg_confidence,
                face_indices=combined_face_indices
            )
        
        else:
            # Generic merge
            merged = replace(
                group[0],
                location=avg_location,
                axis=avg_axis,
                diameter=avg_diameter,
                length=avg_length,
                confidence=avg_confidence,
                face_indices=combined_face_indices
            )
        
        return merged


# Integration helper
def apply_semantic_merging(features: List, 
                          axis_tol: float = 0.5,
                          position_tol: float = 2.0) -> List:
    """
    Convenience function to apply semantic merging to turning features.
    
    Args:
        features: List of TurningFeature objects
        axis_tol: Coaxial tolerance (mm)
        position_tol: Adjacent position tolerance (mm)
        
    Returns:
        List of merged features with duplicates removed
    """
    merger = TurningFeatureMerger(
        axis_tolerance=axis_tol,
        position_tolerance=position_tol
    )
    return merger.merge_turning_features(features)
