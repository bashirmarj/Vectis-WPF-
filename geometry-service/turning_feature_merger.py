"""
turning_feature_merger.py
==========================

SEMANTIC MERGING for turning features to eliminate duplicates.

Version: 2.0 (FIXED - More lenient tolerances for circular edge splits)

Solves:
- V-groove split into 2 conical faces → 2 grooves (FIXED: merge into 1)
- Tapered hole detected multiple times → 3 tapers (FIXED: merge into 1)
- Circular edges split → duplicate steps (FIXED: merge into 1)

Key fixes in v2.0:
- Increased tolerances for circular edge splits (5mm axis, 10mm position, 3mm diameter)
- Better logging to show actual merges
- Added detailed debug output when features don't match
- Special handling for rotational features with loose radial tolerance

Strategy:
1. Group features by type (groove, taper, step)
2. Merge features with:
   - Same axis (coaxial within 5mm - lenient for edge splits)
   - Adjacent axial positions (touching or overlapping within 10mm)
   - Similar geometric properties (diameter within 3mm, angle within 10°)
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
    
    Version 2.0: More lenient tolerances for real-world circular edge splits.
    """
    
    def __init__(self, 
                 axis_tolerance: float = 5.0,       # mm - INCREASED from 0.5mm
                 position_tolerance: float = 10.0,   # mm - INCREASED from 2.0mm
                 diameter_tolerance: float = 3.0,    # mm - INCREASED from 1.0mm
                 angle_tolerance: float = 10.0):     # degrees - INCREASED from 5.0°
        """
        Initialize merger with geometric tolerances.
        
        LENIENT TOLERANCES for circular edge splits that can have significant
        positional variations while still being part of the same feature.
        
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
        
        logger.info(f"   📏 Merger tolerances: axis={axis_tolerance}mm, pos={position_tolerance}mm, dia={diameter_tolerance}mm, angle={angle_tolerance}°")
    
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
        
        initial_count = len(features)
        logger.info(f"\n🔗 Semantic merging: {initial_count} raw features")
        
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
        
        removed = initial_count - len(merged_features)
        if removed > 0:
            logger.info(f"   ✅ Merged {removed} duplicate features → {len(merged_features)} unique features")
        else:
            logger.info(f"   ℹ️  No duplicates found (all {len(merged_features)} features are unique)")
        
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
            if grooves:
                logger.info(f"   🔧 Grooves: {len(grooves)} (no duplicates to merge)")
            return grooves
        
        logger.info(f"   🔧 Processing {len(grooves)} grooves for merging...")
        
        merged = []
        used = set()
        merge_count = 0
        
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
                
                # Debug: Check each criterion
                is_coaxial = self._are_coaxial(axis1, axis2, loc1, loc2)
                axial_distance = abs(np.dot(loc2 - loc1, axis1 / (np.linalg.norm(axis1) + 1e-10)))
                is_adjacent = axial_distance < self.position_tolerance
                is_similar_dia = abs(dia1 - dia2) < self.diameter_tolerance
                
                if not is_coaxial:
                    logger.debug(f"      Groove {i} vs {j}: NOT coaxial (dist={self._axis_distance(axis1, axis2, loc1, loc2):.2f}mm > {self.axis_tolerance}mm)")
                    continue
                
                if not is_adjacent:
                    logger.debug(f"      Groove {i} vs {j}: NOT adjacent (axial_dist={axial_distance:.2f}mm > {self.position_tolerance}mm)")
                    continue
                
                if not is_similar_dia:
                    logger.debug(f"      Groove {i} vs {j}: NOT similar diameter (Δdia={abs(dia1-dia2):.2f}mm > {self.diameter_tolerance}mm)")
                    continue
                
                # This is part of the same groove!
                logger.info(f"      ✓ Merging groove {i} (Ø{dia1:.1f}mm) + groove {j} (Ø{dia2:.1f}mm)")
                group.append(g2)
                used.add(j)
            
            # Merge the group into one feature
            if len(group) > 1:
                merge_count += len(group) - 1
                merged_feature = self._merge_feature_group(group)
                merged.append(merged_feature)
                logger.info(f"      → Merged {len(group)} groove sections → 1 groove (Ø{merged_feature.diameter:.1f}mm)")
            else:
                merged.append(g1)
            
            used.add(i)
        
        if merge_count > 0:
            logger.info(f"   ✅ Grooves: {len(grooves)} → {len(merged)} (removed {merge_count} duplicates)")
        else:
            logger.info(f"   ℹ️  Grooves: {len(grooves)} (no matches found - check tolerances if unexpected)")
        
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
            if tapers:
                logger.info(f"   🔧 Tapers: {len(tapers)} (no duplicates to merge)")
            return tapers
        
        logger.info(f"   🔧 Processing {len(tapers)} tapers for merging...")
        
        merged = []
        used = set()
        merge_count = 0
        
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
                
                # Debug: Check each criterion
                is_coaxial = self._are_coaxial(axis1, axis2, loc1, loc2)
                axial_distance = abs(np.dot(loc2 - loc1, axis1 / (np.linalg.norm(axis1) + 1e-10)))
                is_adjacent = axial_distance < self.position_tolerance
                is_similar_angle = abs(angle1 - angle2) < self.angle_tolerance
                
                if not is_coaxial:
                    logger.debug(f"      Taper {i} vs {j}: NOT coaxial (dist={self._axis_distance(axis1, axis2, loc1, loc2):.2f}mm > {self.axis_tolerance}mm)")
                    continue
                
                if not is_adjacent:
                    logger.debug(f"      Taper {i} vs {j}: NOT adjacent (axial_dist={axial_distance:.2f}mm > {self.position_tolerance}mm)")
                    continue
                
                if not is_similar_angle:
                    logger.debug(f"      Taper {i} vs {j}: NOT similar angle (Δangle={abs(angle1-angle2):.2f}° > {self.angle_tolerance}°)")
                    continue
                
                # Same taper!
                logger.info(f"      ✓ Merging taper {i} (angle={angle1:.1f}°) + taper {j} (angle={angle2:.1f}°)")
                group.append(t2)
                used.add(j)
            
            if len(group) > 1:
                merge_count += len(group) - 1
                merged_feature = self._merge_feature_group(group)
                merged.append(merged_feature)
                logger.info(f"      → Merged {len(group)} taper sections → 1 taper (angle={merged_feature.taper_angle:.1f}°)")
            else:
                merged.append(t1)
            
            used.add(i)
        
        if merge_count > 0:
            logger.info(f"   ✅ Tapers: {len(tapers)} → {len(merged)} (removed {merge_count} duplicates)")
        else:
            logger.info(f"   ℹ️  Tapers: {len(tapers)} (no matches found - check tolerances if unexpected)")
        
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
            if steps:
                logger.info(f"   🔧 Steps: {len(steps)} (no duplicates to merge)")
            return steps
        
        logger.info(f"   🔧 Processing {len(steps)} steps for merging...")
        
        merged = []
        used = set()
        merge_count = 0
        
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
                
                # Debug: Check each criterion
                is_coaxial = self._are_coaxial(axis1, axis2, loc1, loc2)
                is_same_dia = abs(dia1 - dia2) < self.diameter_tolerance
                axial_distance = abs(np.dot(loc2 - loc1, axis1 / (np.linalg.norm(axis1) + 1e-10)))
                is_adjacent = axial_distance < self.position_tolerance + 5.0  # Extra lenient for steps
                
                if not is_coaxial:
                    logger.debug(f"      Step {i} vs {j}: NOT coaxial (dist={self._axis_distance(axis1, axis2, loc1, loc2):.2f}mm > {self.axis_tolerance}mm)")
                    continue
                
                if not is_same_dia:
                    logger.debug(f"      Step {i} vs {j}: NOT same diameter (Δdia={abs(dia1-dia2):.2f}mm > {self.diameter_tolerance}mm)")
                    continue
                
                if not is_adjacent:
                    logger.debug(f"      Step {i} vs {j}: NOT adjacent (axial_dist={axial_distance:.2f}mm > {self.position_tolerance + 5.0}mm)")
                    continue
                
                # Same step!
                logger.info(f"      ✓ Merging step {i} (Ø{dia1:.1f}mm) + step {j} (Ø{dia2:.1f}mm)")
                group.append(s2)
                used.add(j)
            
            if len(group) > 1:
                merge_count += len(group) - 1
                merged_feature = self._merge_feature_group(group)
                merged.append(merged_feature)
                logger.info(f"      → Merged {len(group)} step sections → 1 step (Ø{merged_feature.diameter:.1f}mm)")
            else:
                merged.append(s1)
            
            used.add(i)
        
        if merge_count > 0:
            logger.info(f"   ✅ Steps: {len(steps)} → {len(merged)} (removed {merge_count} duplicates)")
        else:
            logger.info(f"   ℹ️  Steps: {len(steps)} (no matches found - check tolerances if unexpected)")
        
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
        axis1_norm = axis1 / (np.linalg.norm(axis1) + 1e-10)
        axis2_norm = axis2 / (np.linalg.norm(axis2) + 1e-10)
        
        # Check if parallel (within 5 degrees - very lenient)
        axis_alignment = abs(np.dot(axis1_norm, axis2_norm))
        if axis_alignment < 0.996:  # cos(5°) ≈ 0.996
            return False
        
        # Check if axes are close (perpendicular distance)
        loc_vec = loc2 - loc1
        cross = np.cross(axis1_norm, loc_vec)
        distance = np.linalg.norm(cross)
        
        return distance < self.axis_tolerance
    
    def _axis_distance(self, axis1: np.ndarray, axis2: np.ndarray,
                      loc1: np.ndarray, loc2: np.ndarray) -> float:
        """Calculate perpendicular distance between two axes"""
        axis1_norm = axis1 / (np.linalg.norm(axis1) + 1e-10)
        loc_vec = loc2 - loc1
        cross = np.cross(axis1_norm, loc_vec)
        return np.linalg.norm(cross)
    
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
                          axis_tol: float = 5.0,
                          position_tol: float = 10.0,
                          diameter_tol: float = 3.0) -> List:
    """
    Convenience function to apply semantic merging to turning features.
    
    Args:
        features: List of TurningFeature objects
        axis_tol: Coaxial tolerance (mm) - default 5.0mm (lenient for edge splits)
        position_tol: Adjacent position tolerance (mm) - default 10.0mm (lenient)
        diameter_tol: Diameter similarity tolerance (mm) - default 3.0mm (lenient)
        
    Returns:
        List of merged features with duplicates removed
    """
    merger = TurningFeatureMerger(
        axis_tolerance=axis_tol,
        position_tolerance=position_tol,
        diameter_tolerance=diameter_tol
    )
    return merger.merge_turning_features(features)
