"""
turning_feature_merger.py
=========================

Semantic merging for turning features to eliminate split circular edges.

Version: 2.2 - Relaxed Tolerances
Author: Production Feature Recognition Team

CHANGES IN v2.2:
- ✅ RELAXED all tolerances 3-5x for real-world CAD files
- ✅ axis_tolerance: 1.0mm → 5.0mm
- ✅ position_tolerance: 10.0mm → 50.0mm  
- ✅ diameter_tolerance: 3.0mm → 10.0mm
- ✅ angle_tolerance: 5° → 15°
- ✅ V-groove same-Z: 0.5mm → 2.0mm
- ✅ Axis parallelism: 8° → 15°

CHANGES IN v2.1:
- ✅ V-groove merging: Detects 2 grooves at same Z but opposite radial sides
- ✅ Special handling for circumferential grooves (same axial position)
- ✅ Fixes V-groove pulley detection (2 grooves → 1 v-groove)

Solves:
- Circular edges split into segments → Multiple detections
- V-groove detected as 2 separate grooves
- Tapered hole split into multiple tapers
- Steps split across circular edge boundaries

Uses geometric relationships:
- Coaxial alignment (same axis)
- Adjacent positions (axially close OR same Z-position)
- Similar dimensions (diameter, angle)
- Merges face indices from split features
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class TurningFeatureMerger:
    """
    Merges turning features split by circular edge segmentation.
    
    Example: V-groove with 2 conical faces → 1 groove feature
    """

    def __init__(self,
                 axis_tolerance: float = 5.0,      # Coaxial tolerance (mm) - RELAXED from 1.0
                 position_tolerance: float = 50.0,  # Adjacent tolerance (mm) - RELAXED from 10.0
                 diameter_tolerance: float = 10.0,  # Diameter match (mm) - RELAXED from 3.0
                 angle_tolerance: float = 15.0):    # Angle match (degrees) - RELAXED from 5.0
        """
        Initialize merger with lenient tolerances.
        
        Args:
            axis_tolerance: How far off-axis to still consider coaxial (mm)
            position_tolerance: Axial distance to consider adjacent (mm)
            diameter_tolerance: Diameter difference for matching (mm)
            angle_tolerance: Angle difference for matching (degrees)
        """
        self.axis_tolerance = axis_tolerance
        self.position_tolerance = position_tolerance
        self.diameter_tolerance = diameter_tolerance
        self.angle_tolerance = angle_tolerance

    def merge_turning_features(self, features: List) -> List:
        """
        Main entry point: Merge turning features by type.
        
        Args:
            features: List of TurningFeature objects
            
        Returns:
            List of merged features with duplicates removed
        """
        if not features:
            return features

        logger.info(f"🔗 Semantic merging: {len(features)} turning features")
        logger.info(f"   Tolerances: axis={self.axis_tolerance}mm, pos={self.position_tolerance}mm, "
                   f"dia={self.diameter_tolerance}mm, angle={self.angle_tolerance}°")

        # Group by feature type
        by_type = {}
        for feat in features:
            feat_type = feat.feature_type.value
            if feat_type not in by_type:
                by_type[feat_type] = []
            by_type[feat_type].append(feat)

        merged_all = []
        total_merged = 0

        # Merge each type separately
        for feat_type, flist in by_type.items():
            if len(flist) <= 1:
                merged_all.extend(flist)
                continue

            if feat_type == 'groove':
                merged = self._merge_grooves(flist)
            elif feat_type == 'taper':
                merged = self._merge_tapers(flist)
            elif feat_type == 'step':
                merged = self._merge_steps(flist)
            elif feat_type == 'base_cylinder':
                merged = self._merge_bases(flist)
            else:
                merged = flist

            removed = len(flist) - len(merged)
            if removed > 0:
                logger.info(f"   ✓ Merged {removed} {feat_type}(s): {len(flist)} → {len(merged)}")
                total_merged += removed

            merged_all.extend(merged)

        if total_merged > 0:
            logger.info(f"✅ Merged {total_merged} duplicates → {len(merged_all)} unique features")
        else:
            logger.info(f"ℹ️  No duplicates found - features already unique")

        return merged_all

    def _merge_grooves(self, grooves: List) -> List:
        """
        Merge grooves with special V-groove handling.
        
        V-groove pattern: 2 grooves at SAME axial position but opposite sides
        Regular groove: Adjacent axial positions (< position_tolerance)
        """
        if len(grooves) <= 1:
            return grooves

        logger.info(f"   Processing {len(grooves)} grooves for merging...")

        merged = []
        used = set()

        for i, g1 in enumerate(grooves):
            if i in used:
                continue

            # Start merge group with g1
            merge_group = [g1]
            axis1 = np.array(g1.axis)
            loc1 = np.array(g1.location)
            z1 = np.dot(loc1, axis1)  # Axial position

            for j in range(i + 1, len(grooves)):
                if j in used:
                    continue

                g2 = grooves[j]
                axis2 = np.array(g2.axis)
                loc2 = np.array(g2.location)
                z2 = np.dot(loc2, axis2)

                # Check if coaxial
                if not self._axes_parallel(axis1, axis2):
                    continue

                # Check diameter similarity
                if not self._diameters_similar(g1.diameter, g2.diameter):
                    continue

                # V-GROOVE FIX v2.1: Check for same axial position (circumferential groove)
                z_distance = abs(z1 - z2)
                
                if z_distance < 2.0:  # Same Z-position (within 2.0mm) - RELAXED from 0.5mm
                    # V-groove pattern: 2 grooves at same axial position
                    # They're on opposite sides of the rotation axis
                    logger.info(f"      V-groove detected: groove {i} + groove {j} at same Z-position (Δz={z_distance:.2f}mm)")
                    merge_group.append(g2)
                    used.add(j)
                elif z_distance < self.position_tolerance:
                    # Adjacent grooves (regular pattern)
                    merge_group.append(g2)
                    used.add(j)

            if len(merge_group) > 1:
                merged.append(self._merge_group(merge_group))
            else:
                merged.append(g1)

        if len(merged) == len(grooves):
            logger.info(f"   ℹ️  Grooves: {len(grooves)} (no matches found - check tolerances if unexpected)")

        return merged

    def _merge_tapers(self, tapers: List) -> List:
        """
        Merge tapers with same axis, adjacent positions, similar angles.
        """
        if len(tapers) <= 1:
            return tapers

        logger.info(f"   Processing {len(tapers)} tapers for merging...")

        merged = []
        used = set()

        for i, t1 in enumerate(tapers):
            if i in used:
                continue

            merge_group = [t1]
            axis1 = np.array(t1.axis)
            loc1 = np.array(t1.location)
            z1 = np.dot(loc1, axis1)
            angle1 = t1.taper_angle or 0

            for j in range(i + 1, len(tapers)):
                if j in used:
                    continue

                t2 = tapers[j]
                axis2 = np.array(t2.axis)
                loc2 = np.array(t2.location)
                z2 = np.dot(loc2, axis2)
                angle2 = t2.taper_angle or 0

                # Coaxial + similar angle + adjacent
                if (self._axes_parallel(axis1, axis2) and
                    abs(angle1 - angle2) < self.angle_tolerance and
                    abs(z1 - z2) < self.position_tolerance):
                    
                    merge_group.append(t2)
                    used.add(j)

            if len(merge_group) > 1:
                merged.append(self._merge_group(merge_group))
            else:
                merged.append(t1)

        if len(merged) == len(tapers):
            logger.info(f"   ℹ️  Tapers: {len(tapers)} (no matches found)")

        return merged

    def _merge_steps(self, steps: List) -> List:
        """
        Merge steps at same radial position (diameter).
        """
        if len(steps) <= 1:
            return steps

        logger.info(f"   Processing {len(steps)} steps for merging...")

        merged = []
        used = set()

        for i, s1 in enumerate(steps):
            if i in used:
                continue

            merge_group = [s1]
            axis1 = np.array(s1.axis)
            loc1 = np.array(s1.location)
            z1 = np.dot(loc1, axis1)

            for j in range(i + 1, len(steps)):
                if j in used:
                    continue

                s2 = steps[j]
                axis2 = np.array(s2.axis)
                loc2 = np.array(s2.location)
                z2 = np.dot(loc2, axis2)

                # Same diameter + coaxial + adjacent/overlapping
                if (self._diameters_similar(s1.diameter, s2.diameter) and
                    self._axes_parallel(axis1, axis2) and
                    abs(z1 - z2) < self.position_tolerance):
                    
                    merge_group.append(s2)
                    used.add(j)

            if len(merge_group) > 1:
                merged.append(self._merge_group(merge_group))
            else:
                merged.append(s1)

        if len(merged) == len(steps):
            logger.info(f"   ℹ️  Steps: {len(steps)} (no matches found)")

        return merged

    def _merge_bases(self, bases: List) -> List:
        """
        Merge coaxial base cylinders.
        """
        if len(bases) <= 1:
            return bases

        merged = []
        used = set()

        for i, b1 in enumerate(bases):
            if i in used:
                continue

            merge_group = [b1]
            axis1 = np.array(b1.axis)

            for j in range(i + 1, len(bases)):
                if j in used:
                    continue

                b2 = bases[j]
                axis2 = np.array(b2.axis)

                # Coaxial + similar diameter
                if (self._axes_parallel(axis1, axis2) and
                    self._diameters_similar(b1.diameter, b2.diameter)):
                    
                    merge_group.append(b2)
                    used.add(j)

            if len(merge_group) > 1:
                merged.append(self._merge_group(merge_group))
            else:
                merged.append(b1)

        return merged

    def _merge_group(self, features: List):
        """
        Merge a group of features into a single feature.
        Combines face indices and takes average dimensions.
        """
        if len(features) == 1:
            return features[0]

        # Use first feature as base
        merged = features[0]

        # Merge face indices
        all_face_indices = []
        for f in features:
            all_face_indices.extend(f.face_indices)
        merged.face_indices = list(set(all_face_indices))  # Remove duplicates

        # Average dimensions
        merged.length = sum(f.length for f in features) / len(features)

        # For grooves, sum widths (if they're segments of same groove)
        if merged.feature_type.value == 'groove':
            merged.groove_width = sum(f.groove_width or 0 for f in features)

        return merged

    def _axes_parallel(self, axis1: np.ndarray, axis2: np.ndarray) -> bool:
        """Check if two axes are parallel"""
        axis1_norm = axis1 / np.linalg.norm(axis1)
        axis2_norm = axis2 / np.linalg.norm(axis2)
        dot = abs(np.dot(axis1_norm, axis2_norm))
        return dot > 0.97  # ~15 degrees tolerance - RELAXED from 0.99 (~8 degrees)

    def _diameters_similar(self, d1: float, d2: float) -> bool:
        """Check if two diameters are similar"""
        return abs(d1 - d2) < self.diameter_tolerance


def merge_turning_features(features: List,
                           axis_tol: float = 1.0,
                           position_tol: float = 10.0) -> List:
    """
    Convenience function for semantic merging.
    
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


if __name__ == "__main__":
    # Example usage / testing
    print("✅ Turning Feature Merger v2.1 loaded")
    print("   V-groove merging: ENABLED")
    print("   Same Z-position detection: ENABLED")
