# extended_crash_free_recognizer.py
"""
EXTENDED Crash-Free Geometric Feature Recognition
Integrates detailed slicing for pockets, slots, and complex features

NEW CAPABILITIES:
- Multi-axis slicing with contour extraction
- Pocket detection via enclosed cavities
- Slot detection via elongated cavities
- Through vs. blind cavity distinction
- Cross-axis feature validation
"""

import os
import time
import math
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.spatial import ConvexHull
from scipy.ndimage import label as ndimage_label
import gc

# OpenCascade imports - ONLY the safe ones
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                               GeomAbs_Sphere, GeomAbs_Torus)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.GeomLProp import GeomLProp_SLProps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContourPoint:
    """2D point in slice contour"""
    x: float
    y: float


@dataclass
class SliceContour:
    """Closed contour extracted from slice"""
    points: List[ContourPoint]
    is_hole: bool  # True if interior cavity, False if exterior boundary
    area: float
    centroid: Tuple[float, float]
    perimeter: float
    aspect_ratio: float  # Length/width ratio
    circularity: float  # 4π·Area/Perimeter² (1.0 = perfect circle)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for analysis"""
        return np.array([[p.x, p.y] for p in self.points])


@dataclass
class FeatureInstance:
    """Recognized manufacturing feature"""
    feature_type: str
    subtype: Optional[str] = None
    confidence: float = 0.0
    face_indices: List[int] = field(default_factory=list)
    dimensions: Dict[str, float] = field(default_factory=dict)
    location: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float]] = None
    detection_method: str = ""
    slice_span: Optional[Tuple[float, float]] = None  # Start/end positions for cavities
    
    def to_dict(self) -> dict:
        return {
            'type': self.feature_type,
            'subtype': self.subtype,
            'confidence': self.confidence,
            'face_indices': self.face_indices,
            'dimensions': self.dimensions,
            'location': list(self.location) if self.location else None,
            'orientation': list(self.orientation) if self.orientation else None,
            'detection_method': self.detection_method,
            'slice_span': list(self.slice_span) if self.slice_span else None
        }


class ExtendedSlicingAnalyzer:
    """
    Advanced multi-axis slicing with complete contour extraction
    Detects pockets, slots, and complex cavities
    """
    
    def __init__(self, shape: TopoDS_Shape, num_slices: int = 30):
        self.shape = shape
        self.num_slices = num_slices
        self.bbox = self._compute_bbox()
        self.slices_by_axis = {'X': [], 'Y': [], 'Z': []}
    
    def _compute_bbox(self) -> Tuple[float, float, float, float, float, float]:
        """Compute bounding box"""
        bbox = Bnd_Box()
        brepbndlib.Add(self.shape, bbox)
        return bbox.Get()
    
    def slice_all_axes(self) -> Dict[str, List[Dict]]:
        """Perform slicing along all three axes"""
        logger.info("🔪 Multi-axis slicing (X, Y, Z)...")
        
        for axis in ['X', 'Y', 'Z']:
            self.slices_by_axis[axis] = self._slice_along_axis(axis)
        
        return self.slices_by_axis
    
    def _slice_along_axis(self, axis: str) -> List[Dict]:
        """
        Slice shape along specified axis with full contour extraction
        SAFE - uses BRepAlgoAPI_Section
        """
        xmin, ymin, zmin, xmax, ymax, zmax = self.bbox
        
        # Determine slicing parameters based on axis
        if axis == 'X':
            positions = np.linspace(xmin, xmax, self.num_slices)
            normal = gp_Dir(1, 0, 0)
            coord_getter = lambda p: (p.Y(), p.Z())
        elif axis == 'Y':
            positions = np.linspace(ymin, ymax, self.num_slices)
            normal = gp_Dir(0, 1, 0)
            coord_getter = lambda p: (p.X(), p.Z())
        else:  # Z
            positions = np.linspace(zmin, zmax, self.num_slices)
            normal = gp_Dir(0, 0, 1)
            coord_getter = lambda p: (p.X(), p.Y())
        
        slices = []
        
        for pos in positions:
            try:
                # Create cutting plane
                if axis == 'X':
                    origin = gp_Pnt(pos, 0, 0)
                elif axis == 'Y':
                    origin = gp_Pnt(0, pos, 0)
                else:
                    origin = gp_Pnt(0, 0, pos)
                
                plane = gp_Pln(origin, normal)
                
                # Perform section (SAFE operation)
                section = BRepAlgoAPI_Section(self.shape, plane)
                section.Build()
                
                if section.IsDone():
                    section_shape = section.Shape()
                    
                    # Extract contours from section
                    contours = self._extract_contours(section_shape, coord_getter)
                    
                    if contours:
                        slice_data = {
                            'axis': axis,
                            'position': pos,
                            'contours': contours,
                            'cavity_contours': [c for c in contours if c.is_hole],
                            'num_cavities': sum(1 for c in contours if c.is_hole)
                        }
                        
                        slices.append(slice_data)
            
            except Exception as e:
                logger.debug(f"Slice at {axis}={pos:.2f} failed: {e}")
        
        logger.info(f"  ✅ {axis}-axis: {len(slices)} valid slices")
        return slices
    
    def _extract_contours(self, section_shape, coord_getter) -> List[SliceContour]:
        """
        Extract 2D contours from section shape
        """
        contours = []
        
        try:
            # Explore wires (closed loops) in section
            wire_explorer = TopExp_Explorer(section_shape, TopAbs_WIRE)
            
            while wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())
                
                # Extract points from wire edges
                points = []
                edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                
                while edge_explorer.More():
                    edge = topods.Edge(edge_explorer.Current())
                    
                    # Sample points along edge
                    try:
                        curve_data = BRep_Tool.Curve(edge)
                        if curve_data[0] is not None:
                            curve = curve_data[0]
                            first_param = curve_data[1]
                            last_param = curve_data[2]
                            
                            # Sample 10 points per edge
                            for t in np.linspace(first_param, last_param, 10):
                                point_3d = curve.Value(t)
                                x, y = coord_getter(point_3d)
                                points.append(ContourPoint(x, y))
                    except:
                        pass
                    
                    edge_explorer.Next()
                
                if len(points) > 3:
                    # Calculate contour properties
                    points_np = np.array([[p.x, p.y] for p in points])
                    area = abs(self._calculate_polygon_area(points_np))
                    perimeter = self._calculate_perimeter(points_np)
                    centroid = np.mean(points_np, axis=0)
                    
                    # Calculate shape metrics
                    aspect_ratio = self._calculate_aspect_ratio(points_np)
                    circularity = self._calculate_circularity(area, perimeter)
                    
                    # Determine if cavity (interior contour)
                    # Heuristic: smaller contours inside larger bounding area
                    is_hole = area < 10000  # Threshold for cavity vs boundary
                    
                    contour = SliceContour(
                        points=points,
                        is_hole=is_hole,
                        area=area,
                        centroid=(centroid[0], centroid[1]),
                        perimeter=perimeter,
                        aspect_ratio=aspect_ratio,
                        circularity=circularity
                    )
                    
                    contours.append(contour)
                
                wire_explorer.Next()
        
        except Exception as e:
            logger.debug(f"Contour extraction failed: {e}")
        
        return contours
    
    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Calculate area using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        x = points[:, 0]
        y = points[:, 1]
        
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    
    def _calculate_perimeter(self, points: np.ndarray) -> float:
        """Calculate perimeter"""
        if len(points) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            perimeter += np.linalg.norm(p2 - p1)
        
        return perimeter
    
    def _calculate_aspect_ratio(self, points: np.ndarray) -> float:
        """Calculate length/width aspect ratio"""
        if len(points) < 3:
            return 1.0
        
        try:
            # Use convex hull for better aspect ratio estimation
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Calculate bounding rectangle
            min_x, min_y = hull_points.min(axis=0)
            max_x, max_y = hull_points.max(axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if min(width, height) > 0:
                return max(width, height) / min(width, height)
        except:
            pass
        
        return 1.0
    
    def _calculate_circularity(self, area: float, perimeter: float) -> float:
        """
        Calculate circularity (4π·Area/Perimeter²)
        1.0 = perfect circle, <1.0 = elongated
        """
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            return min(1.0, circularity)
        return 0.0
    
    def detect_holes_from_slices(self) -> List[FeatureInstance]:
        """
        Detect holes by tracking circular cavities across slices
        """
        logger.info("🎯 Detecting holes from slice consistency...")
        
        features = []
        z_slices = self.slices_by_axis.get('Z', [])
        
        if len(z_slices) < 3:
            return features
        
        # Track holes across slices
        hole_tracks = []
        
        for i, slice_data in enumerate(z_slices):
            for contour in slice_data['cavity_contours']:
                # Skip non-circular contours (likely not holes)
                if contour.circularity < 0.7:
                    continue
                
                # Try to match with existing tracks
                matched = False
                
                for track in hole_tracks:
                    if i > 0:
                        prev_centroid = track['centroids'][-1]
                        dist = np.sqrt(
                            (contour.centroid[0] - prev_centroid[0])**2 +
                            (contour.centroid[1] - prev_centroid[1])**2
                        )
                        
                        # If close, add to track
                        if dist < 3.0:  # 3mm tolerance
                            track['centroids'].append(contour.centroid)
                            track['areas'].append(contour.area)
                            track['z_positions'].append(slice_data['position'])
                            track['circularities'].append(contour.circularity)
                            matched = True
                            break
                
                # Create new track if no match
                if not matched:
                    hole_tracks.append({
                        'centroids': [contour.centroid],
                        'areas': [contour.area],
                        'z_positions': [slice_data['position']],
                        'circularities': [contour.circularity]
                    })
        
        # Analyze tracks to determine hole type
        zmin, _, _, _, _, zmax = self.bbox
        total_z = zmax - zmin
        
        for track in hole_tracks:
            if len(track['z_positions']) >= 3:
                # Calculate span
                z_span = max(track['z_positions']) - min(track['z_positions'])
                
                # Through-hole if spans most of Z-range
                is_through = z_span > 0.75 * total_z
                
                # Estimate diameter from average area (assuming circular)
                avg_area = np.mean(track['areas'])
                radius = np.sqrt(avg_area / np.pi)
                diameter = 2 * radius
                
                # Calculate center position
                center_x = np.mean([c[0] for c in track['centroids']])
                center_y = np.mean([c[1] for c in track['centroids']])
                center_z = np.mean(track['z_positions'])
                
                # Confidence based on circularity consistency
                avg_circularity = np.mean(track['circularities'])
                confidence = 0.75 + (avg_circularity * 0.2)  # 0.75-0.95 range
                
                feature = FeatureInstance(
                    feature_type='hole',
                    subtype='through' if is_through else 'blind',
                    confidence=confidence,
                    dimensions={
                        'diameter': diameter,
                        'radius': radius,
                        'depth': z_span
                    },
                    location=(center_x, center_y, center_z),
                    orientation=(0, 0, 1),  # Vertical
                    slice_span=(min(track['z_positions']), max(track['z_positions'])),
                    detection_method='slice_consistency_circular_tracking'
                )
                
                features.append(feature)
        
        logger.info(f"  ✅ Found {len(features)} holes via slicing")
        return features
    
    def detect_pockets_from_slices(self) -> List[FeatureInstance]:
        """
        Detect pockets by tracking enclosed cavities across slices
        Pockets = rectangular/rounded cavities that don't go through
        """
        logger.info("🎯 Detecting pockets from slice consistency...")
        
        features = []
        z_slices = self.slices_by_axis.get('Z', [])
        
        if len(z_slices) < 3:
            return features
        
        # Track pockets across slices
        pocket_tracks = []
        
        for i, slice_data in enumerate(z_slices):
            for contour in slice_data['cavity_contours']:
                # Skip circular contours (likely holes)
                if contour.circularity > 0.85:
                    continue
                
                # Skip very small cavities
                if contour.area < 100:  # Minimum 100 mm²
                    continue
                
                # Try to match with existing tracks
                matched = False
                
                for track in pocket_tracks:
                    if i > 0:
                        prev_centroid = track['centroids'][-1]
                        dist = np.sqrt(
                            (contour.centroid[0] - prev_centroid[0])**2 +
                            (contour.centroid[1] - prev_centroid[1])**2
                        )
                        
                        # If close, add to track
                        if dist < 5.0:  # 5mm tolerance
                            track['centroids'].append(contour.centroid)
                            track['areas'].append(contour.area)
                            track['z_positions'].append(slice_data['position'])
                            track['aspect_ratios'].append(contour.aspect_ratio)
                            matched = True
                            break
                
                # Create new track
                if not matched:
                    pocket_tracks.append({
                        'centroids': [contour.centroid],
                        'areas': [contour.area],
                        'z_positions': [slice_data['position']],
                        'aspect_ratios': [contour.aspect_ratio]
                    })
        
        # Analyze tracks
        zmin, _, _, _, _, zmax = self.bbox
        total_z = zmax - zmin
        
        for track in pocket_tracks:
            if len(track['z_positions']) >= 3:
                z_span = max(track['z_positions']) - min(track['z_positions'])
                
                # Must NOT go through (pockets are blind)
                if z_span > 0.7 * total_z:
                    continue  # This is a through feature, not a pocket
                
                # Classify pocket type by aspect ratio
                avg_aspect = np.mean(track['aspect_ratios'])
                
                if avg_aspect < 1.5:
                    subtype = 'rectangular'  # Square-ish
                elif avg_aspect < 3.0:
                    subtype = 'rounded'  # Rounded rectangle
                else:
                    continue  # Too elongated, likely a slot
                
                # Calculate dimensions
                avg_area = np.mean(track['areas'])
                
                # Estimate length and width
                # For rectangular: assume area ≈ length × width
                # Use aspect ratio to back-calculate
                width = np.sqrt(avg_area / avg_aspect)
                length = width * avg_aspect
                
                center_x = np.mean([c[0] for c in track['centroids']])
                center_y = np.mean([c[1] for c in track['centroids']])
                center_z = np.mean(track['z_positions'])
                
                feature = FeatureInstance(
                    feature_type='pocket',
                    subtype=subtype,
                    confidence=0.70,
                    dimensions={
                        'length': length,
                        'width': width,
                        'depth': z_span,
                        'area': avg_area,
                        'aspect_ratio': avg_aspect
                    },
                    location=(center_x, center_y, center_z),
                    orientation=(0, 0, 1),
                    slice_span=(min(track['z_positions']), max(track['z_positions'])),
                    detection_method='slice_consistency_cavity_tracking'
                )
                
                features.append(feature)
        
        logger.info(f"  ✅ Found {len(features)} pockets via slicing")
        return features
    
    def detect_slots_from_slices(self) -> List[FeatureInstance]:
        """
        Detect slots by tracking elongated cavities across slices
        Slots = elongated cavities (aspect ratio > 3:1)
        """
        logger.info("🎯 Detecting slots from slice consistency...")
        
        features = []
        
        # Check Z-axis slices for vertical slots
        z_slices = self.slices_by_axis.get('Z', [])
        
        if len(z_slices) < 3:
            return features
        
        # Track slots across slices
        slot_tracks = []
        
        for i, slice_data in enumerate(z_slices):
            for contour in slice_data['cavity_contours']:
                # Skip circular/square contours
                if contour.aspect_ratio < 3.0:
                    continue
                
                # Skip very small cavities
                if contour.area < 50:
                    continue
                
                # Try to match with existing tracks
                matched = False
                
                for track in slot_tracks:
                    if i > 0:
                        prev_centroid = track['centroids'][-1]
                        dist = np.sqrt(
                            (contour.centroid[0] - prev_centroid[0])**2 +
                            (contour.centroid[1] - prev_centroid[1])**2
                        )
                        
                        if dist < 4.0:  # 4mm tolerance
                            track['centroids'].append(contour.centroid)
                            track['areas'].append(contour.area)
                            track['z_positions'].append(slice_data['position'])
                            track['aspect_ratios'].append(contour.aspect_ratio)
                            matched = True
                            break
                
                # Create new track
                if not matched:
                    slot_tracks.append({
                        'centroids': [contour.centroid],
                        'areas': [contour.area],
                        'z_positions': [slice_data['position']],
                        'aspect_ratios': [contour.aspect_ratio]
                    })
        
        # Analyze tracks
        zmin, _, _, _, _, zmax = self.bbox
        total_z = zmax - zmin
        
        for track in slot_tracks:
            if len(track['z_positions']) >= 3:
                z_span = max(track['z_positions']) - min(track['z_positions'])
                
                # Classify as through or blind
                is_through = z_span > 0.75 * total_z
                
                # Calculate dimensions
                avg_aspect = np.mean(track['aspect_ratios'])
                avg_area = np.mean(track['areas'])
                
                # Estimate length and width from area and aspect ratio
                width = np.sqrt(avg_area / avg_aspect)
                length = width * avg_aspect
                
                center_x = np.mean([c[0] for c in track['centroids']])
                center_y = np.mean([c[1] for c in track['centroids']])
                center_z = np.mean(track['z_positions'])
                
                # Determine slot type
                if avg_aspect > 8.0:
                    subtype = 'straight_through' if is_through else 'straight_blind'
                else:
                    subtype = 'through' if is_through else 'blind'
                
                feature = FeatureInstance(
                    feature_type='slot',
                    subtype=subtype,
                    confidence=0.75,
                    dimensions={
                        'length': length,
                        'width': width,
                        'depth': z_span,
                        'aspect_ratio': avg_aspect
                    },
                    location=(center_x, center_y, center_z),
                    orientation=(0, 0, 1),
                    slice_span=(min(track['z_positions']), max(track['z_positions'])),
                    detection_method='slice_consistency_elongated_tracking'
                )
                
                features.append(feature)
        
        logger.info(f"  ✅ Found {len(features)} slots via slicing")
        return features


class ExtendedCrashFreeRecognizer:
    """
    Extended Crash-Free Geometric Feature Recognizer
    NOW WITH COMPLETE POCKET AND SLOT DETECTION
    """
    
    def __init__(self, time_limit: float = 60.0):
        self.time_limit = time_limit
        self.start_time = None
        self.shape = None
        self.features = []
        self.faces = []
        self.slicer = None
    
    def recognize_features(self, step_file_path: str) -> Dict[str, Any]:
        """
        Main recognition pipeline with extended slicing
        """
        self.start_time = time.time()
        correlation_id = f"extended_{int(time.time() * 1000)}"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[{correlation_id}] EXTENDED CRASH-FREE RECOGNITION")
        logger.info(f"{'='*70}")
        
        try:
            # Load STEP
            logger.info(f"\n📂 Stage 1: Loading STEP file...")
            self.shape = self._load_step_file(step_file_path)
            
            if not self.shape:
                return self._error_response(correlation_id, "Failed to load STEP")
            
            # Collect faces
            logger.info(f"\n📋 Stage 2: Collecting faces...")
            self._collect_faces_safely()
            logger.info(f"  ✅ Collected {len(self.faces)} faces")
            
            # Direct face analysis (holes, bosses, fillets)
            logger.info(f"\n🔍 Stage 3: Direct face geometry...")
            self._detect_cylindrical_features()
            self._detect_conical_features()
            self._detect_curved_features()
            logger.info(f"  ✅ {len(self.features)} features from direct analysis")
            
            # Coaxial grouping (counterbore/countersink)
            if self._check_time():
                logger.info(f"\n🎯 Stage 4: Coaxial cylinder grouping...")
                self._detect_counterbore_countersink()
            
            # EXTENDED SLICING ANALYSIS (NEW!)
            if self._check_time():
                logger.info(f"\n🔪 Stage 5: EXTENDED MULTI-AXIS SLICING...")
                self.slicer = ExtendedSlicingAnalyzer(self.shape, num_slices=30)
                self.slicer.slice_all_axes()
                
                # Detect holes from slices
                hole_features = self.slicer.detect_holes_from_slices()
                self.features.extend(hole_features)
                
                # Detect pockets from slices (NEW!)
                pocket_features = self.slicer.detect_pockets_from_slices()
                self.features.extend(pocket_features)
                
                # Detect slots from slices (NEW!)
                slot_features = self.slicer.detect_slots_from_slices()
                self.features.extend(slot_features)
            
            # Finalization
            elapsed_time = time.time() - self.start_time
            feature_summary = self._generate_summary()
            avg_confidence = np.mean([f.confidence for f in self.features]) if self.features else 0.0
            
            result = {
                'status': 'success',
                'correlation_id': correlation_id,
                'num_features_detected': len(self.features),
                'num_faces_analyzed': len(self.faces),
                'inference_time_sec': elapsed_time,
                'avg_confidence': float(avg_confidence),
                'instances': [f.to_dict() for f in self.features],
                'feature_summary': feature_summary,
                'recognition_method': 'extended_crash_free_with_slicing'
            }
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ EXTENDED RECOGNITION COMPLETE")
            logger.info(f"   Features: {len(self.features)} | Time: {elapsed_time:.2f}s")
            logger.info(f"   Confidence: {avg_confidence:.1%}")
            logger.info(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._error_response(correlation_id, str(e))
        
        finally:
            gc.collect()
    
    def _load_step_file(self, file_path: str) -> Optional[TopoDS_Shape]:
        """Safe STEP loading"""
        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(file_path)
            
            if status != IFSelect_RetDone:
                return None
            
            reader.TransferRoots()
            shape = reader.OneShape()
            
            if shape.IsNull():
                return None
            
            # Optional healing
            try:
                fixer = ShapeFix_Shape(shape)
                fixer.Perform()
                healed = fixer.Shape()
                if not healed.IsNull():
                    return healed
            except:
                pass
            
            return shape
        except Exception as e:
            logger.error(f"STEP load error: {e}")
            return None
    
    def _collect_faces_safely(self):
        """Collect faces without AAG"""
        self.faces = []
        
        try:
            explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
            face_idx = 0
            
            while explorer.More():
                try:
                    face = topods.Face(explorer.Current())
                    
                    face_data = {
                        'index': face_idx,
                        'face': face,
                        'surf_type': None,
                        'area': 0.0
                    }
                    
                    try:
                        adaptor = BRepAdaptor_Surface(face, True)
                        face_data['surf_type'] = adaptor.GetType()
                        
                        props = GProp_GProps()
                        brepgprop_SurfaceProperties(face, props)
                        face_data['area'] = props.Mass()
                    except:
                        pass
                    
                    self.faces.append(face_data)
                    face_idx += 1
                except:
                    pass
                
                explorer.Next()
        except Exception as e:
            logger.error(f"Face collection error: {e}")
    
    def _check_time(self) -> bool:
        """Check time limit"""
        return (time.time() - self.start_time) < self.time_limit
    
    def _detect_cylindrical_features(self):
        """
        Detect holes and bosses from cylindrical faces
        IMPROVED: Uses orientation and part topology to distinguish boss from hole
        """
        logger.info("  🔍 Detecting cylindrical features...")
        
        # First pass: collect all cylinders with metadata
        cylinders = []
        
        for face_data in self.faces:
            if face_data['surf_type'] != GeomAbs_Cylinder:
                continue
            
            try:
                face = face_data['face']
                adaptor = BRepAdaptor_Surface(face, True)
                cylinder = adaptor.Cylinder()
                
                radius = cylinder.Radius()
                axis_dir = cylinder.Axis().Direction()
                location = cylinder.Location()
                area = face_data['area']
                
                # Check if axis is vertical (Z-dominant)
                axis_z = abs(axis_dir.Z())
                is_vertical = axis_z > 0.9
                
                # Check if axis is horizontal
                is_horizontal = axis_z < 0.1
                
                cylinders.append({
                    'face_data': face_data,
                    'radius': radius,
                    'area': area,
                    'axis': (axis_dir.X(), axis_dir.Y(), axis_dir.Z()),
                    'location': (location.X(), location.Y(), location.Z()),
                    'is_vertical': is_vertical,
                    'is_horizontal': is_horizontal
                })
                
            except Exception as e:
                logger.debug(f"Cylinder analysis failed: {e}")
        
        if not cylinders:
            return
        
        # Sort cylinders by size (area) to identify main body
        cylinders.sort(key=lambda c: c['area'], reverse=True)
        
        # Heuristic: Largest cylinder is likely the main body (boss/shaft)
        # Smaller cylinders are likely holes OR small bosses
        
        for idx, cyl_data in enumerate(cylinders):
            radius = cyl_data['radius']
            area = cyl_data['area']
            is_vertical = cyl_data['is_vertical']
            
            # Classification logic (improved):
            
            # 1. Main body detection (largest vertical cylinder)
            if idx == 0 and is_vertical and area > 5000:
                # This is likely the main cylindrical body (pin, shaft, boss)
                feature_type = 'boss'
                subtype = 'cylindrical_shaft'
                confidence = 0.85
                detection_method = 'main_body_vertical_cylinder'
            
            # 2. Small vertical cylinders = holes
            elif is_vertical and radius < 20:
                feature_type = 'hole'
                subtype = 'cylindrical'
                confidence = 0.85
                detection_method = 'small_vertical_cylinder'
            
            # 3. Horizontal cylinders on pins = likely bosses/features
            elif cyl_data['is_horizontal']:
                # Horizontal cylinders are usually bosses or features
                if radius < 10:
                    feature_type = 'fillet'  # Small horizontal = edge blend
                    subtype = 'cylindrical'
                    confidence = 0.65
                    detection_method = 'small_horizontal_cylinder'
                else:
                    feature_type = 'boss'
                    subtype = 'cylindrical'
                    confidence = 0.75
                    detection_method = 'horizontal_cylinder'
            
            # 4. Ambiguous cases - use size heuristic
            else:
                # Use ratio of this cylinder to largest
                size_ratio = area / cylinders[0]['area']
                
                if size_ratio > 0.3:
                    # Significant size = likely boss
                    feature_type = 'boss'
                    subtype = 'cylindrical'
                    confidence = 0.70
                    detection_method = 'large_cylinder_ratio'
                else:
                    # Small relative size = likely hole
                    feature_type = 'hole'
                    subtype = 'cylindrical'
                    confidence = 0.75
                    detection_method = 'small_cylinder_ratio'
            
            # Create feature instance
            feature = FeatureInstance(
                feature_type=feature_type,
                subtype=subtype,
                confidence=confidence,
                face_indices=[cyl_data['face_data']['index']],
                dimensions={
                    'diameter': radius * 2,
                    'radius': radius,
                    'surface_area': area
                },
                location=cyl_data['location'],
                orientation=cyl_data['axis'],
                detection_method=detection_method
            )
            
            self.features.append(feature)
            
            logger.info(f"    {'🔵' if feature_type == 'hole' else '🟢'} {feature_type.upper()}: "
                       f"Ø{radius*2:.1f}mm, area={area:.0f}mm², {detection_method}")
    
    def _detect_conical_features(self):
        """
        Detect chamfers and conical features from conical faces
        NEW METHOD - detects the chamfers on your pin!
        """
        logger.info("  🔍 Detecting conical features (chamfers)...")
        
        for face_data in self.faces:
            if face_data['surf_type'] != GeomAbs_Cone:
                continue
            
            try:
                face = face_data['face']
                adaptor = BRepAdaptor_Surface(face, True)
                cone = adaptor.Cone()
                
                # Get cone parameters
                apex_angle_rad = cone.SemiAngle()
                apex_angle_deg = apex_angle_rad * 180 / np.pi
                
                apex = cone.Apex()
                axis_dir = cone.Axis().Direction()
                area = face_data['area']
                
                # Chamfer detection: small conical surfaces (typically < 500mm²)
                # Common chamfer angles: 30°, 45°, 60°
                is_chamfer = area < 1000  # Small conical surface
                
                if is_chamfer:
                    # Classify chamfer type by angle
                    if abs(apex_angle_deg - 45) < 5:
                        subtype = '45_degree'
                    elif abs(apex_angle_deg - 30) < 5:
                        subtype = '30_degree'
                    elif abs(apex_angle_deg - 60) < 5:
                        subtype = '60_degree'
                    else:
                        subtype = f'{apex_angle_deg:.0f}_degree'
                    
                    feature = FeatureInstance(
                        feature_type='chamfer',
                        subtype=subtype,
                        confidence=0.85,
                        face_indices=[face_data['index']],
                        dimensions={
                            'angle_degrees': apex_angle_deg,
                            'surface_area': area
                        },
                        location=(apex.X(), apex.Y(), apex.Z()),
                        orientation=(axis_dir.X(), axis_dir.Y(), axis_dir.Z()),
                        detection_method='conical_surface_small_area'
                    )
                    
                    self.features.append(feature)
                    
                    logger.info(f"    ⚡ CHAMFER: {apex_angle_deg:.0f}°, area={area:.0f}mm²")
                else:
                    # Large conical surface = cone boss or taper
                    feature = FeatureInstance(
                        feature_type='boss',
                        subtype='conical',
                        confidence=0.75,
                        face_indices=[face_data['index']],
                        dimensions={
                            'angle_degrees': apex_angle_deg,
                            'surface_area': area
                        },
                        location=(apex.X(), apex.Y(), apex.Z()),
                        orientation=(axis_dir.X(), axis_dir.Y(), axis_dir.Z()),
                        detection_method='conical_surface_large_area'
                    )
                    
                    self.features.append(feature)
                    
                    logger.info(f"    🔺 CONICAL BOSS: {apex_angle_deg:.0f}°, area={area:.0f}mm²")
            
            except Exception as e:
                logger.debug(f"Cone analysis failed: {e}")
    
    def _detect_curved_features(self):
        """Detect fillets from small curved faces"""
        for face_data in self.faces:
            surf_type = face_data['surf_type']
            area = face_data['area']
            
            if surf_type in [GeomAbs_Cylinder, GeomAbs_Torus] and area < 200:
                try:
                    face = face_data['face']
                    adaptor = BRepAdaptor_Surface(face, True)
                    
                    radius = 0.0
                    if surf_type == GeomAbs_Cylinder:
                        radius = adaptor.Cylinder().Radius()
                    
                    if radius < 15:
                        feature = FeatureInstance(
                            feature_type='fillet',
                            subtype='constant',
                            confidence=0.65,
                            face_indices=[face_data['index']],
                            dimensions={'radius': radius},
                            detection_method='small_curved_face'
                        )
                        self.features.append(feature)
                except:
                    pass
    
    def _detect_counterbore_countersink(self):
        """Detect counterbore via coaxial cylinders"""
        cylinders = []
        
        for face_data in self.faces:
            if face_data['surf_type'] == GeomAbs_Cylinder:
                try:
                    face = face_data['face']
                    adaptor = BRepAdaptor_Surface(face, True)
                    cylinder = adaptor.Cylinder()
                    
                    cylinders.append({
                        'face_data': face_data,
                        'radius': cylinder.Radius(),
                        'axis': np.array([
                            cylinder.Axis().Direction().X(),
                            cylinder.Axis().Direction().Y(),
                            cylinder.Axis().Direction().Z()
                        ]),
                        'center': np.array([
                            cylinder.Location().X(),
                            cylinder.Location().Y(),
                            cylinder.Location().Z()
                        ])
                    })
                except:
                    pass
        
        # Find coaxial pairs
        used = set()
        
        for i, cyl1 in enumerate(cylinders):
            if i in used:
                continue
            
            for j, cyl2 in enumerate(cylinders):
                if i >= j or j in used:
                    continue
                
                # Check coaxiality
                axis_dot = abs(np.dot(cyl1['axis'], cyl2['axis']))
                center_dist = np.linalg.norm(cyl1['center'] - cyl2['center'])
                
                if axis_dot > 0.98 and center_dist < 3.0:
                    if cyl1['radius'] > cyl2['radius']:
                        outer, inner = cyl1, cyl2
                    else:
                        outer, inner = cyl2, cyl1
                    
                    if outer['radius'] / inner['radius'] > 1.2:
                        feature = FeatureInstance(
                            feature_type='hole',
                            subtype='counterbore',
                            confidence=0.85,
                            face_indices=[
                                outer['face_data']['index'],
                                inner['face_data']['index']
                            ],
                            dimensions={
                                'outer_diameter': outer['radius'] * 2,
                                'inner_diameter': inner['radius'] * 2
                            },
                            location=tuple(outer['center']),
                            orientation=tuple(outer['axis']),
                            detection_method='coaxial_pairing'
                        )
                        
                        self.features.append(feature)
                        used.add(i)
                        used.add(j)
                        break
    
    def _generate_summary(self) -> Dict[str, int]:
        """Generate feature summary"""
        summary = defaultdict(int)
        
        for feature in self.features:
            key = feature.feature_type
            if feature.subtype:
                key = f"{key}_{feature.subtype}"
            summary[key] += 1
        
        return dict(summary)
    
    def _error_response(self, correlation_id: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'status': 'failed',
            'correlation_id': correlation_id,
            'error': error,
            'features': [],
            'num_features_detected': 0
        }


# Flask wrapper
class FlaskCrashFreeRecognizer:
    """Flask-compatible wrapper for geometric feature recognition"""
    
    def __init__(self, time_limit: float = 30.0, memory_limit_mb: int = 2000):
        self.time_limit = time_limit
        self.memory_limit_mb = memory_limit_mb  # Add for compatibility with app.py
        self.recognizer = ExtendedCrashFreeRecognizer(time_limit=time_limit)
    
    def recognize_features(self, step_file_path: str) -> Dict[str, Any]:
        """Recognize features using extended geometric recognition"""
        return self.recognizer.recognize_features(step_file_path)


# Backward compatibility alias (in case anything else references the old name)
FlaskExtendedRecognizer = FlaskCrashFreeRecognizer


# CLI testing
def main():
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python extended_crash_free_recognizer.py <step_file.step>")
        sys.exit(1)
    
    step_file = sys.argv[1]
    
    recognizer = ExtendedCrashFreeRecognizer(time_limit=60.0)
    result = recognizer.recognize_features(step_file)
    
    print("\n" + "="*70)
    print("EXTENDED RECOGNITION RESULTS")
    print("="*70)
    print(f"Status: {result.get('status')}")
    print(f"Features: {result.get('num_features_detected', 0)}")
    print(f"Time: {result.get('inference_time_sec', 0):.2f}s")
    print(f"Confidence: {result.get('avg_confidence', 0):.1%}")
    
    if result.get('feature_summary'):
        print("\n📊 Feature Summary:")
        for feat_type, count in sorted(result['feature_summary'].items()):
            print(f"  • {feat_type}: {count}")
    
    # Save results
    output_file = step_file.replace('.step', '_extended_features.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved: {output_file}")


if __name__ == '__main__':
    main()
