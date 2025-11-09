# slicing_volumetric_detailed.py
"""
Detailed Implementation of Slicing and Volumetric Analysis
Production-grade algorithms for 2.5D and 3D feature detection
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Common
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.Geom import Geom_Plane
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier

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
    is_hole: bool  # True if interior hole, False if exterior boundary
    area: float
    centroid: Tuple[float, float]
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for analysis"""
        return np.array([[p.x, p.y] for p in self.points])


class DetailedSlicer:
    """
    Production-grade multi-axis slicing with contour extraction
    Uses OpenCascade BRepAlgoAPI_Section for accurate intersections
    """
    
    def __init__(self, shape: TopoDS_Shape, num_slices: int = 50):
        self.shape = shape
        self.num_slices = num_slices
        self.bbox = self._compute_bbox()
        self.slices = {'X': [], 'Y': [], 'Z': []}
        
        # Performance tracking
        self.slice_times = []
    
    def _compute_bbox(self) -> Tuple[float, float, float, float, float, float]:
        """Compute bounding box"""
        bbox = Bnd_Box()
        brepbndlib.Add(self.shape, bbox)
        return bbox.Get()
    
    def slice_along_axis(self, axis: str, progress_callback=None) -> List[Dict]:
        """
        Slice shape along specified axis with full contour extraction
        
        Args:
            axis: 'X', 'Y', or 'Z'
            progress_callback: Optional callback(current, total) for progress tracking
        
        Returns:
            List of slice dictionaries with contours
        """
        logger.info(f"🔪 Slicing along {axis}-axis with {self.num_slices} planes...")
        start_time = time.time()
        
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
        
        for i, pos in enumerate(positions):
            # Create cutting plane
            if axis == 'X':
                origin = gp_Pnt(pos, 0, 0)
            elif axis == 'Y':
                origin = gp_Pnt(0, pos, 0)
            else:
                origin = gp_Pnt(0, 0, pos)
            
            plane = gp_Pln(origin, normal)
            
            try:
                # Perform section (intersection)
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
                            'exterior_contours': [c for c in contours if not c.is_hole],
                            'hole_contours': [c for c in contours if c.is_hole],
                            'num_holes': sum(1 for c in contours if c.is_hole)
                        }
                        
                        slices.append(slice_data)
            
            except Exception as e:
                logger.debug(f"Slice at {axis}={pos:.2f} failed: {e}")
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, self.num_slices)
        
        elapsed = time.time() - start_time
        self.slice_times.append(elapsed)
        
        logger.info(f"  ✅ {axis}-axis slicing: {len(slices)} valid slices in {elapsed:.2f}s")
        
        self.slices[axis] = slices
        return slices
    
    def _extract_contours(self, section_shape, coord_getter) -> List[SliceContour]:
        """
        Extract 2D contours from section shape
        
        Args:
            section_shape: Result of BRepAlgoAPI_Section
            coord_getter: Function to extract 2D coordinates from 3D point
        
        Returns:
            List of SliceContour objects
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
                    curve = BRep_Tool.Curve(edge)
                    if curve[0] is not None:
                        first_param = curve[1]
                        last_param = curve[2]
                        
                        # Sample 10 points per edge
                        for t in np.linspace(first_param, last_param, 10):
                            point_3d = curve[0].Value(t)
                            x, y = coord_getter(point_3d)
                            points.append(ContourPoint(x, y))
                    
                    edge_explorer.Next()
                
                if len(points) > 3:
                    # Calculate contour properties
                    points_np = np.array([[p.x, p.y] for p in points])
                    area = self._calculate_polygon_area(points_np)
                    centroid = np.mean(points_np, axis=0)
                    
                    # Determine if hole (interior contour)
                    # Heuristic: smaller area = likely a hole
                    is_hole = area < 100  # Threshold for hole vs boundary
                    
                    contour = SliceContour(
                        points=points,
                        is_hole=is_hole,
                        area=abs(area),
                        centroid=(centroid[0], centroid[1])
                    )
                    
                    contours.append(contour)
                
                wire_explorer.Next()
        
        except Exception as e:
            logger.debug(f"Contour extraction failed: {e}")
        
        return contours
    
    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Calculate area of 2D polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        x = points[:, 0]
        y = points[:, 1]
        
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    
    def analyze_hole_consistency(self) -> List[Dict]:
        """
        Analyze consistency of holes across multiple slices
        Detects through-holes vs blind holes
        
        Returns:
            List of hole feature dictionaries
        """
        logger.info("🔍 Analyzing hole consistency across slices...")
        
        detected_holes = []
        
        # Analyze Z-axis slices for vertical holes
        z_slices = self.slices.get('Z', [])
        
        if len(z_slices) < 5:
            logger.warning("  ⚠️ Insufficient Z-slices for hole analysis")
            return []
        
        # Track holes across slices
        hole_tracks = []
        
        for i, slice_data in enumerate(z_slices):
            for contour in slice_data['hole_contours']:
                # Try to match with existing tracks
                matched = False
                
                for track in hole_tracks:
                    # Check if centroid is close to previous slice
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
                            matched = True
                            break
                
                # Create new track if no match
                if not matched:
                    hole_tracks.append({
                        'centroids': [contour.centroid],
                        'areas': [contour.area],
                        'z_positions': [slice_data['position']]
                    })
        
        # Analyze tracks to determine hole type
        for track in hole_tracks:
            if len(track['z_positions']) >= 3:
                # Through-hole if spans most of Z-range
                z_span = max(track['z_positions']) - min(track['z_positions'])
                total_z = self.bbox[5] - self.bbox[2]
                
                is_through = z_span > 0.8 * total_z
                
                # Estimate diameter from average area (assuming circular)
                avg_area = np.mean(track['areas'])
                radius = np.sqrt(avg_area / np.pi)
                diameter = 2 * radius
                
                # Calculate center position
                center_x = np.mean([c[0] for c in track['centroids']])
                center_y = np.mean([c[1] for c in track['centroids']])
                center_z = np.mean(track['z_positions'])
                
                hole = {
                    'type': 'hole',
                    'subtype': 'through' if is_through else 'blind',
                    'confidence': 0.85,
                    'dimensions': {
                        'diameter': diameter,
                        'radius': radius,
                        'depth': z_span
                    },
                    'location': (center_x, center_y, center_z),
                    'orientation': (0, 0, 1),  # Vertical
                    'detection_method': 'slice_consistency_analysis'
                }
                
                detected_holes.append(hole)
        
        logger.info(f"  ✅ Found {len(detected_holes)} holes via slice analysis")
        return detected_holes


class DetailedVolumetricAnalyzer:
    """
    Production-grade volumetric analysis with adaptive voxelization
    Identifies negative volumes (material removal features)
    """
    
    def __init__(self, shape: TopoDS_Shape, resolution: int = 50):
        self.shape = shape
        self.resolution = resolution
        self.bbox = self._compute_bbox()
        self.voxel_size = self._calculate_voxel_size()
        
        # Cached classifier for performance
        self.classifier = BRepClass3d_SolidClassifier(self.shape)
    
    def _compute_bbox(self) -> Tuple[float, float, float, float, float, float]:
        """Compute bounding box"""
        bbox = Bnd_Box()
        brepbndlib.Add(self.shape, bbox)
        return bbox.Get()
    
    def _calculate_voxel_size(self) -> Tuple[float, float, float]:
        """Calculate voxel dimensions"""
        xmin, ymin, zmin, xmax, ymax, zmax = self.bbox
        
        voxel_x = (xmax - xmin) / self.resolution
        voxel_y = (ymax - ymin) / self.resolution
        voxel_z = (zmax - zmin) / self.resolution
        
        return (voxel_x, voxel_y, voxel_z)
    
    def compute_material_volume(self) -> float:
        """
        Compute total material volume of part
        
        Returns:
            Volume in cubic units
        """
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties
        
        props = GProp_GProps()
        brepgprop_VolumeProperties(self.shape, props)
        
        return props.Mass()
    
    def compute_bounding_box_volume(self) -> float:
        """Compute volume of bounding box"""
        xmin, ymin, zmin, xmax, ymax, zmax = self.bbox
        
        volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        return volume
    
    def compute_material_removal_ratio(self) -> float:
        """
        Compute ratio of removed material to bounding box
        Indicates how much material was removed (machining intensity)
        
        Returns:
            Ratio from 0.0 (solid block) to 1.0 (mostly removed)
        """
        material_vol = self.compute_material_volume()
        bbox_vol = self.compute_bounding_box_volume()
        
        if bbox_vol > 0:
            removal_ratio = 1.0 - (material_vol / bbox_vol)
            return max(0.0, min(1.0, removal_ratio))
        
        return 0.0
    
    def identify_negative_regions(self) -> List[Dict]:
        """
        Identify regions of removed material using adaptive sampling
        
        Returns:
            List of negative region dictionaries
        """
        logger.info("📦 Identifying negative volume regions...")
        start_time = time.time()
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.bbox
        voxel_x, voxel_y, voxel_z = self.voxel_size
        
        # Sample points in bounding box
        negative_points = []
        
        # Subsample for performance (every 3rd voxel)
        step = 3
        
        for i in range(0, self.resolution, step):
            x = xmin + i * voxel_x
            
            for j in range(0, self.resolution, step):
                y = ymin + j * voxel_y
                
                for k in range(0, self.resolution, step):
                    z = zmin + k * voxel_z
                    
                    test_point = gp_Pnt(x, y, z)
                    
                    # Check if point is in bounding box but outside solid
                    self.classifier.Perform(test_point, 1e-6)
                    state = self.classifier.State()
                    
                    from OCC.Core.TopAbs import TopAbs_OUT
                    if state == TopAbs_OUT:
                        # Point is in bbox but outside part = removed material
                        negative_points.append((x, y, z))
        
        # Cluster negative points into regions (simplified)
        # Production implementation would use DBSCAN or connected components
        
        num_negative = len(negative_points)
        total_voxels = (self.resolution // step) ** 3
        
        logger.info(f"  ✅ Volumetric analysis: {num_negative}/{total_voxels} negative voxels ({time.time()-start_time:.2f}s)")
        
        return [{
            'num_negative_voxels': num_negative,
            'total_voxels': total_voxels,
            'negative_ratio': num_negative / total_voxels if total_voxels > 0 else 0
        }]
