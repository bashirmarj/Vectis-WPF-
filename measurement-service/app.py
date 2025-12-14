# measurement-service/app.py
# Standalone Flask application for CAD measurement calculations
# Version 1.0.0 - SolidWorks-style measurement service
#
# This is a COMPLETELY STANDALONE service that does not modify any existing code.
# It provides REST API endpoints for measurement calculations on CAD geometry.

import os
import io
import time
import math
import hashlib
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import json

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# OpenCascade imports for CAD geometry processing
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods_Vertex
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface

# === Configuration ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# === Enums and Data Classes ===

class MeasurementType(Enum):
    """Types of measurements supported"""
    POINT_TO_POINT = "point_to_point"
    EDGE_LENGTH = "edge_length"
    EDGE_DIAMETER = "edge_diameter"
    EDGE_RADIUS = "edge_radius"
    FACE_AREA = "face_area"
    FACE_TO_FACE = "face_to_face"
    ANGLE = "angle"
    COORDINATE = "coordinate"


class UnitSystem(Enum):
    """Supported unit systems"""
    METRIC = "metric"
    IMPERIAL = "imperial"


@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def to_gp_pnt(self) -> gp_Pnt:
        return gp_Pnt(self.x, self.y, self.z)

    @classmethod
    def from_gp_pnt(cls, pnt: gp_Pnt) -> 'Point3D':
        return cls(pnt.X(), pnt.Y(), pnt.Z())

    @classmethod
    def from_list(cls, coords: List[float]) -> 'Point3D':
        return cls(coords[0], coords[1], coords[2])

    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


@dataclass
class Vector3D:
    """3D vector representation"""
    x: float
    y: float
    z: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def normalize(self) -> 'Vector3D':
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag < 1e-10:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)

    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @classmethod
    def from_gp_vec(cls, vec: gp_Vec) -> 'Vector3D':
        return cls(vec.X(), vec.Y(), vec.Z())

    @classmethod
    def from_gp_dir(cls, dir: gp_Dir) -> 'Vector3D':
        return cls(dir.X(), dir.Y(), dir.Z())


@dataclass
class MeasurementResult:
    """Standard measurement result"""
    success: bool
    measurement_type: str
    value: float
    unit: str
    label: str
    display_value: str

    # Additional data
    delta_x: Optional[float] = None
    delta_y: Optional[float] = None
    delta_z: Optional[float] = None

    # For edge measurements
    edge_type: Optional[str] = None
    diameter: Optional[float] = None
    radius: Optional[float] = None
    center: Optional[List[float]] = None

    # For face measurements
    face_id: Optional[int] = None
    face_type: Optional[str] = None
    surface_type: Optional[str] = None
    area: Optional[float] = None
    normal: Optional[List[float]] = None

    # For face-to-face measurements
    face1_id: Optional[int] = None
    face2_id: Optional[int] = None
    angle: Optional[float] = None
    is_parallel: Optional[bool] = None
    perpendicular_distance: Optional[float] = None

    # Points for visualization
    point1: Optional[List[float]] = None
    point2: Optional[List[float]] = None

    # Metadata
    confidence: float = 1.0
    backend_match: bool = True
    processing_time_ms: int = 0
    error: Optional[str] = None


@dataclass
class EdgeInfo:
    """Edge geometry information"""
    edge_id: int
    edge_type: str  # "line", "circle", "arc", "ellipse", "spline"
    start_point: List[float]
    end_point: List[float]
    length: Optional[float] = None
    radius: Optional[float] = None
    diameter: Optional[float] = None
    center: Optional[List[float]] = None
    axis: Optional[List[float]] = None
    angle: Optional[float] = None  # For arcs


@dataclass
class FaceInfo:
    """Face geometry information"""
    face_id: int
    surface_type: str  # "plane", "cylinder", "cone", "sphere", "torus", "other"
    center: List[float]
    normal: List[float]
    area: float
    radius: Optional[float] = None
    axis: Optional[List[float]] = None

    # For planes
    plane_distance: Optional[float] = None  # Distance from origin

    # For cylinders
    cylinder_axis: Optional[List[float]] = None
    cylinder_radius: Optional[float] = None


# === Unit Conversion ===

class UnitConverter:
    """Handle unit conversions between metric and imperial"""

    MM_TO_INCH = 0.0393701
    INCH_TO_MM = 25.4
    MM2_TO_INCH2 = MM_TO_INCH ** 2
    INCH2_TO_MM2 = INCH_TO_MM ** 2

    @staticmethod
    def mm_to_inch(mm: float) -> float:
        """Convert millimeters to inches"""
        return mm * UnitConverter.MM_TO_INCH

    @staticmethod
    def inch_to_mm(inch: float) -> float:
        """Convert inches to millimeters"""
        return inch * UnitConverter.INCH_TO_MM

    @staticmethod
    def mm2_to_inch2(mm2: float) -> float:
        """Convert square millimeters to square inches"""
        return mm2 * UnitConverter.MM2_TO_INCH2

    @staticmethod
    def inch2_to_mm2(inch2: float) -> float:
        """Convert square inches to square millimeters"""
        return inch2 * UnitConverter.INCH2_TO_MM2

    @staticmethod
    def format_length(value_mm: float, unit_system: UnitSystem, decimals: int = 3) -> str:
        """Format length with appropriate unit"""
        if unit_system == UnitSystem.IMPERIAL:
            value = UnitConverter.mm_to_inch(value_mm)
            return f"{value:.{decimals}f} in"
        return f"{value_mm:.{decimals}f} mm"

    @staticmethod
    def format_area(value_mm2: float, unit_system: UnitSystem, decimals: int = 3) -> str:
        """Format area with appropriate unit"""
        if unit_system == UnitSystem.IMPERIAL:
            value = UnitConverter.mm2_to_inch2(value_mm2)
            return f"{value:.{decimals}f} in\u00b2"
        return f"{value_mm2:.{decimals}f} mm\u00b2"

    @staticmethod
    def format_angle(degrees: float, decimals: int = 2) -> str:
        """Format angle in degrees"""
        return f"{degrees:.{decimals}f}\u00b0"


# === Geometry Analysis ===

class GeometryAnalyzer:
    """Analyze CAD geometry for measurements"""

    def __init__(self, shape=None):
        self.shape = shape
        self.faces: Dict[int, Any] = {}
        self.edges: Dict[int, Any] = {}
        self.face_info: Dict[int, FaceInfo] = {}
        self.edge_info: Dict[int, EdgeInfo] = {}

        if shape:
            self._index_geometry()

    def _index_geometry(self):
        """Index all faces and edges in the shape"""
        # Index faces
        face_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        face_id = 0
        while face_explorer.More():
            face = topods_Face(face_explorer.Current())
            self.faces[face_id] = face
            self.face_info[face_id] = self._analyze_face(face, face_id)
            face_id += 1
            face_explorer.Next()

        # Index edges
        edge_explorer = TopExp_Explorer(self.shape, TopAbs_EDGE)
        edge_id = 0
        while edge_explorer.More():
            edge = topods_Edge(edge_explorer.Current())
            self.edges[edge_id] = edge
            self.edge_info[edge_id] = self._analyze_edge(edge, edge_id)
            edge_id += 1
            edge_explorer.Next()

        logger.info(f"Indexed {len(self.faces)} faces and {len(self.edges)} edges")

    def _analyze_face(self, face, face_id: int) -> FaceInfo:
        """Analyze a face and extract geometric properties"""
        adaptor = BRepAdaptor_Surface(face)
        surface_type = adaptor.GetType()

        # Compute area
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass() * 1000000  # Convert to mm^2

        # Get center of mass
        center_pnt = props.CentreOfMass()
        center = [center_pnt.X(), center_pnt.Y(), center_pnt.Z()]

        # Get surface normal at center
        u_min, u_max, v_min, v_max = adaptor.FirstUParameter(), adaptor.LastUParameter(), \
                                      adaptor.FirstVParameter(), adaptor.LastVParameter()
        u_mid = (u_min + u_max) / 2
        v_mid = (v_min + v_max) / 2

        pnt = gp_Pnt()
        normal_vec = gp_Vec()
        adaptor.D1(u_mid, v_mid, pnt, gp_Vec(), normal_vec)

        # Normalize
        if normal_vec.Magnitude() > 1e-10:
            normal_vec.Normalize()
        normal = [normal_vec.X(), normal_vec.Y(), normal_vec.Z()]

        # Determine surface type
        if surface_type == GeomAbs_Plane:
            surf_type_str = "plane"
            plane = adaptor.Plane()
            plane_dist = plane.Distance(gp_Pnt(0, 0, 0))
            return FaceInfo(
                face_id=face_id,
                surface_type=surf_type_str,
                center=center,
                normal=normal,
                area=area,
                plane_distance=plane_dist
            )
        elif surface_type == GeomAbs_Cylinder:
            surf_type_str = "cylinder"
            cylinder = adaptor.Cylinder()
            radius = cylinder.Radius()
            axis = cylinder.Axis()
            axis_dir = axis.Direction()
            return FaceInfo(
                face_id=face_id,
                surface_type=surf_type_str,
                center=center,
                normal=normal,
                area=area,
                radius=radius,
                cylinder_axis=[axis_dir.X(), axis_dir.Y(), axis_dir.Z()],
                cylinder_radius=radius
            )
        elif surface_type == GeomAbs_Cone:
            surf_type_str = "cone"
        elif surface_type == GeomAbs_Sphere:
            surf_type_str = "sphere"
            sphere = adaptor.Sphere()
            radius = sphere.Radius()
            return FaceInfo(
                face_id=face_id,
                surface_type=surf_type_str,
                center=center,
                normal=normal,
                area=area,
                radius=radius
            )
        elif surface_type == GeomAbs_Torus:
            surf_type_str = "torus"
        else:
            surf_type_str = "other"

        return FaceInfo(
            face_id=face_id,
            surface_type=surf_type_str,
            center=center,
            normal=normal,
            area=area
        )

    def _analyze_edge(self, edge, edge_id: int) -> EdgeInfo:
        """Analyze an edge and extract geometric properties"""
        adaptor = BRepAdaptor_Curve(edge)
        curve_type = adaptor.GetType()

        # Get start and end points
        first = adaptor.FirstParameter()
        last = adaptor.LastParameter()
        start_pnt = adaptor.Value(first)
        end_pnt = adaptor.Value(last)

        start_point = [start_pnt.X(), start_pnt.Y(), start_pnt.Z()]
        end_point = [end_pnt.X(), end_pnt.Y(), end_pnt.Z()]

        # Compute length
        length = GCPnts_AbscissaPoint.Length(adaptor)

        if curve_type == GeomAbs_Line:
            return EdgeInfo(
                edge_id=edge_id,
                edge_type="line",
                start_point=start_point,
                end_point=end_point,
                length=length
            )
        elif curve_type == GeomAbs_Circle:
            circle = adaptor.Circle()
            radius = circle.Radius()
            center = circle.Location()
            axis = circle.Axis().Direction()

            # Determine if full circle or arc
            angle = last - first
            is_full_circle = abs(angle - 2 * math.pi) < 0.01

            return EdgeInfo(
                edge_id=edge_id,
                edge_type="circle" if is_full_circle else "arc",
                start_point=start_point,
                end_point=end_point,
                length=length,
                radius=radius,
                diameter=radius * 2,
                center=[center.X(), center.Y(), center.Z()],
                axis=[axis.X(), axis.Y(), axis.Z()],
                angle=math.degrees(angle) if not is_full_circle else 360.0
            )
        elif curve_type == GeomAbs_Ellipse:
            ellipse = adaptor.Ellipse()
            return EdgeInfo(
                edge_id=edge_id,
                edge_type="ellipse",
                start_point=start_point,
                end_point=end_point,
                length=length,
                radius=ellipse.MinorRadius()  # Use minor radius as reference
            )
        else:
            return EdgeInfo(
                edge_id=edge_id,
                edge_type="spline",
                start_point=start_point,
                end_point=end_point,
                length=length
            )

    def find_closest_edge(self, point: Point3D, threshold: float = 5.0) -> Optional[EdgeInfo]:
        """Find the closest edge to a given point"""
        closest_edge = None
        min_distance = threshold

        pnt = point.to_gp_pnt()

        for edge_id, edge in self.edges.items():
            # Get distance from point to edge
            adaptor = BRepAdaptor_Curve(edge)
            first = adaptor.FirstParameter()
            last = adaptor.LastParameter()

            # Sample points along edge
            samples = 20
            for i in range(samples + 1):
                t = first + (last - first) * i / samples
                edge_pnt = adaptor.Value(t)
                dist = pnt.Distance(edge_pnt)
                if dist < min_distance:
                    min_distance = dist
                    closest_edge = self.edge_info[edge_id]

        return closest_edge

    def find_closest_face(self, point: Point3D, threshold: float = 10.0) -> Optional[FaceInfo]:
        """Find the closest face to a given point"""
        closest_face = None
        min_distance = threshold

        pnt = point.to_gp_pnt()

        for face_id, face in self.faces.items():
            # Get distance from point to face using center
            face_info = self.face_info[face_id]
            center_pnt = gp_Pnt(face_info.center[0], face_info.center[1], face_info.center[2])
            dist = pnt.Distance(center_pnt)

            if dist < min_distance:
                min_distance = dist
                closest_face = face_info

        return closest_face

    def calculate_face_to_face_distance(self, face1_id: int, face2_id: int) -> MeasurementResult:
        """Calculate distance between two faces"""
        start_time = time.time()

        if face1_id not in self.faces or face2_id not in self.faces:
            return MeasurementResult(
                success=False,
                measurement_type="face_to_face",
                value=0,
                unit="mm",
                label="Error",
                display_value="Invalid face ID",
                error="One or both face IDs not found"
            )

        face1 = self.faces[face1_id]
        face2 = self.faces[face2_id]
        face1_info = self.face_info[face1_id]
        face2_info = self.face_info[face2_id]

        # Use BRepExtrema for accurate distance calculation
        extrema = BRepExtrema_DistShapeShape(face1, face2)
        extrema.Perform()

        if not extrema.IsDone():
            return MeasurementResult(
                success=False,
                measurement_type="face_to_face",
                value=0,
                unit="mm",
                label="Error",
                display_value="Could not calculate distance",
                error="Distance calculation failed"
            )

        distance = extrema.Value()

        # Get closest points
        pnt1 = extrema.PointOnShape1(1)
        pnt2 = extrema.PointOnShape2(1)

        # Calculate angle between face normals
        n1 = Vector3D(*face1_info.normal)
        n2 = Vector3D(*face2_info.normal)
        dot = abs(n1.dot(n2))
        angle = math.degrees(math.acos(min(1.0, max(-1.0, dot))))

        # Check if parallel (angle close to 0 or 180 degrees)
        is_parallel = angle < 5 or angle > 175

        # For parallel planes, calculate perpendicular distance
        perp_distance = None
        if is_parallel and face1_info.surface_type == "plane" and face2_info.surface_type == "plane":
            # Project center of face2 onto face1's plane
            center_diff = Vector3D(
                face2_info.center[0] - face1_info.center[0],
                face2_info.center[1] - face1_info.center[1],
                face2_info.center[2] - face1_info.center[2]
            )
            perp_distance = abs(center_diff.dot(n1))

        processing_time = int((time.time() - start_time) * 1000)

        return MeasurementResult(
            success=True,
            measurement_type="face_to_face",
            value=distance,
            unit="mm",
            label=f"Distance: {distance:.3f} mm",
            display_value=f"{distance:.3f} mm ({angle:.1f}\u00b0)",
            face1_id=face1_id,
            face2_id=face2_id,
            angle=angle,
            is_parallel=is_parallel,
            perpendicular_distance=perp_distance,
            point1=[pnt1.X(), pnt1.Y(), pnt1.Z()],
            point2=[pnt2.X(), pnt2.Y(), pnt2.Z()],
            processing_time_ms=processing_time
        )


# === Measurement Calculations ===

class MeasurementCalculator:
    """Calculate various measurements on CAD geometry"""

    @staticmethod
    def point_to_point_distance(
        p1: Point3D,
        p2: Point3D,
        unit_system: UnitSystem = UnitSystem.METRIC
    ) -> MeasurementResult:
        """Calculate distance between two points with XYZ deltas"""
        start_time = time.time()

        # Calculate distance
        distance = p1.distance_to(p2)

        # Calculate deltas
        delta_x = abs(p2.x - p1.x)
        delta_y = abs(p2.y - p1.y)
        delta_z = abs(p2.z - p1.z)

        # Format display
        display_value = UnitConverter.format_length(distance, unit_system)

        processing_time = int((time.time() - start_time) * 1000)

        return MeasurementResult(
            success=True,
            measurement_type="point_to_point",
            value=distance,
            unit="mm",
            label=f"Distance: {display_value}",
            display_value=display_value,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_z=delta_z,
            point1=p1.to_list(),
            point2=p2.to_list(),
            processing_time_ms=processing_time
        )

    @staticmethod
    def edge_measurement(edge_info: EdgeInfo, unit_system: UnitSystem = UnitSystem.METRIC) -> MeasurementResult:
        """Measure an edge (length, diameter, or radius based on type)"""
        start_time = time.time()

        if edge_info.edge_type == "circle":
            # Full circle - show diameter
            value = edge_info.diameter or 0
            label = f"\u00d8 {UnitConverter.format_length(value, unit_system)}"
            measurement_type = "edge_diameter"
        elif edge_info.edge_type == "arc":
            # Arc - show radius
            value = edge_info.radius or 0
            label = f"R {UnitConverter.format_length(value, unit_system)}"
            measurement_type = "edge_radius"
        else:
            # Line or other - show length
            value = edge_info.length or 0
            label = UnitConverter.format_length(value, unit_system)
            measurement_type = "edge_length"

        display_value = UnitConverter.format_length(value, unit_system)
        processing_time = int((time.time() - start_time) * 1000)

        return MeasurementResult(
            success=True,
            measurement_type=measurement_type,
            value=value,
            unit="mm",
            label=label,
            display_value=display_value,
            edge_type=edge_info.edge_type,
            diameter=edge_info.diameter,
            radius=edge_info.radius,
            center=edge_info.center,
            point1=edge_info.start_point,
            point2=edge_info.end_point,
            processing_time_ms=processing_time
        )

    @staticmethod
    def face_area(face_info: FaceInfo, unit_system: UnitSystem = UnitSystem.METRIC) -> MeasurementResult:
        """Calculate face area"""
        start_time = time.time()

        display_value = UnitConverter.format_area(face_info.area, unit_system)
        processing_time = int((time.time() - start_time) * 1000)

        return MeasurementResult(
            success=True,
            measurement_type="face_area",
            value=face_info.area,
            unit="mm\u00b2",
            label=f"Area: {display_value}",
            display_value=display_value,
            face_id=face_info.face_id,
            surface_type=face_info.surface_type,
            area=face_info.area,
            normal=face_info.normal,
            point1=face_info.center,
            processing_time_ms=processing_time
        )

    @staticmethod
    def angle_between_vectors(v1: Vector3D, v2: Vector3D) -> MeasurementResult:
        """Calculate angle between two vectors"""
        start_time = time.time()

        v1_norm = v1.normalize()
        v2_norm = v2.normalize()

        dot = v1_norm.dot(v2_norm)
        angle_rad = math.acos(min(1.0, max(-1.0, dot)))
        angle_deg = math.degrees(angle_rad)

        display_value = UnitConverter.format_angle(angle_deg)
        processing_time = int((time.time() - start_time) * 1000)

        return MeasurementResult(
            success=True,
            measurement_type="angle",
            value=angle_deg,
            unit="deg",
            label=f"Angle: {display_value}",
            display_value=display_value,
            angle=angle_deg,
            processing_time_ms=processing_time
        )


# === API Endpoints ===

def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracking"""
    return f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"


# Global cache for analyzed shapes
shape_cache: Dict[str, GeometryAnalyzer] = {}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "vectis-measurement-service",
        "version": "1.0.0",
        "capabilities": [
            "point_to_point",
            "edge_measurement",
            "face_area",
            "face_to_face",
            "angle",
            "unit_conversion"
        ],
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/measure/point-to-point', methods=['POST'])
def measure_point_to_point():
    """
    Calculate point-to-point distance with XYZ deltas

    Request body:
    {
        "point1": [x1, y1, z1],
        "point2": [x2, y2, z2],
        "unit_system": "metric" | "imperial" (optional, default: metric)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data or 'point1' not in data or 'point2' not in data:
            return jsonify({
                "success": False,
                "error": "Missing point1 or point2 in request body",
                "correlation_id": correlation_id
            }), 400

        p1 = Point3D.from_list(data['point1'])
        p2 = Point3D.from_list(data['point2'])
        unit_system = UnitSystem.IMPERIAL if data.get('unit_system') == 'imperial' else UnitSystem.METRIC

        result = MeasurementCalculator.point_to_point_distance(p1, p2, unit_system)

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            **asdict(result)
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/measure/edge', methods=['POST'])
def measure_edge():
    """
    Measure an edge (length, diameter, or radius)

    Request body:
    {
        "edge_info": {
            "edge_id": 0,
            "edge_type": "circle" | "arc" | "line",
            "start_point": [x, y, z],
            "end_point": [x, y, z],
            "length": float (optional),
            "radius": float (optional),
            "diameter": float (optional),
            "center": [x, y, z] (optional)
        },
        "unit_system": "metric" | "imperial" (optional)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data or 'edge_info' not in data:
            return jsonify({
                "success": False,
                "error": "Missing edge_info in request body",
                "correlation_id": correlation_id
            }), 400

        edge_data = data['edge_info']
        edge_info = EdgeInfo(
            edge_id=edge_data.get('edge_id', 0),
            edge_type=edge_data.get('edge_type', 'line'),
            start_point=edge_data.get('start_point', [0, 0, 0]),
            end_point=edge_data.get('end_point', [0, 0, 0]),
            length=edge_data.get('length'),
            radius=edge_data.get('radius'),
            diameter=edge_data.get('diameter'),
            center=edge_data.get('center')
        )

        unit_system = UnitSystem.IMPERIAL if data.get('unit_system') == 'imperial' else UnitSystem.METRIC

        result = MeasurementCalculator.edge_measurement(edge_info, unit_system)

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            **asdict(result)
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/measure/face-area', methods=['POST'])
def measure_face_area():
    """
    Calculate face area

    Request body:
    {
        "face_info": {
            "face_id": 0,
            "surface_type": "plane" | "cylinder" | etc,
            "center": [x, y, z],
            "normal": [x, y, z],
            "area": float
        },
        "unit_system": "metric" | "imperial" (optional)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data or 'face_info' not in data:
            return jsonify({
                "success": False,
                "error": "Missing face_info in request body",
                "correlation_id": correlation_id
            }), 400

        face_data = data['face_info']
        face_info = FaceInfo(
            face_id=face_data.get('face_id', 0),
            surface_type=face_data.get('surface_type', 'other'),
            center=face_data.get('center', [0, 0, 0]),
            normal=face_data.get('normal', [0, 0, 1]),
            area=face_data.get('area', 0)
        )

        unit_system = UnitSystem.IMPERIAL if data.get('unit_system') == 'imperial' else UnitSystem.METRIC

        result = MeasurementCalculator.face_area(face_info, unit_system)

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            **asdict(result)
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/measure/face-to-face', methods=['POST'])
def measure_face_to_face():
    """
    Calculate distance and angle between two faces

    Request body (option 1 - with STEP file):
    {
        "face1_id": int,
        "face2_id": int,
        "file_hash": str (to use cached geometry)
    }

    Request body (option 2 - with face info):
    {
        "face1": {
            "center": [x, y, z],
            "normal": [x, y, z]
        },
        "face2": {
            "center": [x, y, z],
            "normal": [x, y, z]
        },
        "unit_system": "metric" | "imperial" (optional)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Missing request body",
                "correlation_id": correlation_id
            }), 400

        unit_system = UnitSystem.IMPERIAL if data.get('unit_system') == 'imperial' else UnitSystem.METRIC

        # Option 1: Use cached geometry with face IDs
        if 'file_hash' in data and 'face1_id' in data and 'face2_id' in data:
            file_hash = data['file_hash']
            if file_hash not in shape_cache:
                return jsonify({
                    "success": False,
                    "error": "Geometry not found in cache. Please analyze the STEP file first.",
                    "correlation_id": correlation_id
                }), 404

            analyzer = shape_cache[file_hash]
            result = analyzer.calculate_face_to_face_distance(
                data['face1_id'],
                data['face2_id']
            )

            return jsonify({
                "success": result.success,
                "correlation_id": correlation_id,
                **asdict(result)
            })

        # Option 2: Calculate from provided face info (simplified calculation)
        if 'face1' in data and 'face2' in data:
            face1 = data['face1']
            face2 = data['face2']

            # Calculate distance between centers
            c1 = Point3D.from_list(face1['center'])
            c2 = Point3D.from_list(face2['center'])
            distance = c1.distance_to(c2)

            # Calculate angle between normals
            n1 = Vector3D(*face1['normal'])
            n2 = Vector3D(*face2['normal'])
            dot = abs(n1.dot(n2))
            angle = math.degrees(math.acos(min(1.0, max(-1.0, dot))))

            is_parallel = angle < 5 or angle > 175

            # Calculate perpendicular distance for parallel planes
            perp_distance = None
            if is_parallel:
                center_diff = Vector3D(
                    c2.x - c1.x,
                    c2.y - c1.y,
                    c2.z - c1.z
                )
                perp_distance = abs(center_diff.dot(n1))

            display_value = UnitConverter.format_length(distance, unit_system)

            return jsonify({
                "success": True,
                "correlation_id": correlation_id,
                "measurement_type": "face_to_face",
                "value": distance,
                "unit": "mm",
                "label": f"Distance: {display_value}",
                "display_value": f"{display_value} ({angle:.1f}\u00b0)",
                "angle": angle,
                "is_parallel": is_parallel,
                "perpendicular_distance": perp_distance,
                "point1": face1['center'],
                "point2": face2['center']
            })

        return jsonify({
            "success": False,
            "error": "Invalid request. Provide either (file_hash + face IDs) or (face1 + face2 info)",
            "correlation_id": correlation_id
        }), 400

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/measure/angle', methods=['POST'])
def measure_angle():
    """
    Calculate angle between two vectors or three points

    Request body (option 1 - two vectors):
    {
        "vector1": [x, y, z],
        "vector2": [x, y, z]
    }

    Request body (option 2 - three points, angle at middle point):
    {
        "point1": [x, y, z],
        "vertex": [x, y, z],
        "point2": [x, y, z]
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Missing request body",
                "correlation_id": correlation_id
            }), 400

        # Option 1: Two vectors
        if 'vector1' in data and 'vector2' in data:
            v1 = Vector3D(*data['vector1'])
            v2 = Vector3D(*data['vector2'])
            result = MeasurementCalculator.angle_between_vectors(v1, v2)

            return jsonify({
                "success": True,
                "correlation_id": correlation_id,
                **asdict(result)
            })

        # Option 2: Three points
        if 'point1' in data and 'vertex' in data and 'point2' in data:
            p1 = Point3D.from_list(data['point1'])
            vertex = Point3D.from_list(data['vertex'])
            p2 = Point3D.from_list(data['point2'])

            # Create vectors from vertex to each point
            v1 = Vector3D(p1.x - vertex.x, p1.y - vertex.y, p1.z - vertex.z)
            v2 = Vector3D(p2.x - vertex.x, p2.y - vertex.y, p2.z - vertex.z)

            result = MeasurementCalculator.angle_between_vectors(v1, v2)
            result.point1 = p1.to_list()
            result.point2 = p2.to_list()

            return jsonify({
                "success": True,
                "correlation_id": correlation_id,
                **asdict(result)
            })

        return jsonify({
            "success": False,
            "error": "Invalid request. Provide either (vector1 + vector2) or (point1 + vertex + point2)",
            "correlation_id": correlation_id
        }), 400

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/measure/coordinate', methods=['POST'])
def measure_coordinate():
    """
    Get coordinate information for a point

    Request body:
    {
        "point": [x, y, z],
        "unit_system": "metric" | "imperial" (optional)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data or 'point' not in data:
            return jsonify({
                "success": False,
                "error": "Missing point in request body",
                "correlation_id": correlation_id
            }), 400

        point = Point3D.from_list(data['point'])
        unit_system = UnitSystem.IMPERIAL if data.get('unit_system') == 'imperial' else UnitSystem.METRIC

        if unit_system == UnitSystem.IMPERIAL:
            x_val = UnitConverter.mm_to_inch(point.x)
            y_val = UnitConverter.mm_to_inch(point.y)
            z_val = UnitConverter.mm_to_inch(point.z)
            unit = "in"
        else:
            x_val = point.x
            y_val = point.y
            z_val = point.z
            unit = "mm"

        display_value = f"X: {x_val:.3f} {unit}, Y: {y_val:.3f} {unit}, Z: {z_val:.3f} {unit}"

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            "measurement_type": "coordinate",
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "unit": unit,
            "label": "Coordinate",
            "display_value": display_value,
            "point1": data['point']
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/analyze-geometry', methods=['POST'])
def analyze_geometry():
    """
    Analyze STEP file geometry for measurements

    Request: multipart/form-data with 'file' field containing STEP file

    Response:
    {
        "success": bool,
        "file_hash": str (for caching),
        "face_count": int,
        "edge_count": int,
        "faces": [...],
        "edges": [...]
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())
    start_time = time.time()

    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided",
                "correlation_id": correlation_id
            }), 400

        file = request.files['file']
        if not file.filename.lower().endswith(('.step', '.stp')):
            return jsonify({
                "success": False,
                "error": "Only STEP files supported (.step, .stp)",
                "correlation_id": correlation_id
            }), 400

        # Read and hash file
        file_content = file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Check cache
        if file_hash in shape_cache:
            analyzer = shape_cache[file_hash]
            logger.info(f"[{correlation_id}] Using cached geometry for {file_hash[:8]}")
        else:
            # Parse STEP file
            with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            try:
                reader = STEPControl_Reader()
                read_status = reader.ReadFile(tmp_path)

                if read_status != 1:
                    return jsonify({
                        "success": False,
                        "error": f"STEP read failed with status {read_status}",
                        "correlation_id": correlation_id
                    }), 400

                reader.TransferRoots()
                shape = reader.OneShape()

                if shape.IsNull():
                    return jsonify({
                        "success": False,
                        "error": "Failed to extract shape from STEP file",
                        "correlation_id": correlation_id
                    }), 400

                # Tessellate for visualization
                mesh = BRepMesh_IncrementalMesh(shape, 0.005, False, 12.0, True)
                mesh.Perform()

                # Create analyzer and cache
                analyzer = GeometryAnalyzer(shape)
                shape_cache[file_hash] = analyzer
                logger.info(f"[{correlation_id}] Cached geometry for {file_hash[:8]}")

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        processing_time = int((time.time() - start_time) * 1000)

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            "file_hash": file_hash,
            "face_count": len(analyzer.faces),
            "edge_count": len(analyzer.edges),
            "faces": [asdict(f) for f in analyzer.face_info.values()],
            "edges": [asdict(e) for e in analyzer.edge_info.values()],
            "processing_time_ms": processing_time
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/convert-units', methods=['POST'])
def convert_units():
    """
    Convert measurement values between unit systems

    Request body:
    {
        "value": float,
        "from_unit": "mm" | "in" | "mm2" | "in2" | "deg",
        "to_unit": "mm" | "in" | "mm2" | "in2" | "deg",
        "decimals": int (optional, default: 3)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        data = request.get_json()

        if not data or 'value' not in data or 'from_unit' not in data or 'to_unit' not in data:
            return jsonify({
                "success": False,
                "error": "Missing value, from_unit, or to_unit in request body",
                "correlation_id": correlation_id
            }), 400

        value = data['value']
        from_unit = data['from_unit']
        to_unit = data['to_unit']
        decimals = data.get('decimals', 3)

        # Perform conversion
        if from_unit == to_unit:
            converted = value
        elif from_unit == 'mm' and to_unit == 'in':
            converted = UnitConverter.mm_to_inch(value)
        elif from_unit == 'in' and to_unit == 'mm':
            converted = UnitConverter.inch_to_mm(value)
        elif from_unit == 'mm2' and to_unit == 'in2':
            converted = UnitConverter.mm2_to_inch2(value)
        elif from_unit == 'in2' and to_unit == 'mm2':
            converted = UnitConverter.inch2_to_mm2(value)
        elif from_unit == 'deg' and to_unit == 'deg':
            converted = value
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported conversion: {from_unit} to {to_unit}",
                "correlation_id": correlation_id
            }), 400

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": round(converted, decimals),
            "converted_unit": to_unit,
            "display_value": f"{round(converted, decimals):.{decimals}f} {to_unit}"
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


@app.route('/measure/from-mesh', methods=['POST'])
def measure_from_mesh():
    """
    Calculate measurements from mesh data (for frontend integration)

    This endpoint works with the existing mesh data format from the main app.py

    Request body:
    {
        "measurement_type": "point_to_point" | "edge" | "face_area" | "face_to_face",
        "data": {
            // For point_to_point:
            "point1": [x, y, z],
            "point2": [x, y, z],

            // For edge:
            "edge_type": "line" | "circle" | "arc",
            "start": [x, y, z],
            "end": [x, y, z],
            "diameter": float (optional),
            "radius": float (optional),
            "length": float (optional),

            // For face_area:
            "face_id": int,
            "area": float,
            "center": [x, y, z],
            "normal": [x, y, z],
            "surface_type": "plane" | "cylinder" | etc,

            // For face_to_face:
            "face1": { center, normal },
            "face2": { center, normal }
        },
        "unit_system": "metric" | "imperial" (optional)
    }
    """
    correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

    try:
        request_data = request.get_json()

        if not request_data or 'measurement_type' not in request_data or 'data' not in request_data:
            return jsonify({
                "success": False,
                "error": "Missing measurement_type or data in request body",
                "correlation_id": correlation_id
            }), 400

        measurement_type = request_data['measurement_type']
        data = request_data['data']
        unit_system = UnitSystem.IMPERIAL if request_data.get('unit_system') == 'imperial' else UnitSystem.METRIC

        if measurement_type == 'point_to_point':
            p1 = Point3D.from_list(data['point1'])
            p2 = Point3D.from_list(data['point2'])
            result = MeasurementCalculator.point_to_point_distance(p1, p2, unit_system)

        elif measurement_type == 'edge':
            edge_info = EdgeInfo(
                edge_id=data.get('edge_id', 0),
                edge_type=data.get('edge_type', 'line'),
                start_point=data.get('start', [0, 0, 0]),
                end_point=data.get('end', [0, 0, 0]),
                length=data.get('length'),
                radius=data.get('radius'),
                diameter=data.get('diameter'),
                center=data.get('center')
            )
            result = MeasurementCalculator.edge_measurement(edge_info, unit_system)

        elif measurement_type == 'face_area':
            face_info = FaceInfo(
                face_id=data.get('face_id', 0),
                surface_type=data.get('surface_type', 'other'),
                center=data.get('center', [0, 0, 0]),
                normal=data.get('normal', [0, 0, 1]),
                area=data.get('area', 0),
                radius=data.get('radius')
            )
            result = MeasurementCalculator.face_area(face_info, unit_system)

        elif measurement_type == 'face_to_face':
            face1 = data['face1']
            face2 = data['face2']

            c1 = Point3D.from_list(face1['center'])
            c2 = Point3D.from_list(face2['center'])
            distance = c1.distance_to(c2)

            n1 = Vector3D(*face1['normal'])
            n2 = Vector3D(*face2['normal'])
            dot = abs(n1.dot(n2))
            angle = math.degrees(math.acos(min(1.0, max(-1.0, dot))))

            is_parallel = angle < 5 or angle > 175
            perp_distance = None
            if is_parallel:
                center_diff = Vector3D(c2.x - c1.x, c2.y - c1.y, c2.z - c1.z)
                perp_distance = abs(center_diff.dot(n1))

            display_value = UnitConverter.format_length(distance, unit_system)

            result = MeasurementResult(
                success=True,
                measurement_type="face_to_face",
                value=distance,
                unit="mm",
                label=f"Distance: {display_value}",
                display_value=f"{display_value} ({angle:.1f}\u00b0)",
                angle=angle,
                is_parallel=is_parallel,
                perpendicular_distance=perp_distance,
                point1=face1['center'],
                point2=face2['center']
            )
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported measurement type: {measurement_type}",
                "correlation_id": correlation_id
            }), 400

        return jsonify({
            "success": True,
            "correlation_id": correlation_id,
            **asdict(result)
        })

    except Exception as e:
        logger.error(f"[{correlation_id}] Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "correlation_id": correlation_id
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('MEASUREMENT_SERVICE_PORT', 8081))
    logger.info(f"Starting Vectis Measurement Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
