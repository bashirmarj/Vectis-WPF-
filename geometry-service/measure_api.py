"""
Standalone Measurement API for SolidWorks-style measurements.
All values returned in MILLIMETERS.

Endpoints:
  POST /api/measure - Perform measurement based on mode

Modes:
  - edge: Measure edge (returns length, diameter, radius)
  - point_to_point: Distance between two points
  - face_to_face: Distance between two faces
"""

import logging
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify
import math

# OpenCascade imports
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

logger = logging.getLogger(__name__)

# Create Blueprint for measure API
measure_blueprint = Blueprint('measure', __name__)


@measure_blueprint.route('/measure', methods=['POST'])
def measure():
    """
    Perform measurement based on mode.
    
    Request JSON:
    {
        "mode": "edge" | "point_to_point" | "face_to_face",
        "edge_id": int (for edge mode),
        "point1": [x, y, z] (for point_to_point),
        "point2": [x, y, z] (for point_to_point),
        "face1_id": int (for face_to_face),
        "face2_id": int (for face_to_face)
    }
    
    Response JSON:
    {
        "success": bool,
        "measurement_type": str,
        "value_mm": float,
        "label": str,
        "metadata": {}
    }
    """
    try:
        data = request.get_json()
        mode = data.get('mode', 'edge')
        
        if mode == 'edge':
            return measure_edge(data)
        elif mode == 'point_to_point':
            return measure_point_to_point(data)
        elif mode == 'face_to_face':
            return measure_face_to_face(data)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown mode: {mode}'
            }), 400
            
    except Exception as e:
        logger.error(f"Measurement error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def measure_edge(data: Dict) -> tuple:
    """
    Measure edge properties from edge_id.
    Returns length, diameter, radius depending on edge type.
    All values in MM.
    """
    edge_id = data.get('edge_id')
    
    # For now, use pre-computed edge data from tagged_edges
    # This data is already extracted during /analyze
    tagged_edge = data.get('edge_data')
    
    if not tagged_edge:
        return jsonify({
            'success': False,
            'error': 'No edge data provided. Use tagged_edges from /analyze response.'
        }), 400
    
    edge_type = tagged_edge.get('type', 'unknown')
    
    result = {
        'success': True,
        'measurement_type': 'edge',
        'edge_type': edge_type,
        'metadata': {}
    }
    
    # All values already in MM from backend
    if edge_type == 'circle':
        if tagged_edge.get('is_full_circle'):
            # Full circle - return diameter
            diameter = tagged_edge.get('diameter', 0)
            result['value_mm'] = diameter
            result['label'] = f'Ø {diameter:.2f} mm'
            result['metadata']['is_full_circle'] = True
            result['metadata']['radius_mm'] = tagged_edge.get('radius', diameter / 2)
        else:
            # Arc - return radius and arc length
            radius = tagged_edge.get('radius', 0)
            arc_length = tagged_edge.get('arc_length', 0)
            result['value_mm'] = radius
            result['label'] = f'R {radius:.2f} mm'
            if arc_length:
                result['label'] += f' | Arc: {arc_length:.2f} mm'
            result['metadata']['is_full_circle'] = False
            result['metadata']['arc_length_mm'] = arc_length
            
    elif edge_type == 'line':
        length = tagged_edge.get('length', 0)
        result['value_mm'] = length
        result['label'] = f'{length:.2f} mm'
        
        # Include XYZ deltas
        start = tagged_edge.get('start', [0, 0, 0])
        end = tagged_edge.get('end', [0, 0, 0])
        result['metadata']['delta_x_mm'] = abs(end[0] - start[0])
        result['metadata']['delta_y_mm'] = abs(end[1] - start[1])
        result['metadata']['delta_z_mm'] = abs(end[2] - start[2])
        
    elif edge_type == 'ellipse':
        major = tagged_edge.get('major_radius', 0)
        minor = tagged_edge.get('minor_radius', 0)
        result['value_mm'] = major
        result['label'] = f'Ellipse: {major:.2f} × {minor:.2f} mm'
        result['metadata']['major_radius_mm'] = major
        result['metadata']['minor_radius_mm'] = minor
        
    else:
        # Generic edge - return length
        length = tagged_edge.get('length', 0)
        result['value_mm'] = length
        result['label'] = f'{length:.2f} mm'
    
    # Add center point if available
    if 'center' in tagged_edge:
        result['metadata']['center_mm'] = tagged_edge['center']
    
    return jsonify(result), 200


def measure_point_to_point(data: Dict) -> tuple:
    """
    Calculate distance between two points.
    Points expected in MM.
    """
    point1 = data.get('point1')
    point2 = data.get('point2')
    
    if not point1 or not point2:
        return jsonify({
            'success': False,
            'error': 'point1 and point2 required'
        }), 400
    
    # Calculate distance (all in MM)
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    return jsonify({
        'success': True,
        'measurement_type': 'point_to_point',
        'value_mm': round(distance, 4),
        'label': f'{distance:.2f} mm',
        'metadata': {
            'delta_x_mm': round(abs(dx), 4),
            'delta_y_mm': round(abs(dy), 4),
            'delta_z_mm': round(abs(dz), 4),
            'point1_mm': point1,
            'point2_mm': point2
        }
    }), 200


def measure_face_to_face(data: Dict) -> tuple:
    """
    Calculate minimum distance between two faces.
    Uses face data from /analyze response.
    """
    face1_data = data.get('face1_data')
    face2_data = data.get('face2_data')
    
    if not face1_data or not face2_data:
        return jsonify({
            'success': False,
            'error': 'face1_data and face2_data required'
        }), 400
    
    # For parallel planar faces, calculate perpendicular distance
    # This is a simplified implementation
    # Full implementation would use BRepExtrema_DistShapeShape
    
    normal1 = face1_data.get('normal', [0, 0, 1])
    normal2 = face2_data.get('normal', [0, 0, -1])
    center1 = face1_data.get('center', [0, 0, 0])
    center2 = face2_data.get('center', [0, 0, 0])
    
    # Check if faces are parallel (normals are same or opposite direction)
    dot = abs(normal1[0]*normal2[0] + normal1[1]*normal2[1] + normal1[2]*normal2[2])
    is_parallel = dot > 0.99  # ~8 degrees tolerance
    
    if is_parallel:
        # Project distance along normal
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        dz = center2[2] - center1[2]
        distance = abs(dx*normal1[0] + dy*normal1[1] + dz*normal1[2])
        
        return jsonify({
            'success': True,
            'measurement_type': 'face_to_face',
            'value_mm': round(distance, 4),
            'label': f'{distance:.2f} mm (parallel)',
            'metadata': {
                'is_parallel': True,
                'perpendicular_distance_mm': round(distance, 4)
            }
        }), 200
    else:
        # Non-parallel faces - return center-to-center distance
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        dz = center2[2] - center1[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Calculate angle between faces
        angle_rad = math.acos(min(1.0, max(-1.0, dot)))
        angle_deg = math.degrees(angle_rad)
        
        return jsonify({
            'success': True,
            'measurement_type': 'face_to_face',
            'value_mm': round(distance, 4),
            'label': f'{distance:.2f} mm ({angle_deg:.1f}°)',
            'metadata': {
                'is_parallel': False,
                'center_distance_mm': round(distance, 4),
                'angle_deg': round(angle_deg, 2)
            }
        }), 200
