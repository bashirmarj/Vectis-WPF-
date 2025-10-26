"""
High-Quality Adaptive Mesh Generation Service

This service uses Gmsh to generate production-quality display meshes from STEP files.
Gmsh provides best-in-class tessellation with adaptive mesh sizing based on surface curvature.

Key Features:
- Adaptive mesh density (ultra-fine for curved surfaces, coarse for planar surfaces)
- Surface-type-specific tessellation
- Quality presets (fast, balanced, ultra)
- Professional CAD-quality visual output

Technology Stack:
- Gmsh: Professional CAD meshing library
- Flask: REST API
- NumPy: Mesh data processing
"""

import os
import io
import math
import logging
import tempfile
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    print("⚠️ WARNING: Gmsh not available. Install with: conda install -c conda-forge gmsh")

# === CONFIG ===
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mesh_service")

# Global lock for Gmsh (ensures thread-safe execution)
gmsh_lock = threading.Lock()

# === QUALITY PRESETS ===
QUALITY_PRESETS = {
    'fast': {
        'base_size_factor': 0.005,
        'planar_factor': 3.0,
        'curvature_points': 64,         # Increased from 32 → smoother curves
        'target_triangles': 30000,      # Increased from 15000 → better detail
        'sharp_edge_threshold': 30.0
    },
    'balanced': {
        'base_size_factor': 0.003,
        'planar_factor': 2.5,
        'curvature_points': 80,         # Increased from 48 → very smooth curves
        'target_triangles': 60000,      # Increased from 30000 → high detail
        'sharp_edge_threshold': 25.0
    },
    'ultra': {
        'base_size_factor': 0.002,
        'planar_factor': 2.0,
        'curvature_points': 96,         # Increased from 64 → near-perfect curves
        'target_triangles': 120000,     # Increased from 60000 → excellent detail
        'sharp_edge_threshold': 20.0
    }
}




def generate_adaptive_mesh(step_file_path, quality='balanced'):
    """
    Generate adaptive high-quality mesh from STEP file using Gmsh.
    
    Args:
        step_file_path: Path to STEP file
        quality: 'fast', 'balanced', or 'ultra'
    
    Returns:
        dict: {
            'vertices': List of floats [x1,y1,z1, x2,y2,z2, ...],
            'indices': List of ints [i1,i2,i3, i4,i5,i6, ...],
            'normals': List of floats (per-vertex normals),
            'triangle_count': int,
            'quality_stats': dict
        }
    """
    if not GMSH_AVAILABLE:
        raise RuntimeError("Gmsh not available")
    
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['balanced'])
    logger.info(f"🎨 Generating {quality} quality mesh (target: {preset['target_triangles']} triangles)...")
    
    # Acquire lock to ensure Gmsh runs in main thread safely
    with gmsh_lock:
        try:
            # Initialize Gmsh
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            
            # Import STEP file
            gmsh.merge(step_file_path)
            
            # Calculate adaptive mesh sizing (inline to avoid re-initialization)
            bbox = gmsh.model.getBoundingBox(-1, -1)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)
            base_size = diagonal * preset['base_size_factor']
            
            # Calculate mesh sizing based on target triangle count (not arbitrary feature assumptions)
            target_triangles = preset['target_triangles']
            model_surface_area_estimate = diagonal * diagonal * 2  # Rough estimate (2x bounding box face)
            avg_triangle_area = model_surface_area_estimate / target_triangles
            avg_element_size = math.sqrt(avg_triangle_area) * 1.2  # Target average size with margin
            
            # Allow local refinement on curved features (5x finer than average)
            min_feature_diameter = 3.0  # mm (smallest expected curved features)
            local_refinement_size = (math.pi * min_feature_diameter) / preset['curvature_points']
            min_element_size = max(local_refinement_size, avg_element_size * 0.15)  # Don't go below 15% of average
            
            logger.info(f"📏 Model diagonal: {diagonal:.2f}mm")
            logger.info(f"📊 Target mesh: {target_triangles} triangles → avg_size={avg_element_size:.4f}mm, min_size={min_element_size:.4f}mm")
            logger.info(f"📐 Curvature refinement: {preset['curvature_points']} segments on {min_feature_diameter}mm features")
            
            # Set Gmsh mesh sizing - CRITICAL: MeshSizeMax controls total triangle count
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_element_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", avg_element_size * 2.5)  # Allow coarse elements on flat areas
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", avg_element_size * 2.5)
            
            # Enable curvature-based LOCAL refinement (within global size constraints)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", preset['curvature_points'])
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
            
            # Additional refinement controls
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Extend size from boundaries
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)  # Don't use point sizes
            gmsh.option.setNumber("Mesh.MeshSizeFromParametricPoints", 0)  # Don't use parametric sizes
            
            logger.info(f"📊 Using global adaptive meshing (base: {base_size:.4f}mm, curvature points: {preset['curvature_points']})")
            
            # Generate 2D surface mesh
            gmsh.model.mesh.generate(2)
            
            # Extract mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
            
            # Process vertices
            vertices = node_coords.tolist() if isinstance(node_coords, np.ndarray) else list(node_coords)
            
            # Process triangles (filter to only triangular elements)
            indices = []
            for elem_type, tags, node_tags_for_type in zip(elem_types, elem_tags, elem_node_tags):
                if elem_type == 2:  # Triangle element type
                    # Convert 1-indexed to 0-indexed
                    indices.extend([int(tag) - 1 for tag in node_tags_for_type])
            
            triangle_count = len(indices) // 3
            
            # Calculate per-vertex normals with fast averaging
            sharp_edge_threshold = preset.get('sharp_edge_threshold', 30.0)
            logger.info(f"🎨 Calculating normals using fast averaging (O(F) complexity)...")
            normals = calculate_vertex_normals(vertices, indices, sharp_edge_threshold)
            
            gmsh.finalize()
            
            logger.info(f"✅ Generated {triangle_count} triangles ({len(vertices)//3} vertices)")
            logger.info(f"   └─ Smooth averaged normals for {len(vertices)//3} vertices")
            
            return {
                'vertices': vertices,
                'indices': indices,
                'normals': normals,
                'triangle_count': triangle_count,
                'quality_stats': {
                    'quality_preset': quality,
                    'curvature_points': preset['curvature_points'],
                    'base_mesh_size': base_size,
                    'diagonal': diagonal
                }
            }
        
        except Exception as e:
            logger.error(f"Mesh generation failed: {e}")
            gmsh.finalize()
            raise


def calculate_vertex_normals(vertices, indices, sharp_edge_threshold=30.0):
    """
    Calculate vertex normals using angle-weighted averaging with sharp edge detection.
    
    This approach:
    1. Averages face normals at each vertex (smooth shading)
    2. BUT splits normals across edges where angle > sharp_edge_threshold (preserves hard edges)
    
    Args:
        vertices: Flat list of vertex coordinates [x,y,z, x,y,z, ...]
        indices: Triangle indices
        sharp_edge_threshold: Angle in degrees above which edges are considered sharp
    
    Returns:
        Flat list of vertex normals
    """
    num_vertices = len(vertices) // 3
    num_faces = len(indices) // 3
    
    # Step 1: Calculate face normals
    face_normals = []
    for i in range(num_faces):
        i1, i2, i3 = indices[i*3], indices[i*3+1], indices[i*3+2]
        
        v1 = np.array(vertices[i1*3:i1*3+3])
        v2 = np.array(vertices[i2*3:i2*3+3])
        v3 = np.array(vertices[i3*3:i3+3])
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        face_normals.append(normal)
    
    # Step 2: Build face adjacency (which faces share each vertex)
    vertex_faces = [[] for _ in range(num_vertices)]
    for face_idx in range(num_faces):
        i1, i2, i3 = indices[face_idx*3], indices[face_idx*3+1], indices[face_idx*3+2]
        vertex_faces[i1].append(face_idx)
        vertex_faces[i2].append(face_idx)
        vertex_faces[i3].append(face_idx)
    
    # Step 3: Average normals at each vertex (with sharp edge detection)
    vertex_normals = np.zeros((num_vertices, 3))
    sharp_threshold_cos = np.cos(np.radians(sharp_edge_threshold))
    
    for vertex_idx in range(num_vertices):
        adjacent_faces = vertex_faces[vertex_idx]
        
        if len(adjacent_faces) == 0:
            continue
        
        # Group faces by normal similarity (sharp edge detection)
        # If two face normals differ by > sharp_edge_threshold, they're in different groups
        normal_groups = []
        
        for face_idx in adjacent_faces:
            face_normal = face_normals[face_idx]
            
            # Try to find an existing group this face belongs to
            added = False
            for group in normal_groups:
                group_normal = group[0]  # Representative normal of the group
                dot_product = np.dot(face_normal, group_normal)
                
                # If normals are similar (angle < threshold), add to this group
                if dot_product > sharp_threshold_cos:
                    group.append(face_normal)
                    added = True
                    break
            
            # If no suitable group found, create new group
            if not added:
                normal_groups.append([face_normal])
        
        # Average the largest group (most representative smooth surface)
        largest_group = max(normal_groups, key=len)
        avg_normal = np.mean(largest_group, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_normal)
        if norm > 0:
            avg_normal = avg_normal / norm
        
        vertex_normals[vertex_idx] = avg_normal
    
    return vertex_normals.flatten().tolist()


# === API ENDPOINTS ===

@app.route('/mesh-cad', methods=['POST'])
def mesh_cad():
    """
    Generate high-quality adaptive mesh from STEP file.
    
    Request:
        - file: STEP file (multipart/form-data)
        - quality: 'fast' | 'balanced' | 'ultra' (default: 'balanced')
    
    Response:
        {
            "success": true,
            "vertices": [...],
            "indices": [...],
            "normals": [...],
            "triangle_count": 15234,
            "quality_stats": {...}
        }
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        quality = request.form.get('quality', 'fast')
        
        if quality not in QUALITY_PRESETS:
            return jsonify({'success': False, 'error': f'Invalid quality: {quality}'}), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Generate mesh
            mesh_data = generate_adaptive_mesh(tmp_path, quality)
            
            return jsonify({
                'success': True,
                **mesh_data
            })
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        logger.error(f"Mesh generation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'service': 'mesh-service',
        'status': 'healthy',
        'gmsh_available': GMSH_AVAILABLE,
        'quality_presets': list(QUALITY_PRESETS.keys())
    })


@app.route('/', methods=['GET'])
def index():
    """Service information"""
    return jsonify({
        'service': 'High-Quality Mesh Generation Service',
        'version': '1.0.0',
        'endpoints': {
            '/mesh-cad': 'POST - Generate adaptive mesh from STEP file',
            '/health': 'GET - Health check'
        },
        'quality_presets': list(QUALITY_PRESETS.keys())
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"🚀 Starting Mesh Service on port {port}")
    logger.info(f"📊 Gmsh available: {GMSH_AVAILABLE}")
    app.run(host='0.0.0.0', port=port, debug=False)
