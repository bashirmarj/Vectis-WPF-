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
        'base_size_factor': 0.003,      # 0.3% of diagonal (slightly finer for smoother curves)
        'planar_factor': 3.0,           # 3x coarser on flat surfaces (efficient)
        'curvature_points': 64,         # 64 elements per 2π = ~5.6° between points (very smooth)
        'target_triangles': 40000,      # Target ~40k triangles (balanced)
        'sharp_edge_threshold': 30.0    # 30° angle for sharp edge detection
    },
    'balanced': {
        'base_size_factor': 0.0015,     # 0.15% of diagonal (fine detail)
        'planar_factor': 2.5,           # 2.5x coarser on flats
        'curvature_points': 80,         # 80 elements per 2π = ~4.5° between points (excellent)
        'target_triangles': 150000,     # Target ~150k triangles
        'sharp_edge_threshold': 25.0    # 25° angle for sharp edge detection
    },
    'ultra': {
        'base_size_factor': 0.0006,     # 0.06% of diagonal (very fine)
        'planar_factor': 2.0,           # Less coarsening
        'curvature_points': 96,         # 96 elements per 2π = ~3.75° between points (ultra smooth)
        'target_triangles': 500000,     # Target ~500k triangles
        'sharp_edge_threshold': 20.0    # 20° angle for sharp edge detection (preserve more detail)
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
            
            logger.info(f"📏 Model diagonal: {diagonal:.2f}mm, base mesh size: {base_size:.4f}mm")
            
            # Use Gmsh's built-in curvature-adaptive meshing (industry standard)
            # CRITICAL FIX: Calculate MeshSizeMax to GUARANTEE curvature_points is respected
            #
            # For a hole to have N segments, element size must be: (π * diameter) / N
            # Conservative: Ensure even 3mm holes get proper segmentation
            import math
            min_feature_diameter = 3.0  # mm (assume smallest holes are 3mm)
            required_element_size_for_curvature = (math.pi * min_feature_diameter) / preset['curvature_points']
            
            # Use the SMALLER of:
            # 1. What's needed for curvature refinement
            # 2. What's allowed by planar face sizing
            max_element_size = min(required_element_size_for_curvature, base_size * preset['planar_factor'])
            
            logger.info(f"📐 Curvature refinement: max_element={max_element_size:.4f}mm (ensures {preset['curvature_points']} segments on 3mm features)")
            
            # Set curvature refinement parameters
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", preset['curvature_points'])
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)  # Enable curvature-based sizing
            
            # Set mesh size constraints
            gmsh.option.setNumber("Mesh.MeshSizeMin", base_size * 0.05)  # Very fine for small features
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_element_size)  # Respects curvature target
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", base_size * 0.05)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_element_size)
            
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
            
            # Calculate per-vertex normals with sharp edge detection
            sharp_edge_threshold = preset.get('sharp_edge_threshold', 30.0)
            logger.info(f"🎨 Calculating normals with {sharp_edge_threshold}° sharp edge threshold...")
            normals = calculate_vertex_normals(vertices, indices, sharp_edge_threshold)
            
            gmsh.finalize()
            
            logger.info(f"✅ Generated {triangle_count} triangles ({len(vertices)//3} vertices)")
            logger.info(f"   └─ Smooth shading on curves, sharp edges preserved at {sharp_edge_threshold}°")
            
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
    Calculate vertex normals with angle-based sharp edge detection.
    
    This is the industry-standard approach used by SolidWorks, Fusion 360, and Onshape:
    - Smooth surfaces: Average normals when angle < threshold
    - Sharp edges: Don't average when angle > threshold
    
    Args:
        vertices: Flat list of vertex coordinates [x,y,z, x,y,z, ...]
        indices: Triangle indices
        sharp_edge_threshold: Angle in degrees (default 30°)
    
    Returns:
        Flat list of vertex normals
    """
    num_vertices = len(vertices) // 3
    threshold_rad = np.radians(sharp_edge_threshold)
    threshold_cos = np.cos(threshold_rad)
    
    # Step 1: Calculate face normals
    num_faces = len(indices) // 3
    face_normals = np.zeros((num_faces, 3))
    
    for i in range(num_faces):
        i1, i2, i3 = indices[i*3], indices[i*3+1], indices[i*3+2]
        
        v1 = np.array(vertices[i1*3:i1*3+3])
        v2 = np.array(vertices[i2*3:i2*3+3])
        v3 = np.array(vertices[i3*3:i3+3])
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal)
        
        if length > 1e-10:
            face_normals[i] = normal / length
        else:
            face_normals[i] = np.array([0, 0, 1])
    
    # Step 2: Build vertex-to-faces connectivity
    vertex_faces = [[] for _ in range(num_vertices)]
    for face_idx in range(num_faces):
        i1, i2, i3 = indices[face_idx*3], indices[face_idx*3+1], indices[face_idx*3+2]
        vertex_faces[i1].append(face_idx)
        vertex_faces[i2].append(face_idx)
        vertex_faces[i3].append(face_idx)
    
    # Step 3: Calculate vertex normals with sharp edge detection
    vertex_normals = np.zeros((num_vertices, 3))
    
    for v_idx in range(num_vertices):
        adjacent_faces = vertex_faces[v_idx]
        
        if not adjacent_faces:
            vertex_normals[v_idx] = np.array([0, 0, 1])
            continue
        
        # For each face, check if it should contribute to this vertex's normal
        accumulated_normal = np.zeros(3)
        
        for face_idx in adjacent_faces:
            face_normal = face_normals[face_idx]
            
            # Check angle with all other adjacent faces
            should_smooth = True
            for other_face_idx in adjacent_faces:
                if face_idx == other_face_idx:
                    continue
                
                other_normal = face_normals[other_face_idx]
                
                # Calculate angle between normals using dot product
                dot_product = np.dot(face_normal, other_normal)
                dot_product = np.clip(dot_product, -1.0, 1.0)  # Handle numerical errors
                
                # If angle > threshold, this is a sharp edge - don't smooth
                if dot_product < threshold_cos:
                    should_smooth = False
                    break
            
            if should_smooth:
                accumulated_normal += face_normal
            else:
                # Sharp edge: use face normal directly (no averaging)
                accumulated_normal = face_normal
                break
        
        # Normalize the accumulated normal
        length = np.linalg.norm(accumulated_normal)
        if length > 1e-10:
            vertex_normals[v_idx] = accumulated_normal / length
        else:
            vertex_normals[v_idx] = face_normals[adjacent_faces[0]]
    
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
        quality = request.form.get('quality', 'balanced')
        
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
