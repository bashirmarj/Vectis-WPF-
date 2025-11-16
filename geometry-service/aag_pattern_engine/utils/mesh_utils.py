"""Mesh tessellation and processing utilities"""

import numpy as np
from typing import Dict, List, Tuple
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location

def tessellate_shape(
    shape,
    linear_deflection: float = 0.005,
    angular_deflection: float = 12.0
) -> Dict:
    """
    Tessellate OpenCascade shape into triangle mesh with face mapping
    
    Args:
        shape: OpenCascade TopoDS_Shape
        linear_deflection: Linear deflection in meters (0.001 = 1mm)
        angular_deflection: Angular deflection in degrees (12° standard)
    
    Returns:
        Dict with vertices, indices, normals, and face_mapping
    """
    # Perform tessellation
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    
    if not mesh.IsDone():
        raise ValueError("Tessellation failed")
    
    vertices = []
    indices = []
    face_mapping = {}
    vertex_map = {}  # Deduplication map
    
    global_vertex_index = 0
    global_triangle_index = 0
    
    # Iterate through faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    
    while face_explorer.More():
        face = face_explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is not None:
            face_triangle_start = global_triangle_index
            transformation = location.Transformation()
            
            # Extract vertices for this face
            face_vertex_map = {}
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(transformation)
                
                # Round coordinates for deduplication
                coord = (
                    round(pnt.X(), 6),
                    round(pnt.Y(), 6),
                    round(pnt.Z(), 6)
                )
                
                if coord not in vertex_map:
                    vertex_map[coord] = global_vertex_index
                    vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])
                    global_vertex_index += 1
                
                face_vertex_map[i] = vertex_map[coord]
            
            # Extract triangles
            face_triangles = []
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                
                # Respect face orientation
                if face.Orientation() == 1:  # REVERSED
                    indices.extend([
                        face_vertex_map[n1],
                        face_vertex_map[n3],
                        face_vertex_map[n2]
                    ])
                else:
                    indices.extend([
                        face_vertex_map[n1],
                        face_vertex_map[n2],
                        face_vertex_map[n3]
                    ])
                
                face_triangles.append(global_triangle_index)
                global_triangle_index += 1
            
            # Store face mapping
            face_mapping[face_id] = {
                "triangle_indices": face_triangles,
                "triangle_range": [face_triangle_start, global_triangle_index - 1]
            }
        
        face_id += 1
        face_explorer.Next()
    
    # Compute per-vertex normals
    normals = compute_vertex_normals(vertices, indices, len(vertices) // 3)
    
    return {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "face_mapping": face_mapping,
        "face_count": face_id,
        "triangle_count": len(indices) // 3,
        "vertex_count": len(vertices) // 3
    }

def compute_vertex_normals(
    vertices: List[float],
    indices: List[int],
    vertex_count: int
) -> List[float]:
    """
    Compute smooth vertex normals from triangle data
    
    Args:
        vertices: Flat list of vertex coordinates [x1,y1,z1, x2,y2,z2, ...]
        indices: Triangle indices
        vertex_count: Number of vertices
    
    Returns:
        Flat list of vertex normals [nx1,ny1,nz1, nx2,ny2,nz2, ...]
    """
    normals = np.zeros((vertex_count, 3), dtype=np.float32)
    
    # Accumulate face normals at each vertex
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
        
        v0 = np.array(vertices[i0*3:i0*3+3])
        v1 = np.array(vertices[i1*3:i1*3+3])
        v2 = np.array(vertices[i2*3:i2*3+3])
        
        # Compute face normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Accumulate at each vertex
        normals[i0] += normal
        normals[i1] += normal
        normals[i2] += normal
    
    # Normalize all normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms
    
    return normals.flatten().tolist()
