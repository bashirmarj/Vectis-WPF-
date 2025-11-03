"""
Minimal OCCWL stub for graph building
Simplified implementation based on References code
"""
import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomLProp import GeomLProp_SLProps, GeomLProp_CLProps
from OCC.Core.BRepTools import breptools
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.gp import gp_Pnt
import networkx as nx


class Solid:
    """Wrapper for OCC solid shape"""
    def __init__(self, shape):
        self.shape = shape


class Face:
    """Wrapper for OCC face"""
    def __init__(self, face):
        self.face = face
        self.adaptor = BRepAdaptor_Surface(face)
    
    def surface_type(self):
        return self.adaptor.GetType()


class Edge:
    """Wrapper for OCC edge"""
    def __init__(self, edge):
        self.edge = edge
        self.adaptor = BRepAdaptor_Curve(edge)
    
    def has_curve(self):
        try:
            return self.adaptor.GetType() is not None
        except:
            return False


def face_adjacency(solid):
    """Build face adjacency graph"""
    shape = solid.shape
    
    # Build edge-face map
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    from OCC.Core.TopExp import topexp
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    # Extract faces
    faces = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        faces.append(face_explorer.Current())
        face_explorer.Next()
    
    # Build graph
    G = nx.Graph()
    
    # Add nodes (faces)
    for i, face in enumerate(faces):
        G.add_node(i, face=Face(face))
    
    # Add edges (adjacencies)
    edge_list = []
    for i in range(1, edge_face_map.Size() + 1):
        face_list = edge_face_map.FindFromIndex(i)
        if face_list.Size() == 2:
            # Get the edge
            edge_shape = edge_face_map.FindKey(i)
            edge_obj = Edge(edge_shape)
            
            # Find face indices
            face1 = face_list.First()
            face2 = face_list.Last()
            
            idx1 = None
            idx2 = None
            for j, f in enumerate(faces):
                if f.IsEqual(face1):
                    idx1 = j
                if f.IsEqual(face2):
                    idx2 = j
            
            if idx1 is not None and idx2 is not None:
                edge_list.append((idx1, idx2))
                G.add_edge(idx1, idx2, edge=edge_obj)
    
    return G


def uvgrid(face, method="point", num_u=10, num_v=10):
    """Sample UV grid on face surface"""
    adaptor = face.adaptor
    
    try:
        u_min = adaptor.FirstUParameter()
        u_max = adaptor.LastUParameter()
        v_min = adaptor.FirstVParameter()
        v_max = adaptor.LastVParameter()
    except:
        # Default parameters if bounds not available
        u_min, u_max = 0, 1
        v_min, v_max = 0, 1
    
    u_values = np.linspace(u_min, u_max, num_u)
    v_values = np.linspace(v_min, v_max, num_v)
    
    result = []
    
    for u in u_values:
        row = []
        for v in v_values:
            try:
                if method == "point":
                    pnt = adaptor.Value(u, v)
                    row.append([pnt.X(), pnt.Y(), pnt.Z()])
                
                elif method == "normal":
                    props = GeomLProp_SLProps(adaptor, u, v, 1, 1e-6)
                    if props.IsNormalDefined():
                        normal = props.Normal()
                        row.append([normal.X(), normal.Y(), normal.Z()])
                    else:
                        row.append([0, 0, 1])
                
                elif method == "visibility_status":
                    # Simplified: assume all points are inside/on boundary
                    row.append([0])  # 0 = inside
                
                else:
                    row.append([0, 0, 0])
            
            except:
                # Fallback for invalid UV parameters
                if method == "point":
                    row.append([0, 0, 0])
                elif method == "normal":
                    row.append([0, 0, 1])
                else:
                    row.append([0])
        
        result.append(row)
    
    return np.array(result)


def ugrid(edge, method="point", num_u=10):
    """Sample U grid on edge curve"""
    adaptor = edge.adaptor
    
    try:
        u_min = adaptor.FirstParameter()
        u_max = adaptor.LastParameter()
    except:
        u_min, u_max = 0, 1
    
    u_values = np.linspace(u_min, u_max, num_u)
    result = []
    
    for u in u_values:
        try:
            if method == "point":
                pnt = adaptor.Value(u)
                result.append([pnt.X(), pnt.Y(), pnt.Z()])
            
            elif method == "tangent":
                props = GeomLProp_CLProps(adaptor, u, 1, 1e-6)
                if props.IsTangentDefined():
                    tangent = props.Tangent()
                    result.append([tangent.X(), tangent.Y(), tangent.Z()])
                else:
                    result.append([1, 0, 0])
            
            else:
                result.append([0, 0, 0])
        
        except:
            if method == "point":
                result.append([0, 0, 0])
            elif method == "tangent":
                result.append([1, 0, 0])
            else:
                result.append([0, 0, 0])
    
    return np.array(result)
