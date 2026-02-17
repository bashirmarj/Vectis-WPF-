"""
VTK Face-Level Tessellator

Converts OCP/OpenCascade shapes to VTK polydata with per-cell face IDs.
This enables:
- Per-face coloring via VTK lookup tables
- Face picking via VTK cell picker
- Feature highlighting in the 3D viewer

Adapted from app.py tessellate_shape() function.
"""

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdTypeArray, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray

# Use OCP directly (not OCC shim) for the local viewer code
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location


def tessellate_to_vtk(shape, linear_deflection=0.005, angular_deflection=12.0):
    """
    Tessellate OCP shape into VTK polydata with face-level cell data.

    Args:
        shape: OCP TopoDS_Shape
        linear_deflection: Linear tessellation tolerance in meters (0.005 = 5mm)
        angular_deflection: Angular tessellation tolerance in degrees

    Returns:
        vtkPolyData with:
            - Points (vertices)
            - Polys (triangles)
            - CellData "face_id" array (int per triangle)
            - CellData "feature_type" array (string per triangle, initially "unrecognized")
    """
    # Perform tessellation
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()

    if not mesh.IsDone():
        raise ValueError("Tessellation failed")

    # VTK data structures
    vtk_points = vtkPoints()
    vtk_cells = vtkCellArray()
    face_id_array = vtkIntArray()
    face_id_array.SetName("face_id")
    face_id_array.SetNumberOfComponents(1)

    # Track vertices for deduplication
    vertex_map = {}  # {(x,y,z): vtk_point_id}
    global_vertex_index = 0
    global_triangle_count = 0

    # Iterate over faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0

    while face_explorer.More():
        face_shape = face_explorer.Current()
        # Cast to TopoDS_Face (OCP requires explicit type)
        from OCP.TopoDS import TopoDS
        face = TopoDS.Face_s(face_shape)

        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is not None:
            transformation = location.Transformation()

            # Extract vertices for this face
            face_vertex_map = {}  # {local_node_id: global_vtk_id}

            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(transformation)

                # Round to avoid floating point precision issues
                coord = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))

                if coord not in vertex_map:
                    vertex_map[coord] = global_vertex_index
                    vtk_points.InsertNextPoint(pnt.X(), pnt.Y(), pnt.Z())
                    global_vertex_index += 1

                face_vertex_map[i] = vertex_map[coord]

            # Extract triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()

                # Respect face orientation (TopAbs_REVERSED = 1)
                if face.Orientation() == 1:
                    # Reversed orientation - flip winding order
                    vtk_triangle = [
                        face_vertex_map[n1],
                        face_vertex_map[n3],
                        face_vertex_map[n2]
                    ]
                else:
                    vtk_triangle = [
                        face_vertex_map[n1],
                        face_vertex_map[n2],
                        face_vertex_map[n3]
                    ]

                # Add triangle to VTK cell array
                vtk_cells.InsertNextCell(3, vtk_triangle)

                # Tag this triangle with the face_id
                face_id_array.InsertNextValue(face_id)
                global_triangle_count += 1

        face_id += 1
        face_explorer.Next()

    # Build VTK polydata
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    polydata.GetCellData().AddArray(face_id_array)
    polydata.GetCellData().SetActiveScalars("face_id")

    print(f"Tessellation: {vtk_points.GetNumberOfPoints()} vertices, "
          f"{global_triangle_count} triangles, {face_id} faces")

    return polydata


def compute_face_bounds(polydata, face_id):
    """
    Compute bounding box for all triangles belonging to a specific face.

    Args:
        polydata: vtkPolyData with face_id cell data
        face_id: Face ID to compute bounds for

    Returns:
        (xmin, xmax, ymin, ymax, zmin, zmax) or None if face not found
    """
    face_id_array = polydata.GetCellData().GetArray("face_id")
    if not face_id_array:
        return None

    points = polydata.GetPoints()
    cells = polydata.GetPolys()

    # Find all cells (triangles) belonging to this face
    bounds = [float('inf'), float('-inf')] * 3  # [xmin, xmax, ymin, ymax, zmin, zmax]
    found = False

    for cell_id in range(polydata.GetNumberOfCells()):
        if face_id_array.GetValue(cell_id) == face_id:
            found = True
            cell = polydata.GetCell(cell_id)
            for i in range(cell.GetNumberOfPoints()):
                pt = points.GetPoint(cell.GetPointId(i))
                bounds[0] = min(bounds[0], pt[0])  # xmin
                bounds[1] = max(bounds[1], pt[0])  # xmax
                bounds[2] = min(bounds[2], pt[1])  # ymin
                bounds[3] = max(bounds[3], pt[1])  # ymax
                bounds[4] = min(bounds[4], pt[2])  # zmin
                bounds[5] = max(bounds[5], pt[2])  # zmax

    return tuple(bounds) if found else None


def get_face_center(polydata, face_id):
    """
    Compute centroid of all triangles belonging to a specific face.

    Args:
        polydata: vtkPolyData with face_id cell data
        face_id: Face ID to compute center for

    Returns:
        (x, y, z) or None if face not found
    """
    bounds = compute_face_bounds(polydata, face_id)
    if not bounds:
        return None

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2
    )


def tessellate_face_to_vtk(face, linear_deflection=0.005, angular_deflection=12.0):
    """
    Tessellate a single OCP face into VTK polydata.
    Used for face highlighting.

    Args:
        face: OCP TopoDS_Face
        linear_deflection: Linear tessellation tolerance
        angular_deflection: Angular tessellation tolerance in degrees

    Returns:
        vtkPolyData containing just this face's triangulation
    """
    # VTK data structures
    vtk_points = vtkPoints()
    vtk_cells = vtkCellArray()

    # Get existing triangulation (should already exist from main tessellation)
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)

    if triangulation is None:
        # If no triangulation exists, create one
        from OCP.TopoDS import TopoDS_Shape
        shape = TopoDS_Shape(face)
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is None:
            raise ValueError("No triangulation available for face")

    transformation = location.Transformation()

    # Extract vertices
    vertex_map = {}  # {local_node_id: global_vtk_id}
    for i in range(1, triangulation.NbNodes() + 1):
        pnt = triangulation.Node(i)
        pnt.Transform(transformation)
        vertex_map[i] = vtk_points.InsertNextPoint(pnt.X(), pnt.Y(), pnt.Z())

    # Extract triangles
    for i in range(1, triangulation.NbTriangles() + 1):
        triangle = triangulation.Triangle(i)
        n1, n2, n3 = triangle.Get()

        # Respect face orientation
        if face.Orientation() == 1:
            vtk_triangle = [vertex_map[n1], vertex_map[n3], vertex_map[n2]]
        else:
            vtk_triangle = [vertex_map[n1], vertex_map[n2], vertex_map[n3]]

        vtk_cells.InsertNextCell(3, vtk_triangle)

    # Build VTK polydata
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    return polydata
