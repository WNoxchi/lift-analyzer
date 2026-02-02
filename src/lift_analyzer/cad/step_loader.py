"""STEP file loader and mesh conversion utilities."""

from pathlib import Path

import numpy as np
import pyvista as pv
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Shape, topods


def load_step(filepath: str | Path) -> TopoDS_Shape:
    """Load a STEP file and return the TopoDS_Shape.

    Parameters
    ----------
    filepath : str | Path
        Path to the STEP file.

    Returns
    -------
    TopoDS_Shape
        The loaded CAD geometry.

    Raises
    ------
    FileNotFoundError
        If the STEP file does not exist.
    ValueError
        If the STEP file cannot be read or contains no shapes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"STEP file not found: {filepath}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(filepath))

    if status != 1:  # IFSelect_RetDone
        raise ValueError(f"Failed to read STEP file: {filepath}")

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise ValueError(f"No shapes found in STEP file: {filepath}")

    return shape


def shape_to_mesh(shape: TopoDS_Shape, deflection: float = 0.1) -> pv.PolyData:
    """Convert a TopoDS_Shape to a PyVista PolyData mesh.

    Parameters
    ----------
    shape : TopoDS_Shape
        The CAD geometry to convert.
    deflection : float, optional
        Mesh deflection parameter controlling tessellation quality.
        Smaller values produce finer meshes. Default is 0.1.

    Returns
    -------
    pv.PolyData
        The triangulated mesh suitable for visualization.
    """
    # Tessellate the shape
    mesh = BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    mesh.Perform()

    vertices = []
    faces = []
    vertex_offset = 0

    # Extract triangles from each face
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is not None:
            # Get transformation matrix
            transform = location.Transformation()

            # Extract vertices
            num_nodes = triangulation.NbNodes()
            for i in range(1, num_nodes + 1):
                node = triangulation.Node(i)
                # Apply transformation
                node_transformed = node.Transformed(transform)
                vertices.append([node_transformed.X(), node_transformed.Y(), node_transformed.Z()])

            # Extract triangles
            num_triangles = triangulation.NbTriangles()
            for i in range(1, num_triangles + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                # Adjust indices for vertex offset and convert to 0-based
                faces.append([3, vertex_offset + n1 - 1, vertex_offset + n2 - 1, vertex_offset + n3 - 1])

            vertex_offset += num_nodes

        explorer.Next()

    if not vertices:
        raise ValueError("No triangulation data could be extracted from shape")

    # Create PyVista mesh
    vertices_array = np.array(vertices, dtype=np.float64)
    faces_array = np.array(faces, dtype=np.int64).flatten()

    return pv.PolyData(vertices_array, faces_array)


def load_step_as_mesh(filepath: str | Path, deflection: float = 0.1) -> pv.PolyData:
    """Convenience function to load a STEP file directly as a PyVista mesh.

    Parameters
    ----------
    filepath : str | Path
        Path to the STEP file.
    deflection : float, optional
        Mesh deflection parameter. Default is 0.1.

    Returns
    -------
    pv.PolyData
        The triangulated mesh.
    """
    shape = load_step(filepath)
    return shape_to_mesh(shape, deflection)
