"""Center of Lift calculation using geometric projected area method.

This is a simplified approach that estimates the center of lift (center of pressure)
by computing the area-weighted centroid of surfaces facing the airflow.

For accurate aerodynamic analysis, use a proper VLM or panel method instead.
"""

from enum import Enum
from typing import NamedTuple

import numpy as np
import pyvista as pv


class AirflowDirection(Enum):
    """Predefined airflow directions.

    The direction indicates where the air is flowing TO (not from).
    For an aircraft flying forward with nose pointing -Y, airflow is -Y to +Y.
    """
    POSITIVE_X = (1.0, 0.0, 0.0)
    NEGATIVE_X = (-1.0, 0.0, 0.0)
    POSITIVE_Y = (0.0, 1.0, 0.0)   # Default: aircraft nose at -Y, air flows to +Y
    NEGATIVE_Y = (0.0, -1.0, 0.0)
    POSITIVE_Z = (0.0, 0.0, 1.0)
    NEGATIVE_Z = (0.0, 0.0, -1.0)


class CenterOfLiftResult(NamedTuple):
    """Result of center of lift calculation."""
    position: np.ndarray        # 3D position [x, y, z]
    y_coordinate: float         # Y-axis position (primary output for longitudinal CoL)
    total_projected_area: float # Total area facing the flow
    num_faces_considered: int   # Number of mesh faces that faced the flow


def calculate_center_of_lift(
    mesh: pv.PolyData,
    airflow_direction: AirflowDirection | tuple[float, float, float] = AirflowDirection.POSITIVE_Y,
) -> CenterOfLiftResult:
    """Calculate the center of lift using geometric projected area method.

    This method computes the area-weighted centroid of all mesh faces that
    face into the airflow. It's a geometric approximation that assumes
    uniform pressure on forward-facing surfaces.

    Parameters
    ----------
    mesh : pv.PolyData
        The triangulated mesh to analyze.
    airflow_direction : AirflowDirection or tuple
        Direction the air is flowing TO. Default is POSITIVE_Y, meaning
        the aircraft nose points toward -Y and air flows from -Y to +Y.

    Returns
    -------
    CenterOfLiftResult
        Named tuple containing:
        - position: 3D center of lift coordinates
        - y_coordinate: Y-axis position (longitudinal CoL)
        - total_projected_area: Sum of projected areas
        - num_faces_considered: Number of faces facing the flow

    Notes
    -----
    This is a simplified geometric method. For accurate results, use a
    Vortex Lattice Method (VLM) or panel method.
    """
    # Normalize airflow direction
    if isinstance(airflow_direction, AirflowDirection):
        flow_dir = np.array(airflow_direction.value, dtype=np.float64)
    else:
        flow_dir = np.array(airflow_direction, dtype=np.float64)
    flow_dir = flow_dir / np.linalg.norm(flow_dir)

    # Compute face normals and centers
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)

    # Extract face data
    face_normals = mesh.cell_data["Normals"]  # Shape: (n_faces, 3)
    face_centers = mesh.cell_centers().points   # Shape: (n_faces, 3)

    # Calculate face areas
    sized_mesh = mesh.compute_cell_sizes(area=True)
    face_areas = sized_mesh.cell_data["Area"]  # Shape: (n_faces,)

    # Find faces facing into the airflow
    # A face faces the flow if its normal points opposite to the flow direction
    # i.e., normal · flow_direction < 0
    dot_products = np.dot(face_normals, flow_dir)
    facing_flow_mask = dot_products < 0

    if not np.any(facing_flow_mask):
        # No faces facing the flow - return mesh centroid as fallback
        centroid = mesh.center
        return CenterOfLiftResult(
            position=np.array(centroid),
            y_coordinate=centroid[1],
            total_projected_area=0.0,
            num_faces_considered=0,
        )

    # Calculate projected area for each face
    # Projected area = face_area * |cos(angle)| = face_area * |normal · flow_dir|
    projected_areas = face_areas * np.abs(dot_products)

    # Apply mask to only consider faces facing the flow
    projected_areas_facing = projected_areas[facing_flow_mask]
    centers_facing = face_centers[facing_flow_mask]

    # Calculate area-weighted centroid
    total_projected_area = np.sum(projected_areas_facing)
    weighted_center = np.sum(
        centers_facing * projected_areas_facing[:, np.newaxis],
        axis=0
    ) / total_projected_area

    return CenterOfLiftResult(
        position=weighted_center,
        y_coordinate=weighted_center[1],
        total_projected_area=total_projected_area,
        num_faces_considered=int(np.sum(facing_flow_mask)),
    )
