"""Numba-accelerated panel method influence calculations.

This module provides JIT-compiled versions of the computationally intensive
influence coefficient calculations.

The hot loops in the panel method are O(n^2) where n is the number of panels.
Numba provides 10-100x speedup on these nested loops compared to pure Python.

Numba is a hard requirement - without it the panel method is too slow to be useful.
"""
# ruff: noqa: N803, N806  # Physics naming conventions (Cp, V, A matrices)

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(cache=True)
def _compute_local_coords_numba(
    vertices: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute local coordinate system for a panel (Numba-accelerated).

    Parameters
    ----------
    vertices : ndarray, shape (3, 3)
        Vertices of the triangular panel.
    normal : ndarray, shape (3,)
        Unit normal vector of the panel.

    Returns
    -------
    l_vec, m_vec, n_vec : tuple of ndarray
        Local coordinate axes.
    """
    # l-axis: along first edge
    edge1 = vertices[1] - vertices[0]
    edge1_norm = np.sqrt(edge1[0]**2 + edge1[1]**2 + edge1[2]**2)

    if edge1_norm < 1e-12:
        edge1 = vertices[2] - vertices[0]
        edge1_norm = np.sqrt(edge1[0]**2 + edge1[1]**2 + edge1[2]**2)
        if edge1_norm < 1e-12:
            l_vec = np.array([1.0, 0.0, 0.0])
            m_vec = np.array([0.0, 1.0, 0.0])
            return l_vec, m_vec, normal.copy()

    l_vec = edge1 / edge1_norm
    n_vec = normal.copy()

    # m-axis: perpendicular to both
    m_vec = np.array([
        n_vec[1] * l_vec[2] - n_vec[2] * l_vec[1],
        n_vec[2] * l_vec[0] - n_vec[0] * l_vec[2],
        n_vec[0] * l_vec[1] - n_vec[1] * l_vec[0],
    ])
    m_norm = np.sqrt(m_vec[0]**2 + m_vec[1]**2 + m_vec[2]**2)

    if m_norm < 1e-12:
        if abs(n_vec[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        m_vec = np.array([
            n_vec[1] * perp[2] - n_vec[2] * perp[1],
            n_vec[2] * perp[0] - n_vec[0] * perp[2],
            n_vec[0] * perp[1] - n_vec[1] * perp[0],
        ])
        m_norm = np.sqrt(m_vec[0]**2 + m_vec[1]**2 + m_vec[2]**2)

    m_vec = m_vec / m_norm

    return l_vec, m_vec, n_vec


@njit(cache=True)
def _point_to_local_numba(
    point: NDArray[np.float64],
    origin: NDArray[np.float64],
    l_vec: NDArray[np.float64],
    m_vec: NDArray[np.float64],
    n_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform a point to local panel coordinates (Numba-accelerated).

    Parameters
    ----------
    point : ndarray, shape (3,)
        Point in global coordinates.
    origin : ndarray, shape (3,)
        Origin of local system.
    l_vec, m_vec, n_vec : ndarray, shape (3,)
        Local coordinate axes.

    Returns
    -------
    local_point : ndarray, shape (3,)
        Point in local coordinates.
    """
    rel = point - origin
    return np.array([
        rel[0] * l_vec[0] + rel[1] * l_vec[1] + rel[2] * l_vec[2],
        rel[0] * m_vec[0] + rel[1] * m_vec[1] + rel[2] * m_vec[2],
        rel[0] * n_vec[0] + rel[1] * n_vec[1] + rel[2] * n_vec[2],
    ])


@njit(cache=True)
def source_panel_influence_numba(
    panel_vertices: NDArray[np.float64],
    panel_normal: NDArray[np.float64],
    panel_centroid: NDArray[np.float64],
    field_point: NDArray[np.float64],
    sigma: float = 1.0,
) -> float:
    """Calculate potential induced by a constant-strength source panel (Numba-accelerated).

    Parameters
    ----------
    panel_vertices : ndarray, shape (3, 3)
        Vertices of the triangular panel.
    panel_normal : ndarray, shape (3,)
        Unit outward normal.
    panel_centroid : ndarray, shape (3,)
        Panel center point.
    field_point : ndarray, shape (3,)
        Point where potential is evaluated.
    sigma : float
        Source strength per unit area.

    Returns
    -------
    phi : float
        Induced velocity potential.
    """
    l_vec, m_vec, n_vec = _compute_local_coords_numba(panel_vertices, panel_normal)

    # Transform vertices to local coordinates
    local_verts = np.zeros((3, 3))
    for i in range(3):
        local_verts[i] = _point_to_local_numba(
            panel_vertices[i], panel_centroid, l_vec, m_vec, n_vec
        )

    local_p = _point_to_local_numba(field_point, panel_centroid, l_vec, m_vec, n_vec)
    xi, eta, zeta = local_p[0], local_p[1], local_p[2]

    # Numerical stability for points in panel plane
    if abs(zeta) < 1e-10:
        if zeta >= 0:
            zeta = 1e-10
        else:
            zeta = -1e-10

    phi = 0.0
    n_edges = 3

    for i in range(n_edges):
        x1, y1 = local_verts[i, 0], local_verts[i, 1]
        j = (i + 1) % n_edges
        x2, y2 = local_verts[j, 0], local_verts[j, 1]

        dx = x2 - x1
        dy = y2 - y1
        d = np.sqrt(dx**2 + dy**2)

        if d < 1e-12:
            continue

        s1 = ((xi - x1) * dx + (eta - y1) * dy) / d
        s2 = ((xi - x2) * dx + (eta - y2) * dy) / d
        h = ((xi - x1) * dy - (eta - y1) * dx) / d

        r1 = np.sqrt((xi - x1)**2 + (eta - y1)**2 + zeta**2)
        r2 = np.sqrt((xi - x2)**2 + (eta - y2)**2 + zeta**2)
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        denom = r1 + r2 + d
        numer = r1 + r2 - d
        if abs(numer) > 1e-10 and denom > 1e-10:
            log_term = h * np.log(numer / denom)
        else:
            log_term = 0.0

        h_sq = h**2 + zeta**2
        if h_sq > 1e-20:
            atan1 = np.arctan2(h * s1, h_sq + abs(zeta) * r1)
            atan2 = np.arctan2(h * s2, h_sq + abs(zeta) * r2)
            atan_term = abs(zeta) * (atan1 - atan2)
        else:
            atan_term = 0.0

        phi += log_term - atan_term

    phi *= sigma / (4.0 * np.pi)
    return phi


@njit(cache=True)
def doublet_panel_influence_numba(
    panel_vertices: NDArray[np.float64],
    panel_normal: NDArray[np.float64],
    panel_centroid: NDArray[np.float64],
    field_point: NDArray[np.float64],
    mu: float = 1.0,
) -> float:
    """Calculate potential induced by a constant-strength doublet panel (Numba-accelerated).

    Parameters
    ----------
    panel_vertices : ndarray, shape (3, 3)
        Vertices of the triangular panel.
    panel_normal : ndarray, shape (3,)
        Unit outward normal.
    panel_centroid : ndarray, shape (3,)
        Panel center point.
    field_point : ndarray, shape (3,)
        Point where potential is evaluated.
    mu : float
        Doublet strength per unit area.

    Returns
    -------
    phi : float
        Induced velocity potential.
    """
    l_vec, m_vec, n_vec = _compute_local_coords_numba(panel_vertices, panel_normal)

    # Transform vertices to local coordinates
    local_verts = np.zeros((3, 3))
    for i in range(3):
        local_verts[i] = _point_to_local_numba(
            panel_vertices[i], panel_centroid, l_vec, m_vec, n_vec
        )

    local_p = _point_to_local_numba(field_point, panel_centroid, l_vec, m_vec, n_vec)
    xi, eta, zeta = local_p[0], local_p[1], local_p[2]

    omega = 0.0
    n_edges = 3

    for i in range(n_edges):
        x1, y1 = local_verts[i, 0], local_verts[i, 1]
        j = (i + 1) % n_edges
        x2, y2 = local_verts[j, 0], local_verts[j, 1]

        dx = x2 - x1
        dy = y2 - y1
        d = np.sqrt(dx**2 + dy**2)

        if d < 1e-12:
            continue

        s1 = ((xi - x1) * dx + (eta - y1) * dy) / d
        s2 = ((xi - x2) * dx + (eta - y2) * dy) / d
        h = ((xi - x1) * dy - (eta - y1) * dx) / d

        r1 = np.sqrt((xi - x1)**2 + (eta - y1)**2 + zeta**2)
        r2 = np.sqrt((xi - x2)**2 + (eta - y2)**2 + zeta**2)
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        h_sq = h**2 + zeta**2
        if h_sq > 1e-20:
            atan1 = np.arctan2(h * s1, h_sq + abs(zeta) * r1)
            atan2 = np.arctan2(h * s2, h_sq + abs(zeta) * r2)
            omega += atan1 - atan2

    if zeta < 0:
        omega = -omega

    phi = -mu * omega / (4.0 * np.pi)
    return phi


@njit(cache=True)
def _far_field_source_numba(
    area: float,
    centroid: NDArray[np.float64],
    field_point: NDArray[np.float64],
) -> float:
    """Far-field approximation for source panel (Numba-accelerated)."""
    r_vec = field_point - centroid
    r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    if r < 1e-10:
        return 0.0
    return area / (4.0 * np.pi * r)


@njit(cache=True)
def _far_field_doublet_numba(
    area: float,
    centroid: NDArray[np.float64],
    normal: NDArray[np.float64],
    field_point: NDArray[np.float64],
) -> float:
    """Far-field approximation for doublet panel (Numba-accelerated)."""
    r_vec = field_point - centroid
    r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    if r < 1e-10:
        return 0.0
    r_dot_n = r_vec[0] * normal[0] + r_vec[1] * normal[1] + r_vec[2] * normal[2]
    return -area * r_dot_n / (4.0 * np.pi * r**3)


@njit(parallel=True, cache=True)
def compute_influence_matrices_numba(
    vertices: NDArray[np.float64],
    centroids: NDArray[np.float64],
    normals: NDArray[np.float64],
    areas: NDArray[np.float64],
    control_points: NDArray[np.float64],
    panel_sizes: NDArray[np.float64],
    far_field_ratio: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute influence matrices using Numba parallel acceleration.

    Parameters
    ----------
    vertices : ndarray, shape (n_panels, 3, 3)
        Vertices of each triangular panel.
    centroids : ndarray, shape (n_panels, 3)
        Panel center points.
    normals : ndarray, shape (n_panels, 3)
        Panel unit normals.
    areas : ndarray, shape (n_panels,)
        Panel areas.
    control_points : ndarray, shape (n_panels, 3)
        Internal control points for Dirichlet BC.
    panel_sizes : ndarray, shape (n_panels,)
        Characteristic panel sizes (sqrt of area).
    far_field_ratio : float
        Use far-field approximation when distance > ratio * panel_size.

    Returns
    -------
    A_source, A_doublet : tuple of ndarray
        Source and doublet influence matrices, shape (n_panels, n_panels).
    """
    n_panels = len(areas)
    A_source = np.zeros((n_panels, n_panels))
    A_doublet = np.zeros((n_panels, n_panels))

    for i in prange(n_panels):
        cp = control_points[i]

        for j in range(n_panels):
            # Distance from control point to panel j centroid
            r_vec = cp - centroids[j]
            dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            threshold = far_field_ratio * panel_sizes[j]

            if dist > threshold and i != j:
                # Far-field approximation
                A_source[i, j] = _far_field_source_numba(areas[j], centroids[j], cp)
                A_doublet[i, j] = _far_field_doublet_numba(
                    areas[j], centroids[j], normals[j], cp
                )
            else:
                # Near-field: full panel integration
                A_source[i, j] = source_panel_influence_numba(
                    vertices[j], normals[j], centroids[j], cp, 1.0
                )
                A_doublet[i, j] = doublet_panel_influence_numba(
                    vertices[j], normals[j], centroids[j], cp, 1.0
                )

    return A_source, A_doublet


