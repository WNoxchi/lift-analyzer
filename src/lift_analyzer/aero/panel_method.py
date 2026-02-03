"""Source + Doublet Panel Method for Center of Lift calculation.

Implementation based on:
- Katz & Plotkin, "Low Speed Aerodynamics", Chapter 10
- NASA TP 2995, "Panel Methods - An Introduction"

The panel method solves the Laplace equation for potential flow by distributing
singularities (sources and doublets) on the body surface. For 3D lifting bodies,
doublet panels are preferred because the doublet strength μ is a scalar quantity.

Theory:
- Sources represent thickness effects (non-lifting)
- Doublets represent lifting effects (circulation)
- Dirichlet BC: internal perturbation potential = 0
- Kutta condition: smooth flow at trailing edge

Note: This module uses physics naming conventions (Cp, V_surface, A_doublet, etc.)
following Katz & Plotkin notation.
"""
# ruff: noqa: N803, N806  # Physics naming conventions (Cp, V, A matrices)

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from lift_analyzer.aero.panel_method_numba import compute_influence_matrices_numba


class PanelMethodResult(NamedTuple):
    """Result from panel method calculation."""

    center_of_lift: NDArray[np.float64]  # 3D position [x, y, z]
    total_lift: float  # Total lift force (N, normalized by dynamic pressure)
    pressure_coefficients: NDArray[np.float64]  # Cp at each panel
    doublet_strengths: NDArray[np.float64]  # μ at each panel
    source_strengths: NDArray[np.float64]  # σ at each panel


@dataclass
class PanelGeometry:
    """Geometric data for all panels in the mesh.

    All arrays have shape (n_panels, ...) where n_panels is the number
    of triangular panels in the mesh.
    """

    vertices: NDArray[np.float64]  # (n_panels, 3, 3) - 3 vertices per triangle
    centroids: NDArray[np.float64]  # (n_panels, 3) - panel center points
    normals: NDArray[np.float64]  # (n_panels, 3) - outward unit normals
    areas: NDArray[np.float64]  # (n_panels,) - panel areas

    @classmethod
    def from_mesh(cls, mesh: pv.PolyData) -> "PanelGeometry":
        """Extract panel geometry from a PyVista triangular mesh.

        Parameters
        ----------
        mesh : pv.PolyData
            Triangulated surface mesh.

        Returns
        -------
        PanelGeometry
            Panel geometric data.
        """
        # Ensure we have triangles
        mesh = mesh.triangulate()

        # Compute normals and cell data
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
        sized_mesh = mesh.compute_cell_sizes(area=True)

        # Extract arrays
        normals = mesh.cell_data["Normals"].astype(np.float64)
        areas = sized_mesh.cell_data["Area"].astype(np.float64)
        centroids = mesh.cell_centers().points.astype(np.float64)

        # Extract vertex coordinates for each triangle
        # mesh.faces format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # (n_panels, 3) indices
        points = mesh.points.astype(np.float64)
        vertices = points[faces]  # (n_panels, 3, 3)

        return cls(
            vertices=vertices,
            centroids=centroids,
            normals=normals,
            areas=areas,
        )


def _compute_local_coords(
    panel_vertices: NDArray[np.float64],
    panel_normal: NDArray[np.float64],
    panel_centroid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute local coordinate system for a panel.

    Parameters
    ----------
    panel_vertices : ndarray, shape (3, 3)
        Vertices of the triangular panel.
    panel_normal : ndarray, shape (3,)
        Unit normal vector of the panel.
    panel_centroid : ndarray, shape (3,)
        Center point of the panel.

    Returns
    -------
    l_vec : ndarray, shape (3,)
        Local x-axis (tangent, along first edge).
    m_vec : ndarray, shape (3,)
        Local y-axis (tangent, perpendicular to l in panel plane).
    n_vec : ndarray, shape (3,)
        Local z-axis (normal).
    """
    # l-axis: along first edge of triangle
    edge1 = panel_vertices[1] - panel_vertices[0]
    edge1_norm = np.linalg.norm(edge1)
    if edge1_norm < 1e-12:
        # Degenerate edge - try second edge
        edge1 = panel_vertices[2] - panel_vertices[0]
        edge1_norm = np.linalg.norm(edge1)
        if edge1_norm < 1e-12:
            # Fully degenerate triangle - return arbitrary orthonormal basis
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), panel_normal
    l_vec = edge1 / edge1_norm

    # n-axis: panel normal
    n_vec = panel_normal

    # m-axis: perpendicular to both (right-hand rule)
    m_vec = np.cross(n_vec, l_vec)
    m_norm = np.linalg.norm(m_vec)
    if m_norm < 1e-12:
        # l_vec parallel to normal - use arbitrary perpendicular
        if abs(n_vec[0]) < 0.9:
            m_vec = np.cross(n_vec, np.array([1.0, 0.0, 0.0]))
        else:
            m_vec = np.cross(n_vec, np.array([0.0, 1.0, 0.0]))
        m_norm = np.linalg.norm(m_vec)
    m_vec = m_vec / m_norm

    return l_vec, m_vec, n_vec


def _point_to_local(
    point: NDArray[np.float64],
    origin: NDArray[np.float64],
    l_vec: NDArray[np.float64],
    m_vec: NDArray[np.float64],
    n_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform a point to local panel coordinates.

    Parameters
    ----------
    point : ndarray, shape (3,) or (N, 3)
        Point(s) in global coordinates.
    origin : ndarray, shape (3,)
        Origin of local system (panel centroid).
    l_vec, m_vec, n_vec : ndarray, shape (3,)
        Local coordinate axes.

    Returns
    -------
    local_point : ndarray, shape (3,) or (N, 3)
        Point(s) in local coordinates (ξ, η, ζ).
    """
    rel = point - origin
    if rel.ndim == 1:
        return np.array([np.dot(rel, l_vec), np.dot(rel, m_vec), np.dot(rel, n_vec)])
    else:
        return np.column_stack([
            np.dot(rel, l_vec),
            np.dot(rel, m_vec),
            np.dot(rel, n_vec),
        ])


def source_panel_influence(
    panel_vertices: NDArray[np.float64],
    panel_normal: NDArray[np.float64],
    panel_centroid: NDArray[np.float64],
    field_point: NDArray[np.float64],
    sigma: float = 1.0,
) -> float:
    """Calculate potential induced by a constant-strength source panel.

    Based on Katz & Plotkin Equations 10.89-10.97 for quadrilateral panels,
    adapted for triangular panels.

    For a triangular panel with vertices P1, P2, P3, the induced potential at
    point P is computed by integrating the source influence over the panel.

    Parameters
    ----------
    panel_vertices : ndarray, shape (3, 3)
        Vertices of the triangular panel in global coordinates.
    panel_normal : ndarray, shape (3,)
        Unit outward normal of the panel.
    panel_centroid : ndarray, shape (3,)
        Center point of the panel.
    field_point : ndarray, shape (3,)
        Point where potential is evaluated.
    sigma : float, optional
        Source strength per unit area. Default is 1.0.

    Returns
    -------
    phi : float
        Induced velocity potential at the field point.

    Notes
    -----
    The source panel induces a potential:
        φ = (σ/4π) ∫∫ (1/r) dS

    Using the formulation from Katz & Plotkin, this is computed analytically
    for flat polygonal panels.
    """
    # Set up local coordinate system
    l_vec, m_vec, n_vec = _compute_local_coords(
        panel_vertices, panel_normal, panel_centroid
    )

    # Transform vertices and field point to local coordinates
    # In local coords, panel lies in z=0 plane (approximately, centered at origin)
    local_verts = np.array([
        _point_to_local(v, panel_centroid, l_vec, m_vec, n_vec)
        for v in panel_vertices
    ])
    local_p = _point_to_local(field_point, panel_centroid, l_vec, m_vec, n_vec)

    xi, eta, zeta = local_p

    # For numerical stability when field point is in the panel plane
    zeta = max(abs(zeta), 1e-10) * np.sign(zeta) if abs(zeta) < 1e-10 else zeta

    # Compute influence using the edge-by-edge formula (Eq. 10.89-10.97)
    # For a triangular panel with 3 edges
    phi = 0.0
    n_edges = 3

    for i in range(n_edges):
        # Edge from vertex i to vertex (i+1) mod 3
        x1, y1, _ = local_verts[i]
        x2, y2, _ = local_verts[(i + 1) % n_edges]

        # Edge parameters
        dx = x2 - x1
        dy = y2 - y1
        d = np.sqrt(dx**2 + dy**2)

        if d < 1e-12:
            continue

        # Projections (Eq. 10.90-10.91)
        # s = tangential distance along edge
        # n = normal distance from edge
        s1 = ((xi - x1) * dx + (eta - y1) * dy) / d
        s2 = ((xi - x2) * dx + (eta - y2) * dy) / d

        # Perpendicular distance from field point projection to edge
        h = ((xi - x1) * dy - (eta - y1) * dx) / d

        # Distances from field point to edge endpoints
        r1 = np.sqrt((xi - x1)**2 + (eta - y1)**2 + zeta**2)
        r2 = np.sqrt((xi - x2)**2 + (eta - y2)**2 + zeta**2)

        # Avoid numerical issues
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        # Source influence formula (Eq. 10.97)
        # First term: logarithmic contribution from line integral
        denom = r1 + r2 + d
        numer = r1 + r2 - d
        if abs(numer) > 1e-10 and denom > 1e-10:
            log_term = h * np.log(numer / denom)
        else:
            log_term = 0.0

        # Second term: arctangent contribution (solid angle)
        # This accounts for the 3D nature of the problem
        h_sq = h**2 + zeta**2
        if h_sq > 1e-20:
            atan1 = np.arctan2(h * s1, h_sq + abs(zeta) * r1)
            atan2 = np.arctan2(h * s2, h_sq + abs(zeta) * r2)
            atan_term = abs(zeta) * (atan1 - atan2)
        else:
            atan_term = 0.0

        phi += log_term - atan_term

    # Include the 1/(4π) factor and source strength
    phi *= sigma / (4.0 * np.pi)

    return phi


def doublet_panel_influence(
    panel_vertices: NDArray[np.float64],
    panel_normal: NDArray[np.float64],
    panel_centroid: NDArray[np.float64],
    field_point: NDArray[np.float64],
    mu: float = 1.0,
) -> float:
    """Calculate potential induced by a constant-strength doublet panel.

    Based on Katz & Plotkin Equations 10.104-10.109 for quadrilateral panels,
    adapted for triangular panels.

    A doublet panel is equivalent to a vortex ring at the panel boundary.
    The potential is related to the solid angle subtended by the panel.

    Parameters
    ----------
    panel_vertices : ndarray, shape (3, 3)
        Vertices of the triangular panel in global coordinates.
    panel_normal : ndarray, shape (3,)
        Unit outward normal of the panel.
    panel_centroid : ndarray, shape (3,)
        Center point of the panel.
    field_point : ndarray, shape (3,)
        Point where potential is evaluated.
    mu : float, optional
        Doublet strength per unit area. Default is 1.0.

    Returns
    -------
    phi : float
        Induced velocity potential at the field point.

    Notes
    -----
    The doublet panel induces a potential:
        φ = -(μ/4π) Ω

    where Ω is the solid angle subtended by the panel at the field point.
    For a flat polygonal panel, this is computed as a sum over edges.
    """
    # Set up local coordinate system
    l_vec, m_vec, n_vec = _compute_local_coords(
        panel_vertices, panel_normal, panel_centroid
    )

    # Transform vertices and field point to local coordinates
    local_verts = np.array([
        _point_to_local(v, panel_centroid, l_vec, m_vec, n_vec)
        for v in panel_vertices
    ])
    local_p = _point_to_local(field_point, panel_centroid, l_vec, m_vec, n_vec)

    xi, eta, zeta = local_p

    # Compute solid angle using edge formula (Eq. 10.108-10.109)
    # Solid angle Ω = Σ arctan(...) for each edge
    omega = 0.0
    n_edges = 3

    for i in range(n_edges):
        # Edge from vertex i to vertex (i+1) mod 3
        x1, y1, _ = local_verts[i]
        x2, y2, _ = local_verts[(i + 1) % n_edges]

        # Edge parameters
        dx = x2 - x1
        dy = y2 - y1
        d = np.sqrt(dx**2 + dy**2)

        if d < 1e-12:
            continue

        # Projections
        s1 = ((xi - x1) * dx + (eta - y1) * dy) / d
        s2 = ((xi - x2) * dx + (eta - y2) * dy) / d

        # Perpendicular distance from field point projection to edge
        h = ((xi - x1) * dy - (eta - y1) * dx) / d

        # Distances from field point to edge endpoints
        r1 = np.sqrt((xi - x1)**2 + (eta - y1)**2 + zeta**2)
        r2 = np.sqrt((xi - x2)**2 + (eta - y2)**2 + zeta**2)

        # Avoid division by zero
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        # Doublet influence (solid angle contribution from this edge)
        # Eq. 10.109 from Katz & Plotkin
        h_sq = h**2 + zeta**2
        if h_sq > 1e-20:
            atan1 = np.arctan2(h * s1, h_sq + abs(zeta) * r1)
            atan2 = np.arctan2(h * s2, h_sq + abs(zeta) * r2)
            omega += atan1 - atan2

    # Apply sign based on which side of panel we're on
    if zeta < 0:
        omega = -omega

    # Doublet potential: φ = -μΩ/(4π)
    phi = -mu * omega / (4.0 * np.pi)

    return phi


def _clean_coincident_panels(mesh: pv.PolyData, tolerance: float = 1e-3) -> pv.PolyData:
    """Remove or merge panels that are nearly coincident.

    Coincident panels (e.g., upper/lower wing surfaces collapsed together
    by decimation) cause numerical instability in the panel method solver,
    producing huge opposite singularity strengths.

    Parameters
    ----------
    mesh : pv.PolyData
        Input triangulated mesh.
    tolerance : float, optional
        Distance tolerance for considering panels coincident.
        Default is 1e-3 (relative to mesh scale).

    Returns
    -------
    pv.PolyData
        Cleaned mesh with coincident panels removed.
    """
    mesh = mesh.triangulate()

    # Compute cell centers and normals for comparison
    centers = mesh.cell_centers().points
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    normals = mesh.cell_data["Normals"]

    # Scale tolerance by mesh size
    bounds = mesh.bounds
    mesh_scale = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    abs_tolerance = tolerance * mesh_scale

    # Find coincident panels using a spatial approach
    n_cells = mesh.n_cells
    keep_mask = np.ones(n_cells, dtype=bool)

    # Build a simple spatial hash for efficiency
    # Group cells by their approximate position
    cell_groups: dict[tuple[int, int, int], list[int]] = {}
    grid_size = abs_tolerance * 2

    for i in range(n_cells):
        # Grid cell for this panel's center
        gx = int(centers[i, 0] / grid_size)
        gy = int(centers[i, 1] / grid_size)
        gz = int(centers[i, 2] / grid_size)
        key = (gx, gy, gz)

        if key not in cell_groups:
            cell_groups[key] = []
        cell_groups[key].append(i)

    # Check for coincident panels within each grid cell AND neighboring cells
    n_removed = 0
    checked_pairs: set[tuple[int, int]] = set()

    for key, indices in cell_groups.items():
        # Get indices from this cell and all 26 neighboring cells
        all_nearby: list[int] = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if neighbor_key in cell_groups:
                        all_nearby.extend(cell_groups[neighbor_key])

        # Check all pairs
        for i, idx1 in enumerate(all_nearby):
            if not keep_mask[idx1]:
                continue

            for idx2 in all_nearby[i + 1:]:
                if not keep_mask[idx2]:
                    continue

                # Skip if already checked
                pair = (min(idx1, idx2), max(idx1, idx2))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Check distance between centers
                dist = np.linalg.norm(centers[idx1] - centers[idx2])
                if dist < abs_tolerance:
                    # Also check if normals are opposite (indicating upper/lower surface pair)
                    normal_dot = np.dot(normals[idx1], normals[idx2])
                    if normal_dot < 0.5:  # Normals are not aligned (opposite or perpendicular)
                        # Coincident with different orientations - remove one
                        keep_mask[idx2] = False
                        n_removed += 1

    if n_removed > 0:
        print(f"    Removed {n_removed} coincident panels")
        # Extract cells to keep and convert back to PolyData
        mesh = mesh.extract_cells(np.where(keep_mask)[0])
        mesh = mesh.extract_surface()  # Convert UnstructuredGrid back to PolyData

    return mesh


def _decimate_mesh_iteratively(
    mesh: pv.PolyData,
    target_faces: int,
    max_reduction_per_pass: float = 0.85,
) -> pv.PolyData:
    """Iteratively decimate mesh until target face count is reached.

    PyVista's decimate() can silently fail for large (>95%) reductions in a
    single pass. This function performs multiple smaller reduction passes.

    Parameters
    ----------
    mesh : pv.PolyData
        Input triangulated mesh.
    target_faces : int
        Target number of faces.
    max_reduction_per_pass : float, optional
        Maximum reduction ratio per pass (0.85 = 85% reduction max).
        Default is 0.85.

    Returns
    -------
    pv.PolyData
        Decimated mesh with approximately target_faces faces.
    """
    mesh = mesh.triangulate()

    for iteration in range(10):  # Max 10 iterations to prevent infinite loops
        current_faces = mesh.n_cells

        if current_faces <= target_faces:
            break

        # Calculate reduction needed
        reduction_needed = 1.0 - (target_faces / current_faces)

        # Cap at max_reduction_per_pass
        reduction = min(reduction_needed, max_reduction_per_pass)

        # Use decimate_pro which is more aggressive than decimate
        mesh = mesh.decimate_pro(reduction, preserve_topology=False)
        mesh = mesh.triangulate()

        new_faces = mesh.n_cells

        # Detect stall (no progress)
        if new_faces >= current_faces * 0.99:
            print(f"    Decimation stalled at {new_faces} faces (target: {target_faces})")
            break

        print(f"    Pass {iteration + 1}: {current_faces} -> {new_faces} faces "
              f"({100 * (1 - new_faces / current_faces):.1f}% reduction)")

    return mesh


def _compute_influence_matrices(
    panels: PanelGeometry,
    internal_offset: float = 0.001,
    far_field_ratio: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Aerodynamic Influence Coefficient (AIC) matrices.

    For Dirichlet boundary conditions, we evaluate the potential at points
    slightly inside the body surface (internal points).

    Uses Numba JIT compilation with parallel execution for performance.

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    internal_offset : float, optional
        Distance to offset control points inward (as fraction of mesh size).
        Default is 0.001.
    far_field_ratio : float, optional
        Use far-field approximation when distance > ratio * panel_size.
        Default is 5.0.

    Returns
    -------
    A_source : ndarray, shape (n_panels, n_panels)
        Source influence matrix. A_source[i,j] is the potential at panel i
        control point due to unit source strength on panel j.
    A_doublet : ndarray, shape (n_panels, n_panels)
        Doublet influence matrix. A_doublet[i,j] is the potential at panel i
        control point due to unit doublet strength on panel j.
    """
    n_panels = len(panels.areas)

    # Compute internal control points (slightly inside the surface)
    # Offset inward along the normal
    mesh_scale = np.sqrt(np.mean(panels.areas))
    offset_dist = internal_offset * mesh_scale
    control_points = panels.centroids - offset_dist * panels.normals

    # Panel characteristic sizes for far-field criterion
    panel_sizes = np.sqrt(panels.areas)

    # Use Numba-accelerated computation
    A_source, A_doublet = compute_influence_matrices_numba(
        panels.vertices.astype(np.float64),
        panels.centroids.astype(np.float64),
        panels.normals.astype(np.float64),
        panels.areas.astype(np.float64),
        control_points.astype(np.float64),
        panel_sizes.astype(np.float64),
        far_field_ratio,
    )

    return A_source, A_doublet


def _compute_rhs_freestream(
    panels: PanelGeometry,
    freestream: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute RHS from freestream potential at control points.

    For Dirichlet BC: internal potential = 0
    => doublet_potential + source_potential + freestream_potential = 0

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    freestream : ndarray, shape (3,)
        Freestream velocity vector.

    Returns
    -------
    rhs : ndarray, shape (n_panels,)
        Right-hand side vector for the linear system.
    """
    # Freestream potential at control points: φ∞ = V∞ · r
    # For internal control points offset inward
    mesh_scale = np.sqrt(np.mean(panels.areas))
    offset_dist = 0.001 * mesh_scale
    control_points = panels.centroids - offset_dist * panels.normals

    # φ∞ = V∞ · x (potential from freestream)
    phi_freestream = np.dot(control_points, freestream)

    return -phi_freestream


def _solve_source_strengths(
    panels: PanelGeometry,
    freestream: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Determine source strengths from zero normal flow condition.

    For a closed body in potential flow, the source strength equals
    the negative of the normal component of freestream velocity:
        σ = -V∞ · n

    This ensures zero normal velocity through the surface (impermeability).

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    freestream : ndarray, shape (3,)
        Freestream velocity vector.

    Returns
    -------
    sigma : ndarray, shape (n_panels,)
        Source strength at each panel.
    """
    # σ = -V∞ · n (zero normal flow through body)
    return -np.dot(panels.normals, freestream)


def _solve_doublet_strengths(
    A_doublet: NDArray[np.float64],
    A_source: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve for doublet strengths using Dirichlet boundary condition.

    The linear system is:
        A_doublet · μ = -A_source · σ - φ∞

    where φ∞ is the freestream potential (included in rhs).

    Parameters
    ----------
    A_doublet : ndarray, shape (n, n)
        Doublet influence matrix.
    A_source : ndarray, shape (n, n)
        Source influence matrix.
    sigma : ndarray, shape (n,)
        Source strengths.
    rhs : ndarray, shape (n,)
        Right-hand side from freestream.

    Returns
    -------
    mu : ndarray, shape (n,)
        Doublet strengths at each panel.
    """
    # Complete RHS: -A_source · σ + rhs (where rhs = -φ∞)
    b = rhs - np.dot(A_source, sigma)

    # Solve A_doublet · μ = b
    # Use least squares for robustness (may be singular for closed bodies)
    mu, residuals, rank, s = np.linalg.lstsq(A_doublet, b, rcond=None)

    return mu


def _compute_surface_velocity(
    panels: PanelGeometry,
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    freestream: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute surface velocity at each panel.

    The surface velocity is computed from the gradient of the doublet
    distribution (which is equivalent to a surface vorticity) plus the
    freestream component tangent to the surface.

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    mu : ndarray, shape (n_panels,)
        Doublet strengths.
    sigma : ndarray, shape (n_panels,)
        Source strengths.
    freestream : ndarray, shape (3,)
        Freestream velocity vector.

    Returns
    -------
    V_surface : ndarray, shape (n_panels, 3)
        Surface velocity vectors at each panel centroid.
    """
    n_panels = len(panels.areas)
    V_surface = np.zeros((n_panels, 3))

    # For each panel, compute the surface gradient of μ
    # This requires looking at neighboring panels
    # Using a simplified approach: finite difference with nearest neighbors

    # Precompute normal dot products for neighbor filtering
    # Only use neighbors on the same surface (similar normal direction)
    normal_threshold = 0.7  # ~45 degrees

    for i in range(n_panels):
        # Freestream tangential component
        n = panels.normals[i]
        V_inf_tangent = freestream - np.dot(freestream, n) * n

        # Doublet gradient contribution
        # For constant-strength panels, the velocity jump is related to
        # the gradient of doublet strength along the surface
        # V_tangent = ∇s(μ) where ∇s is the surface gradient

        # Find neighbors on the SAME surface (similar normal direction)
        # This prevents mixing upper/lower wing surfaces in gradient calc
        normal_dots = np.dot(panels.normals, n)
        same_surface = normal_dots > normal_threshold

        # Compute distances only to same-surface panels
        distances = np.linalg.norm(
            panels.centroids - panels.centroids[i], axis=1
        )
        distances[i] = np.inf  # Exclude self
        distances[~same_surface] = np.inf  # Exclude opposite-surface panels

        # Get nearest same-surface neighbors
        valid_count = np.sum(same_surface) - 1  # -1 for self
        n_neighbors = min(6, valid_count)

        if n_neighbors >= 3:
            neighbor_idx = np.argpartition(distances, n_neighbors)[:n_neighbors]

            # Compute surface gradient using least squares fit
            rel_pos = panels.centroids[neighbor_idx] - panels.centroids[i]
            mu_diff = mu[neighbor_idx] - mu[i]

            # Use rcond to filter out ill-conditioned components
            grad_mu, residuals, rank, s = np.linalg.lstsq(rel_pos, mu_diff, rcond=1e-6)

            # Clamp gradient magnitude to prevent numerical blowup
            # Physical limit: gradient shouldn't exceed ~10 * freestream / panel_size
            max_grad = 10.0 * np.linalg.norm(freestream) / np.sqrt(panels.areas[i])
            grad_mag = np.linalg.norm(grad_mu)
            if grad_mag > max_grad:
                grad_mu = grad_mu * (max_grad / grad_mag)

            # Project gradient onto panel surface
            grad_mu_tangent = grad_mu - np.dot(grad_mu, n) * n

            # Surface velocity from doublet is the surface gradient
            V_doublet = grad_mu_tangent
        else:
            V_doublet = np.zeros(3)

        V_surface[i] = V_inf_tangent + V_doublet

    return V_surface


def _compute_pressure_coefficient(
    V_surface: NDArray[np.float64],
    V_inf_mag: float,
) -> NDArray[np.float64]:
    """Compute pressure coefficient from surface velocity.

    Using Bernoulli's equation for incompressible flow:
        Cp = 1 - (V/V∞)²

    Parameters
    ----------
    V_surface : ndarray, shape (n_panels, 3)
        Surface velocity at each panel.
    V_inf_mag : float
        Magnitude of freestream velocity.

    Returns
    -------
    Cp : ndarray, shape (n_panels,)
        Pressure coefficient at each panel.
    """
    V_mag = np.linalg.norm(V_surface, axis=1)
    Cp = 1.0 - (V_mag / V_inf_mag) ** 2
    return Cp


def _compute_lift_force(
    panels: PanelGeometry,
    Cp: NDArray[np.float64],
    lift_direction: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Compute total lift and lift distribution from pressure.

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    Cp : ndarray, shape (n_panels,)
        Pressure coefficient at each panel.
    lift_direction : ndarray, shape (3,)
        Direction of lift (typically vertical, Z-axis).

    Returns
    -------
    total_lift : float
        Total lift coefficient (CL * Sref).
    lift_per_panel : ndarray, shape (n_panels,)
        Lift contribution from each panel.
    """
    # Force per panel: F = -Cp * A * n (pressure acts inward on normal)
    # Lift component: L = F · lift_dir
    lift_per_panel = -Cp * panels.areas * np.dot(panels.normals, lift_direction)
    total_lift = np.sum(lift_per_panel)

    return total_lift, lift_per_panel


def _compute_lift_from_doublet_strength(
    panels: PanelGeometry,
    mu: NDArray[np.float64],
    freestream: NDArray[np.float64],
    lift_direction: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Compute lift distribution directly from doublet strength.

    This method bypasses the problematic surface velocity calculation.
    In panel method theory, doublet strength μ represents the velocity
    potential jump across the surface, which is directly related to
    circulation and thus lift (Kutta-Joukowski theorem).

    For a panel, the lift contribution is proportional to:
    - The doublet strength μ (represents local circulation)
    - The panel's "span" (sqrt of area as characteristic length)
    - The freestream velocity magnitude
    - The panel's orientation relative to lift direction

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    mu : ndarray, shape (n_panels,)
        Doublet strengths from the panel method solution.
    freestream : ndarray, shape (3,)
        Freestream velocity vector.
    lift_direction : ndarray, shape (3,)
        Direction of lift (typically vertical, Z-axis).

    Returns
    -------
    total_lift : float
        Total lift (proportional to actual lift).
    lift_per_panel : ndarray, shape (n_panels,)
        Lift contribution from each panel.

    Notes
    -----
    The Kutta-Joukowski theorem states: L' = ρ V∞ Γ (lift per unit span)

    For a panel with doublet strength μ:
    - μ represents the velocity potential jump (related to circulation)
    - Effective span ≈ sqrt(area)
    - Lift contribution ∝ μ × V∞ × span × orientation_factor

    This gives relative lift distribution, not absolute values, but the
    CENTER of lift should be accurate since it depends on distribution.
    """
    V_inf_mag = np.linalg.norm(freestream)
    freestream_dir = freestream / V_inf_mag

    # Characteristic "span" for each panel
    panel_span = np.sqrt(panels.areas)

    # For lift calculation from doublet strength:
    # - μ represents the velocity potential jump (related to circulation)
    # - The sign of μ combined with panel orientation determines lift direction
    # - Panels facing the lift direction with positive μ generate positive lift
    #
    # Using signed normal·lift alignment (not abs) preserves the physics:
    # - Upper surface panel (normal up) + positive μ → positive lift
    # - Lower surface panel (normal down) + positive μ → negative lift
    # - The combination captures the pressure difference correctly
    normal_lift_component = np.dot(panels.normals, lift_direction)

    # Filter out panels with extreme μ values (numerical instability indicators)
    # These typically occur at coincident/degenerate panels
    mu_abs = np.abs(mu)
    mu_median = np.median(mu_abs)
    mu_threshold = mu_median * 1000  # Panels with μ > 1000x median are suspect

    valid_mask = mu_abs < mu_threshold
    n_filtered = np.sum(~valid_mask)

    if n_filtered > 0:
        print(f"    Filtering {n_filtered} panels with extreme μ values")

    # Lift per panel using Kutta-Joukowski relationship:
    # L = ρ V∞ Γ × span, where Γ ∝ μ
    # Note: Sign convention - negative μ indicates circulation that creates
    # positive lift for upward-facing panels
    lift_per_panel = -mu * V_inf_mag * panel_span * normal_lift_component

    # Zero out lift from filtered panels
    lift_per_panel[~valid_mask] = 0.0

    total_lift = np.sum(lift_per_panel)

    return total_lift, lift_per_panel


def _compute_center_of_lift(
    panels: PanelGeometry,
    lift_per_panel: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute center of lift as lift-weighted centroid.

    The center of lift is where a single force equal to total lift would
    produce the same pitching moment. Panels with negative lift (downforce)
    shift the CoL away from them.

    Parameters
    ----------
    panels : PanelGeometry
        Panel geometric data.
    lift_per_panel : ndarray, shape (n_panels,)
        Lift force from each panel (positive = upward).

    Returns
    -------
    col : ndarray, shape (3,)
        Center of lift position.
    """
    total_lift = np.sum(lift_per_panel)
    total_abs_lift = np.sum(np.abs(lift_per_panel))

    # Check if lift is well-defined (not too much cancellation)
    # If |total_lift| << sum(|lift|), the CoL is numerically unstable
    if abs(total_lift) < 1e-10:
        # No net lift - return geometric center
        print("    Warning: Net lift ~0, CoL undefined. Using geometric center.")
        return np.mean(panels.centroids, axis=0)

    lift_ratio = abs(total_lift) / total_abs_lift if total_abs_lift > 0 else 0
    if lift_ratio < 0.1:
        # Severe cancellation - CoL is unreliable
        print(f"    Warning: Lift cancellation ({lift_ratio:.1%} net/gross). "
              "CoL may be unreliable.")

    # Lift-weighted centroid using signed values
    # CoL = Σ(position × lift) / Σ(lift)
    col = np.sum(
        panels.centroids * lift_per_panel[:, np.newaxis],
        axis=0,
    ) / total_lift

    # Sanity check: CoL should be within or near the body bounds
    centroid_min = panels.centroids.min(axis=0)
    centroid_max = panels.centroids.max(axis=0)
    body_size = centroid_max - centroid_min
    margin = body_size * 2  # Allow 2x body size margin

    if np.any(col < centroid_min - margin) or np.any(col > centroid_max + margin):
        print(f"    Warning: CoL {col} is far outside body bounds. "
              "Result may be unreliable.")

    return col


def solve_panel_method(
    mesh: pv.PolyData,
    freestream_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    freestream_magnitude: float = 1.0,
    lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_panels: int = 5000,
) -> PanelMethodResult:
    """Solve the Source + Doublet panel method for a 3D body.

    This implements a Dirichlet boundary condition panel method where:
    - Sources represent the body thickness (non-lifting effect)
    - Doublets represent the lifting effect (circulation)

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated surface mesh of the body.
    freestream_direction : tuple, optional
        Direction of freestream velocity. Default is (0, 1, 0) meaning
        flow from -Y toward +Y (aircraft nose at -Y).
    freestream_magnitude : float, optional
        Magnitude of freestream velocity. Default is 1.0 (gives Cp directly).
    lift_direction : tuple, optional
        Direction in which lift is measured. Default is (0, 0, 1) for +Z.
    max_panels : int, optional
        Maximum number of panels. Mesh will be decimated if larger.
        Default is 5000 (25M influence calculations). Set to 0 to disable.

    Returns
    -------
    PanelMethodResult
        Named tuple containing:
        - center_of_lift: 3D position of center of lift
        - total_lift: Total lift coefficient * reference area
        - pressure_coefficients: Cp at each panel
        - doublet_strengths: μ at each panel
        - source_strengths: σ at each panel

    Notes
    -----
    This method assumes:
    1. Incompressible, inviscid, irrotational flow (potential flow)
    2. Steady state
    3. No wake modeling (suitable for bodies with small lift gradients)

    For wings with significant lift, wake panels should be added.
    """
    # Normalize directions
    freestream_dir = np.array(freestream_direction, dtype=np.float64)
    freestream_dir = freestream_dir / np.linalg.norm(freestream_dir)
    freestream = freestream_dir * freestream_magnitude

    lift_dir = np.array(lift_direction, dtype=np.float64)
    lift_dir = lift_dir / np.linalg.norm(lift_dir)

    # Triangulate and potentially decimate the mesh
    mesh = mesh.triangulate()
    n_faces_original = mesh.n_cells

    if max_panels > 0 and n_faces_original > max_panels:
        print(f"Decimating mesh: {n_faces_original} -> ~{max_panels} panels")
        mesh = _decimate_mesh_iteratively(mesh, target_faces=max_panels)
        print(f"    Final: {mesh.n_cells} panels")

    # Clean mesh to remove coincident panels (common after decimation of thin surfaces)
    mesh = _clean_coincident_panels(mesh)

    # Extract panel geometry
    panels = PanelGeometry.from_mesh(mesh)
    n_panels = len(panels.areas)

    print(f"Panel method: {n_panels} panels")

    # Step 1: Compute influence matrices
    print("  Computing influence matrices...")
    A_source, A_doublet = _compute_influence_matrices(panels)

    # Step 2: Determine source strengths (impermeability condition)
    print("  Computing source strengths...")
    sigma = _solve_source_strengths(panels, freestream)

    # Step 3: Set up RHS from freestream
    rhs = _compute_rhs_freestream(panels, freestream)

    # Step 4: Solve for doublet strengths
    print("  Solving for doublet strengths...")
    mu = _solve_doublet_strengths(A_doublet, A_source, sigma, rhs)

    # Step 5: Compute surface velocities
    print("  Computing surface velocities...")
    V_surface = _compute_surface_velocity(panels, mu, sigma, freestream)

    # Step 6: Compute pressure coefficients with outlier filtering
    Cp = _compute_pressure_coefficient(V_surface, freestream_magnitude)

    # Filter outlier Cp values using robust statistics
    # Use median and IQR to identify outliers (more robust than mean/std)
    Cp_median = np.median(Cp)
    Cp_q1 = np.percentile(Cp, 25)
    Cp_q3 = np.percentile(Cp, 75)
    Cp_iqr = Cp_q3 - Cp_q1

    # Outliers are beyond 3*IQR from quartiles (very conservative)
    Cp_lower = Cp_q1 - 3 * Cp_iqr
    Cp_upper = Cp_q3 + 3 * Cp_iqr

    outlier_mask = (Cp < Cp_lower) | (Cp > Cp_upper)
    n_outliers = np.sum(outlier_mask)

    if n_outliers > 0:
        print(f"    Filtering {n_outliers} panels with outlier Cp values")
        # Replace outliers with median
        Cp[outlier_mask] = Cp_median

    # Step 7: Compute lift and center of lift
    print("  Computing lift distribution...")
    total_lift, lift_per_panel = _compute_lift_force(panels, Cp, lift_dir)
    col = _compute_center_of_lift(panels, lift_per_panel)

    # Diagnostic output
    print(f"\n  === Diagnostics ===")
    print(f"  Cp: min={Cp.min():.3f}, max={Cp.max():.3f}, mean={Cp.mean():.3f}")
    print(f"  V_surface magnitude: min={np.linalg.norm(V_surface, axis=1).min():.3f}, "
          f"max={np.linalg.norm(V_surface, axis=1).max():.3f}")

    # Lift distribution by Y-coordinate (fore/aft)
    y_coords = panels.centroids[:, 1]
    y_mid = (y_coords.min() + y_coords.max()) / 2
    fore_mask = y_coords < y_mid
    aft_mask = ~fore_mask

    fore_lift = np.sum(lift_per_panel[fore_mask])
    aft_lift = np.sum(lift_per_panel[aft_mask])
    print(f"  Lift (fore Y<{y_mid:.2f}): {fore_lift:.4f} ({100*fore_lift/total_lift if abs(total_lift)>1e-10 else 0:.1f}%)")
    print(f"  Lift (aft Y>={y_mid:.2f}): {aft_lift:.4f} ({100*aft_lift/total_lift if abs(total_lift)>1e-10 else 0:.1f}%)")

    # Find panels contributing most to lift
    top_lift_idx = np.argsort(np.abs(lift_per_panel))[-5:]
    print(f"  Top 5 lift contributors (panel idx, position, lift, Cp):")
    for idx in reversed(top_lift_idx):
        pos = panels.centroids[idx]
        print(f"    Panel {idx}: Y={pos[1]:.3f}, Z={pos[2]:.3f}, lift={lift_per_panel[idx]:.4f}, Cp={Cp[idx]:.3f}")

    print(f"  ===================\n")
    print(f"  Total lift coefficient: {total_lift:.4f}")
    print(f"  Center of lift: ({col[0]:.4f}, {col[1]:.4f}, {col[2]:.4f})")

    return PanelMethodResult(
        center_of_lift=col,
        total_lift=total_lift,
        pressure_coefficients=Cp,
        doublet_strengths=mu,
        source_strengths=sigma,
    )


def calculate_center_of_lift_panel_method(
    mesh: pv.PolyData,
    airflow_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    max_panels: int = 10000,
) -> PanelMethodResult:
    """Calculate center of lift using the Source + Doublet panel method.

    This is the main entry point for panel method based CoL calculation.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangulated surface mesh.
    airflow_direction : tuple, optional
        Direction of airflow (where air flows TO). Default is (0, 1, 0)
        meaning aircraft nose at -Y, air flows from -Y to +Y.
    max_panels : int, optional
        Maximum number of panels. Mesh will be decimated if larger.
        Default is 10000 (best numerical stability for most models).
        Set to 0 to disable decimation.

    Returns
    -------
    PanelMethodResult
        Full panel method results including center of lift.
    """
    return solve_panel_method(
        mesh,
        freestream_direction=airflow_direction,
        freestream_magnitude=1.0,
        lift_direction=(0.0, 0.0, 1.0),  # Lift in +Z direction
        max_panels=max_panels,
    )
