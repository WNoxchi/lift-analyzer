"""Tests for the Source + Doublet panel method implementation."""

import numpy as np
import pytest
import pyvista as pv

from lift_analyzer.aero.panel_method import (
    PanelGeometry,
    PanelMethodResult,
    _decimate_mesh_iteratively,
    calculate_center_of_lift_panel_method,
    doublet_panel_influence,
    solve_panel_method,
    source_panel_influence,
)

from lift_analyzer.aero.panel_method_numba import (
    doublet_panel_influence_numba,
    source_panel_influence_numba,
)


class TestPanelGeometry:
    """Tests for panel geometry extraction."""

    def test_from_mesh_box(self):
        """Test extracting panel geometry from a simple box."""
        box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
        panels = PanelGeometry.from_mesh(box)

        # A box has 6 faces, each triangulated into 2 triangles = 12 triangles
        assert panels.vertices.shape[0] == 12
        assert panels.vertices.shape[1] == 3  # 3 vertices per triangle
        assert panels.vertices.shape[2] == 3  # 3 coordinates per vertex

        assert panels.centroids.shape == (12, 3)
        assert panels.normals.shape == (12, 3)
        assert panels.areas.shape == (12,)

        # All normals should be unit vectors
        norms = np.linalg.norm(panels.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # Total area should be 6 faces * 4 = 24
        np.testing.assert_allclose(np.sum(panels.areas), 24.0, rtol=1e-10)

    def test_from_mesh_sphere(self):
        """Test extracting panel geometry from a sphere."""
        sphere = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        panels = PanelGeometry.from_mesh(sphere)

        # Check shapes
        assert panels.vertices.ndim == 3
        assert panels.centroids.ndim == 2
        assert panels.normals.ndim == 2

        # All normals should be unit vectors (within floating point tolerance)
        norms = np.linalg.norm(panels.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

        # For a unit sphere, normals should approximately point radially outward
        # (centroid direction from origin should match normal)
        centroid_dirs = panels.centroids / np.linalg.norm(
            panels.centroids, axis=1, keepdims=True
        )
        dot_products = np.sum(centroid_dirs * panels.normals, axis=1)
        # Should be close to 1.0 (parallel) for most panels
        assert np.mean(dot_products) > 0.9


class TestSourcePanelInfluence:
    """Tests for source panel influence calculation."""

    def test_far_field_behavior(self):
        """Source influence should decay as 1/r in far field."""
        # Simple triangular panel
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)

        # Compute influence at increasing distances
        distances = [10.0, 20.0, 40.0]
        potentials = []
        for d in distances:
            field_point = centroid + np.array([0.0, 0.0, d])
            phi = source_panel_influence(
                vertices, normal, centroid, field_point, sigma=1.0
            )
            potentials.append(phi)

        # In far field, source panel acts like a point source
        # phi ~ sigma * Area / (4*pi*r), so phi * r should be constant
        phi_times_r = [p * d for p, d in zip(potentials, distances)]
        np.testing.assert_allclose(
            phi_times_r[0], phi_times_r[1], rtol=0.1
        )
        np.testing.assert_allclose(
            phi_times_r[1], phi_times_r[2], rtol=0.1
        )

    def test_symmetry(self):
        """Influence should be symmetric about panel plane."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)

        # Points above and below panel at same distance
        p_above = centroid + np.array([0.0, 0.0, 1.0])
        p_below = centroid + np.array([0.0, 0.0, -1.0])

        phi_above = source_panel_influence(vertices, normal, centroid, p_above)
        phi_below = source_panel_influence(vertices, normal, centroid, p_below)

        # Source influence is symmetric about panel plane
        np.testing.assert_allclose(phi_above, phi_below, rtol=1e-10)


class TestDoubletPanelInfluence:
    """Tests for doublet panel influence calculation."""

    def test_far_field_behavior(self):
        """Doublet influence should decay as 1/r² in far field."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)

        # Compute influence at increasing distances along normal
        distances = [10.0, 20.0, 40.0]
        potentials = []
        for d in distances:
            field_point = centroid + np.array([0.0, 0.0, d])
            phi = doublet_panel_influence(
                vertices, normal, centroid, field_point, mu=1.0
            )
            potentials.append(abs(phi))

        # In far field, doublet panel acts like a point doublet
        # phi ~ mu * Area * cos(theta) / (4*pi*r²)
        # Along normal, cos(theta) = 1, so phi * r² should be constant
        phi_times_r2 = [p * d**2 for p, d in zip(potentials, distances)]
        np.testing.assert_allclose(
            phi_times_r2[0], phi_times_r2[1], rtol=0.15
        )
        np.testing.assert_allclose(
            phi_times_r2[1], phi_times_r2[2], rtol=0.15
        )

    def test_antisymmetry(self):
        """Doublet influence should be antisymmetric about panel plane."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)

        # Points above and below panel at same distance
        p_above = centroid + np.array([0.0, 0.0, 1.0])
        p_below = centroid + np.array([0.0, 0.0, -1.0])

        phi_above = doublet_panel_influence(vertices, normal, centroid, p_above)
        phi_below = doublet_panel_influence(vertices, normal, centroid, p_below)

        # Doublet influence is antisymmetric: opposite sign on each side
        np.testing.assert_allclose(phi_above, -phi_below, rtol=1e-10)


class TestSolvePanelMethod:
    """Tests for the full panel method solver."""

    def test_sphere_no_lift(self):
        """A sphere in uniform flow should have zero lift."""
        sphere = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)

        result = solve_panel_method(
            sphere,
            freestream_direction=(0.0, 1.0, 0.0),
            freestream_magnitude=1.0,
            lift_direction=(0.0, 0.0, 1.0),
        )

        # Sphere has no net lift due to symmetry
        # (might not be exactly zero due to mesh asymmetries)
        assert abs(result.total_lift) < 1.0  # Very small compared to mesh area

    def test_returns_valid_result(self):
        """Test that solver returns valid result structure."""
        box = pv.Box(bounds=(-1, 1, -2, 2, -0.5, 0.5))

        result = solve_panel_method(box)

        assert isinstance(result, PanelMethodResult)
        assert result.center_of_lift.shape == (3,)
        assert isinstance(result.total_lift, float)
        assert result.pressure_coefficients.ndim == 1
        assert result.doublet_strengths.ndim == 1
        assert result.source_strengths.ndim == 1

    def test_col_within_body(self):
        """Center of lift should be within or near the body."""
        box = pv.Box(bounds=(-1, 1, -2, 2, -0.5, 0.5))

        result = solve_panel_method(box)

        # CoL should be within bounds (with some margin)
        col = result.center_of_lift
        assert -2 < col[0] < 2
        assert -3 < col[1] < 3
        assert -1 < col[2] < 1


class TestCalculateCenterOfLiftPanelMethod:
    """Tests for the main CoL calculation function."""

    def test_basic_call(self):
        """Test basic function call works."""
        mesh = pv.Box(bounds=(-1, 1, -2, 2, -0.5, 0.5))
        result = calculate_center_of_lift_panel_method(mesh)

        assert isinstance(result, PanelMethodResult)
        assert result.center_of_lift is not None

    def test_airflow_direction_affects_result(self):
        """Different airflow directions should give different results."""
        # Asymmetric shape
        mesh = pv.Box(bounds=(0, 2, -1, 1, -0.5, 0.5))

        result_y = calculate_center_of_lift_panel_method(
            mesh, airflow_direction=(0.0, 1.0, 0.0)
        )
        result_x = calculate_center_of_lift_panel_method(
            mesh, airflow_direction=(1.0, 0.0, 0.0)
        )

        # Results should differ for asymmetric body with different flow directions
        # The doublet strengths (which drive lift) should be different
        assert not np.allclose(result_y.doublet_strengths, result_x.doublet_strengths)


class TestDecimation:
    """Tests for iterative mesh decimation."""

    def test_decimate_reduces_to_target(self):
        """Test that decimation achieves target face count."""
        # Create a high-resolution sphere (many faces)
        sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
        sphere = sphere.triangulate()
        original_faces = sphere.n_cells

        # Should have many faces
        assert original_faces > 2000

        # Decimate to target
        target = 500
        decimated = _decimate_mesh_iteratively(sphere, target_faces=target)

        # Should be close to target (within 20%)
        assert decimated.n_cells <= target * 1.2
        assert decimated.n_cells >= target * 0.5  # At least half of target

    def test_decimate_no_op_when_below_target(self):
        """Test that decimation is a no-op when already below target."""
        box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
        box = box.triangulate()
        original_faces = box.n_cells

        # Target is higher than current
        target = original_faces * 2
        decimated = _decimate_mesh_iteratively(box, target_faces=target)

        # Should be unchanged
        assert decimated.n_cells == original_faces

    def test_decimate_large_reduction(self):
        """Test that large reductions (>95%) work correctly."""
        # Create a very high-resolution mesh
        sphere = pv.Sphere(radius=1.0, theta_resolution=100, phi_resolution=100)
        sphere = sphere.triangulate()
        original_faces = sphere.n_cells

        # Request 98% reduction (this is what was failing before)
        target = int(original_faces * 0.02)  # 2% of original
        decimated = _decimate_mesh_iteratively(sphere, target_faces=target)

        # Should achieve significant reduction (at least 80%)
        reduction = 1.0 - (decimated.n_cells / original_faces)
        assert reduction > 0.80, f"Only achieved {reduction*100:.1f}% reduction"


class TestNumbaConsistency:
    """Tests for Numba vs pure Python consistency."""

    def test_source_panel_consistency(self):
        """Test that Numba source influence matches pure Python."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)
        field_point = centroid + np.array([0.3, 0.2, 0.5])

        # Compare Python and Numba versions
        phi_python = source_panel_influence(vertices, normal, centroid, field_point)
        phi_numba = source_panel_influence_numba(vertices, normal, centroid, field_point)

        np.testing.assert_allclose(phi_python, phi_numba, rtol=1e-10)

    def test_doublet_panel_consistency(self):
        """Test that Numba doublet influence matches pure Python."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)
        field_point = centroid + np.array([0.3, 0.2, 0.5])

        # Compare Python and Numba versions
        phi_python = doublet_panel_influence(vertices, normal, centroid, field_point)
        phi_numba = doublet_panel_influence_numba(vertices, normal, centroid, field_point)

        np.testing.assert_allclose(phi_python, phi_numba, rtol=1e-10)

    def test_source_multiple_field_points(self):
        """Test Numba consistency across multiple field points."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.mean(vertices, axis=0)

        # Test various field points
        field_points = [
            centroid + np.array([0.0, 0.0, 0.1]),   # Close, above
            centroid + np.array([0.0, 0.0, -0.1]),  # Close, below
            centroid + np.array([5.0, 0.0, 0.0]),   # Far away
            centroid + np.array([0.1, 0.1, 0.1]),   # Near edge
        ]

        for fp in field_points:
            phi_python = source_panel_influence(vertices, normal, centroid, fp)
            phi_numba = source_panel_influence_numba(vertices, normal, centroid, fp)
            np.testing.assert_allclose(phi_python, phi_numba, rtol=1e-9)
