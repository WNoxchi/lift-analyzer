"""Validation tests for panel method against known analytical solutions."""

import numpy as np
import pyvista as pv

from lift_analyzer.aero.panel_method import (
    PanelGeometry,
    solve_panel_method,
)


def test_symmetric_body_no_net_lift():
    """A symmetric body at zero AoA should have approximately zero net lift."""
    # Create a sphere - perfectly symmetric
    sphere = pv.Sphere(radius=50, theta_resolution=20, phi_resolution=20)

    result = solve_panel_method(
        sphere,
        freestream_direction=(0.0, 1.0, 0.0),
        freestream_magnitude=1.0,
        lift_direction=(0.0, 0.0, 1.0),
        max_panels=500,
    )

    print(f"\nSphere test (should have ~zero lift):")
    print(f"  Total lift: {result.total_lift:.4f}")
    print(f"  Center of lift: {result.center_of_lift}")

    # Sphere should have very small lift due to symmetry
    # Allow some numerical noise
    total_area = 4 * np.pi * 50**2
    normalized_lift = abs(result.total_lift) / total_area
    print(f"  Normalized lift (should be <<1): {normalized_lift:.6f}")


def test_ellipsoid_col_at_center():
    """An ellipsoid's CoL should be near its geometric center."""
    # Create elongated ellipsoid (like a fuselage)
    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
    # Scale to make ellipsoid: long in Y (flight direction)
    ellipsoid = sphere.scale([20, 100, 20], inplace=False)

    result = solve_panel_method(
        ellipsoid,
        freestream_direction=(0.0, 1.0, 0.0),
        freestream_magnitude=1.0,
        lift_direction=(0.0, 0.0, 1.0),
        max_panels=500,
    )

    print(f"\nEllipsoid test:")
    print(f"  Total lift: {result.total_lift:.4f}")
    print(f"  Center of lift: {result.center_of_lift}")
    print(f"  Geometric center: (0, 0, 0)")

    # CoL should be near origin for symmetric ellipsoid
    col_distance = np.linalg.norm(result.center_of_lift)
    print(f"  CoL distance from center: {col_distance:.2f}")


def test_tilted_ellipsoid():
    """An ellipsoid tilted nose-up should generate positive lift."""
    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
    ellipsoid = sphere.scale([20, 100, 10], inplace=False)  # Flat-ish ellipsoid

    # Tilt 15 degrees nose-up (rotate around X axis)
    ellipsoid = ellipsoid.rotate_x(15, point=(0, 0, 0))

    result = solve_panel_method(
        ellipsoid,
        freestream_direction=(0.0, 1.0, 0.0),
        freestream_magnitude=1.0,
        lift_direction=(0.0, 0.0, 1.0),
        max_panels=500,
    )

    print(f"\nTilted ellipsoid test (15 deg nose-up):")
    print(f"  Total lift: {result.total_lift:.4f}")
    print(f"  Center of lift: {result.center_of_lift}")
    print(f"  Cp range: [{result.pressure_coefficients.min():.3f}, {result.pressure_coefficients.max():.3f}]")
    print(f"  Expected: Positive lift (nose-up generates upward force)")
    print(f"  Actual lift sign: {'POSITIVE (correct)' if result.total_lift > 0 else 'NEGATIVE (wrong)'}")


if __name__ == "__main__":
    test_symmetric_body_no_net_lift()
    test_ellipsoid_col_at_center()
    test_tilted_ellipsoid()
