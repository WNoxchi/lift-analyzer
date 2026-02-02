"""PyVista-based STEP file viewer with interactive controls."""

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from lift_analyzer.cad.step_loader import load_step_as_mesh


class StepViewer:
    """Interactive 3D viewer for STEP files.

    Features:
    - Click and drag to rotate the model
    - Orientation compass in upper-right corner
    - Home button to reset view to default orientation
    - Default view: isometric from front-right-top

    Coordinate System (matches Autodesk Fusion export):
    - X: Right (starboard)
    - Y: Aft (-Y is forward/nose)
    - Z: Up
    """

    # ==========================================================================
    # COORDINATE SYSTEM CONFIGURATION
    # Adjust these values to change the default "home" view orientation.
    # Azimuth: rotation around Z axis (0=+X, 90=+Y, 180=-X, 270=-Y)
    # Elevation: tilt up/down from horizontal (positive = looking down from above)
    # ==========================================================================
    DEFAULT_AZIMUTH = 270.0    # View from +X, +Y quadrant (front-right of aircraft)
    DEFAULT_ELEVATION = 0.0   # Slight top-down angle

    def __init__(self, title: str = "Lift Analyzer - STEP Viewer") -> None:
        """Initialize the viewer.

        Parameters
        ----------
        title : str, optional
            Window title. Default is "Lift Analyzer - STEP Viewer".
        """
        self.title = title
        self.plotter: pv.Plotter | None = None
        self.mesh: pv.PolyData | None = None
        self._default_camera_position: Any = None

    def load(self, filepath: str | Path, deflection: float = 0.1) -> None:
        """Load a STEP file for visualization.

        Parameters
        ----------
        filepath : str | Path
            Path to the STEP file.
        deflection : float, optional
            Mesh quality parameter. Smaller = finer mesh. Default is 0.1.
        """
        self.mesh = load_step_as_mesh(filepath, deflection)

    def load_mesh(self, mesh: pv.PolyData) -> None:
        """Load a PyVista mesh directly (useful for testing).

        Parameters
        ----------
        mesh : pv.PolyData
            The mesh to display.
        """
        self.mesh = mesh

    def _reset_camera(self) -> None:
        """Reset camera to default orientation (45° azimuth, 45° elevation)."""
        if self.plotter is None or self.mesh is None:
            return

        # Reset to fit the model in view
        self.plotter.reset_camera()

        # Apply default orientation
        self.plotter.camera.azimuth = self.DEFAULT_AZIMUTH
        self.plotter.camera.elevation = self.DEFAULT_ELEVATION

        # Store for future resets
        self._default_camera_position = self.plotter.camera_position

    def _on_home_key(self) -> None:
        """Callback for home key press."""
        if self._default_camera_position is not None and self.plotter is not None:
            self.plotter.camera_position = self._default_camera_position
            self.plotter.render()

    def _add_home_button(self) -> None:
        """Add a clickable home button below the compass."""
        if self.plotter is None:
            return

        def home_callback(state: bool) -> None:
            self._on_home_key()

        # Add checkbox widget styled as a button
        # Position below the orientation widget (upper right)
        self.plotter.add_checkbox_button_widget(
            home_callback,
            value=False,
            position=(10, 10),  # Will be repositioned after window sizing
            size=30,
            border_size=2,
            color_on="white",
            color_off="lightgray",
            background_color="gray",
        )

        # Add text label for the home button
        self.plotter.add_text(
            "H: Home",
            position="lower_left",
            font_size=10,
            color="white",
            shadow=True,
        )

    def show(self, interactive: bool = True) -> None:
        """Display the viewer window.

        Parameters
        ----------
        interactive : bool, optional
            If True, show interactive window. If False, return immediately
            (useful for testing). Default is True.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load() or load_mesh() first.")

        # Create plotter
        self.plotter = pv.Plotter(title=self.title)

        # Enable trackball-style rotation (click and drag)
        self.plotter.enable_trackball_style()

        # Add the mesh with nice default appearance
        self.plotter.add_mesh(
            self.mesh,
            color="lightblue",
            show_edges=False,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
        )

        # Add orientation widget (compass) in upper-right corner
        # Coordinate system: X=Right, -Y=Forward (nose), Z=Up
        self.plotter.add_axes(
            interactive=True,
            line_width=3,
            color="white",
            x_color="red",      # X = Right (starboard)
            y_color="green",    # Y = Aft (-Y = forward/nose)
            z_color="blue",     # Z = Up
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            label_size=(0.1, 0.025),
            viewport=(0.75, 0.75, 1.0, 1.0),  # Upper-right quadrant
        )

        # Set up the default camera position
        self._reset_camera()

        # Add key binding for home
        self.plotter.add_key_event("h", lambda: self._on_home_key())
        self.plotter.add_key_event("H", lambda: self._on_home_key())

        # Add home button and instructions
        self._add_home_button()

        # Show the window
        if interactive:
            self.plotter.show()
        else:
            # Non-blocking for testing
            self.plotter.show(auto_close=False, interactive_update=True)

    def close(self) -> None:
        """Close the viewer window."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None


def create_test_mesh() -> pv.PolyData:
    """Create a simple test mesh (box) for testing without a STEP file.

    Returns
    -------
    pv.PolyData
        A simple box mesh.
    """
    return pv.Box(bounds=(-1, 1, -0.5, 0.5, -0.2, 0.2))


def main_demo() -> None:
    """Run a demo with a test box (no STEP file required)."""
    viewer = StepViewer(title="Lift Analyzer - Demo")
    viewer.load_mesh(create_test_mesh())
    viewer.show()


if __name__ == "__main__":
    main_demo()
