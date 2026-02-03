"""PyVista-based STEP file viewer with interactive controls."""

from pathlib import Path
from typing import Any

import pyvista as pv

from lift_analyzer.aero.center_of_lift import (
    AirflowDirection,
    CenterOfLiftResult,
    calculate_center_of_lift,
)
from lift_analyzer.aero.panel_method import (
    PanelMethodResult,
    calculate_center_of_lift_panel_method,
)
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
    DEFAULT_AZIMUTH = 270.0    # View from +X, -Y quadrant (front-left of aircraft)
    DEFAULT_ELEVATION = 0.0   # Slight top-down angle

    # ==========================================================================
    # AIRFLOW CONFIGURATION
    # Default airflow direction: from -Y to +Y (aircraft nose points -Y)
    # ==========================================================================
    DEFAULT_AIRFLOW = AirflowDirection.POSITIVE_Y

    def __init__(
        self,
        title: str = "Lift Analyzer - STEP Viewer",
        use_panel_method: bool = False,
        max_panels: int = 2000,
    ) -> None:
        """Initialize the viewer.

        Parameters
        ----------
        title : str, optional
            Window title. Default is "Lift Analyzer - STEP Viewer".
        use_panel_method : bool, optional
            If True, use Source+Doublet panel method for CoL calculation.
            If False (default), use geometric projected area method.
        max_panels : int, optional
            Maximum panels for panel method. More panels = better detail
            but slower computation. Default is 2000.
        """
        self.title = title
        self.use_panel_method = use_panel_method
        self.max_panels = max_panels
        self.plotter: pv.Plotter | None = None
        self.mesh: pv.PolyData | None = None
        self.col_result: CenterOfLiftResult | None = None
        self.panel_result: PanelMethodResult | None = None
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
        self._calculate_col()

    def load_mesh(self, mesh: pv.PolyData) -> None:
        """Load a PyVista mesh directly (useful for testing).

        Parameters
        ----------
        mesh : pv.PolyData
            The mesh to display.
        """
        self.mesh = mesh
        self._calculate_col()

    def _calculate_col(self) -> None:
        """Calculate center of lift for the loaded mesh."""
        if self.mesh is None:
            return

        if self.use_panel_method:
            # Use Source + Doublet panel method
            airflow_dir = (
                self.DEFAULT_AIRFLOW.value
                if isinstance(self.DEFAULT_AIRFLOW, AirflowDirection)
                else self.DEFAULT_AIRFLOW
            )
            self.panel_result = calculate_center_of_lift_panel_method(
                self.mesh, airflow_direction=airflow_dir, max_panels=self.max_panels
            )
            # Create a compatible result for the visualization
            self.col_result = CenterOfLiftResult(
                position=self.panel_result.center_of_lift,
                y_coordinate=self.panel_result.center_of_lift[1],
                total_projected_area=self.panel_result.total_lift,
                num_faces_considered=len(self.panel_result.pressure_coefficients),
            )
        else:
            # Use geometric projected area method
            self.col_result = calculate_center_of_lift(self.mesh, self.DEFAULT_AIRFLOW)

    def _add_col_line(self) -> None:
        """Add a red horizontal line showing the Center of Lift Y-position."""
        if self.plotter is None or self.mesh is None or self.col_result is None:
            return

        # Get mesh bounds to determine line length and reference point
        bounds = self.mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        x_min, x_max = bounds[0], bounds[1]
        y_min = bounds[2]  # Front of aircraft (nose at -Y)
        z_center = (bounds[4] + bounds[5]) / 2  # Center Z for visibility

        # Extend line slightly beyond mesh bounds
        x_padding = (x_max - x_min) * 0.1
        line_x_min = x_min - x_padding
        line_x_max = x_max + x_padding

        # Create line at the CoL Y-coordinate
        col_y = self.col_result.y_coordinate
        line = pv.Line(
            pointa=(line_x_min, col_y, z_center),
            pointb=(line_x_max, col_y, z_center),
        )

        # Add the line to the plotter
        self.plotter.add_mesh(
            line,
            color="red",
            line_width=4,
            label="Center of Lift",
        )

        # Calculate distance from front of aircraft (nose)
        col_distance_from_nose = col_y - y_min

        # Add a label showing distance from nose and method used
        method_label = "Panel Method" if self.use_panel_method else "Geometric"
        self.plotter.add_text(
            f"CoL ({method_label}): {col_distance_from_nose:.3f} from front",
            position="upper_left",
            font_size=12,
            color="red",
            shadow=True,
        )

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

        # Add Center of Lift visualization (red line)
        self._add_col_line()

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


def main_demo(use_panel_method: bool = False) -> None:
    """Run a demo with a test box (no STEP file required).

    Parameters
    ----------
    use_panel_method : bool, optional
        If True, use panel method for CoL calculation. Default is False.
    """
    viewer = StepViewer(title="Lift Analyzer - Demo", use_panel_method=use_panel_method)
    viewer.load_mesh(create_test_mesh())
    viewer.show()


if __name__ == "__main__":
    main_demo()
