"""Main entry point for the Lift Analyzer application."""

import argparse
import sys
from pathlib import Path

from lift_analyzer.viz.viewer import StepViewer, main_demo


def main() -> int:
    """Run the Lift Analyzer visualizer.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        prog="lift-analyzer",
        description="Visualize STEP files and analyze center of lift.",
    )
    parser.add_argument(
        "step_file",
        nargs="?",
        type=Path,
        help="Path to STEP file to visualize. If not provided, runs demo mode.",
    )
    parser.add_argument(
        "--deflection",
        "-d",
        type=float,
        default=0.1,
        help="Mesh deflection parameter (smaller = finer mesh). Default: 0.1",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with a test shape (no STEP file required).",
    )
    parser.add_argument(
        "--panel-method",
        "-p",
        action="store_true",
        help="Use Source+Doublet panel method for CoL (default: geometric).",
    )
    parser.add_argument(
        "--max-panels",
        "-n",
        type=int,
        default=10000,
        help="Maximum panels for panel method (more = slower but preserves detail). Default: 10000",
    )

    args = parser.parse_args()

    # Demo mode
    if args.demo or args.step_file is None:
        print("Running in demo mode (no STEP file provided)")
        if args.panel_method:
            print("Using Source+Doublet panel method for CoL calculation")
        print("Controls:")
        print("  - Click and drag to rotate")
        print("  - Press 'H' to reset to home view")
        print("  - Close window to exit")
        main_demo(use_panel_method=args.panel_method)
        return 0

    # Load and display STEP file
    step_path = args.step_file
    if not step_path.exists():
        print(f"Error: STEP file not found: {step_path}", file=sys.stderr)
        return 1

    if step_path.suffix.lower() not in (".step", ".stp"):
        print(f"Warning: File may not be a STEP file: {step_path}", file=sys.stderr)

    print(f"Loading STEP file: {step_path}")
    if args.panel_method:
        print("Using Source+Doublet panel method for CoL calculation")
    print("Controls:")
    print("  - Click and drag to rotate")
    print("  - Press 'H' to reset to home view")
    print("  - Close window to exit")

    try:
        viewer = StepViewer(
            title=f"Lift Analyzer - {step_path.name}",
            use_panel_method=args.panel_method,
            max_panels=args.max_panels,
        )
        viewer.load(step_path, deflection=args.deflection)
        viewer.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
