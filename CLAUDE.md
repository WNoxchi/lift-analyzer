# Lift Analyzer

Finite element analysis tool for aircraft design. Ingests STEP files, displays CAD geometry, and calculates Center of Lift (Center of Pressure).

# Tech Stack
- Language: Python 3.11+
- CAD/Geometry: pythonocc-core (OpenCASCADE wrapper) for STEP file parsing
- Visualization: PyVista for 3D rendering
- Numerics: NumPy, SciPy for computations
- Acceleration: Numba for JIT-compiled panel method (required dependency)
- GUI: PyVista interactive viewer

# Python Environment Requirements

**IMPORTANT**: All Python package installations and executions MUST use the conda environment named `col`. Do NOT install packages to the system's global Python environment.

### Environment Setup

Before running any Python commands, check if the `col` environment exists and create it if needed:

```bash
# Check if environment exists
conda env list | grep col

# If it does NOT exist, create it:
conda create -n col python=3.11 -y
conda activate col
conda install -c conda-forge pythonocc-core -y
pip install -r requirements.txt
pip install -e .

# If it exists, just activate it:
conda activate col
```

### Installing New Dependencies

When adding new dependencies:
1. Activate the `col` environment first: `conda activate col`
2. For conda-forge packages (like pythonocc-core): `conda install -c conda-forge <package>`
3. For pip packages: `pip install <package>`
4. Update requirements.txt if adding pip dependencies

# Development Commands

**Note**: Always ensure `conda activate col` has been run before executing these commands.

```bash
# Install pythonocc-core (conda required)
conda install -c conda-forge pythonocc-core

# Install package in editable mode (required once, then changes take effect immediately)
pip install -e .

# Run application
python -m lift_analyzer model.step                    # geometric method (fast)
python -m lift_analyzer model.step --panel-method     # panel method (slower, more accurate)
python -m lift_analyzer model.step -p -n 5000         # panel method with 5000 panels
python -m lift_analyzer --demo                        # no STEP file needed

# Run tests (single file preferred)
pytest tests/test_file.py -v

# Run all tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

# Code Style
- Follow PEP 8, enforced via Ruff
- Type hints required for all public functions
- Docstrings: NumPy style for scientific/engineering context
- 88-character line length (Black default)

# Project Structure
```
lift_analyzer/
├── src/lift_analyzer/
│   ├── __init__.py
│   ├── cad/           # STEP file parsing, geometry handling
│   ├── fea/           # Finite element analysis, mesh generation
│   ├── aero/          # Aerodynamic calculations, pressure distribution
│   ├── viz/           # 3D visualization, rendering
│   └── ui/            # User interface components
├── tests/
├── data/              # Sample STEP files for testing
└── docs/
```

# Domain Conventions
- Use SI units internally (meters, Pascals, Newtons)
- Coordinate system: X = right (starboard), -Y = forward (nose), Z = up (matches Autodesk Fusion export)
- Center of Pressure (CoP) = Center of Lift (CoL) - use interchangeably
- Distinguish between reference area (Sref) and wetted area

# Architecture Notes
- STEP import produces a TopoDS_Shape (OpenCASCADE)
- Mesh generation converts geometry to triangular surface mesh
- Pressure distribution computed via Source+Doublet panel method
- CoL calculated as pressure-weighted centroid

# Panel Method Implementation

The panel method (`src/lift_analyzer/aero/panel_method.py`) implements a Source+Doublet Dirichlet formulation:

## Pipeline
1. **Mesh decimation** - Iterative reduction to target panel count (default 10k)
2. **Coincident panel removal** - Detects/removes overlapping panels from thin surfaces
3. **Influence matrix computation** - Numba-accelerated O(n²) calculation
4. **Linear solve** - Least-squares solution for doublet strengths
5. **Surface velocity** - Computed from doublet gradient (neighbor-based)
6. **Pressure coefficients** - From Bernoulli equation, with IQR-based outlier filtering
7. **Lift distribution** - From Cp × area × normal projection
8. **Center of lift** - Lift-weighted centroid

## Key Files
- `panel_method.py` - Main solver, pure Python reference implementations
- `panel_method_numba.py` - Numba JIT-compiled influence functions (required)

## Known Limitations
- Results oscillate with panel count rather than converging perfectly
- Numerical instability from thin CAD surfaces (mitigated by outlier filtering)
- No wake modeling (limits accuracy for high-lift configurations)
- Best results at ~10,000 panels for typical aircraft models

## Planned Improvements
- Consolidate panel_method.py and panel_method_numba.py into single file
- Consider geometric lift estimation as fallback method
- Better mesh preprocessing for thin surfaces

# Testing
- Unit tests for numerical computations must include tolerance checks
- Use pytest-approx for floating point comparisons
- Validation tests in `tests/test_panel_validation.py` for known geometries
- Sample STEP files in `data/` for integration tests

# Current State of the Project:
  Working Features:
  - Iterative mesh decimation (fixes the 98% reduction bug)
  - Numba-accelerated panel method solver (10-100x speedup)
  - IQR-based outlier filtering for numerical stability
  - Coincident panel removal for thin CAD surfaces
  - Default 10,000 panels (best stability based on testing)

  Key Files:
  - src/lift_analyzer/aero/panel_method.py - Main solver with pure Python reference
  - src/lift_analyzer/aero/panel_method_numba.py - Required Numba JIT functions
  - tests/test_panel_method.py - Unit tests including decimation and Numba consistency
  - tests/test_panel_validation.py - Validation tests for known geometries

  Next Steps (when ready to continue):
  1. Consolidate panel_method.py and panel_method_numba.py into a single file
  2. Consider adding geometric lift estimation as a comparison/fallback method
  3. Further investigate convergence behavior with panel count

  The project is in a usable state with python -m lift_analyzer model.step --panel-method providing
  reasonable CoL estimates at the default 10k panel count.
