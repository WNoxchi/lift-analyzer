# Lift Analyzer

Finite element analysis tool for aircraft design. Ingests STEP files, displays CAD geometry, and calculates Center of Lift (Center of Pressure).

# Tech Stack
- Language: Python 3.11+
- CAD/Geometry: pythonocc-core (OpenCASCADE wrapper) for STEP file parsing
- Visualization: PyVista or trimesh for 3D rendering
- Numerics: NumPy, SciPy for FEA computations
- GUI: PyQt6 or browser-based (Dash/Streamlit)

# Development Commands
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run application
python -m lift_analyzer

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
- Coordinate system: X = forward, Y = starboard, Z = up (aircraft body frame)
- Center of Pressure (CoP) = Center of Lift (CoL) - use interchangeably
- Distinguish between reference area (Sref) and wetted area

# Architecture Notes
- STEP import produces a TopoDS_Shape (OpenCASCADE)
- Mesh generation converts geometry to triangular surface mesh
- Pressure distribution applied via panel method or CFD import
- CoL calculated as pressure-weighted centroid

# Testing
- Unit tests for numerical computations must include tolerance checks
- Use pytest-approx for floating point comparisons
- Sample STEP files in `data/` for integration tests
