# drone_detection_rf_2026
Drone detection using downlink rf signal challenge 

## Quick start (install from source)

Recommended: Python 3.10+. From the repository root:

```bash
# create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# install the package and its runtime dependencies
pip install --upgrade pip
pip install .
```

Run the simulation (non-interactive in CI / servers):

```bash
# run as a module (preserves package imports)
MPLBACKEND=Agg python -m simulation.tdoa_sim

# or use the provided wrapper which ensures repo root is on PYTHONPATH
MPLBACKEND=Agg python scripts/run_tdoa_sim.py
```

Notes:
- The simulation displays a Matplotlib plot by default. For headless runs set `MPLBACKEND=Agg` or run in an environment with a display.
- If you want to install directly from GitHub, adding packaging metadata (pyproject.toml) in this repo allows `pip install git+https://...`.

## Continuous Integration

This repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Installs the package on Ubuntu/macOS/Windows
- Runs import smoke tests
- Runs the simulation in headless mode (Matplotlib Agg backend)

