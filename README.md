# Drone Detection RF 2026

RF-based drone detection and localization system using Time Difference of Arrival (TDoA) analysis and machine learning classification.

## Overview

This project provides tools for detecting and localizing drones through analysis of their downlink RF signals. The system combines signal processing techniques (TDoA-based multilateration) with machine learning for robust drone identification and positioning.

## Repository Structure

```
drone_detection_rf_2026/
├── src/                          # Main source code package
│   ├── core/                     # Core algorithms and utilities
│   │   ├── tdoa.py              # TDoA localization solver implementation
│   │   └── utils.py             # Signal processing and visualization utilities
│   ├── models/                   # Machine learning models and training pipelines
│   └── hardware/                 # Hardware integration and receiver node software
├── simulation/                   # Simulation and testing environment
│   ├── tdoa_sim.py              # Main TDoA simulation script
│   └── scenarios.py             # Test scenarios and configurations
├── scripts/                      # Executable scripts for common workflows
│   ├── run_tdoa_sim.py          # Script to execute TDoA simulations
│   ├── train_model.py           # Script for training ML models
│   └── deploy_hardware.py       # Hardware deployment utilities
├── tests/                        # Unit tests and validation
│   └── test_core.py             # Tests for core algorithms
├── data/                         # Datasets and data processing
│   └── drone_dataset_npz/       # Processed RF signal datasets (NPZ format)
├── models/                       # Trained model artifacts and checkpoints
├── docs/                         # Documentation
│   ├── architecture.md          # System architecture documentation
│   └── hardware/                # Hardware-specific documentation
└── pyproject.toml               # Python project configuration
```

### Folder Descriptions

- **`src/`**: Contains the main source code organized as a Python package
  - **`core/`**: Fundamental algorithms including TDoA solvers and signal processing utilities
  - **`models/`**: Machine learning model implementations and training pipelines
  - **`hardware/`**: Code for integrating with physical receiver hardware

- **`simulation/`**: Environment for testing and validating algorithms
  - Contains simulation scripts that can run without hardware
  - Useful for development, debugging, and algorithm validation

- **`scripts/`**: Entry points for common operations
  - Provides convenient command-line interfaces to key functionality
  - Handles path setup and module imports automatically

- **`tests/`**: Unit tests ensuring code correctness
  - Validates core algorithms with known inputs/outputs
  - Should be run after any code changes

- **`data/`**: Datasets used for training and testing
  - Contains processed RF signal data in NPZ format
  - Includes preprocessing notebooks and documentation

- **`models/`**: Storage for trained machine learning models
  - Contains serialized model files and training artifacts
  - Used by deployment scripts

- **`docs/`**: Documentation beyond this README
  - Architecture decisions, hardware setup guides, etc.

## Installation

### Requirements
- Python 3.10 or higher
- Dependencies: numpy, scipy, matplotlib

### Setup Steps

1. **Clone the repository** (if not already done)

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
   Or alternatively:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running TDoA Simulations

The main simulation demonstrates drone localization using TDoA:

```bash
python scripts/run_tdoa_sim.py
```

This will:
- Generate a synthetic drone position
- Simulate RF signals received at multiple nodes
- Compute TDoA measurements with realistic noise
- Solve for position using least-squares optimization
- Display visualization of hyperbolas and estimated position

For headless environments (no display):
```bash
MPLBACKEND=Agg python scripts/run_tdoa_sim.py
```

### Training Machine Learning Models

To train drone detection models:

```bash
python scripts/train_model.py
```

Note: This script is currently a placeholder and will be implemented with the full training pipeline.

### Running Tests

Validate that the core algorithms work correctly:

```bash
python -m pytest tests/
```

Or run specific test file:
```bash
python tests/test_core.py
```

### Custom Simulations

Modify `simulation/scenarios.py` to create custom test scenarios, then run:

```bash
python simulation/scenarios.py
```

## Development Workflow

### Adding New Algorithms

1. Implement new functions in appropriate `src/` submodules
2. Add corresponding tests in `tests/`
3. Update any relevant scripts in `scripts/` if needed
4. Run tests to ensure everything works

### Working with the Codebase

- All imports should use absolute paths from the repository root
- Scripts in `scripts/` handle path setup automatically
- For development work, ensure the virtual environment is activated
- Run tests frequently to catch regressions

### Key Modules

- **`src.core.tdoa`**: TDoA position solving algorithms
- **`src.core.utils`**: Signal generation, hyperbola plotting, utilities
- **`simulation.tdoa_sim`**: Main simulation orchestration

## License

See [LICENSE](LICENSE) file.
