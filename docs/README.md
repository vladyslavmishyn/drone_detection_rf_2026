# TDoA Localization System - Technical Overview

## Function Map

### Core Modules (`src/core/`)

**gcc.py** - Cross-correlation algorithms
- `gcc_phat(sig1, sig2, fs)` → (correlation, lags)
  - Inputs: Two complex signals, sample rate
  - Computes GCC-PHAT for TDoA estimation
  - Returns: Correlation function and lag array

**iq_handler.py** - Signal generation & I/O
- `generate_signal(fs, duration, f0, phase)` → complex array
  - Creates complex baseband tone
- `generate_chirp(fs, n_samples, bandwidth, phase)` → complex array
  - Creates linear FM chirp
- `save_iq_file(filepath, data)` → None
  - Writes interleaved I/Q to binary file
- `write_metadata(filepath, fs, n_samples, n_packets, receivers, tx_pos)` → None
  - Saves plain-text metadata with receiver/TX positions
- `load_iq_dataset(directory)` → (signals_list, metadata_dict)
  - Loads all .iq files and metadata from directory

**utils.py** - Coordinate & visualization utilities
- `lonlat_to_xy_m(lonlat_array)` → xy_meters
  - Converts GPS to EPSG:3857 meters
- `xy_m_to_lonlat(xy_array)` → lonlat_degrees
  - Inverse coordinate transform
- `get_bounds(points, extra_points, margin)` → (xmin, xmax, ymin, ymax)
  - Calculates map bounds with margin
- `match_fig_aspect(xmin, xmax, ymin, ymax, width, height)` → adjusted_bounds
  - Corrects bounds for figure aspect ratio

**tdoa.py** - Localization solver
- `solve_tdoa_pairwise(receivers_xy, tdoa_pairs, c)` → (estimated_xy, result)
  - Inputs: Receiver positions (m), TDoA measurements (s), speed of light
  - Solves hyperbolic multilateration via least-squares
  - Returns: Estimated TX position and optimization result

### Notebooks (`docs/notebooks/`)

**signal_gen_real.ipynb** - Generates test data
- Uses `generate_chirp()` to create TX signal
- Applies geometric delays via `apply_delay()` (local FIR-based function)
- Saves to `.iq` files via `save_iq_file()` and `write_metadata()`

**gcc_tdoa_iq.ipynb** - Processes data & localizes
- Loads data via `load_iq_dataset()`
- Computes TDoA using `gcc_phat()` for all receiver pairs
- Estimates TX position via `solve_tdoa_pairwise()`
- Visualizes results using coordinate transforms

## System Integration

```
Signal Generation Flow:
generate_chirp() → apply_delay() → save_iq_file() + write_metadata()

Processing Flow:
load_iq_dataset() → gcc_phat() → solve_tdoa_pairwise() → visualization
```

## Optimization Recommendations

### Current Issues
1. **Duplicate functions**: `lonlat_to_xy_m()` exists in both notebook and utils.py
2. **Missing integration**: `apply_delay()` only in notebook, should be in iq_handler.py
3. **No error handling**: File I/O and coordinate transforms lack validation
4. **Hardcoded values**: Speed of light, default packet sizes scattered across files

### Suggested Structure

```
src/core/
├── signal.py          # generate_signal, generate_chirp, apply_delay
├── gcc.py             # gcc_phat, advanced weighting functions
├── tdoa.py            # solve_tdoa_pairwise, residual functions
├── io.py              # save_iq_file, load_iq_dataset, write_metadata
├── geo.py             # lonlat_to_xy_m, xy_m_to_lonlat
└── visualization.py   # get_bounds, match_fig_aspect, plotting helpers

src/config.py          # Constants (c, default fs, packet sizes)
```

### Priority Actions
1. **Move `apply_delay()` to signal.py** - Needed by both notebooks
2. **Consolidate coordinate functions** - Remove duplicates
3. **Add validation** - Check file existence, coordinate bounds, signal lengths
4. **Create config.py** - Centralize constants
5. **Add unit tests** - Verify GCC, coordinate transforms, file I/O
6. **Document API** - Add docstrings with equations for TDoA algorithms

### Research-Ready Enhancements
- **Logging**: Add structured logging for experiment tracking
- **Configuration files**: YAML/JSON for experiment parameters
- **Batch processing**: Process multiple datasets automatically
- **Performance metrics**: SNR, GDOP, Cramér-Rao bounds
- **Export results**: CSV/HDF5 format for analysis in R/MATLAB
