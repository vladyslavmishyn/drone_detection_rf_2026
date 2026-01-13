# System Architecture

## Overview

The drone detection system uses Time Difference of Arrival (TDoA) multilateration combined with machine learning for detection and localization.

## Signal Processing Pipeline

1. **RF Signal Reception** - Receiver nodes capture downlink signals from drones
2. **Time Synchronization** - Precise timing alignment across receiver array
3. **TDoA Measurement** - Calculate time differences between receiver pairs
4. **Multilateration** - Solve for drone position using TDoA constraints
5. **Position Estimation** - Least-squares optimization for target location

## Machine Learning Pipeline

1. **Feature Extraction** - Extract discriminative features from RF signals
2. **Training** - Train random forest classifier on labeled samples
3. **Classification** - Identify drone presence and type
4. **Validation** - Performance metrics on test dataset

## System Components

### TDoA Localization (`src/core/tdoa.py`)
- Least-squares solver for pairwise TDoA constraints
- Handles multiple receiver configurations
- Noise-tolerant optimization

### Signal Processing (`src/core/utils.py`)
- TDoA measurement simulation
- Visualization of hyperbolic loci
- Test signal generation

### ML Models (`src/models/`)
- Classification pipeline (to be implemented)
- Feature engineering
- Model persistence and deployment

### Hardware Integration (`src/hardware/`)
- Receiver node software (to be implemented)
- Device communication protocols
- Configuration management

## Dataflow

```
Receiver Array
    ↓
Signal Capture
    ↓
TDoA Measurement
    ↓
Multilateration    →    ML Classification
    ↓                          ↓
Position Estimate      Drone Detection
```

## Coordinate System

- Receiver positions: Cartesian coordinates (meters)
- Target position: 2D or 3D coordinates
- Speed of propagation: 3×10⁸ m/s (RF in free space)
