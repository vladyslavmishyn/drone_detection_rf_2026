# TDoA Drone Localization

A Python simulation of Time Difference of Arrival (TDoA) based drone localization using multiple RF receiver nodes.

## Overview

This program simulates a drone detection system that uses TDoA (Time Difference of Arrival) measurements from multiple RF receiver nodes to estimate the position of a drone. The system:

1. Simulates RF signal arrival times at multiple receiver nodes
2. Computes time differences between nodes
3. Uses least squares optimization to estimate the drone's position
4. Visualizes the results and calculates localization error

## Requirements

- Python 3.7 or higher
- Required libraries:
  - numpy >= 1.21.0
  - scipy >= 1.7.0
  - matplotlib >= 3.4.0
  - contextily
  - pyproj

## Installation

1. Install the required dependencies:
```bash
pip install numpy scipy matplotlib contextily pyproj # Can be also pip3 in some systems
```

## Running the Program

Run the simulation with:
```bash
python main.py # Can be also python3 in some systems etc
```