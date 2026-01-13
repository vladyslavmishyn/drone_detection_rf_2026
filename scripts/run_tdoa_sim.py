"""Run simulation.tdoa_sim with the repo root on sys.path."""
import os
import runpy
import sys


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    # Put repo root on sys.path so package imports resolve.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Run the simulation module (preserves package context).
    return runpy.run_module("simulation.tdoa_sim", run_name="__main__")


if __name__ == "__main__":
    main()
