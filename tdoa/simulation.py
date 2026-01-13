"""Runner shim that invokes simulation.tdoa_sim."""
import os
import runpy


def _find_simulation_path():
    """Return path to simulation/tdoa_sim.py (relative to this file)."""
    p = os.path.join(os.path.dirname(__file__), "..", "simulation", "tdoa_sim.py")
    return os.path.normpath(p)


def main():
    try:
        # Prefer import when available.
        from simulation.tdoa_sim import main as _sim_main
        return _sim_main()
    except Exception:
        # Fallback: run the script file directly.
        sim_path = _find_simulation_path()
        return runpy.run_path(sim_path, run_name="__main__")


if __name__ == "__main__":
    main()
