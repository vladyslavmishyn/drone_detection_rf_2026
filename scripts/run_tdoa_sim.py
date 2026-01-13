"""Run TDOA simulation with proper module imports."""
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    
    # Import and run simulation
    from simulation.tdoa_sim import main as sim_main
    return sim_main()


if __name__ == "__main__":
    main()

