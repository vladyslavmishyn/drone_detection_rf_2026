"""Run various test scenarios."""
import sys
from pathlib import Path
from src.core import generate_pairwise_tdoa, solve_tdoa_pairwise
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def scenario_4_node_square():
    """Basic 4-node square scenario."""
    nodes = np.array([
        [-20.0, -10.0],
        [20.0, -10.0],
        [20.0, 10.0],
        [-20.0, 10.0]
    ])
    true_pos = np.array([-10, -100])
    
    pairs = [(i, j) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    pairs_dt = generate_pairwise_tdoa(nodes, true_pos, pairs=pairs, 
                                      sigma_t=5e-11, seed=42)
    
    est_pos, _ = solve_tdoa_pairwise(nodes, pairs_dt)
    error = float(np.linalg.norm(est_pos - true_pos))
    
    print("Scenario: 4-node square")
    print(f"  True position: {true_pos}")
    print(f"  Est. position: {est_pos}")
    print(f"  Error (m): {error:.3f}")
    return error < 1.0  # Pass if error < 1m


if __name__ == "__main__":
    all_pass = scenario_4_node_square()
    sys.exit(0 if all_pass else 1)
