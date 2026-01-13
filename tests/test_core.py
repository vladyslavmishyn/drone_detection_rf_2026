"""Test suite for drone detection algorithms."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tdoa_import():
    """Test that core TDOA module imports correctly."""
    from src.core import solve_tdoa_pairwise, generate_pairwise_tdoa
    assert solve_tdoa_pairwise is not None
    assert generate_pairwise_tdoa is not None
    print("✓ TDOA imports work")


def test_tdoa_basic():
    """Test basic TDOA localization."""
    from src.core import solve_tdoa_pairwise, generate_pairwise_tdoa
    import numpy as np
    
    nodes = np.array([[-20, -10], [20, -10], [20, 10], [-20, 10]], dtype=float)
    true_pos = np.array([0, 0], dtype=float)
    
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pairs_dt = generate_pairwise_tdoa(nodes, true_pos, pairs=pairs, sigma_t=0)
    
    est_pos, _ = solve_tdoa_pairwise(nodes, pairs_dt)
    error = np.linalg.norm(est_pos - true_pos)
    
    assert error < 0.1, f"TDOA error too large: {error}"
    print(f"✓ TDOA localization works (error: {error:.4f}m)")


if __name__ == "__main__":
    test_tdoa_import()
    test_tdoa_basic()
    print("\nAll tests passed!")
