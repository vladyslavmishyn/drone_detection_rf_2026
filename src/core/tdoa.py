"""Time Difference of Arrival (TDoA) localization algorithms."""
import numpy as np
from scipy.optimize import least_squares


def tdoa_pairwise_residual(x, nodes, pairs, c=3e8):
    """Residuals for pairwise TDoA constraints.

    For each (i, j, dt):
        ||x - nodes[i]|| - ||x - nodes[j]||  -  c*dt
    Returns residuals in meters.

    Args:
        x: (D,) estimated target position
        nodes: (N, D) receiver node positions
        pairs: list of (i, j, dt_seconds) tuples
        c: speed of propagation (m/s)

    Returns:
        (M,) residual vector in meters
    """
    x = np.asarray(x, dtype=float)
    nodes = np.asarray(nodes, dtype=float)

    r = np.empty(len(pairs), dtype=float)
    for k, (i, j, dt) in enumerate(pairs):
        di = np.linalg.norm(x - nodes[i])
        dj = np.linalg.norm(x - nodes[j])
        r[k] = (di - dj) - c * dt
    return r


def solve_tdoa_pairwise(nodes, pairs, c=3e8, x0=None, loss="soft_l1"):
    """Estimate target position from pairwise TDoA measurements.

    Uses least-squares optimization to find the position that best fits
    the time difference of arrival constraints.

    Args:
        nodes: (N, D) array of receiver positions (meters)
        pairs: iterable of (i, j, dt_seconds) tuples where dt = toa[i] - toa[j]
        c: speed of propagation (m/s, default 3e8 for RF)
        x0: (D,) initial guess; defaults to centroid of nodes
        loss: loss function for least_squares ('soft_l1', 'linear', etc.)

    Returns:
        (x, res) tuple where:
            - x: (D,) estimated target position
            - res: scipy OptimizeResult object with details
    """
    nodes = np.asarray(nodes, dtype=float)

    if x0 is None:
        x0 = nodes.mean(axis=0)

    def fun(x):
        return tdoa_pairwise_residual(x, nodes, pairs, c=c)
    
    res = least_squares(fun, x0, loss=loss)
    return res.x, res
