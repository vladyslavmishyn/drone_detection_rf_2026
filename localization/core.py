import numpy as np
from scipy.optimize import least_squares

def tdoa_pairwise_residual(x, nodes, pairs, c=3e8):
    """
    Residuals for pairwise TDoA constraints.

    For each (i, j, dt):
        ||x - nodes[i]|| - ||x - nodes[j]||  -  c*dt
    Returns residuals in meters.
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
    """
    Estimate TX position from pairwise TDoA delays.

    nodes: (N, D) array
    pairs: iterable of (i, j, dt_seconds)
    """
    nodes = np.asarray(nodes, dtype=float)

    if x0 is None:
        x0 = nodes.mean(axis=0)

    fun = lambda x: tdoa_pairwise_residual(x, nodes, pairs, c=c)
    res = least_squares(fun, x0, loss=loss)
    return res.x, res
