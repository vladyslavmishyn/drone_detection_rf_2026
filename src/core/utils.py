"""Utilities for TDoA simulation and signal processing."""
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np


def generate_pairwise_tdoa(
    nodes: np.ndarray,
    tx_pos: np.ndarray,
    pairs: Optional[Iterable[Tuple[int, int]]] = None,
    c: float = 3e8,
    sigma_t: float = 0.0,
    seed: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """Generate pairwise TDoA measurements with optional noise.

    Args:
        nodes: (N,2) array of receiver positions (meters)
        tx_pos: (2,) transmitter position (meters)
        pairs: optional iterable of (i,j) indices; if None, uses all i<j pairs
        c: speed of propagation (m/s, default 3e8)
        sigma_t: standard deviation of additive Gaussian noise on dt (seconds)
        seed: RNG seed for reproducibility

    Returns:
        list of tuples (i, j, dt_seconds) where dt = toa[i] - toa[j]
    """
    nodes = np.asarray(nodes, dtype=float)
    tx_pos = np.asarray(tx_pos, dtype=float)
    if pairs is None:
        N = nodes.shape[0]
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    rng = np.random.default_rng(seed)
    d = np.linalg.norm(nodes - tx_pos, axis=1)  # distances (m)
    toas = d / float(c)  # seconds

    out: List[Tuple[int, int, float]] = []
    for (i, j) in pairs:
        dt = float(toas[i] - toas[j])  # seconds
        if sigma_t and sigma_t > 0.0:
            dt = dt + float(rng.normal(scale=float(sigma_t)))
        out.append((int(i), int(j), float(dt)))
    return out


def get_bounds_from_nodes(
    nodes: np.ndarray,
    extra_points: Optional[Sequence[np.ndarray]] = None,
    margin: float = 0.6,
    min_margin: float = 5.0,
) -> Tuple[float, float, float, float]:
    """Compute plotting bounds that include nodes and extra points.

    Args:
        nodes: (N,2) array of receiver positions
        extra_points: optional sequence of (2,) arrays (e.g., 
        true/estimated positions)
        margin: fraction of span to add as margin
        min_margin: minimum margin in meters

    Returns:
        (xmin, xmax, ymin, ymax) tuple
    """
    nodes = np.asarray(nodes, dtype=float)
    pts = [nodes]
    if extra_points:
        for p in extra_points:
            pts.append(np.atleast_2d(np.asarray(p, dtype=float)))

    all_pts = np.vstack(pts)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = maxs - mins

    m = np.maximum(span * float(margin), float(min_margin))
    xmin, ymin = mins - m
    xmax, ymax = maxs + m
    return float(xmin), float(xmax), float(ymin), float(ymax)


def hyperbola_field_2d(
    ni: np.ndarray,
    nj: np.ndarray,
    dt: float,
    c: float = 3e8,
    xlim: Tuple[float, float] = (-50.0, 50.0),
    ylim: Tuple[float, float] = (-30.0, 30.0),
    grid: int = 600,
):
    """Generate hyperbola implicit field meshgrid.

    The field represents: Z(x,y) = ||(x,y)-ni|| - ||(x,y)-nj|| - c*dt

    Args:
        ni, nj: (2,) receiver positions
        dt: time difference (seconds)
        c: speed of propagation (m/s)
        xlim, ylim: plot axis limits
        grid: grid resolution

    Returns:
        (X, Y, Z) meshgrids
    """
    ni = np.asarray(ni, dtype=float)
    nj = np.asarray(nj, dtype=float)
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    xs = np.linspace(xmin, xmax, int(grid))
    ys = np.linspace(ymin, ymax, int(grid))
    X, Y = np.meshgrid(xs, ys)
    di = np.sqrt((X - ni[0]) ** 2 + (Y - ni[1]) ** 2)
    dj = np.sqrt((X - nj[0]) ** 2 + (Y - nj[1]) ** 2)
    Z = (di - dj) - (float(c) * float(dt))
    return X, Y, Z


def plot_hyperbola_2d(
    ax,
    ni: np.ndarray,
    nj: np.ndarray,
    dt: float,
    c: float = 3e8,
    xlim: Tuple[float, float] = (-50.0, 50.0),
    ylim: Tuple[float, float] = (-30.0, 30.0),
    grid: int = 600,
    alpha: float = 0.35,
):
    """Plot hyperbola on matplotlib axes as zero contour.

    Args:
        ax: matplotlib Axes object
        ni, nj: (2,) receiver positions
        dt: time difference (seconds)
        c: speed of propagation (m/s)
        xlim, ylim: plot axis limits
        grid: grid resolution
        alpha: contour line transparency

    Returns:
        matplotlib QuadContourSet
    """
    X, Y, Z = hyperbola_field_2d(ni, nj, dt, c=c, xlim=xlim, ylim=ylim, 
                                 grid=grid)
    cs = ax.contour(X, Y, Z, levels=[0.0], linewidths=1.0, alpha=float(alpha), 
                    colors="red")
    return cs


def generate_signal(fs: float, duration: float, 
                    f0: float, phase: float = 0.0) -> np.ndarray:
    """Generate complex baseband sinusoid.

    Args:
        fs: sample rate (Hz)
        duration: signal duration (seconds)
        f0: tone frequency (Hz)
        phase: initial phase (radians)

    Returns:
        complex64 numpy array of length int(fs*duration)
    """
    t = np.arange(int(np.round(fs * duration))) / float(fs)
    sig = np.exp(2j * np.pi * float(f0) * t + 1j * float(phase))
    return sig.astype(np.complex64)
