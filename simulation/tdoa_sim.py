"""TDoA simulation"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core import (
    solve_tdoa_pairwise,
    generate_pairwise_tdoa,
    plot_hyperbola_2d,
    get_bounds,
)


def all_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def plot_scene(
    nodes: np.ndarray,
    true_pos: np.ndarray,
    est_pos: np.ndarray,
    pairs_dt: List[Tuple[int, int, float]],
    show_hyperbolas: bool = True,
) -> None:
    # use a rectangular figure so the visualization frame is rectangular by 
    # default
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(nodes[:, 0], nodes[:, 1], c="C0", marker="o", label="nodes")
    for k, p in enumerate(nodes):
        ax.text(p[0] + 0.3, p[1] + 0.3, f"N{k}", color="C0")
    ax.scatter([true_pos[0]], [true_pos[1]], c="C1",
               marker="*", s=120, label="true TX")
    ax.scatter([est_pos[0]], [est_pos[1]], c="C2",
               marker="x", s=80, label="estimate")
    ax.legend()
    # allow rectangular scaling so the plot frame remains rectangular
    ax.set_aspect("auto")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TDoA simulation")

    if show_hyperbolas:
        # compute bounds from nodes and include the true and estimated 
        # positions
        xmin, xmax, ymin, ymax = get_bounds(
            nodes, extra_points=(true_pos, est_pos))

        # set plot limits explicitly and add a rectangular grid (mesh)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # choose ~10 grid lines along each axis for a clear mesh
        nx_ticks = 10
        xticks = np.linspace(xmin, xmax, nx_ticks + 1)
        yticks = np.linspace(ymin, ymax, nx_ticks + 1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

        for (i, j, dt) in pairs_dt:
            # plot implicit hyperbola curve (contour level 0)
            try:
                plot_hyperbola_2d(
                    ax,
                    nodes[i],
                    nodes[j],
                    dt,
                    c=3e8,
                    xlim=(xmin, xmax),
                    ylim=(ymin, ymax),
                    grid=600,
                    alpha=0.25,
                )
            except Exception:
                # plotting should not break the simulation; continue silently
                pass

    plt.tight_layout()
    plt.show()


def main() -> None:
    # example anchors (meters) - a rectangle / square
    nodes = np.array([[-20.0, -10.0],
                      [20.0, -10.0],
                      [20.0,  10.0],
                      [-20.0,  10.0]])

    # true transmitter position (drone)
    true_pos = np.array([-10, -100])

    # create all unique i<j pairs
    pairs = all_pairs(nodes.shape[0])

    # generate noisy dt measurements (seconds)
    # choose a small timing noise sigma (seconds)
    sigma_t = 5e-11  # ~ 0.015 m uncertainty (sigma_t * c)
    pairs_dt = generate_pairwise_tdoa(
        nodes, true_pos, pairs=pairs, c=3e8, sigma_t=sigma_t, seed=0)

    # estimate using existing solver from core.py
    est_pos, res = solve_tdoa_pairwise(
        nodes, pairs_dt, c=3e8, x0=None, loss="soft_l1")

    err_m = float(np.linalg.norm(est_pos - true_pos))
    print(f"true position: {true_pos}")
    print(f"estimated   : {est_pos}")
    print(f"position error (m): {err_m:.3f}")

    # plot scene with optional hyperbola visualizations
    plot_scene(nodes, true_pos, est_pos, pairs_dt, show_hyperbolas=True)


if __name__ == "__main__":
    main()
