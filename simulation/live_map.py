import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import contextily as cx

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.tdoa_sim import all_pairs
from src.core import (
    generate_pairwise_tdoa,
    plot_hyperbola_2d,
    get_bounds,
)

"""
run plot_map(receivers_coords) once to visualize the map
run plot_drone((lat, lon), drone_id, show_hyperbolas) multiple times to update drone positions

"""


nodes = None
fig = None
ax = None
drone_scatter = None


drones = {}
drone_labels = {}

def match_fig_aspect(xmin, xmax, ymin, ymax, fig_w, fig_h):
    """Expand bounds so (xrange/yrange) matches (fig_w/fig_h)"""
    dx = xmax - xmin
    dy = ymax - ymin
    target = fig_w / fig_h
    current = dx / dy

    if current < target:
        # too narrow, widen x
        new_dx = dy * target
        extra = (new_dx - dx) / 2.0
        xmin -= extra
        xmax += extra
    else:
        # too wide, increase y
        new_dy = dx / target
        extra = (new_dy - dy) / 2.0
        ymin -= extra
        ymax += extra

    return xmin, xmax, ymin, ymax


def plot_map(receivers_coords):
    """
    receivers_coords: array of (lon, lat) in degrees (EPSG:4326)
    """

    global fig, ax, drone_scatter, nodes
    nodes = np.array(receivers_coords)

    gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in nodes],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_position([0, 0, 1, 1])

    gdf.plot(ax=ax, markersize=90)

    for k, p in enumerate(gdf.geometry):
        ax.text(p.x + 80, p.y + 80, f"N{k}")

    xy = np.array([(p.x, p.y) for p in gdf.geometry], dtype=float)
    xmin, xmax, ymin, ymax = get_bounds(xy)
    xmin, xmax, ymin, ymax = match_fig_aspect(
        xmin, xmax, ymin, ymax,
        fig.get_figwidth(), fig.get_figheight()
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("auto")

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, reset_extent=False)

    ax.set_axis_off()


def plot_drone(coordinates, drone_id, show_hyperbolas=True):
    """
    coordinates: (latitude, longitude) in degrees
    """
    global fig, ax, drones, drone_labels

    if fig is None or ax is None:
        raise RuntimeError("Call plot_map() before plot_drone(...).")

    #convert drone to meters
    lat, lon = float(coordinates[0]), float(coordinates[1])
    g = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
    x, y = float(g.iloc[0].x), float(g.iloc[0].y)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    inside = (xmin <= x <= xmax) and (ymin <= y <= ymax)

    # create the drone if not exists
    if drone_id not in drones:
        drones[drone_id] = ax.scatter([x], [y], marker="*", s=300, c="red", 
                                      edgecolors="black", linewidths=2, zorder=10)
        drone_labels[drone_id] = ax.text(
            x + 150, y + 150,
            str(drone_id),
            zorder=20,
            fontsize=12,
            weight="bold",
            color="red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8)
        )
        drones[drone_id]._hyperbolas = []  # store artists here

    # hide if outside
    if not inside:
        drones[drone_id].set_visible(False)
        drone_labels[drone_id].set_visible(False)

        # remove hyperbolas
        for art in drones[drone_id]._hyperbolas:
            try:
                art.remove()
            except Exception:
                pass
        drones[drone_id]._hyperbolas.clear()
        return

    
    # if inside, update the position
    drones[drone_id].set_visible(True)
    drone_labels[drone_id].set_visible(True)
    drones[drone_id].set_offsets([[x, y]])
    drone_labels[drone_id].set_position((x + 100, y + 100))


    # Hyperbolas
    if show_hyperbolas:
        # convert nodes to meters
        nodes_m = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in nodes],
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        nodes_xy = np.array([(p.x, p.y) for p in nodes_m])

        # clear old hyperbolas
        for art in drones[drone_id]._hyperbolas:
            try:
                art.remove()
            except Exception:
                pass
        drones[drone_id]._hyperbolas.clear()

        sigma_t = 5e-11
        pairs_dt = generate_pairwise_tdoa(
            nodes_xy,
            np.array([x, y]),
            pairs=all_pairs(nodes_xy.shape[0]),
            c=3e8,
            sigma_t=sigma_t,
            seed=0,
        )

        for (i, j, dt) in pairs_dt:
            try:
                cs = plot_hyperbola_2d(
                    ax,
                    nodes_xy[i],
                    nodes_xy[j],
                    dt,
                    c=3e8,
                    xlim=(xmin, xmax),
                    ylim=(ymin, ymax),
                    grid=300,
                    alpha=0.15,
                )
                if hasattr(cs, "collections"):
                    drones[drone_id]._hyperbolas.extend(cs.collections)
            except Exception:
                pass


def simulate_drone_updates():
    # center (roughly Sumy)
    center_lat = 50.922
    center_lon = 34.800

    t = 0.0
    while True:
        # Drone A
        lat_a = center_lat + 0.10 * np.sin(t)
        lon_a = center_lon + 0.15 * np.cos(t)

        # Drone B
        lat_b = center_lat + 0.06 * np.sin(t + 1.5)
        lon_b = center_lon + 0.10 * np.cos(t + 1.5)

        # Drone C
        lat_c = center_lat + 0.006 * np.sin(t + 1.5)
        lon_c = center_lon + 0.010 * np.cos(t + 1.5)

        plot_drone((lat_a, lon_a), drone_id="A")
        #plot_drone((lat_b, lon_b), drone_id="B")
        plot_drone((lat_c, lon_c), drone_id="C")

        plt.pause(0.05) 
        t += 0.08


if __name__ == "__main__":
    # (lon, lat) around Sumy
    receivers = np.array([
        [34.7680, 50.9430],  # N0
        [34.8380, 50.9360],  # N1
        [34.8250, 50.8980],  # N2
        [34.7750, 50.9060],  # N3
    ])
    plot_map(receivers)

    simulate_drone_updates()