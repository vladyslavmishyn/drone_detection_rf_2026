import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import contextily as cx


fig = None
ax = None
drone_scatter = None


drones = {}
drone_labels = {}

def bounds_with_padding(xy: np.ndarray, pad_ratio: float = 0.35):
    """
    xy: (x, y) in meters (EPSG:3857)
    """
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 100.0
    if dy == 0:
        dy = 100.0

    padx = dx * pad_ratio
    pady = dy * pad_ratio

    return xmin - padx, xmax + padx, ymin - pady, ymax + pady


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


def plot_map(receivers_coords: np.ndarray) -> None:
    """
    receivers_coords: array of (lon, lat) in degrees (EPSG:4326)
    """

    global fig, ax, drone_scatter

    gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in receivers_coords],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_position([0, 0, 1, 1])

    gdf.plot(ax=ax, markersize=90)

    for k, p in enumerate(gdf.geometry):
        ax.text(p.x + 80, p.y + 80, f"N{k}")

    xy = np.array([(p.x, p.y) for p in gdf.geometry], dtype=float)
    xmin, xmax, ymin, ymax = bounds_with_padding(xy, pad_ratio=0.35)
    xmin, xmax, ymin, ymax = match_fig_aspect(
        xmin, xmax, ymin, ymax,
        fig.get_figwidth(), fig.get_figheight()
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("auto")

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, reset_extent=False)

    ax.set_axis_off()
    plt.show(block=False)



def plot_drone(coordinates, drone_id, hyperbolas=None):
    """
    Plot the drone / update it's coordinates on the live map given its coordinates and id
    coordinates is an array (latitude, longitude)
    """
    #if the map has not been initialized yet, raise an error
    #if the drone with drone_id was already plotted, update its position
    #if position outside of bounds, do not plot

    global fig, ax, drones, drone_labels

    if fig is None or ax is None:
        raise RuntimeError("Call plot_map(receivers_coords) before plot_drone(...).")

    lat, lon = float(coordinates[0]), float(coordinates[1])

    g = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
    x, y = float(g.iloc[0].x), float(g.iloc[0].y)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    inside = (xmin <= x <= xmax) and (ymin <= y <= ymax)

    if drone_id not in drones:
        drones[drone_id] = ax.scatter(
            [x], [y],
            marker="o",
            s=90,
            zorder=10,
        )
        drone_labels[drone_id] = ax.text(
            x + 100, y + 100,
            str(drone_id),
            zorder=20,
            fontsize=9,
            weight="bold",
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    if not inside:
        drones[drone_id].set_visible(False)
        drone_labels[drone_id].set_visible(False)
        return

    drones[drone_id].set_visible(True)
    drone_labels[drone_id].set_visible(True)

    drones[drone_id].set_offsets([[x, y]])
    drone_labels[drone_id].set_position((x + 100, y + 100))
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# (lon, lat) around Sumy (approx square)
nodes = np.array([
    [34.7680, 50.9430],  # N0
    [34.8380, 50.9360],  # N1
    [34.8250, 50.8980],  # N2
    [34.7750, 50.9060],  # N3
])


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

        plot_drone((lat_a, lon_a), drone_id="A")
        plot_drone((lat_b, lon_b), drone_id="B")

        plt.pause(0.05) 
        t += 0.08


if __name__ == "__main__":
    plot_map(nodes)

    simulate_drone_updates()