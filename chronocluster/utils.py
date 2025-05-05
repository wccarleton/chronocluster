#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster

import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
import rasterio
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from scipy.ndimage import zoom
from statsmodels.distributions.empirical_distribution import ECDF
from pyproj import CRS
from shapely.geometry import Point as ShapelyPoint

from chronocluster.clustering import Point


def _largest_divisor(n, max_divisors=10):
    "Utility function for legible axis tick mark density."
    for i in range(max_divisors, 0, -1):
        if n % i == 0:
            return i
    return 1


def clustering_heatmap(
    results, distances, time_slices, result_type="K", save=None, **kwargs
):
    """
    Plot a heatmap of either Ripley's K function or the pair correlation function over time and distance.

    Parameters:
    -----------
    results (np.ndarray): A 3D array where the first dimension is the distance, the second dimension is the time slice,
                          and the third dimension is the iteration.
    distances (array-like): Array of distances at which K or g was calculated.
    time_slices (array-like): Array of time slices.
    result_type (str): The type of results being plotted ('K' for Ripley's K function or 'g' for pair correlation function).
    save (str or None): If a string is provided, specifies the filename to save the plot (e.g., 'heatmap.png').
    **kwargs : dict: Additional keyword arguments passed to `plt.savefig`.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Matplotlib figure instance.
    ax : matplotlib.axes._subplots.AxesSubplot
        The Axes object for the plot.
    """
    mean_values = np.mean(results, axis=2)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the heatmap
    sns.heatmap(
        mean_values,
        xticklabels=time_slices,
        yticklabels=distances,
        cmap="viridis",
        cbar_kws={"label": f"Mean {result_type}(d)"},
        ax=ax,
    )

    # Adjust x and y axis ticks using the largest divisor
    x_divisor = _largest_divisor(len(time_slices))
    y_divisor = _largest_divisor(len(distances))

    x_ticks_indices = np.arange(0, len(time_slices), len(time_slices) // x_divisor)
    y_ticks_indices = np.arange(0, len(distances), len(distances) // y_divisor)

    ax.set_xticks(x_ticks_indices)
    ax.set_xticklabels(np.round(time_slices[x_ticks_indices], 2))

    ax.set_yticks(y_ticks_indices)
    ax.set_yticklabels(np.round(distances[y_ticks_indices], 2))

    # Set axis labels and title
    ax.set_xlabel("Time Slices")
    ax.set_ylabel("Distances")
    ax.set_title(f"Heatmap of Mean {result_type}(d) Function Over Time and Distance")

    # Invert the y-axis for proper heatmap orientation
    ax.invert_yaxis()

    # Save the plot if the save parameter is specified
    if save:
        plt.savefig(fname=save, **kwargs)

    return fig, ax


def pdiff_heatmap(
    p_diff_array, time_slices, support, condition="greater", save=None, **kwargs
):
    """
    Plot the heatmap of probability values.

    Parameters:
    -----------
    p_diff_array (np.ndarray): Array of probability values with shape (distances, time_slices).
    time_slices (np.ndarray): Array of time slice values.
    support (np.ndarray): Array of pairwise distance values.
    condition (str): Condition to apply for probability calculation ('greater' or 'less').
    save (str or None): If a string is provided, specifies the filename to save the plot (e.g., 'heatmap.png').
    **kwargs : dict: Additional keyword arguments passed to `plt.savefig`.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Matplotlib figure instance.
    ax : matplotlib.axes._subplots.AxesSubplot
        The Axes object for the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the condition label for the color bar
    if condition == "greater":
        cond_label = {"label": "P( > observed)"}
    elif condition == "less":
        cond_label = {"label": "P( <= observed)"}
    else:
        raise ValueError("Invalid condition. Use 'greater' or 'less'.")

    # Plot the heatmap
    sns.heatmap(
        p_diff_array,
        xticklabels=time_slices,
        yticklabels=support,
        cmap="viridis_r",
        vmin=0,
        cbar_kws=cond_label,
        ax=ax,
    )

    # Set axis labels and title
    ax.set_xlabel("Time Slices")
    ax.set_ylabel("Pairwise Distances")
    ax.set_title("Probability Heat Map")

    # Adjust x and y axis ticks using the largest divisor
    x_divisor = _largest_divisor(len(time_slices))
    y_divisor = _largest_divisor(len(support))

    x_ticks_indices = np.arange(0, len(time_slices), len(time_slices) // x_divisor)
    y_ticks_indices = np.arange(0, len(support), len(support) // y_divisor)

    ax.set_xticks(x_ticks_indices)
    ax.set_xticklabels(np.round(time_slices[x_ticks_indices], 2))

    ax.set_yticks(y_ticks_indices)
    ax.set_yticklabels(np.round(support[y_ticks_indices], 2))

    # Invert the y-axis for proper heatmap orientation
    ax.invert_yaxis()

    # Save the plot if a filename is provided
    if save:
        plt.savefig(fname=save, **kwargs)

    return fig, ax


def plot_derivative(results, distances, time_slices, t_index=0, result_type="K"):
    """
    Plot the derivative of Ripley's K or L function over distance for a specific time slice.

    Parameters:
    results (np.ndarray): A 3D array where the first dimension is the distance, the second dimension is the time slice, and the third dimension is the iteration.
    distances (array-like): Array of distances at which K or L was calculated.
    time_slices (array-like): Array of time slices.
    t_index (int): Index of the time slice desired for plotting.
    result_type (str): The type of results being plotted ('K' for Ripley's K function, 'L' for Ripley's L function).
    """
    mean_values = np.mean(results[:, t_index, :], axis=1)
    derivative_values = np.gradient(mean_values, distances)

    plt.figure(figsize=(8, 6))
    plt.plot(distances, derivative_values, label=f"d{result_type}/dd")
    plt.xlabel("Distance")
    plt.ylabel(f"d{result_type}/dd")
    plt.title(f"Derivative of {result_type}(d) for Time Slice {time_slices[t_index]}")
    plt.legend()
    plt.show()


def plot_l_diff(l_results, distances, time_slices):
    """
    Plot the difference d - L(d) over distance for a specific time slice.

    Parameters:
    l_results (np.ndarray): A 3D array of L function values.
    distances (array-like): Array of distances at which L was calculated.
    time_slices (array-like): Array of time slices.
    """
    t_index = 5  # For example, the 6th time slice
    mean_l_values = np.mean(l_results[:, t_index, :], axis=1)
    l_diff = mean_l_values - distances

    plt.figure(figsize=(8, 6))
    plt.plot(distances, l_diff, label="d - L(d)")
    plt.axhline(0, color="gray", linestyle="--", label="Reference Line")
    plt.xlabel("Distance (d)")
    plt.ylabel("d - L(d)")
    plt.title(f"d - L(d) for Time Slice {time_slices[t_index]}")
    plt.legend()
    plt.show()


def plot_mc_points(simulations, iter=0, t=0):
    """
    Plot points from the simulations for a given iteration and time slice.

    Parameters:
    simulations (list): List of simulated point sets from mc_samples.
    iter (int): Index of the simulation iteration to plot.
    t (int): Index of the time slice to plot.
    """
    time_slice, points = simulations[iter][t]

    if points.shape == (0,):
        raise NoPointsInTimeSliceException(
            f"No points available in iteration {iter}, time slice {t}."
        )

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Simulation Points for Iteration {iter} and Time Slice {t}")
    plt.show()


def get_box(points, buffer=0):
    """
    Calculate the bounding box that contains all points with an optional buffer.

    Parameters:
    points (list of Point): A list of Point objects, each with 'x' and 'y' attributes.
    buffer (float, optional): The buffer to add to each side of the bounding box. Default is 0.

    Returns:
    tuple: A tuple (min_x, min_y, max_x, max_y) representing the coordinates of the bounding box.

    Raises:
    ValueError: If the points list is empty.

    Example:
    >>> points = [Point(1, 2), Point(2, 3), Point(3, 4)]
    >>> buffer = 1
    >>> calculate_bounding_box(points, buffer)
    (0, 1, 4, 5)
    """

    if not points:
        raise ValueError("The points list cannot be empty.")

    # Extract x and y coordinates
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]

    # Determine the minimum and maximum coordinates with the buffer
    min_x = min(x_coords) - buffer
    max_x = max(x_coords) + buffer
    min_y = min(y_coords) - buffer
    max_y = max(y_coords) + buffer

    # Return the bounding box as a tuple: (min_x, min_y, max_x, max_y)
    return (min_x, min_y, max_x, max_y)

def chrono_plot2d(
    points,
    time,
    ax=None,
    style_params=None,
    plot_limits=None,
    save=None,
    basemap_provider=None,
    crs=None,
    **kwargs,
):
    """
    Plot a 2D scatter of spatial points with alpha (transparency) scaled by inclusion probability at a given time slice.

    Optionally includes a basemap (via contextily) and supports coordinate reference system (CRS) transformations.

    Parameters
    ----------
    points : list of Point or Point
        A single Point or list of Points, each of which must implement a `calculate_inclusion_probability(time)` method
        and have `.x` and `.y` attributes (e.g., shapely.geometry.Point-like).
    time : float
        The time slice at which to evaluate and visualize inclusion probability.
    ax : matplotlib.axes.Axes, optional
        The matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    style_params : dict, optional
        Dictionary to override default plotting styles. Recognized keys include 'point_color' and 'point_size'.
    plot_limits : tuple of tuple, optional
        ((xmin, xmax), (ymin, ymax)) axis limits to apply to the plot.
    save : str or Path, optional
        If provided, saves the figure to the given file path.
    basemap_provider : contextily tile provider, optional
        If provided, a basemap will be added using `contextily.add_basemap`.
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input points. Required if `basemap_provider` is specified.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the plotted data.
    fig : matplotlib.figure.Figure
        The Figure object containing the Axes.
    """

    if isinstance(points, Point):
        points = [points]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    default_style = {
        "point_color": "black",
        "point_size": 50,
    }
    if style_params:
        default_style.update(style_params)

    point_color = default_style["point_color"]
    point_size = default_style["point_size"]

    # Robust GeoDataFrame construction
    geometries = [ShapelyPoint(p.x, p.y) for p in points]
    inclusion_probs = [p.calculate_inclusion_probability(time) for p in points]

    gdf = gpd.GeoDataFrame(
        {"inclusion_prob": inclusion_probs},
        geometry=gpd.GeoSeries(geometries, crs=CRS.from_user_input(crs))
    )

    # Reproject if using a basemap
    if basemap_provider is not None:
        if gdf.crs is None:
            raise ValueError("CRS must be provided if using a basemap.")
        gdf = gdf.to_crs(epsg=3857)

    # Dummy plot to fix extent
    if basemap_provider is not None:
        gdf.plot(ax=ax, alpha=0, markersize=0)
        ctx.add_basemap(ax, source=basemap_provider, crs=gdf.crs)

    # Plot points with inclusion-based alpha
    gdf.plot(
        ax=ax,
        color=point_color,
        alpha=gdf["inclusion_prob"],
        markersize=point_size
    )

    # Optional axis limits
    if plot_limits is not None:
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if fig is not None:
        fig.tight_layout()
        if save:
            fig.savefig(save, bbox_inches="tight", **kwargs)

    return ax, fig

def inclusion_legend(
    ax=None,
    alphas=[0.2, 0.5, 0.8, 1.0],
    color="black",
    title="Inclusion Probability",
    fontsize=8,
    below=True,
    shared=False,
    fig=None
):
    """
    Adds a horizontal alpha-based legend with circular markers.

    Parameters:
    - ax: Matplotlib axis
    - alphas: List of alpha values to show
    - color: Color of the points
    - title: Legend title
    - fontsize: Size of text
    - below: If True, places legend below plot (horizontal layout)
    - shared: If True, adds a shared legend outside the plot (used for side-by-side)
    - fig: Matplotlib figure, required if shared=True
    """
    handles = [
        Line2D(
            [], [], 
            marker="o", 
            linestyle="None",
            markersize=8,
            color=color,
            alpha=a,
            label=f"{a:.1f}"
        )
        for a in alphas
    ]
    
    # For shared legend
    if shared:
        if fig is None:
            raise ValueError("For shared legends, 'fig' must be provided.")
        fig.legend(
            handles=handles,
            title=title,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            fontsize=fontsize,
            title_fontsize=fontsize,
            handletextpad=1.0,
            handlelength=1.5,
            borderpad=0.6,
            labelspacing=0.5,
            ncol=len(alphas)
        )
    else:
        # For individual plot legends (below or standard inside)
        if ax is None:
            raise ValueError("For individual legends, 'ax' must be provided.")
        
        if below:
            legend = ax.legend(
                handles=handles,
                title=title,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                frameon=True,
                fontsize=fontsize,
                title_fontsize=fontsize,
                handletextpad=1.0,
                handlelength=1.5,
                borderpad=0.6,
                labelspacing=0.5,
                ncol=len(alphas)
            )
            legend.get_frame().set_alpha(0.0)
            legend.get_frame().set_edgecolor("none")
        else:
            ax.legend(handles=handles, title=title, fontsize=fontsize)

def chrono_plot(
    points,
    ax=None,
    style_params=None,
    time_slice=None,
    plot_limits=None,
    save=None,
    **kwargs,
):
    """
    Plots a list of points in 3D, with the z-axis representing time and cylinders representing temporal distributions.
    Also adds a shadow layer to show point locations on the x-y plane at the bottom and an optional time slice plane.

    Parameters:
    -----------
    points : list of Point or single Point
        The list of Point instances to plot. Can also be a single Point instance.
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        The 3D axes to plot on. If None, a new figure and axes will be created.
    style_params : dict, optional
        A dictionary of styling parameters.
    time_slice : float or None, optional
        The z-axis coordinate for the time slice plane. If None, no time slice plane is added.
    plot_limits : list of tuples or None, optional
        List of tuples specifying the (min, max) for each axis: [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
        If None, axis limits will be determined automatically.
    save : str or None, optional
        If a string is provided, specifies the filename to save the plot (e.g., 'plot.png').
    **kwargs : dict
        Additional arguments passed to `plt.savefig`.

    Returns:
    --------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes with the plot.
    fig : matplotlib.figure.Figure or None
        The Matplotlib figure instance (if a new figure was created).
    """

    # Ensure points is always a list
    if isinstance(points, Point):
        points = [points]

    # Create a new figure and axes if not provided
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Determine the bottom Z-axis level for shadows
    if plot_limits is not None:
        z_bottom = plot_limits[2][0]  # Use the minimum Z limit from plot_limits
    else:
        # Determine z_bottom by finding the lowest point.start_distribution.mean()
        z_top = max(point.end_distribution.mean() for point in points)
        z_bottom = min(point.start_distribution.mean() for point in points)

    # Default styling parameters
    default_style = {
        "start_mean_color": (
            0.12156862745098039,
            0.4666666666666667,
            0.7058823529411765,
        ),  # Blue
        "end_mean_color": (1.0, 0.4980392156862745, 0.054901960784313725),  # Orange
        "mean_point_size": 50,
        "cylinder_color": (0.6, 0.6, 0.6),  # Grey
        "ppf_limits": (0.01, 0.99),
        "shadow_color": (0.2, 0.2, 0.2),  # Dark grey
        "shadow_size": 30,
        "shadow_bottom_offset": (z_top - z_bottom)
        * 0.5,  # buffer between oldest point and plot shadow
        "time_slice_color": (0.5, 0.5, 0.5),  # Grey
        "time_slice_alpha": 0.3,
        "time_slice_point_color": (0, 0, 0),  # Black
    }

    # Update defaults with any provided style parameters
    if style_params is not None:
        default_style.update(style_params)

    # Extract parameters
    start_mean_color = default_style["start_mean_color"]
    end_mean_color = default_style["end_mean_color"]
    mean_point_size = default_style["mean_point_size"]
    cylinder_color = default_style["cylinder_color"]
    shadow_color = default_style["shadow_color"]
    shadow_size = default_style["shadow_size"]
    shadow_bottom_offset = default_style["shadow_bottom_offset"]
    time_slice_color = default_style["time_slice_color"]
    time_slice_alpha = default_style["time_slice_alpha"]
    time_slice_point_color = default_style["time_slice_point_color"]
    ppf_limits = default_style["ppf_limits"]

    # Plot each point
    for point in points:
        x, y = point.x, point.y
        z_start = point.start_distribution.ppf(ppf_limits[0])
        z_end = point.end_distribution.ppf(ppf_limits[1])

        # Plot cylinder
        ax.plot([x, x], [y, y], [z_start, z_end], color=cylinder_color)

        # Plot mean points
        if start_mean_color is not None:
            start_mean = point.start_distribution.mean()
            ax.scatter(
                [x], [y], [start_mean], color=start_mean_color, s=mean_point_size
            )
        if end_mean_color is not None:
            end_mean = point.end_distribution.mean()
            ax.scatter([x], [y], [end_mean], color=end_mean_color, s=mean_point_size)

        # Plot shadow
        ax.scatter(
            [x],
            [y],
            [z_bottom - shadow_bottom_offset],
            color=shadow_color,
            s=shadow_size,
            alpha=0.5,
        )

    # Add time slice plane
    if time_slice is not None:
        x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
        y_range = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.full_like(xx, time_slice)
        ax.plot_surface(xx, yy, zz, color=time_slice_color, alpha=time_slice_alpha)

        # Plot points on time slice
        for point in points:
            inclusion_prob = point.calculate_inclusion_probability(time_slice)
            ax.scatter(
                [point.x],
                [point.y],
                [time_slice],
                color=time_slice_point_color,
                alpha=inclusion_prob,
            )

    # Apply plot limits if provided
    if plot_limits is not None:
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])
        ax.set_zlim(plot_limits[2] - shadow_bottom_offset)

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")

    # remove background
    if fig is not None:
        fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Tight layout to prevent layout issues
    if fig is not None:
        fig.tight_layout()

    # Save plot if required
    if save:
        plt.savefig(fname=save, bbox_inches="tight", **kwargs)

    return ax, fig


def chrono_plotly(
    points: List["Point"],
    world_line_limits: Tuple[float, float] = (0.05, 0.95),
    geotiff_path: Optional[str] = None,
    basemap_z: Optional[float] = None,
    style_params: Optional[Dict] = None,
    time_slice: Optional[float] = None,
    plot_limits: Optional[List[Tuple[float, float]]] = None,
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """
    Create an interactive 3D visualization of chronological point data using Plotly.

    Parameters:
    -----------
    points : list of Point
        List of Point instances to visualize
    geotiff_path : str, optional
        Path to GeoTIFF basemap file
    basemap_z : float, optional
        Z-coordinate for basemap placement
    style_params : dict, optional
        Visualization styling parameters:
        - 'pipe_color': str (default: 'black')
        - 'pipe_width': int (default: 3)
        - 'mean_point_size': int (default: 10)
        - 'start_color': str (default: 'blue')
        - 'end_color': str (default: 'red')
        - 'shadow_opacity': float (default: 0.5)
        - 'time_slice_color': str (default: 'gray')
    time_slice : float, optional
        Time value for horizontal slice plane
    plot_limits : list of tuples, optional
        [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    width : int
        Width of the plot in pixels
    height : int
        Height of the plot in pixels

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Default style parameters
    default_style = {
        "pipe_color": "black",
        "pipe_width": 2,
        "mean_point_size": 3,
        "start_color": "blue",
        "end_color": "red",
        "shadow_color": "rgba(0, 0, 0, 0.25)",
        "shadow_size": 5,
        "time_slice_color": "rgba(0, 0, 0, 0.25)",
        "time_slice_point_color": "rgba(0, 0, 0, 1)",
        "time_slice_point_size": 5,
    }
    if style_params:
        default_style.update(style_params)

    # Create figure
    fig = go.Figure()

    # Load Basemap if specified
    if geotiff_path:
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            data = src.read(1)
            h, w = data.shape

            # More aggressive resolution reduction
            if max(h, w) > 300:
                data = zoom(data, (300 / h, 300 / w))
                h, w = data.shape

            xx, yy = (
                np.linspace(bounds.left, bounds.right, w),
                np.linspace(bounds.bottom, bounds.top, h),
            )
            xx, yy = np.meshgrid(xx, yy)

            if basemap_z is None:
                min_z = min(
                    point.start_distribution.ppf(world_line_limits[0])
                    for point in points
                )
                max_z = max(
                    point.start_distribution.ppf(world_line_limits[1])
                    for point in points
                )
                basemap_z = min_z - ((max_z - min_z) * 0.25)

            fig.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=np.full_like(xx, basemap_z),
                    surfacecolor=data,
                    colorscale="Viridis",
                    showscale=False,
                    opacity=1,
                )
            )

    # Collect data for batch plotting
    x_lines, y_lines, z_lines = [], [], []
    x_points, y_points, z_points, color_points = [], [], [], []
    slice_x, slice_y, slice_z = [], [], []

    for point in points:
        x, y = point.x, point.y
        z_start, z_end = (
            point.start_distribution.ppf(world_line_limits[0]),
            point.end_distribution.ppf(world_line_limits[1]),
        )
        mean_start, mean_end = (
            point.start_distribution.mean(),
            point.end_distribution.mean(),
        )

        # Store line coordinates
        x_lines.extend([x, x, None])
        y_lines.extend([y, y, None])
        z_lines.extend([z_start, z_end, None])

        # Store point coordinates
        x_points.extend([x, x])
        y_points.extend([y, y])
        z_points.extend([mean_start, mean_end])
        color_points.extend([default_style["start_color"], default_style["end_color"]])

        # Time slice intersection points
        if time_slice is not None and z_start <= time_slice <= z_end:
            slice_x.append(x)
            slice_y.append(y)
            slice_z.append(time_slice)

    # Add batched world-line traces
    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(
                color=default_style["pipe_color"], width=default_style["pipe_width"]
            ),
            showlegend=False,
        )
    )

    # Add point base shadows
    fig.add_trace(
        go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode="markers",
            marker=dict(size=default_style["mean_point_size"], color=color_points),
            showlegend=False,
        )
    )

    # Add point base shadows
    fig.add_trace(
        go.Scatter3d(
            x=slice_x,
            y=slice_y,
            z=np.full_like(slice_x, basemap_z),
            mode="markers",
            marker=dict(
                size=default_style["shadow_size"], color=default_style["shadow_color"]
            ),
            showlegend=False,
        )
    )

    # Add Time Slice (if specified)
    if time_slice is not None:
        x_range = (
            np.linspace(bounds.left, bounds.right, 10)
            if geotiff_path
            else [point.x for point in points]
        )
        y_range = (
            np.linspace(bounds.bottom, bounds.top, 10)
            if geotiff_path
            else [point.y for point in points]
        )
        xx, yy = np.meshgrid(x_range, y_range)

        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=np.full_like(xx, time_slice),
                colorscale=[
                    [0, default_style["time_slice_color"]],
                    [1, default_style["time_slice_color"]],
                ],
                showscale=False,
            )
        )

    # Add time slice intersection points
    if time_slice is not None:
        fig.add_trace(
            go.Scatter3d(
                x=slice_x,
                y=slice_y,
                z=slice_z,
                mode="markers",
                marker=dict(
                    size=default_style["time_slice_point_size"],
                    color=default_style["time_slice_point_color"],
                ),
                showlegend=False,
            )
        )

    # Configure Layout
    fig.update_layout(
        width=width,
        height=height,
        scene=dict(
            aspectmode="cube",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Time",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        showlegend=False,
    )

    # Apply Plot Limits (if specified)
    if plot_limits:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=plot_limits[0]),
                yaxis=dict(range=plot_limits[1]),
                zaxis=dict(range=plot_limits[2]),
            )
        )

    return fig


def calrc_plot(mydist, plot_type="pdf", num_samples=10000, bins=50):
    """
    Plot either the PDF and histogram of samples or the CDF and empirical CDF (ECDF) from samples for a calibrated radiocarbon distribution object.

    Parameters:
    - mydist: instance of calrcarbon distribution
    - plot_type: 'pdf' or 'cdf' to specify the type of plot
    - num_samples: number of samples to generate for histogram and ECDF
    - bins: number of bins for the histogram
    """
    t_values, _ = mydist._get_pdf_values(mydist.c14_mean, mydist.c14_err)
    t_min, t_max = t_values.min(), t_values.max()
    tau_range = np.linspace(t_min, t_max, 1000)

    if plot_type == "pdf":
        pdf_values = mydist.pdf(tau_range)
        samples = mydist.rvs(size=num_samples)

        plt.figure(figsize=(10, 6))

        # Plot PDF
        plt.plot(tau_range, pdf_values, label="PDF", color="blue")

        # Plot histogram of samples
        plt.hist(
            samples,
            bins=bins,
            density=True,
            alpha=0.6,
            color="orange",
            label="Sample Histogram",
        )

        plt.xlim(t_min, t_max)
        plt.xlabel("Tau")
        plt.ylabel("Density")
        plt.title("PDF and Sample Histogram")
        plt.legend()

    elif plot_type == "cdf":
        cdf_values = mydist.cdf(tau_range)
        samples = mydist.rvs(size=num_samples)
        ecdf = ECDF(samples)

        plt.figure(figsize=(10, 6))

        # Plot CDF
        plt.plot(tau_range, cdf_values, label="CDF", color="orange", linewidth=10)

        # Plot ECDF
        plt.step(ecdf.x, ecdf.y, where="post", label="Empirical CDF", color="blue")

        plt.xlim(t_min, t_max)
        plt.xlabel("Tau")
        plt.ylabel("CDF")
        plt.title("CDF and Empirical CDF")
        plt.legend()

    else:
        raise ValueError("Invalid plot_type. Use 'pdf' or 'cdf'.")


def draw_ellipses(ax, peaks_df, std_devs=[1, 2], **kwargs):
    """
    Draws ellipses representing GMM components based on mean coordinates and covariance matrices.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the ellipses.
    peaks_df : pd.DataFrame
        DataFrame containing the 'Coordinates' and 'Covariances' for each component.
    std_devs : list
        List of standard deviations to draw for each ellipse (e.g., [1, 2] for 1 SD and 2 SD ellipses).
    kwargs : Additional arguments for the Ellipse patch.
    """
    for i, (mean, cov) in peaks_df[["Coordinates", "Covariances"]].iterrows():
        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)

        # The angle of the ellipse in degrees
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        # Draw an ellipse for each specified standard deviation
        for std_dev in std_devs:
            width, height = (
                2 * np.sqrt(eigvals) * std_dev
            )  # Scale by the standard deviation

            # Create the ellipse and add it to the plot
            ellipse = Ellipse(
                xy=mean, width=width, height=height, angle=angle, **kwargs
            )
            ax.add_patch(ellipse)


def plot_pdd(
    time_slices,
    time_slice_idx,
    support,
    density_arrays,
    quantiles=[0.025, 0.975],
    density_names=None,
    colors=None,
    ax=None,
    save=None,
    **kwargs,
):
    """
    Plots pairwise distance densities for multiple datasets with quantile error envelopes.

    Parameters:
    -----------
    time_slices : array-like
        Array of time slices.
    time_slice_idx : int
        Index of the time slice in the data arrays.
    support : array-like
        Array of distance values (x-axis).
    density_arrays : list of np.ndarray
        List of 3D arrays containing density values.
    quantiles : list of float, optional
        Quantile range for the error envelope (default is 95% confidence interval [0.025, 0.975]).
    density_names : list of str, optional
        Names of the density datasets for the legend. Defaults to "Dataset 1", "Dataset 2", etc.
    colors : list of str, optional
        Colors for the plots. Defaults to a cycling set of colors.
    ax : matplotlib.axes.Axes, optional
        An existing plot panel to plot on.
    save : str or None, optional
        If a string is provided, specifies the filename to save the plot (e.g., 'plot.png').
    **kwargs : dict
        Additional keyword arguments passed to `plt.savefig`.

    Returns:
    --------
    tuple:
        - fig : matplotlib.figure.Figure
            The figure object for the plot.
        - ax : matplotlib.axes.Axes
            The axes object for the plot.
    """
    time_slice = time_slices[time_slice_idx]

    # Default names and colors
    if density_names is None:
        density_names = [f"Dataset {i+1}" for i in range(len(density_arrays))]
    if colors is None:
        colors = itertools.cycle(["blue", "orange", "green", "red", "purple", "brown"])

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    # Iterate over densities
    for density, name, color in zip(density_arrays, density_names, colors):
        # Calculate mean and quantiles
        mean_density = np.mean(density[:, time_slice_idx, :], axis=1)
        lower_quantile = np.quantile(
            density[:, time_slice_idx, :], quantiles[0], axis=1
        )
        upper_quantile = np.quantile(
            density[:, time_slice_idx, :], quantiles[1], axis=1
        )

        # Plot mean and quantiles
        ax.plot(support, mean_density, label=name, color=color)
        ax.fill_between(support, lower_quantile, upper_quantile, color=color, alpha=0.2)

    # Add labels and title
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Pairwise Distance Density for Time Slice {time_slice}")

    # Add legend
    ax.legend()

    # Save the plot if a filename is provided
    if save:
        plt.savefig(fname=save, **kwargs)

    # Return the plot objects
    return fig, ax


# Exceptions


class NoPointsInTimeSliceException(Exception):
    """Exception raised when the selected time slice has no points."""

    def __init__(self, message="No points in the selected time slice."):
        self.message = message
        super().__init__(self.message)
