import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from chronocluster.clustering import Point
from statsmodels.distributions.empirical_distribution import ECDF

def _largest_divisor(n, max_divisors=10):
    'Utility function for legible axis tick mark density.'
    for i in range(max_divisors, 0, -1):
        if n % i == 0:
            return i
    return 1

def clustering_heatmap(results,
                       distances,
                       time_slices,
                       result_type = 'K'):
    """
    Plot a heatmap of either Ripley's K function or the pair correlation function over time and distance.
    
    Parameters:
    results (np.ndarray): A 3D array where the first dimension is the distance, the second dimension is the time slice, and the third dimension is the iteration.
    time_slices (array-like): Array of time slices.
    distances (array-like): Array of distances at which K or g was calculated.
    result_type (str): The type of results being plotted ('K' for Ripley's K function or 'g' for pair correlation function).
    """
    mean_values = np.mean(results, axis=2)
    
    plt.figure(figsize=(12, 6))

    ax = sns.heatmap(mean_values, 
                    xticklabels = time_slices, 
                    yticklabels = distances, 
                    cmap = 'viridis', 
                    cbar_kws = {'label': f"Mean {result_type}(d)"})
    
    # Adjust x and y axis ticks using the largest divisor
    x_divisor = _largest_divisor(len(time_slices))
    y_divisor = _largest_divisor(len(distances))

    x_ticks_indices = np.arange(0, len(time_slices), len(time_slices) // x_divisor)
    y_ticks_indices = np.arange(0, len(distances), len(distances) // y_divisor)
    
    ax.set_xticks(x_ticks_indices)
    ax.set_xticklabels(np.round(time_slices[x_ticks_indices], 2))
    
    ax.set_yticks(y_ticks_indices)
    ax.set_yticklabels(np.round(distances[y_ticks_indices], 2))

    plt.xlabel('Time Slices')
    plt.ylabel('Distances')
    plt.title(f"Heatmap of Mean {result_type}(d) Function Over Time and Distance")

    plt.gca().invert_yaxis()
    plt.show()

def pdiff_heatmap(p_diff_array, time_slices, support, condition = 'greater'):
    """
    Plot the heatmap of probability values.
    
    Parameters:
    p_diff_array (np.ndarray): Array of probability values with shape (distances, time_slices).
    time_slices (np.ndarray): Array of time slice values.
    support (np.ndarray): Array of pairwise distance values.
    condition (str): Condition to apply for probability calculation ('greater' or 'less').
    """
    plt.figure(figsize=(12, 6))
    if condition == 'greater':
        cond_label = {'label': 'P( > observed)'}
    elif condition == 'less':
        cond_label = {'label': 'P( <= observed)'}
    else:
        raise ValueError("Invalid condition. Use 'greater' or 'less'.")
    ax = sns.heatmap(p_diff_array, 
                     xticklabels=time_slices, 
                     yticklabels=support, 
                     cmap='viridis_r',
                     vmin=0, 
                     cbar_kws=cond_label)
    plt.xlabel('Time Slices')
    plt.ylabel('Pairwise Distances')
    plt.title('Probability Heat Map')
    
    # Adjust x and y axis ticks using the largest divisor
    x_divisor = _largest_divisor(len(time_slices))
    y_divisor = _largest_divisor(len(support))

    x_ticks_indices = np.arange(0, len(time_slices), len(time_slices) // x_divisor)
    y_ticks_indices = np.arange(0, len(support), len(support) // y_divisor)
    
    ax.set_xticks(x_ticks_indices)
    ax.set_xticklabels(np.round(time_slices[x_ticks_indices], 2))
    
    ax.set_yticks(y_ticks_indices)
    ax.set_yticklabels(np.round(support[y_ticks_indices], 2))
    
    plt.gca().invert_yaxis()
    plt.show()

def plot_derivative(results, 
                    distances, 
                    time_slices, 
                    t_index = 0, 
                    result_type='K'):
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
    plt.plot(distances, derivative_values, label=f'd{result_type}/dd')
    plt.xlabel('Distance')
    plt.ylabel(f'd{result_type}/dd')
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
    plt.plot(distances, l_diff, label='d - L(d)')
    plt.axhline(0, color='gray', linestyle='--', label='Reference Line')
    plt.xlabel('Distance (d)')
    plt.ylabel('d - L(d)')
    plt.title(f"d - L(d) for Time Slice {time_slices[t_index]}")
    plt.legend()
    plt.show()
    
def plot_mc_points(simulations, 
                   iter = 0, 
                   t = 0):
    """
    Plot points from the simulations for a given iteration and time slice.
    
    Parameters:
    simulations (list): List of simulated point sets from mc_samples.
    iter (int): Index of the simulation iteration to plot.
    t (int): Index of the time slice to plot.
    """
    time_slice, points = simulations[iter][t]
    
    if points.shape == (0,):
        raise NoPointsInTimeSliceException(f"No points available in iteration {iter}, time slice {t}.")

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Simulation Points for Iteration {iter} and Time Slice {t}')
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

def chrono_plot(points, ax=None, style_params=None, time_slice=None, plot_limits=None):
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
        A dictionary of styling parameters. Possible keys:
            - 'start_mean_color': Color for the start mean points.
            - 'end_mean_color': Color for the end mean points.
            - 'mean_point_size': Size of the mean points.
            - 'cylinder_color': Color for the cylinders.
            - 'ppf_limits': Tuple of two floats for the ppf limits (default is (0.01, 0.99)).
            - 'shadow_color': Color for the shadow points on the x-y plane.
            - 'shadow_size': Size of the shadow points.
            - 'time_slice_color': Color for the time slice plane.
            - 'time_slice_alpha': Transparency for the time slice plane.
            - 'time_slice_point_color': Color for the points at the time slice plane.
    time_slice : float or None, optional
        The z-axis coordinate for the time slice plane. If None, no time slice plane is added.
    plot_limits : list of tuples or None, optional
        List of tuples specifying the (min, max) for each axis: [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
        If None, axis limits will be determined automatically.

    Returns:
    --------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes with the plot.
    """
    # Ensure points is always a list
    if isinstance(points, Point):
        points = [points]

    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Default styling parameters
    default_style = {
        'start_mean_color': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
        'end_mean_color': (1.0, 0.4980392156862745, 0.054901960784313725),  # Orange
        'mean_point_size': 50,
        'cylinder_color': (0.6, 0.6, 0.6),  # Grey
        'ppf_limits': (0.01, 0.99),
        'shadow_color': (0.2, 0.2, 0.2),  # Dark grey
        'shadow_size': 30,
        'time_slice_color': (0.5, 0.5, 0.5),  # Grey
        'time_slice_alpha': 0.3,
        'time_slice_point_color': (0, 0, 0),  # Black
    }

    # Update defaults with any provided style parameters
    if style_params is not None:
        default_style.update(style_params)

    start_mean_color = default_style['start_mean_color']
    end_mean_color = default_style['end_mean_color']
    mean_point_size = default_style['mean_point_size']
    cylinder_color = default_style['cylinder_color']
    ppf_limits = default_style['ppf_limits']
    shadow_color = default_style['shadow_color']
    shadow_size = default_style['shadow_size']
    time_slice_color = default_style['time_slice_color']
    time_slice_alpha = default_style['time_slice_alpha']
    time_slice_point_color = default_style['time_slice_point_color']

    legend_labels = set()
    if ax.get_legend():
        legend_labels = set([t.get_text() for t in ax.get_legend().get_texts()])

    z_bottom = float('inf')
    for point in points:
        # Define cylinder base
        x = point.x
        y = point.y
        z_start = point.start_distribution.ppf(ppf_limits[0])
        z_end = point.end_distribution.ppf(ppf_limits[1])

        # Plot cylinder as a vertical line
        ax.plot([x, x], [y, y], [z_start, z_end], color=cylinder_color)

        # Annotate the mean start and end times
        start_mean = point.start_distribution.mean()
        end_mean = point.end_distribution.mean()
        if start_mean_color is not None:
            if 'Start Mean' not in legend_labels:
                ax.scatter([x], [y], [start_mean], color=start_mean_color, s=mean_point_size, label='Start Mean')
                legend_labels.add('Start Mean')
            else:
                ax.scatter([x], [y], [start_mean], color=start_mean_color, s=mean_point_size)
        
        if end_mean_color is not None:
            if 'End Mean' not in legend_labels:
                ax.scatter([x], [y], [end_mean], color=end_mean_color, s=mean_point_size, label='End Mean')
                legend_labels.add('End Mean')
            else:
                ax.scatter([x], [y], [end_mean], color=end_mean_color, s=mean_point_size)

        # Update z_bottom
        z_bottom = min(z_bottom, z_start)

    # Plot shadow points on the x-y plane at the bottom
    for point in points:
        x = point.x
        y = point.y
        ax.scatter([x], [y], [z_bottom - 10], color=shadow_color, s=shadow_size, alpha=0.5, label='Shadow' if 'Shadow' not in legend_labels else "")
    if 'Shadow' not in legend_labels:
        legend_labels.add('Shadow')

    # Plot points at the time slice
    if time_slice is not None:
        for point in points:
            x = point.x
            y = point.y
            presence_prob = point.calculate_inclusion_probability(time_slice)
            ax.scatter([x], [y], [time_slice], color=time_slice_point_color, s=mean_point_size, alpha=presence_prob, label='Time Slice Point' if 'Time Slice Point' not in legend_labels else "")
        if 'Time Slice Point' not in legend_labels:
            legend_labels.add('Time Slice Point')

    # Add time slice plane
    if time_slice is not None:
        if plot_limits is not None:
            x_min, x_max = plot_limits[0]
            y_min, y_max = plot_limits[1]
        else:
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            x_min, x_max = x_limits[0], x_limits[1]
            y_min, y_max = y_limits[0], y_limits[1]
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
        zz = np.full(xx.shape, time_slice)
        ax.plot_surface(xx, yy, zz, color=time_slice_color, alpha=time_slice_alpha)

    # Apply plot limits if specified
    if plot_limits is not None:
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])
        ax.set_zlim(plot_limits[2])

    # axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time', rotation=90)  # Rotate the label

    # Create a legend
    if legend_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    if fig is not None:
        fig.tight_layout()

    return ax

def calrc_plot(mydist, plot_type='pdf', num_samples=10000, bins=50):
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

    if plot_type == 'pdf':
        pdf_values = mydist.pdf(tau_range)
        samples = mydist.rvs(size=num_samples)
        
        plt.figure(figsize=(10, 6))
        
        # Plot PDF
        plt.plot(tau_range, pdf_values, label='PDF', color='blue')
        
        # Plot histogram of samples
        plt.hist(samples, bins=bins, density=True, alpha=0.6, color='orange', label='Sample Histogram')
        
        plt.xlim(t_min, t_max)
        plt.xlabel('Tau')
        plt.ylabel('Density')
        plt.title('PDF and Sample Histogram')
        plt.legend()
        plt.show()

    elif plot_type == 'cdf':
        cdf_values = mydist.cdf(tau_range)
        samples = mydist.rvs(size=num_samples)
        ecdf = ECDF(samples)
        
        plt.figure(figsize=(10, 6))
        
        # Plot CDF
        plt.plot(tau_range, cdf_values, label='CDF', color='blue', linewidth=2)
        
        # Plot ECDF
        plt.step(ecdf.x, ecdf.y, where='post', label='Empirical CDF', color='orange')
        
        plt.xlim(t_min, t_max)
        plt.xlabel('Tau')
        plt.ylabel('CDF')
        plt.title('CDF and Empirical CDF')
        plt.legend()
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Use 'pdf' or 'cdf'.")

# Exceptions

class NoPointsInTimeSliceException(Exception):
    """Exception raised when the selected time slice has no points."""
    def __init__(self, message="No points in the selected time slice."):
        self.message = message
        super().__init__(self.message)
