#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster_dark

from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.stats import norm, uniform

from chronocluster.calcurves import calibration_curves
from chronocluster.clustering import Point
from chronocluster.distributions import calrcarbon, ddelta


def pts_from_csv(file_path):
    """
    Load Point objects from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    list: List of Point objects.
    """
    points = []
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    for row in data:
        x, y, start_mean, start_std, end_mean, end_std = row
        start_distribution = norm(loc=start_mean, scale=start_std)
        end_distribution = norm(loc=end_mean, scale=end_std)
        point = Point(x, y, start_distribution, end_distribution)
        points.append(point)
    return points


def pts_to_csv(points, file_path):
    """
    Save Point objects to a CSV file.

    Parameters:
    points (list): List of Point objects.
    file_path (str): Path to the CSV file.
    """
    data = []
    for point in points:
        mean_start = point.start_distribution.mean()
        std_start = point.start_distribution.std()
        mean_end = point.end_distribution.mean()
        std_end = point.end_distribution.std()
        data.append([point.x, point.y, mean_start, std_start, mean_end, std_end])
    np.savetxt(
        file_path,
        data,
        delimiter=",",
        header="x,y,start_mean,start_std,end_mean,end_std",
        comments="",
    )


def pts_to_df(points: Iterable[Point]) -> pd.DataFrame:
    """
    Convert a list of Point objects to a pandas DataFrame.

    Parameters:
    points (list of Point): List of Point objects.

    Returns:
    pd.DataFrame: DataFrame containing the points data.
    """
    data = [point.to_dict() for point in points]
    return pd.DataFrame(data)


def df_to_pts(df: pd.DataFrame) -> list[Point]:
    """
    Convert a pandas DataFrame to a list of Point objects.

    Parameters:
    df (pd.DataFrame): DataFrame containing the points data.

    Returns:
    list of Point: List of Point objects.
    """
    points = []
    for _, row in df.iterrows():
        x = row["x"]
        y = row["y"]
        start_type = row["start_type"]
        start_params = row["start_params"]
        end_type = row["end_type"]
        end_params = row["end_params"]

        if start_type == "norm":
            start_distribution = norm(loc=start_params[0], scale=start_params[1])
        elif start_type == "uniform":
            start_distribution = uniform(loc=start_params[0], scale=start_params[1])
        elif start_type == "constant":
            start_distribution = ddelta(d=start_params[0])
        elif start_type == "calrcarbon":
            calcurve_name, c14_mean, c14_err = start_params
            calcurve = calibration_curves[calcurve_name]
            start_distribution = calrcarbon(calcurve, c14_mean, c14_err)
        else:
            raise ValueError(f"Unsupported start distribution type: {start_type}")

        if end_type == "norm":
            end_distribution = norm(loc=end_params[0], scale=end_params[1])
        elif end_type == "uniform":
            end_distribution = uniform(loc=end_params[0], scale=end_params[1])
        elif end_type == "constant":
            end_distribution = ddelta(d=end_params[0])
        elif end_type == "calrcarbon":
            calcurve_name, c14_mean, c14_err = end_params
            calcurve = calibration_curves[calcurve_name]
            end_distribution = calrcarbon(calcurve, c14_mean, c14_err)
        else:
            raise ValueError(f"Unsupported end distribution type: {end_type}")

        point = Point(x, y, start_distribution, end_distribution)
        points.append(point)

    return points


def kde_to_geotiff(
    x_mesh, y_mesh, kde_values, epsg_code=4326, output_path="kde_output.tif"
):
    """
    Save KDE values as a GeoTIFF file for use in GIS software.

    Parameters:
    -----------
    x_mesh : np.ndarray
        2D array of x-coordinates for the mesh grid.
    y_mesh : np.ndarray
        2D array of y-coordinates for the mesh grid.
    kde_values : np.ndarray
        2D array of KDE values corresponding to the mesh grid.
    epsg_code : int
        EPSG code for the desired coordinate system. Defails to ESGE:4326, which
        is a global geographic coordinate system in decimal degrees with the
        WGS84 datum.
    output_path : str
        File path to save the GeoTIFF output.
    """
    # Check input consistency
    if x_mesh.shape != y_mesh.shape or x_mesh.shape != kde_values.shape:
        raise ValueError("x_mesh, y_mesh, and kde_values must have the same shape.")

    # Determine resolution (assumes uniform spacing)
    x_res = x_mesh[0, 1] - x_mesh[0, 0]
    y_res = y_mesh[1, 0] - y_mesh[0, 0]

    if x_res <= 0 or y_res <= 0:
        raise ValueError("Mesh grid spacing must be positive.")

    # Get the origin point (top-left corner of the raster)
    x_min = x_mesh[0, 0]
    y_max = y_mesh[0, 0]

    # Define transform based on origin and resolution
    transform = from_origin(x_min, y_max, x_res, y_res)

    # Ensure kde_values is in the expected orientation
    kde_values = np.flipud(
        kde_values
    )  # Flip vertically for correct orientation in GeoTIFF

    # Define metadata for the GeoTIFF
    metadata = {
        "driver": "GTiff",
        "height": kde_values.shape[0],
        "width": kde_values.shape[1],
        "count": 1,
        "dtype": kde_values.dtype.name,
        "crs": f"EPSG:{epsg_code}",
        "transform": transform,
    }

    # Write to GeoTIFF
    try:
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(kde_values, 1)
        print(f"GeoTIFF saved as {output_path}")
    except Exception as e:
        print(f"Error saving GeoTIFF: {e}")


def pts_to_gis(
    points,
    output_path="points.gpkg",
    epsg=4326,
    file_format="GPKG",
    ppf_limits=(0.01, 0.99),
):
    """
    Converts a Point or a list of Points to a GeoDataFrame and exports to a GIS-compatible file.
    Includes additional attributes such as distribution type, mean, and probabilistic limits (ppf).

    Parameters:
    -----------
    points : Point or list of Point
        A single Point object or a list of Point objects to export.
    output_path : str, optional
        File path to save the GIS vector file. Defaults to "points.gpkg".
    epsg : int, optional
        EPSG code for the desired coordinate system. Defaults to 4326 (WGS 84).
    file_format : str, optional
        Format of the output file. Options: "GPKG", "SHP", "GeoJSON".
        Defaults to "GPKG" (GeoPackage).
    ppf_limits : tuple, optional
        Percentile limits for the probabilistic range of the worldline. Defaults to (0.01, 0.99).

    Returns:
    --------
    None
    """
    # Ensure points is a list
    if isinstance(points, Point):
        points = [points]
    elif not isinstance(points, list):
        raise ValueError("Input must be a Point or a list of Point objects.")

    # Extract data from Point objects
    data = []
    for point in points:
        # Get probabilistic limits and distribution attributes
        start_name = point.start_distribution.dist.name
        end_name = point.end_distribution.dist.name
        start_mean = point.start_distribution.mean()
        end_mean = point.end_distribution.mean()
        z_start = point.start_distribution.ppf(ppf_limits[0])  # Lower percentile
        z_end = point.end_distribution.ppf(ppf_limits[1])  # Upper percentile

        # Append attributes to the data list
        data.append(
            {
                "x": point.x,
                "y": point.y,
                "z": z_start,
                "start_dist": start_name,
                "end_dist": end_name,
                "start_mean": start_mean,
                "end_mean": end_mean,
                "z_end": z_end,
            }
        )

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            [row["x"] for row in data], [row["y"] for row in data]
        ),
        crs=f"EPSG:{epsg}",
    )

    # Select the correct driver for the file format
    driver = {"GPKG": "GPKG", "SHP": "ESRI Shapefile", "GEOJSON": "GeoJSON"}.get(
        file_format.upper()
    )
    if driver is None:
        raise ValueError(
            f"Unsupported file format: {file_format}. Choose from 'GPKG', 'SHP', or 'GeoJSON'."
        )

    # Export the GeoDataFrame
    gdf.to_file(output_path, driver=driver)
    print(f"Points exported to {output_path} as {file_format.upper()}.")
