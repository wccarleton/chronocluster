import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
from chronocluster.distributions import ddelta, calrcarbon
from chronocluster.clustering import Point
from chronocluster.calcurves import calibration_curves

def pts_from_csv(file_path):
    """
    Load Point objects from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    list: List of Point objects.
    """
    points = []
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
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
    np.savetxt(file_path, data, delimiter=',', header='x,y,start_mean,start_std,end_mean,end_std', comments='')

def pts_to_df(points):
    """
    Convert a list of Point objects to a pandas DataFrame.
    
    Parameters:
    points (list of Point): List of Point objects.
    
    Returns:
    pd.DataFrame: DataFrame containing the points data.
    """
    data = []
    for point in points:
        start_type = point.start_distribution.dist.name
        start_params = [point.start_distribution.mean(), point.start_distribution.std()]
        end_type = point.end_distribution.dist.name
        end_params = [point.end_distribution.mean(), point.end_distribution.std()]
        data.append({
            'x': point.x,
            'y': point.y,
            'start_type': start_type,
            'start_params': start_params,
            'end_type': end_type,
            'end_params': end_params
        })
    return pd.DataFrame(data)

def df_to_pts(df):
    """
    Convert a pandas DataFrame to a list of Point objects.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the points data.
    
    Returns:
    list of Point: List of Point objects.
    """
    points = []
    for _, row in df.iterrows():
        x = row['x']
        y = row['y']
        start_type = row['start_type']
        start_params = row['start_params']
        end_type = row['end_type']
        end_params = row['end_params']
        
        if start_type == 'norm':
            start_distribution = norm(loc=start_params[0], scale=start_params[1])
        elif start_type == 'uniform':
            start_distribution = uniform(loc=start_params[0], scale=start_params[1])
        elif start_type == 'constant':
            start_distribution = ddelta(d=start_params[0])
        elif start_type == 'calrcarbon':
            calcurve_name, c14_mean, c14_err = start_params
            calcurve = calibration_curves[calcurve_name]
            start_distribution = calrcarbon(calcurve, c14_mean, c14_err)
        else:
            raise ValueError(f"Unsupported start distribution type: {start_type}")

        if end_type == 'norm':
            end_distribution = norm(loc=end_params[0], scale=end_params[1])
        elif end_type == 'uniform':
            end_distribution = uniform(loc=end_params[0], scale=end_params[1])
        elif end_type == 'constant':
            end_distribution = ddelta(d=end_params[0])
        elif end_type == 'calrcarbon':
            calcurve_name, c14_mean, c14_err = end_params
            calcurve = calibration_curves[calcurve_name]
            end_distribution = calrcarbon(calcurve, c14_mean, c14_err)
        else:
            raise ValueError(f"Unsupported end distribution type: {end_type}")

        point = Point(x, y, start_distribution, end_distribution)
        points.append(point)
    
    return points