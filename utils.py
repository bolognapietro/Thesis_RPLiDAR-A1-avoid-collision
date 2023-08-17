#!/usr/bin/env python3
import numpy as np
from math import cos, sin, sqrt, atan, atan2

def polar_to_cartesian(theta, ro):
    """
    Converts polar coordinates to cartesian coordinates.
    
    Args:
        theta (float): The angle in degrees.
        ro (float): The distance from the origin.
        
    Returns:
        tuple: A tuple containing the x and y coordinates in the cartesian system.
    """

    if(type(theta) == float or type(theta) == np.float64):
        t = np.deg2rad(theta)
        x = ro*cos(t)
        y = ro*sin(t)
        return x, y
    elif(type(theta) == list or type(theta) == np.ndarray):
        x_list = [r * np.cos(t) for t, r in zip(theta, ro)]
        y_list = [r * np.sin(t) for t, r in zip(theta, ro)]
        return x_list, y_list

def cartesian_to_polar(x, y):
    """
    Converts Cartesian coordinates to polar coordinates.
    
    Parameters:
    x (float): The x-coordinate of the point in Cartesian system.
    y (float): The y-coordinate of the point in Cartesian system.
    
    Returns:
    tuple: A tuple containing theta and ro values representing polar coordinates.
           theta (float): The angle (in radians) between the positive x-axis and the ray from the origin to the point.
           ro (float): The distance from the origin to the point.
    """
    if(type(x) == float or type(x) == np.float64):
        ro = sqrt(pow(x, 2) + pow(y, 2))
        theta = atan2(y, x)
        return theta, ro
    elif(type(x) == list or type(x) == np.ndarray):
        ro = [np.sqrt(x0**2 + y0**2) for x0, y0 in zip(x, y)]
        theta = [np.arctan2(y0, x0) for x0, y0 in zip(x, y)]
        return theta, ro

def calculate_distance(p1, p2):
    """
    Calculates the distance between two points in polar coordinates.

    Parameters:
    p1 (tuple): A tuple representing the first point in polar coordinates, where p1 = (theta1, radius1).
    p2 (tuple): A tuple representing the second point in polar coordinates, where p2 = (theta2, radius2).

    Returns:
    float: The distance between the two points in polar coordinates.
    """

    t1, r1 = p1
    t2, r2 = p2
    return sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cos(t2 - t1))

def find_nearest_point(target_point, points):
    """
    Finds the nearest point to the target point from a list of points.

    Parameters:
    target_point (tuple): The coordinates of the target point.
    points (list): List of points, each represented as a tuple of coordinates.

    Returns:
    The nearest point (tuple) to the target point.
    """

    nearest_point = None
    min_distance = float('inf')

    for point in points:
        ptr = point[0], point[1]
        distance = calculate_distance(target_point, ptr)
        if distance < min_distance:
            min_distance = distance
            nearest_point = point

    return nearest_point

def compute_centroids(cartesian_points):
    """
    Computes the polar coordinates (theta, r) of the centroid given a list of cartesian points.

    Parameters:
    cartesian_points (list): A list of tuples representing the cartesian points [(x1, y1), (x2, y2), ...]

    Returns:
    tuple: A tuple containing the polar coordinates (theta, r)
    """

    # Extract x and y coordinates from the cartesian points
    x = [c[0] for c in cartesian_points]
    y = [c[1] for c in cartesian_points]

    # Initialize variables for sum of x and y coordinates
    sum_x, sum_y = 0, 0

    # Calculate the sum of x and y coordinates
    for i in x:
        sum_x += i
    for j in y:
        sum_y += j

    # Calculate the mean of x and y coordinates
    mean_x, mean_y = 1, 1  # Default values to prevent division by zero
    if len(x) != 0:
        mean_x = sum_x / len(x)
        mean_y = sum_y / len(y)

    # Calculate the polar coordinates (theta, r) of the centroid
    r = sqrt(pow(mean_x, 2) + pow(mean_y, 2))
    if mean_x > 0:
        theta = atan(mean_y / mean_x)
    if mean_x < 0:
        theta = atan(mean_y / mean_x) + np.pi

    # Return the polar coordinates
    return theta, r

def moving_average_filter(data, window_size):
    """
    Computes the moving average of a given list of data points.
    
    Args:
        data (list): A list of data points.
        window_size (int): The size of the moving window.
        
    Returns:
        list: A list of filtered data points where each data point is the average of the elements in its window.
    """
    
    filtered_data = []
    
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        window = data[start_index:end_index]
        average = sum(window) / len(window)
        filtered_data.append(average)
    
    return filtered_data  

def apply_moving_average_filter(data_list, window_size):
    """
    Applies a moving average filter to a list of data.

    Parameters:
        data_list (list): The input list of data.
        window_size (int): The size of the moving average window.

    Returns:
        filtered_data (list): The filtered data after applying the moving average filter.
    """

    filtered_data = moving_average_filter(data_list, window_size)
    return filtered_data

def millimeter_to_centimeter(value):
    """
    Converts a given value from millimeters to centimeters.

    Parameters:
        value (float): The value in millimeters to be converted.

    Returns:
        float: The converted value in centimeters.
    """

    value = value/10
    return value

def is_line_partially_below_threshold(r1, r2, threshold):
    """
    Checks if a line is partially below a given threshold.

    Parameters:
        r1 (float): The first value of the line.
        r2 (float): The second value of the line.
        threshold (float): The threshold value.

    Returns:
        bool: True if the line is partially below the threshold, False otherwise.
    """

    if r1 < threshold and r2 < threshold:
        return True
    elif min(r1, r2) < threshold < max(r1, r2):
        return True
    else:
        return False

    
def check_collision(rCP, threshold):
    # Check collision i=[b,t,l,r]
    results = []
    for i in range(len(rCP[0])):
        if (i<3):
            is_below = is_line_partially_below_threshold(rCP[1,i], rCP[1,i+1], threshold)
        else:
            is_below = is_line_partially_below_threshold(rCP[1,i], rCP[1,0], threshold)
        if is_below == True:
            results.append(i)

    return results