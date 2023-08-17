#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import DBSCAN
import utils as ut

def dbscan(scan):
    """
    Applies the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm on the given scan data.

    Args:
        scan (list): List of measurements representing the scan data.

    Returns:
        tuple: A tuple containing the following elements:
            - labels (numpy.ndarray): Array of labels assigned by the DBSCAN algorithm.
            - polar (numpy.ndarray): Array of polar coordinates (in radians) from the original scan data.
            - core_samples_mask (numpy.ndarray): Boolean array indicating the core points and noise points.
            - cartesian (numpy.ndarray): Array of converted Cartesian coordinates from the original scan data.
    """
    
    # polar coordinates returned by rplidar
    polar = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
    
    # coordinates converted in cartesian
    cartesian = np.array([ut.polar_to_cartesian(d[1], d[2]) for d in scan])
    
    # apply dbscan algorithm
    db = DBSCAN(eps=100, min_samples=3).fit(cartesian)
    labels = db.labels_
    
    # number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # boolean array to distinguish core points and noise points
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    return labels, polar, core_samples_mask, cartesian