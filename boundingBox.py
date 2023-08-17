#!/usr/bin/env python3
import numpy as np

def getEigenAngle(v):
    return np.rad2deg(np.arctan(v[1]/v[0]))

def inclination(x,y):
    min_index_x = x.index(min(x))
    max_index_x = x.index(max(x))

    extreme_left_point = (x[min_index_x], y[min_index_x])
    extreme_right_point = (x[max_index_x], y[max_index_x])

    delta_y = extreme_right_point[1] - extreme_left_point[1]
    delta_x = extreme_right_point[0] - extreme_left_point[0]

    if delta_x == 0:
        # if delta_x è zero
        theta = float('inf')
    else:
        theta = delta_y / delta_x
    return theta

def boundingBox(x,y, offset):
    if len(x) < 2 or len(y) < 2:
        return 0

    theta = inclination(x,y)
    # Rotation matrix
    rot = lambda theta: np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                                [np.sin(np.deg2rad(theta)),  np.cos(np.deg2rad(theta))]])

    data = np.matmul(rot(0), np.vstack([x, y]))

    # Mean and covariance
    means = np.mean(data, axis=1)
    cov = np.cov(data)

    # Eigenvalues ​​and Eigenvectors of the Covariance Matrix
    eval, evec = np.linalg.eigh(cov)

    centerd_data = data - means[:, np.newaxis]
    xmin, xmax, ymin, ymax = np.min(centerd_data[0,:]), np.max(centerd_data[0,:]), np.min(centerd_data[1,:]), np.max(centerd_data[1,:])

    theta_pc1 = getEigenAngle(evec[:, 0])

    aligned_coords = np.matmul(rot(-theta_pc1), centerd_data)
    xmin, xmax, ymin, ymax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :])
    rectCoords = lambda x1, y1, x2, y2: np.array([[x1, x2, x2, x1],
                                                [y1, y1, y2, y2]])

    # Apply the offset to the bounding box
    xmin = xmin - offset
    xmax = xmax + offset
    ymin = ymin - offset
    ymax = ymax + offset
    rectangleCoordinates = rectCoords(xmin, ymin, xmax, ymax)
    rectangleCoordinates = np.matmul(rot(theta_pc1), rectangleCoordinates)

    # Translate back
    rectangleCoordinates += means[:, np.newaxis]
    return rectangleCoordinates