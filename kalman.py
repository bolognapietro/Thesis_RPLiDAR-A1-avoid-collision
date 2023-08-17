#!/usr/bin/env python3

'''
    File name         : KalmanFilter.py
    Description       : KalmanFilter class used for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import numpy as np
import utils as ut

class KalmanFilter(object):
    """
    This class implements a Kalman filter for tracking objects in 2D space.
    The filter estimates the state of the object based on noisy measurements.
    """

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas, x_0, y_0, vx_0, vy_0):
        """
        Initializes the Kalman filter with the provided parameters.

        :param dt: Sampling time (time for 1 cycle).
        :param u_x: Acceleration in x-direction.
        :param u_y: Acceleration in y-direction.
        :param std_acc: Process noise magnitude.
        :param x_std_meas: Standard deviation of the measurement in x-direction.
        :param y_std_meas: Standard deviation of the measurement in y-direction.
        """

        # Define sampling time
        self.dt = dt

        # Define the control input variables
        self.u = np.matrix([[u_x],[u_y]])

        # Initial State
        self.x = np.matrix([[x_0], [y_0], [vx_0], [vy_0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Define Measurement Mapping Matrix. It's the Transformation matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix(np.eye(4))
        '''self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2'''

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        """
        This function updates the time state of the system using the given equations.
        
        Returns:
            numpy.ndarray: The predicted state vector containing the first two elements.
        """

        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)  # Eq.(9)
        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # Eq.(10)
        # Return the predicted state
        return self.x[0:2]

    def update(self, z):
        """
        Updates the state estimate and error covariance matrix using the Kalman filter update equations.

        Parameters:
        - self: the KalmanFilter object
        - z: the measurement vector

        Returns:
        - the updated state estimate
        """

        # Calculate S: innovation covariance matrix
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain: the optimal blending factor between the predicted state estimate and the measured state estimate
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update the state estimate using the measurement: x = x + K * (z - H * x)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        # Create identity matrix
        I = np.eye(self.H.shape[1])
        # Update the error covariance matrix: P = (I - K * H) * P
        self.P = (I - (K * self.H)) * self.P
        # Return the updated state estimate
        return self.x[0:2]