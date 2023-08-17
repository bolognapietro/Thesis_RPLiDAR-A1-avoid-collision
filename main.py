#!/usr/bin/env python3
import rplidar
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from collections import deque
from copy import deepcopy
import time
import utils as ut
import kalman as km
import boundingBox as bb
import dbscan as db

# Setup variables
PORT_NAME = '/dev/ttyUSB0'
DMAX = 1000
START_TRESHOLD = 10
DELTA_TIME = 1
THRESHOLD = 200

# Global variables 
start = True
num_kalman = 0
total_time = 0
kalman_list = []
space_list = None
velocity = []
space = []

# Lists and FIFO queue
timestamp_fifo = deque([time.time()])
time_list = []
vRelative_list = []

def update(iterator, lidar):
    """
    Update function for processing lidar scans and updating the plot.

    Parameters:
    - frame (int): Frame number of the plot.
    - iterator (Iterator): Iterator object for retrieving lidar scans.
    - ax (Axes): Axes object for the plot.
    - lidar (RPLidar): RPLidar object for accessing lidar data.

    Returns:
    - None
    
    """

    # Set variables 
    global start
    global total_time
    global num_kalman
    global kalman_list
    global space_list
    global velocity
    global space
    vRelative = 0
    centers = []
    kalman_old = []

    # Try to get the next scan from the iterator    
    try:
        scan = next(iterator)
    except rplidar.RPLidarException as re:
        if re.args[0] == 'Check bit not equal to 1':
            return
        elif re.args[0] == 'New scan flags mismatch':
            return
        else:
            raise rplidar.RPLidarException(re.args[0])
    except StopIteration:
        lidar.clean_input()
        return
    except Exception as e:
        e

    # Fix the setup
    lidar.clean_input()

    # Filter the data
    new_scan = []
    for s in scan:
        if (s[2] <= DMAX):
            new_scan.append(s)

    # Check if it's the first cycle 
    if(start==True):
        start = False
        pass
    else:
        # Start the timer for the relative velocity
        new_timestamp = time.time()
        timestamp_fifo.append(new_timestamp)

        # Perform DBSCAN algorithm
        labels, polar, core_samples_mask, cartesian = db.dbscan(new_scan)
        unique_labels = set(labels)

        # Process each cluster separately
        for k in unique_labels:
            # Boolean mask indicating points belonging to the cluster with identifier k
            class_member_mask = labels == k
            cartesian_points = cartesian[class_member_mask & core_samples_mask]

            # Compute Bounding Box
            x_points, y_points = [c[0] for c in cartesian_points], [c[1] for c in cartesian_points]
            rectangleCoordinates = bb.boundingBox(x_points, y_points, offset=20)
            if type(rectangleCoordinates) != int:
                rCt, rCr = ut.cartesian_to_polar(rectangleCoordinates[0], rectangleCoordinates[1])
                rectangleCoordinatesPolar = np.array([rCt, rCr])
                results = ut.check_collision(rectangleCoordinatesPolar, THRESHOLD)
            
            # Compute and plot centroids
            theta, ro = ut.compute_centroids(cartesian_points)
            if (ro >= 20):
                centers.append([theta, ro, k])
            else:
                continue

        # Create instances of the kalman filter
        total_centers = len(centers)
        if (total_centers != num_kalman):
            kalman_list.clear()
            id = 0
            sp = []
            for center in centers:
                real_pos = [center[0], center[1]]
                x_0, y_0 = ut.polar_to_cartesian(np.rad2deg(center[0]), center[1])
                kalman = km.KalmanFilter(DELTA_TIME, 0, 0, 1, 0.1, 0.1, x_0, y_0, 0, 0)
                kalman.id = id 
                id += 1
                predicted_pos = [center[0], center[1]]
                kalman_list.append([real_pos, predicted_pos, kalman])
                sp.append(real_pos)
            space_list = deque([sp])
            num_kalman = total_centers                
        else:
            pass

        kalman_old = deepcopy(kalman_list)
        kalman_list.clear()

        # Apply kalman filter predict and update functions
        old_space = space_list.popleft()
        new_space = []
        vRelative_list = []
        distances = []
        timestamp_prev = timestamp_fifo.popleft()   
        deltaTime = new_timestamp-timestamp_prev   
        for kal in kalman_old: 
            # Compute the real position of the centroids
            pred = kal[1]
            nearest = ut.find_nearest_point((pred[0], pred[1]), centers)
            tCent, rCent = nearest[0], nearest[1]
            realP = [tCent, rCent]
            new_space.append(realP)
            distances.append(rCent)
            
            # Predict the next position using kalman filter
            kalman = kal[2]
            (kx, ky) = kalman.predict()
            tPred, rPred = ut.cartesian_to_polar(kx[(0, 0)], ky[(0, 0)])
            predP = [tPred, rPred]

            # Update the kalman filter
            a, b = ut.polar_to_cartesian(np.rad2deg(tCent), rCent)
            c = np.matrix([[a], [b]])
            (kx1, ky1) = kalman.update(c)
            
            kalman_list.append([realP, predP, kalman])
        
            # Compute the relative velocity [cm/s]
            tReal0, rReal0 = old_space[kalman.id]    # Perv 
            deltaSpace = rCent-rReal0   # in millimetri
            vRelative = ut.millimeter_to_centimeter(deltaSpace)/deltaTime # 1 cm/ms = 1m/s
            vRelative_list.append(vRelative)
            
        # Save position and speed values
        space_list.append(new_space)
        
        # Print some information in the terminal
        print("INFORMATIONS")  
        print("Number of kalman filters: {}".format(num_kalman))
        print("\nSpace and Relative Velocity:\nTHETA\tRO\tVELOCITY")
        final = []
        for n in range(len(new_space)):
            final.append([new_space[n][0], new_space[n][1], vRelative_list[n]])
            print("{}\t{}\t{}".format(int(np.rad2deg(new_space[n][0])),round(new_space[n][1], 2), round(vRelative_list[n], 2)))
        print("-----------------------------")
        
    
def main():
    """
    This is the main function that initializes the RPLIDAR and KalmanFilter objects, 
    sets up the plot, starts the animation, and performs additional operations after scanning.

    Parameters:
        None

    Returns:
        None
    """
    # Initialize RPLIDAR (port, baudrate)
    lidar = rplidar.RPLidar(PORT_NAME, baudrate=115200)
    iterator = lidar.iter_scans(max_buf_meas=False, min_len=False)
    
    try:
        while True:
            update(iterator, lidar)
    except KeyboardInterrupt:
        pass

    # Stop the lidar scanner
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()

if __name__ == "__main__":
    main()