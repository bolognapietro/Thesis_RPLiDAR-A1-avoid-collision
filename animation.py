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

# Set plotting variables
ALL_POINTS = False  # displays all clustered points scanned by the lidar
NOISE_POINTS = False    # displays all noise points scanned by the lidar
CENTROIDS = False   # displays all centroids
BOUNDING_BOX = False    # displays bounding box for each cluster
VELOCITY_GRAPH = False  # displays relative velocity graph

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

def plot_bounding_box(ax, rectangleCoordinatesPolar):
    ax.plot(rectangleCoordinatesPolar[0, 0:2], rectangleCoordinatesPolar[1, 0:2], color='g') # | (up)
    ax.plot(rectangleCoordinatesPolar[0, 1:3], rectangleCoordinatesPolar[1, 1:3], color='g') # -->
    ax.plot(rectangleCoordinatesPolar[0, 2:], rectangleCoordinatesPolar[1, 2:], color='g')    # | (down)
    ax.plot([rectangleCoordinatesPolar[0, 3], rectangleCoordinatesPolar[0, 0]], [rectangleCoordinatesPolar[1, 3], rectangleCoordinatesPolar[1, 0]], color='g')    # <--

def plot_vRelative(filtered_vRelative_list):
    """
    This function creates a new plot to visualize the relative velocity (1920x1080 px).

    Parameters:
        None

    Returns:
        None
    """

    # Create a new plot with specified dimensions
    fig2 = plt.figure("Relative Velocity", figsize=(20, 11.25))

    # Adds subplot ax1 to the figure
    ax1 = fig2.add_subplot(2,1,1)

    # Plot the variation of relative velocity over time
    # ax1.plot(time_list, vRelative_list)
    ax1.plot(time_list, filtered_vRelative_list)
    ax1.axhline(y=0, color='black', linestyle='--', label='y = 0')
    ax1.set_title('Speed/time graph')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Speed [cm/s]')

    # Adds subplot ax2 to the figure
    ax2 = fig2.add_subplot(2,1,2)

    # Plot the variation of space over time
    ax2.plot(time_list, space, label='Variation of space over time')
    ax2.axhline(y=20, color='red', linestyle='--', label='y = Threshold')
    ax2.set_title('Space/time graph')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Space [cm]')

    # Save plots in a PDF files
    plt.savefig('/home/pietro/tirocinio/my/lidar_avoid_collision/velocità_relativa.pdf', dpi=96)

    # Display the plot
    plt.show()

def update(frame, iterator, ax, lidar):
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
    ax.clear()
    ax.set_rlim(0, DMAX)  

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

            # Compute all points
            if (ALL_POINTS == True):
                polar_points = polar[class_member_mask & core_samples_mask]
                ang = [d[0] for d in polar_points]
                dist = [d[1] for d in polar_points]
                ax.scatter(ang, dist, s=15)

            # Compute noise points
            if (NOISE_POINTS == True):
                noise_points = polar[class_member_mask & ~core_samples_mask]
                ang = [d[0] for d in noise_points]
                dist = [d[1] for d in noise_points]
                ax.scatter(ang, dist, s=15, c='black')
                
            # Compute Bounding Box
            x_points, y_points = [c[0] for c in cartesian_points], [c[1] for c in cartesian_points]
            rectangleCoordinates = bb.boundingBox(x_points, y_points, offset=20)
            if type(rectangleCoordinates) != int:
                rCt, rCr = ut.cartesian_to_polar(rectangleCoordinates[0], rectangleCoordinates[1])
                rectangleCoordinatesPolar = np.array([rCt, rCr])
                results = ut.check_collision(rectangleCoordinatesPolar, THRESHOLD)
            
                if (BOUNDING_BOX == True):
                    plot_bounding_box(ax, rectangleCoordinatesPolar)
                    # Plottaggio delle rette parzialmente sotto la soglia
                    for side in results:
                        if side == 0:
                            ax.plot([rectangleCoordinatesPolar[0,0], rectangleCoordinatesPolar[0,1]], [rectangleCoordinatesPolar[1,0], rectangleCoordinatesPolar[1,1]], 'r-')
                        elif side == 1:
                            ax.plot([rectangleCoordinatesPolar[0,1], rectangleCoordinatesPolar[0,2]], [rectangleCoordinatesPolar[1,1], rectangleCoordinatesPolar[1,2]], 'r-')
                        elif side == 2:
                            ax.plot([rectangleCoordinatesPolar[0,2], rectangleCoordinatesPolar[0,3]], [rectangleCoordinatesPolar[1,2], rectangleCoordinatesPolar[1,3]], 'r-')
                        elif side == 3:
                            ax.plot([rectangleCoordinatesPolar[0,3], rectangleCoordinatesPolar[0,0]], [rectangleCoordinatesPolar[1,3], rectangleCoordinatesPolar[1,0]], 'r-')

            # Compute and plot centroids
            theta, ro = ut.compute_centroids(cartesian_points)
            if (ro >= 20):
                centers.append([theta, ro, k])
                if (CENTROIDS == True):
                    ax.scatter(theta, ro, s=60, c="red", lw=1, edgecolors="black")
                    ax.text(theta, ro, f'Centroid {k}', c='red')
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

            # Plotting real position and predicted position 
            ax.scatter(tCent, rCent, label='Centroid', s=80, c="red", lw=1, edgecolor='black')
            ax.scatter(tPred, rPred, label='Predicted point', s=60, c="blue", lw=1, edgecolor='black')
            ax.text(tPred, rPred, f'Kalman {kalman.id}', c='blue')
        
            # Compute the relative velocity [cm/s]
            tReal0, rReal0 = old_space[kalman.id]    # Perv 
            deltaSpace = rCent-rReal0   # in millimetri
            vRelative = ut.millimeter_to_centimeter(deltaSpace)/deltaTime # 1 cm/ms = 1m/s
            vRelative_list.append(vRelative)
            
        # Save position and speed values
        space_list.append(new_space)

        if VELOCITY_GRAPH:
            velocity.append(vRelative_list[0])
            space.append(ut.millimeter_to_centimeter(distances[0]))
            total_time += deltaTime
            time_list.append(total_time)
        
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
    
    # Initialize plot for the real-time visualization
    fig = plt.figure("Real-time RPLIDAR A1 Visualization")
    fig.set_size_inches(12, 8)
    ax = fig.add_subplot(projection='polar')
    ax.grid(True)
    
    # Perform the animation
    ani = animation.FuncAnimation(fig, update, fargs=(iterator, ax, lidar), interval=DELTA_TIME)
    plt.show()

    # Plot the relative velocity graphs  
    if VELOCITY_GRAPH:
        window_size = 2
        filtered_vRelative_list = ut.apply_moving_average_filter(velocity, window_size)
        plot_vRelative(filtered_vRelative_list)

    # Stop the lidar scanner
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()

if __name__ == "__main__":
    main()