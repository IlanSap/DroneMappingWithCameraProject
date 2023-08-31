import csv
import heapq
import multiprocessing
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy import polyfit
import almog_coreset_multi_dim as almog_coreset
import sys

# Input: filename (str) - The name of the .xyz file containing the data.
# Output: (np.ndarray) -  A 2D numpy array containing x, y, and z points from the file.
# Description: Reads a file in the .xyz format and extracts the x, y, and z coordinates of points where the z coordinate lies between -0.9 and 0.9. It then returns these points as a 2D numpy array.
# Done by: us
def read_xyz_file(filename):
    print(f"Reading file '{filename}'...")
    try:
        with open(filename, 'r') as file:
            # Read lines from the file
            lines = file.readlines()

            # Initialize an empty list to store the points
            points = []

            # Parse the lines and extract the x, y, and z coordinates
            for line in lines:
                x, y, z, _, _, _ = map(float, line.strip().split())
                if -0.9 < z < 0.9:
                    points.append([x, y])

            # Convert the list of points to a NumPy array
            points_array = np.array(points)

            return points_array
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None


# Input: points (List[List[float]])- A list or array of points where each point is an array containing x, y, and z values. param (str)- The name of the file to write to.
# Output: None.
# Description: Writes the given points to a file in the .xyz format.
# Done by: us
def write_xyz_file(points, param):
    with open(param, 'w') as file:
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")


# Input: data (np.ndarray)- A 2D numpy array containing the x and y coordinates of the points to be plotted. color (str, optional) - The color of the points. Default is 'blue'.
# Output: None.
# Description: Plots the given points on a scatter plot using matplotlib.
# Done by: us
def plot_points(data, color='blue'):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], label='points', color=color, s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('points')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# Input: data (np.ndarray) - A 2D numpy array of data points in polar coordinates. color (str, optional) - The color of the points. Default is 'blue'.
# Output: None.
# Description: Converts the given polar coordinates to Cartesian coordinates and plots them on a scatter plot.
# Done by: us
def plot_points_reconstructed(data, color='blue'):
    plt.figure(figsize=(10, 10))
    X = data[:, 0]
    Y = data[:, 1]
    rect = np.array([pol2cart(x, y) for x, y in zip(X, Y)])
    Xs = rect[:, 0]
    Ys = rect[:, 1]
    plt.scatter(Xs, Ys, label='room', color=color, s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('points')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# Input: point (List[float]) - A list or array containing the x and y Cartesian coordinates.
# Output: (List[float]) - A list containing the rho (distance from the origin) and phi (angle from the x-axis) polar coordinates
# Description: Converts Cartesian coordinates to polar coordinates.
# Done by: us
def cart2pol(point):
    x, y = point
    rho = np.sqrt((x ** 2 + y ** 2))
    phi = np.arctan2(y, x)
    return [phi, rho]


# Input: theta (float)- The angle from the x-axis in radians. r (float)- The distance from the origin.
# Output: (List[float]) - A list containing the x and y Cartesian coordinates.
# Description: Converts polar coordinates to Cartesian coordinates.
# Done by: us
def pol2cart(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return [x, y]


# Input: n (int)- Number of points on each side of the square. side_length (float)- The length of the sides of the square.
# Output: (np.ndarray)- A 2D numpy array containing the x and y coordinates of the points forming the square.
# Description: Creates a square using the given number of points and side length.
# Done by: us
def create_square(n, side_length):
    # Calculate the step size for the points on each side
    step = side_length / (n - 1)

    # Create the x and y coordinates for the square points on the perimeter
    x_coordinates = np.linspace(-side_length / 2, side_length / 2, n)  # Top and bottom sides
    y_coordinates = np.linspace(-side_length / 2, side_length / 2, n)  # Left and right sides

    # Combine the coordinates to form the square points
    top_points = np.array([(x, side_length / 2) for x in x_coordinates])
    bottom_points = np.array([(x, -side_length / 2) for x in x_coordinates[1:-1]])
    left_points = np.array([(-side_length / 2, y) for y in y_coordinates[1:-1]])
    right_points = np.array([(side_length / 2, y) for y in y_coordinates[1:-1]])

    # Concatenate all points to form the final square ndarray
    square_points = np.concatenate((top_points, right_points, bottom_points[::-1], left_points[::-1]))

    return square_points

# Input: num_points (int)- Number of points per edge to represent the square. side_length( float)- Length of the sides of the square. percentage_to_remove (float)- Percentage of points to remove from the right edge.
# Output: (np.ndarray)- A 2D numpy array containing the x and y coordinates of the points forming the square with an opening.
# Description: Creates a square with an opening by removing a specified percentage of points from the middle of the right edge.
# Done by: us
def generate_square_with_opening(num_points, side_length, percentage_to_remove):
    # Calculate the step size for the points on each side
    step = side_length / (num_points - 1)

    # Create the x and y coordinates for the square points on the perimeter
    x_coordinates = np.linspace(-side_length / 2, side_length / 2, num_points)  # Top and bottom sides
    y_coordinates = np.linspace(-side_length / 2, side_length / 2, num_points)  # Left and right sides

    # Calculate the number of points to remove from the right edge
    points_to_remove = int(num_points * percentage_to_remove / 100)

    # Calculate the middle index
    middle_index = num_points // 2

    # Calculate the number of points to remove on each side of the middle index
    points_to_remove_per_side = points_to_remove // 2

    # Generate the points for the top and bottom sides of the square
    top_points = np.array([(x, side_length / 2) for x in x_coordinates])
    bottom_points = np.array([(x, -side_length / 2) for x in x_coordinates[1:-1]])

    # Generate the points for the left side of the square
    left_points = np.array([(-side_length / 2, y) for y in y_coordinates[1:-1]])

    # Generate the points for the right side of the square (excluding the points to be removed)
    right_points = np.concatenate([
        np.array([(side_length / 2, y) for y in y_coordinates[middle_index + points_to_remove_per_side + 1: -1]]),
        np.array([(side_length / 2, y) for y in y_coordinates[1: middle_index - points_to_remove_per_side]])
    ])

    # Concatenate all points to form the final square ndarray
    square_points = np.concatenate((top_points, right_points, bottom_points[::-1], left_points[::-1]))

    return square_points


# Input: matrix (numpy.ndarray)- The 2D matrix to be written to the CSV file. filename (str)- The name of the CSV file to be created.
# Output: None
# Description: Writes the provided 2D matrix to a CSV file.
# Done by: us
def write_matrix_to_csv(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)


# Input: filename (str)- The name of the CSV file to be read.
# Output: (numpy.ndarray)- The 2D matrix read from the CSV file.
# Description: Reads a 2D matrix from a CSV file.
# Done by: us
def read_matrix_from_csv(filename):
    print("Reading matrix from csv\n")
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        matrix = list(reader)
    return np.array(matrix, dtype=float)


# Input: graph (np.ndarray) - A 2D numpy array representing the weighted adjacency matrix of a graph. source (int)- The index of the source vertex.
# Output: (List[int]) - A list containing the vertices of the shortest path.
# Description: Finds the shortest path from the source vertex to all other vertices in the graph using Dijkstra's algorithm. The function tweaks the traditional algorithm to account for a threshold alpha.
# Done by: us with help from Bar Israel
def dijkstra_shortest_path(graph, source):
    print(f"Finding shortest path...\n")
    low = 0
    high = 100000
    prev = False
    found = False
    while high > low:
        alpha = (high + low) // 2
        n = graph.shape[0]
        dist = [float('inf')] * n
        path = [[0]] * n
        dist[source] = 0
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for u in range(n):
                for v in range(n):
                    if alpha < graph[u][v] < float('inf') and dist[u] + graph[u][v] < dist[v]:
                        if len(path[u]) < k:
                            dist[v] = dist[u] + graph[u][v]
                            path[v] = path[u] + [v]
        if (len(path[n - 1]) >= k):
            # Success
            if found == True:
                print(f"Found the best path! for alpha = {alpha}\n")
                return path[n - 1]
            if prev == True:
                prev = False
                low = alpha + 1
                print(f"Path found with alpha = {alpha}, high = {high}, low = {low}. Trying again with alpha = {(high - low) // 2}...\n")
            else:
                prev = True
                alpha += 1
                print(f"Path found with alpha = {alpha}, high = {high}, low = {low}. Trying again with alpha = {(high - low) // 2}...\n")
        else:
            #Failure
            if prev == True:
                found = True
                alpha -= 1
            else:
                high = alpha - 1
                print(
                    f"Path found with alpha = {alpha}, high = {high}, low = {low}. Trying again with alpha = {(high - low) // 2}...\n")
    return path[n - 1]


# Input: adj_matrix (np.ndarray) - A 2D numpy array representing the adjacency matrix of a graph. max_length (float) - The maximum length for determining the corners.
# Output: (List[int]) - A list containing the vertices that represent corners.
# Description: Determines the corners of a shape based on the adjacency matrix and a maximum length.
# Done by: us
def find_corners(adj_matrix, max_length):

    # create a matrix size number_of_points X size_of_path X the_path_itself and initialize every cell to infinity
    shortest_paths = np.full((len(adj_matrix), max_length+1, 2), np.inf)

    # initialize the weight of the path from vertex 0 to itself as weight 0.
    shortest_paths[0, 0] = 0

    # for every vertex:
    for node_index in range(0, len(adj_matrix)):

        # take all the distances from node_index that create an upper triangle in the matrix. (a row)
        distances_from_here = adj_matrix[node_index:(node_index+1), node_index:]

        # take the distances from the cell that save it. (a row)
        distance_to_here = shortest_paths[node_index:(node_index+1), :, 0]


        total_distance_from_here = distance_to_here + distances_from_here.T

        # create boolean array
        better_distances = shortest_paths[node_index:, 1:, 0] > total_distance_from_here[:, :-1]

        shortest_paths[node_index:, 1:, 0][better_distances] = total_distance_from_here[:, :-1][better_distances]

        shortest_paths[node_index:, 1:, 1][better_distances] = node_index

    corners=list()
    corners.append(shortest_paths[len(adj_matrix)-1, max_length, 1].astype(int))
    for i in range(1, max_length):
        corners.append(shortest_paths[corners[-1], max_length-i, 1].astype(int))
    corners.reverse()
    corners.append(len(adj_matrix)-1)
    return corners


# Description: Represents a room and its various properties, and provides methods for processing and analyzing the room's data.
class Room:

    # Input: None
    # Output: None
    # Description: Initializes an instance of the Room class with various lists and attributes used to store room data and computations.
    # Done by: us
    def __init__(self):
        self.coefficients = []
        self.djicoeffs = []
        self.data = []
        self.rectangle = []
        self.polarRoom = []
        self.sorted_room = []
        self.MSEmat = []
        self.segments = []
        self.coreset_room = []
        self.weights = []
    
    
    # Input: data (np.ndarray) - 2D array containing data points of the room in Cartesian coordinates.
    # Output: None
    # Description: Converts the given data points from Cartesian to polar coordinates, sorts them, and stores the results in respective attributes.
    # Done by: us
    def fit(self, data):
        self.data = data
        self.polarRoom = (np.array([cart2pol(point) for point in self.data]))
        self.sorted_room = self.polarRoom[self.polarRoom[:, 0].argsort()]
    
    
    # Input: None
    # Output: (np.ndarray) - 2D array representing the MSE matrix.
    # Description: Creates the MSE matrix using multithreading for parallel computation.
    # Done by: us
    def create_MSE_matrix_with_threads(self):
        print(f"Creating MSE matrix...")
        size = self.sorted_room.shape[0]
        step = int((size / number_of_threads))
        self.MSEmat = np.full((size, size), float('inf'))
        threads = [threading.Thread(target=self.MSE_thread, args=(i * step, (i + 1) * step)) for i in
                   range(number_of_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return self.MSEmat
    
    
    # Input: None
    # Output: (np.ndarray) - 2D array representing the MSE matrix.
    # Description: Creates the MSE matrix using multiprocessing for parallel computation.
    # Done by: us
    def create_MSE_matrix_with_processes(self) -> object:
        print(f"Creating MSE matrix...")
        # plot_points(self.coreset_room)
        size = self.coreset_room.shape[0]
        step = int(np.ceil((size / number_of_processes)))
        self.MSEmat = multiprocessing.Array('d', size * size)  # Creating a shared array for multiprocessing
        processes = [multiprocessing.Process(target=self.MSE_process, args=(i * step, (i + 1) * step, size * size, i))
                     for i in reversed(range(number_of_processes))]
        for i, process in enumerate(processes):
            process.start()
            print(f"Process {i + 1} started")
        for process in processes:
            process.join()
        return np.frombuffer(self.MSEmat.get_obj()).reshape((size, size))
    
    
    # Input: None
    # Output: (np.ndarray) - 2D array representing the MSE matrix.
    # Description: Creates the MSE matrix without any parallel computation.
    # Done by: us
    def create_MSE_matrix_no_parallel(self):
        size = self.coreset_room.shape[0]
        # step = int((size / number_of_threads))
        self.MSEmat = np.full((size, size), float('inf'))
        for q in tqdm(range(size), desc="Creating MSE matrix"):
            for j in range(q):
                self.MSEmat[j][q] = self.MSE(q, j)
        return self.MSEmat
    
    
    # Input: start (int) - Start index for processing. end (int) - End index for processing. matrix_size (int) - Total size of the MSE matrix. indexi (int) - Index of the current process.
    # Output: None
    # Description: A helper method for create_MSE_matrix_with_processes to compute a segment of the MSE matrix.
    # Done by: us
    def MSE_process(self, start, end, matrix_size, indexi):
        for q in range(start, end):
            for j in range(q):
                index = j * self.sorted_room.shape[0] + q
                if index < matrix_size:
                    self.MSEmat[index] = self.MSE(q, j)
        print(f"Process {indexi + 1} finished")
        return
    
    
    # Input: start (int) - Start index for processing. end (int) - End index for processing.
    # Output: None
    # Description: A helper method for create_MSE_matrix_with_threads to compute a segment of the MSE matrix.
    # Done by: us
    def MSE_thread(self, start, end):
        for q in range(start, end):
            for j in range(q):
                self.MSEmat[q][j] = self.MSE(q, j)
        print(f"Thread {q} finished")
        return
        
    
    # Input: i (int) - Index i of the data point. j (int) - Index j of the data point.
    # Output: (float) - The computed Mean Squared Error (MSE) between points i and j.
    # Description: Computes the MSE between the given indices.
    # Done by: us
    def MSE(self, i, j):
        mse = \
            polyfit(self.coreset_room[j:i, 0], self.coreset_room[j:i, 1], 2, w=np.array(self.weights[j:i]), full=True)[
                1]
        if np.any(mse):
            return mse[0]
        else:
            return 0
            
    
    # Input: None
    # Output: None
    # Description: Cleans up the sorted room data by removing unwanted data points based on certain conditions.
    # Done by: us
    def cleanup(self):
        print(f"Cleaning up...\n")
        print(f"Sorted room size: {self.sorted_room.shape[0]} before deletion\n")
        plot_points_reconstructed(self.sorted_room, 'green')
        dirty_points = []
        for i in range(k - 1):
            if self.segments[i] != self.segments[i + 1]:
                Xc = room.sorted_room[self.segments[i]:self.segments[i + 1], 0]
                Yc = room.sorted_room[self.segments[i]:self.segments[i + 1], 1]
                poly = np.poly1d(polyfit(self.sorted_room[self.segments[i]:self.segments[i + 1], 0],
                                         self.sorted_room[self.segments[i]:self.segments[i + 1], 1], 2))
                j = 0
                for x, y in zip(Xc, Yc):
                    if y < poly(x):
                        dirty_points.append(segments[i] + j)
                    j += 1
        self.sorted_room = np.delete(self.sorted_room, dirty_points, axis=0)
        print(f"Sorted room size: {self.sorted_room.shape[0]} after deletion\n")
        

    # Input: epsilon (float, optional) - Epsilon value for the coreset computation. Default is 0.01. k (int, optional) - Number of segments. Default is 4.
    # Output: None
    # Description: Computes the coreset of the sorted room data based on given parameters.
    # Done by: us
    def coreset(self, epsilon=0.01, k=4):
        self.coreset_room, self.weights, _, _ = almog_coreset.to_coreset([list(point) for point in self.sorted_room],
                                                                         epsilon, k)
        self.coreset_room = np.array(self.coreset_room)


    # Input: None
    # Output: (int) - Index of the sparsest segment.
    # Description: Analyzes segments to identify and return the sparsest one.
    # Done by: us
    def find_sparse_parabola(self):
        min_dens = float('inf')
        mini = 0
        for i in tqdm(range(len(segments) - 1), desc="Plotting segments"):
            if segments[i] != segments[i + 1]:
                X = self.sorted_room[segments[i]:segments[i + 1], 0]
                Y = np.poly1d(polyfit(self.sorted_room[self.segments[i]:self.segments[i + 1], 0],
                                      self.sorted_room[segments[i]:self.segments[i + 1], 1], 2))(X)
                if self.segments[i] != self.segments[i + 1]:
                    dens = ((self.segments[i + 1] - self.segments[i]) / (
                            self.sorted_room[segments[i + 1], 0] - self.sorted_room[segments[i], 0]))
                    if dens < min_dens:
                        min_dens = dens
                        mini = i
                # plt.plot(X, Y, label='points', color='red')
                rect = np.array([pol2cart(x, y) for x, y in zip(X, Y)])
                Xs = rect[:, 0]
                Ys = rect[:, 1]
                plt.plot(Xs, Ys, label='Wall', color='red')
        return mini

    # Input: mini (int) - Index of the sparsest segment.
    # Output: max_col (int), max_row (int) - Indices of points defining the sparsest sub-section.
    # Description: Examines every sub-section of the sparsest segment to identify the sparsest one.
    # Done by: us
    def find_exit(self,mini):
        max_value = float('-inf')
        max_row = -1
        max_col = -1
        for row_idx in tqdm(range(self.segments[mini], self.segments[mini + 1]), desc="Finding exit segment"):
            for col_idx in range(self.segments[mini], row_idx):
                d = (np.abs(room.sorted_room[row_idx, 0] - room.sorted_room[col_idx, 0]))
                n = (np.abs(row_idx - col_idx))
                value = (d / n)
                if value != float('inf') and value > max_value:
                    max_value = value
                    max_row = row_idx
                    max_col = col_idx
        print(f"Max value: {max_value} in row: {max_row} and col: {max_col}")
        return max_row,max_col



# Overview: 
# The script is designed to process point cloud data to identify and visualize specific structural features, such as segments and potential exit points.
# It loads the data, fits it into the Room class, performs computations to create a Mean Square Error (MSE) matrix, and finally visualizes the determined segments and potential exits.
if __name__ == '__main__':
    # Parameters Configuration:
    # k (int)- Number of desired segments. 
    # epsilon (float)- Parameter for the coreset function which likely influences the approximation accuracy.
    # bCleanUp (bool)- A boolean flag to potentially clean data (although not used in this main function). 
    # filename (str)- Accepts the input filename from the command line arguments.
    k = 8
    epsilon = 0.01 
    bCleanUp = False
    number_of_threads = 16
    number_of_processes = 1
    filename = sys.argv[1]
    
    
    # Data Loading & Processing:
    room = Room() 
    # room.fit(pd.read_csv('pointData0.csv', header=None).to_numpy()[::3, :2])
    room.fit(read_xyz_file(filename)[:, :2]) # load xyz file as 2d array
    room.coreset(epsilon, k + 1) # generate a core subset of the data.
    # plot_points(room.data)
    # room.fit(read_xyz_file('point_cloud.xyz')[::2000, :2])
    # room.fit(read_xyz_file('square.xyz')[:, :2])
    # room.fit(create_square(100, 100))
    # room.fit(generate_square_with_opening(100, 100, 20))
    start = time.time()
    
    
    # Mean Square Error (MSE) Matrix Creation:
    matrix = room.create_MSE_matrix_no_parallel() # The MSE matrix is computed for the coresetted room data.
    # matrix = room.create_MSE_matrix_with_threads()
    # matrix = room.create_MSE_matrix_with_processes()
    # write_matrix_to_csv(matrix, 'MSEXY.csv')
    # matrix = read_matrix_from_csv('MSEYZ.csv')
    # matrix = read_matrix_from_csv('MSEXY.csv')
    #segments = dijkstra_shortest_path(matrix, 0, k, alpha)
    
    
    # Determine Segments:
    # Using the MSE matrix, segments within the data are identified and a list of segment indices is returned.
    segments = find_corners(matrix, k)
    room.segments = [np.where(room.sorted_room == point)[0][0] for point in room.coreset_room[segments, :]]
    segments = room.segments

    # Visualization:
    plt.figure(figsize=(10, 10))
    # plt.scatter(room.polarRoom[:, 0], room.polarRoom[:, 1], label='points', color='blue', s=1)
    mini = room.find_sparse_parabola() # The sparsest parabola within the data is identified, which represents the parabola with the fewest number of points.
    # plot all the data points
    X = room.sorted_room[segments[mini]:segments[mini + 1], 0]
    Y = np.poly1d(polyfit(room.sorted_room[segments[mini]:segments[mini + 1], 0],
                          room.sorted_room[segments[mini]:segments[mini + 1], 1], 2))(X)
    rect = np.array([pol2cart(x, y) for x, y in zip(X, Y)])
    Xs = rect[:, 0]
    Ys = rect[:, 1]
    plt.plot(Xs, Ys, label='Exit Segment', color='blue')
    plt.scatter(room.data[:, 0], room.data[:, 1], label='points', color='green', s=1)
    max_row,max_col = room.find_exit(mini) # determining the exit segment (likely representing an opening in the point cloud such as a doorway)

    X = room.sorted_room[max_col:max_row+1, 0]
    poly = np.poly1d(polyfit(room.sorted_room[segments[mini]:segments[mini + 1], 0],
                             room.sorted_room[segments[mini]:segments[mini + 1], 1], 2))(X)
    Y = np.poly1d(polyfit(room.sorted_room[max_col:max_row+1, 0],
                          poly, 2))(X)
    rect = np.array([pol2cart(x, y) for x, y in zip(X, Y)])
    Xs = rect[:, 0]
    Ys = rect[:, 1]
    exit_point = pol2cart(((room.sorted_room[max_col,0]+room.sorted_room[max_row,0])/2),((room.sorted_room[max_col,1]+room.sorted_room[max_row,1])/2)) # calculate the average between start of the exit segment and the end of it, the result is the point in which the drone should exit the room
    print(f"Exit in: ${exit_point}$")
    plt.scatter(exit_point[0],exit_point[1],color='orange')
    plt.plot(Xs, Ys, label='Exit', color='yellow', linewidth=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'points with {k} segments')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
