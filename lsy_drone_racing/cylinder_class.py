from __future__ import annotations
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

import csv

class Cylinder:
    def __init__(self, radius, height, center):
        self.radius = radius
        self.height = height
        self.x_center = center[0]
        self.y_center = center[1]
        self.z_center = center[2]
        self.cylinder_points = self.cylinder_points()
      
    def cylinder_points(self):
        #Numpy array of all points between the base and the top of the cylinder3
        return np.array([[self.x_center, self.y_center, z + (self.height/2)] for z in np.linspace(self.z_center - self.height/2, self.z_center + self.height/2, 10)])
    
    def __str__(self) -> str:
        return str(self.cylinder_points)

def plot_stuff  (obstacles,waypoints):
    #Plot the obstacles and waypoints in 3D in an environment x,y,z where x = [-3, 3], y = [-3, 3], z = [0,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Obstacles and Waypoints')
    for obstacle in obstacles:
        ax.plot(obstacle.cylinder_points[:,0], obstacle.cylinder_points[:,1], obstacle.cylinder_points[:,2], color='r')
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], color='b')
    plt.show()
    

def main():
    
    obstacle_height = 0.6
    obstacle_radius = 0.15
    waypoints = np.array(give_gate())
    #only keep the x,y and z positions from the waypoints
    waypoints = waypoints[:,0:3]
    print("Waypoints",waypoints)
    obstacle_positions = give_obst()
    obs1_pos = obstacle_positions[0][0:3]
    obs2_pos = obstacle_positions[1][0:3]
    obs3_pos = obstacle_positions[2][0:3]
    obs4_pos = obstacle_positions[3][0:3]
    cylinder1 = Cylinder(0.15, 0.6, obs1_pos)
    cylinder2 = Cylinder(0.15, 0.6, obs2_pos)
    cylinder3 = Cylinder(0.15, 0.6, obs3_pos)
    cylinder4 = Cylinder(0.15, 0.6, obs4_pos)

    obstacles = [cylinder1, cylinder2, cylinder3, cylinder4]
    print("Object1",cylinder1)
        
    plot_stuff(obstacles,waypoints)
    
    
if __name__ == "__main__":
    main()