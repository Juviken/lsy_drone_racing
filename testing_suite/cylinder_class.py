from __future__ import annotations
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

from gats_obs import give_gate, give_obst
import csv

class Cylinder:
    def __init__(self, radius, height, center):
        self.radius = radius
        self.height = height
        self.x_center = center[0]
        self.y_center = center[1]
        self.z_center = center[2]
        self.obstacle_points = self.obstacle_points()
      
    def obstacle_points(self):
        #Numpy array of all points between the base and the top of the cylinder3
        return np.array([[self.x_center, self.y_center, z + (self.height/2)] for z in np.linspace(self.z_center - self.height/2, self.z_center + self.height/2, 10)])
    
    def __str__(self) -> str:
        return str(self.obstacle_points)
    
class gate_obstacle:
    """_summary_
    Create a gate object which contains 8 points, one above, one on the bottom and 2 to the sides considering the yaw of the gate
    """
    def __init__(self, waypoint,radius):
        self.x = waypoint[0]
        self.y = waypoint[1]
        self.z = waypoint[2]
        self.yaw = waypoint[5]
        self.radius = radius
        self.obstacle_points = self.gate_points()
        
    def gate_points(self):
        
        top = np.array([self.x, self.y, self.z + self.radius])
        bottom = np.array([self.x, self.y, self.z - self.radius])
        dx = self.radius * np.cos(self.yaw)
        dy = self.radius * np.sin(self.yaw)
        
        #Corners of the gate
        top_left = np.array([self.x - dx, self.y - dy, self.z + self.radius])
        top_right = np.array([self.x + dx, self.y + dy, self.z + self.radius])
        bottom_left = np.array([self.x - dx, self.y - dy, self.z - self.radius])
        bottom_right = np.array([self.x + dx, self.y + dy, self.z - self.radius])
        
        #Generate 5 points in-between each of the corners
        #Top row
        top_left_inter = np.array([self.x - dx/2, self.y - dy/2, self.z + self.radius])
        bottom_left_inter = np.array([self.x - dx/2, self.y - dy/2, self.z - self.radius])
        top_right_inter = np.array([self.x + dx/2, self.y + dy/2, self.z + self.radius])
        bottom_right_inter = np.array([self.x + dx/2, self.y + dy/2, self.z - self.radius])
        left_top_inter = np.array([self.x - dx, self.y - dy, self.z + self.radius/2])
        left_bottom_inter = np.array([self.x - dx, self.y - dy, self.z - self.radius/2])
        right_top_inter = np.array([self.x + dx, self.y + dy, self.z + self.radius/2])
        right_bottom_inter = np.array([self.x + dx, self.y + dy, self.z - self.radius/2])
        
        
        
        left = np.array([self.x - dx, self.y - dy, self.z])
        right = np.array([self.x + dx, self.y + dy, self.z])
        #Calculate the distance between left and right
        diff = np.linalg.norm(left-right)
        print("Diff",diff)
        return np.array([top,bottom,left,right,top_left,top_right,bottom_left,bottom_right,top_left_inter,bottom_left_inter,top_right_inter,bottom_right_inter,left_top_inter,left_bottom_inter, right_top_inter,right_bottom_inter])
        
    def __str__(self) -> str:
        return str(self.gate_points)


    
def waypoint_magic(waypoints: np.ndarray, buffer_distance: float = 0.25) -> np.ndarray:
    """
    Take gate waypoint information like x, y, z, and rotation and return the waypoints with new waypoints
    added just before and after each gate to ensure the drone does not hit the edge of the gate.
    The yaw is in radians between -pi and pi

    Args:
        waypoints (np.ndarray): Array of waypoints where each row represents [x, y, z, yaw] for a gate.
        buffer_distance (float): Distance before and after the gate to place additional waypoints.

    Returns:
        np.ndarray: Modified waypoints array including additional waypoints before and after each gate.
    """
    print("Input to magic:",waypoints)
    new_waypoints = []
    for i in range(len(waypoints)):
        # Extract the current waypoint, order is x, y, z,yaw
        x, y, z = waypoints[i, 0:3]
        yaw = waypoints[i, 5]
        

        # Calculate direction vector for the buffer distance before and after the gate
        dx = buffer_distance * np.cos(yaw+(np.pi/2))
        dy = buffer_distance * np.sin(yaw+(np.pi/2))
        
       
        #Calculations for placement of buffer vector
        #Height of next gatepoint unless it is the last gate



        
        z_next = waypoints[(i+1)%len(waypoints),2]
        
        
        
        if i==3:
            z_after = z
        else:
            z_after = (z_next + z)/2
        
        x_after = x + dx
        y_after = y + dy
        
        if i == 1:
            x_after = x_after-0.2
            y_after = y_after+0.4
        
        waypoint_after= [x_after, y_after, z_after]
        waypoint_before = [x - dx, y - dy, z]
        

        # Original waypoint (at the gate)
        
    
        new_waypoints.append(waypoint_before)
        new_waypoints.append([x, y, z])
        new_waypoints.append(waypoint_after)
        
        #new_waypoints.append([x, y, z])
        
        # Waypoint after the gate
        
        print("New waypoints:",new_waypoints)

    return np.array(new_waypoints)

def plot_stuff  (obstacles,waypoints):
    #Plot the obstacles and waypoints in 3D in an environment x,y,z where x = [-3, 3], y = [-3, 3], z = [0,6]
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
        ax.plot(obstacle.obstacle_points[:,0], obstacle.obstacle_points[:,1], obstacle.obstacle_points[:,2], color='r')
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], color='b')
    #Add limits to plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
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