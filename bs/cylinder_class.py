import numpy as np
from gats_obs import give_obst, give_gate

class Cylinder:
    def __init__(self, radius, height, x_center, y_center, z_center):
        self.radius = radius
        self.height = height
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center

    def center_coordinates(self):
        return [self.x_center, self.y_center, self.z_center]
      
      
def plot_stuff(obstacles):
    obstacle_height = 0.6
    obstacle_radius = 0.15

    waypoints = np.array(give_gate())
    #only keep the x,y and z positions from the waypoints
    waypoints = waypoints[:,0:3]
    print(waypoints)
    obstacle_positions = give_obst()
    obs1_pos = obstacle_positions[0][0:3]
    obs2_pos = obstacle_positions[1][0:3]
    obs3_pos = obstacle_positions[2][0:3]
    obs4_pos = obstacle_positions[3][0:3]
    #Print the obstacle positions
    for i in range(len(obstacle_positions)):
        print(f"Obstacle {i+1} position: {obstacle_positions[i]}")
    t2 = np.linspace(0, 1, waypoints.shape[0])

    obstacles = [cylinder1, cylinder2, cylinder3,cylinder4]

    for i in range(len(obstacles)):
        print(f"Obstacle {i+1} position: {obstacles[i].center_coordinates()}")



def main():
    cylinder1 = Cylinder(0.15, 0.6, 0.0, 0.0, 0.0)
    cylinder2 = Cylinder(0.15, 0.6, 0.0, 0.0, 0.0)
    cylinder3 = Cylinder(0.15, 0.6, 0.0, 0.0, 0.0)
    cylinder4 = Cylinder(0.15, 0.6, 0.0, 0.0, 0.0)

    obstacles = [cylinder1, cylinder2, cylinder3, cylinder4]
    for i in range(len(obstacles)):
        print(f"Obstacle {i+1} position: {obstacles[i].center_coordinates()}")
        
    plot_stuff(obstacles)