from __future__ import annotations
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from geometry import create_cylinder
from gats_obs import give_gate, give_obst



def plot_cylinder(radius,height,position) -> None:
  """_summary_

  Args:
      radius (_type_): radius in meters
      height (_type_): _height in meters
      position (numpy.array): x,y,z coordinates in meters
  """



def main():


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
  
  

  
  
  
if __name__ == '__main__':
    main()