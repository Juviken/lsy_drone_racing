from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_cylinder(grid_size, center, radius, height, axis='z'):
    """
    Create a cylindrical obstacle in a 3D grid.

    Parameters:
    grid_size : tuple of int
        The dimensions of the 3D grid (x, y, z).
    center : tuple of float
        The (x, y, z) coordinates of the cylinder's base center.
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder.
    axis : str
        The axis along which the cylinder extends ('x', 'y', or 'z').

    Returns:
    cylinder : 3D numpy array
        A boolean array where True indicates the presence of the obstacle.
    """
    x, y, z = np.indices(grid_size)
    
    #Reduce all the values in x,y,z to between -3 and 3
    x = x/20
    y = y/20
    z = z/20
    x = x - 3
    y = y - 3
    
    
    if axis == 'z':
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        cylinder = (dist_from_center <= radius) & (z >= center[2]) & (z < center[2] + height)
    elif axis == 'x':
        dist_from_center = np.sqrt((y - center[1])**2 + (z - center[2])**2)
        cylinder = (dist_from_center <= radius) & (x >= center[0]) & (x < center[0] + height)
    elif axis == 'y':
        dist_from_center = np.sqrt((x - center[0])**2 + (z - center[2])**2)
        cylinder = (dist_from_center <= radius) & (y >= center[1]) & (y < center[1] + height)
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    return cylinder


