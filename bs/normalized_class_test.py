from __future__ import annotations
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from geometry import create_cylinder
from gats_obs import give_gate, give_obst
from cylinder_class import Cylinder, gate_obstacle, waypoint_magic
import csv

class TrajGen:
    """Class for generating and optimizing 3D trajectories using cubic splines."""

    def __init__(self, waypoints, obstacles, t2,initial_guess=None, duration=10, ctrl_freq=30, obstacle_margin=2, max_iterations=50,alpha=0.7,use_initial=False):
        self.waypoints = waypoints
        self.t2 = t2
        self.duration = duration
        self.ctrl_freq = ctrl_freq
        self.obstacle_margin = obstacle_margin
        if use_initial:
            self.initial_guess = initial_guess
        else:
            self.initial_guess = self.generate_initial_guess()
        self.current_trajectory = None
        self.intermediate_trajectories = {'Initial Guess': self.initial_guess}
        self.optimization_iterations = 0
        self.obstacles = obstacles
        self.obstacle_tree = self.create_obstacle_tree(obstacles)
        self.max_iterations = max_iterations
        self.alpha=alpha

        

    
    def __str__(self):
        #Return string representation of the last trajectory in intermediate_trajectories
        return str(self.intermediate_trajectories[f"Iteration{self.optimization_iterations-1}"])

    def generate_initial_guess(self):
        """Generate an initial guess for the trajectory based on simple cubic spline interpolation."""
        print("Generating initial guess...")
        cs = CubicSpline(self.t2, self.waypoints)
        t_fine = np.linspace(min(self.t2), max(self.t2), self.duration * self.ctrl_freq)
        return cs(t_fine)

    def create_obstacle_tree(self, obstacles):
        """Convert obstacles to a KD-tree for fast distance calculations."""
        obstacle_points = []
        for obs in obstacles:
            points = obs.obstacle_points  # Convert indices to coordinates
            # Convert indices to coordinates
            obstacle_points.extend(points)  # Add obstacle points to list
        return cKDTree(obstacle_points)     # Create KD-tree from list of obstacle points

    def plot_trajectory(self, trajectory):
        if trajectory is None:
            print("No trajectory data to plot.")
            return
        # Continue with plotting if trajectory is valid
        print("Plotting single trajectory...")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], color='r', label='Waypoints')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Cubic Spline Trajectory')
        #Plot obstacles
        for obs in self.obstacles:
            ax.scatter(obs.obstacle_points[:, 0], obs.obstacle_points[:, 1], obs.obstacle_points[:, 2], color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cubic Spline Interpolation and Trajectory Optimization')
        ax.legend()
        #Set limits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.show()


    def plot_intermediate_trajectories(self,legend=False):
        """Plot intermediate trajectories."""
        print("Plotting intermediate trajectories...")
       
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for iteration, trajectory in self.intermediate_trajectories.items():
            ax.plot(trajectory[:, 0], trajectory[:, 1] , trajectory[:, 2], label=f'Iteration {iteration}')
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], color='r', label='Waypoints')
        #Plot obstacles
        for obs in self.obstacles:
            ax.plot(obs.obstacle_points[:, 0], obs.obstacle_points[:, 1], obs.obstacle_points[:, 2], color='r')
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Optimization Progress of 3D Trajectories')
        ax.legend()
        plt.show()

    def optimize_trajectory(self, plot_val=False):
        """Optimize the trajectory considering obstacles and waypoints."""
        print("Starting optimization...")
        print("Starting lsy optimization")
        def callback(traj):
            self.optimization_iterations += 1
            print(f"Iteration {self.optimization_iterations}")
            self.intermediate_trajectories[f"Iteration{self.optimization_iterations}"] = traj.reshape(-1, 3)
            self.current_trajectory = traj.reshape(-1, 3)

        constraints = [self.inequality_constraint(), self.equality_constraint()]
        initial_guess_flat = self.initial_guess.flatten()
        result = minimize(
            lambda traj: self.smoothness_objective(traj),
            initial_guess_flat,
            constraints=constraints,
            method='SLSQP',
            callback=callback,
            options={'maxiter': self.max_iterations}
        )
        #smoothness_objective, combined_objective(alpha)
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return None
        optimized_trajectory = result.x.reshape(-1, 3)
        self.current_trajectory = optimized_trajectory
        if plot_val:
            self.plot_trajectory(optimized_trajectory)
        return optimized_trajectory

    def combined_objective(self, traj):
        """Calculate combined cost of trajectory
        alpha: weight for total distance vs smoothness, high alpha=more weight on total distance, low alpha=more weight on smoothness
        ."""
        traj = traj.reshape(-1, 3)
        total_distance = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
        smoothness = np.sum(np.diff(traj, n=2, axis=0) ** 2) #L2 norm of second derivative
        return self.alpha * total_distance + (1 - self.alpha) * smoothness
    
    def smoothness_objective(self, traj):
        """Calculate smoothness cost of trajectory."""
        traj = traj.reshape(-1, 3)
        smoothness = np.sum(np.diff(traj, n=2, axis=0) ** 2)
        return smoothness

    def inequality_constraint(self):
        """Create inequality constraints based on obstacle avoidance."""
        def constraint(traj):
            traj = traj.reshape(-1, 3)
            distances, _ = self.obstacle_tree.query(traj)
            return np.min(distances - self.obstacle_margin)
        return {'type': 'ineq', 'fun': constraint}

    def equality_constraint(self):
        """Ensure waypoints are visited by constraining positions."""
        def constraint(traj):
            traj = traj.reshape(-1, 3)
            indices = np.round(self.t2 * self.duration * self.ctrl_freq).astype(int)
            indices = np.clip(indices, 0, traj.shape[0] - 1)
            return (traj[indices] - self.waypoints).flatten()
        return {'type': 'eq', 'fun': constraint}
    
    def save_trajectory(self, filename):
        """Save the optimized trajectory to a CSV file."""
        np.savetxt(filename, self.current_trajectory, delimiter=',')
        
    def plot_from_csv(self, filename):
        """Plot a trajectory from a CSV file."""
        print("Plotting from CSV file...")
        trajectory = np.loadtxt(filename, delimiter=',')
        self.plot_trajectory(trajectory)
    
    def give_current(self):
        return self.current_trajectory

def main():

    
    obstacle_height = 0.9
    obstacle_radius = 0.15
    z_low = 0.525
    z_high = 1.0
    gate_radius = 0.3

    gatepoints = np.array(give_gate())
    waypoints = gatepoints

    #Use waypoint magic
    waypoints = waypoint_magic(waypoints,buffer_distance=0.3)
    #only keep the x,y and z positions from the waypoints
    waypoints = waypoints[:,0:3]
    
    #Add a waypoints at the start of the trajectory
    start = np.array([[1,1,0.3],[1,0,z_low]])
    waypoints = np.concatenate((start,waypoints),axis=0)
    end = np.array([[0.0, -2.0, 0.5]])
    waypoints = np.concatenate((waypoints,end),axis=0)
    obstacle_positions = give_obst()
    obs1_pos = obstacle_positions[0][0:3]
    obs2_pos = obstacle_positions[1][0:3]
    obs3_pos = obstacle_positions[2][0:3]
    obs4_pos = obstacle_positions[3][0:3]
    t2 = np.linspace(0, 1, waypoints.shape[0])  # Time vector for waypoints
    

    cylinder1 = Cylinder(obstacle_radius, obstacle_height, obs1_pos)
    cylinder2 = Cylinder(obstacle_radius, obstacle_height, obs2_pos)
    cylinder3 = Cylinder(obstacle_radius, obstacle_height, obs3_pos)
    cylinder4 = Cylinder(obstacle_radius, obstacle_height, obs4_pos)
    gate1 = gate_obstacle(gatepoints[0],gate_radius)
    gate2 = gate_obstacle(gatepoints[1],gate_radius)
    gate3 = gate_obstacle(gatepoints[2],gate_radius)
    gate4 = gate_obstacle(gatepoints[3],gate_radius)
    obstacles = [cylinder1, cylinder2, cylinder3,cylinder4,gate1,gate2,gate3,gate4]
    
    #Create gate obstacles

    

    traj_gen = TrajGen(waypoints, obstacles, t2, duration=14, ctrl_freq=30, obstacle_margin=obstacle_radius*2, max_iterations=50,alpha=0.2,use_initial=False)
    #traj_gen.plot_trajectory(traj_gen.initial_guess)
    run_optimization = False
    
    if run_optimization:
        optimized_trajectory = traj_gen.optimize_trajectory(plot_val=False)
        #print intermediate trajectories
        traj_gen.plot_intermediate_trajectories()
        #Print final trajectory
        print(traj_gen.intermediate_trajectories[f"Iteration{traj_gen.optimization_iterations}"])
        #Print shape of final trajectory
        print(traj_gen.intermediate_trajectories[f"Iteration{traj_gen.optimization_iterations}"].shape)
        #Save final trajectory
        traj_gen.save_trajectory('optimized_trajectory_test.csv')
        
    else:
        traj_gen.plot_from_csv('optimized_trajectory_test.csv')

if __name__ == '__main__':
    main()
