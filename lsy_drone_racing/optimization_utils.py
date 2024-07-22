from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from lsy_drone_racing.geometry import create_cylinder

class TrajGen:
    def __init__(self, waypoints, obstacles, t2, initial_guess=None, duration=10, ctrl_freq=30, obstacle_margin=2,obstacle_margin_gate=0.2, max_iterations=50, alpha=0.01, use_initial=False):
        """
        Initialize the TrajGen class with waypoints, obstacles, and various parameters for trajectory generation and optimization.

        Args:
            waypoints (np.ndarray): Array of waypoint coordinates.
            obstacles (list): List of obstacles in the form of numpy arrays.
            t2 (np.ndarray): Array of time instances corresponding to waypoints.
            initial_guess (np.ndarray, optional): Initial guess for the trajectory. Defaults to None.
            duration (int, optional): Duration of the trajectory. Defaults to 10.
            ctrl_freq (int, optional): Control frequency. Defaults to 30.
            obstacle_margin (int, optional): Margin for obstacle avoidance. Defaults to 2.
            obstacle_margin_gate (float, optional): Margin for obstacle gate avoidance. Defaults to 0.2.
            max_iterations (int, optional): Maximum iterations for optimization. Defaults to 50.
            alpha (float, optional): Weight for total distance vs smoothness. Defaults to 0.7.
            scaling_factor (int, optional): Scaling factor for the waypoints. Defaults to 1.
            use_initial (bool, optional): Use the provided initial guess if True. Defaults to False.
        """
        self.waypoints = waypoints
        self.t2 = t2
        self.duration = duration
        self.ctrl_freq = ctrl_freq
        self.obstacle_margin = obstacle_margin
        self.obstacle_margin_gate = obstacle_margin_gate
        self.max_iterations = max_iterations
        self.alpha = alpha


        self.initial_guess = initial_guess if use_initial and initial_guess is not None else self.generate_initial_guess() # Generate initial guess if not provided
        self.current_trajectory = self.initial_guess
        self.intermediate_trajectories = {'Initial Guess': self.initial_guess}
        self.optimization_iterations = 0
        self.obstacles = obstacles
        self.obstacle_tree = self.create_obstacle_tree(obstacles[0:4])  # Create KD-tree for obstacles
        self.gate_tree = self.create_obstacle_tree(obstacles[4:8]) # Create KD-tree for gates

        


    def __str__(self):
        """Return the string representation of the TrajGen object

        Returns:
            str: String representation of the TrajGen object.
        """
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
        
        print("Creating obstacle tree...")
        for obs in obstacles:
            obstacle_points.extend(obs.obstacle_points)  # Add obstacle points to list
        return cKDTree(obstacle_points)     # Create KD-tree from list of obstacle points
    
    def plot_trajectory(self, trajectory):
        """Plot a single trajectory in 3D space

        Args:
            trajectory (np.array): Trajectory to plot.
        """
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
            ax.voxels(obs, facecolors='green', edgecolor='black')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cubic Spline Interpolation and Trajectory Optimization')
        ax.legend()
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
            ax.voxels(obs, facecolors='green', edgecolor='black')
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Optimization Progress of 3D Trajectories')
        if legend:
            ax.legend()
        plt.show()

    def optimize_trajectory(self) -> np.ndarray:
        """Optimize the trajectory considering obstacles and waypoints."""
        print("Starting optimization...")

        def callback(traj):
            """"Callback function to store intermediate trajectories."""
            self.ref_x, self.ref_y, self.ref_z = self.current_trajectory[:, 0],self.current_trajectory[:, 1],self.current_trajectory[:, 2]
            #assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"
            self.optimization_iterations += 1
            print(f"Iteration {self.optimization_iterations}")
            self.intermediate_trajectories[f"Iteration{self.optimization_iterations}"] = traj.reshape(-1, 3)
            self.current_trajectory = traj.reshape(-1, 3)

        constraints = [self.obstacle_inequality(),self.gate_inequality(), self.equality_constraint()] # Add constraints
        initial_guess_flat = self.initial_guess.flatten() # Flatten initial guess for optimization
        
        #Start optimization using SciPy's minimize function, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        result = minimize(
            lambda traj: self.combined_objective(traj),   # Objective function
            initial_guess_flat,                             # Initial guess
            constraints=constraints,                        # Constraints
            method='SLSQP',                                 # Optimization method, SLSQP, see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
            callback=callback,                              # Callback function to store intermediate trajectories and count iterations
            options={'maxiter': self.max_iterations}        # Maximum number of iterations
        )
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return None
        optimized_trajectory = result.x.reshape(-1, 3)
        self.current_trajectory = optimized_trajectory

        return optimized_trajectory

    def combined_objective(self, traj):
        """Calculate combined cost of trajectory
        alpha: weight for total distance vs smoothness, high alpha=more weight on total distance, low alpha=more weight on smoothness
        ."""
        traj = traj.reshape(-1, 3)
        total_distance = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1))) #L2 norm of first derivative
        smoothness = np.sum(np.diff(traj, n=2, axis=0) ** 2) #L2 norm of second derivative
        return self.alpha * total_distance + (1 - self.alpha) * smoothness
    

    def smoothness_objective(self, traj):
        """Calculate smoothness cost of trajectory."""
        traj = traj.reshape(-1, 3) #Reshape trajectory
        smoothness = np.sum(np.diff(traj, n=2, axis=0) ** 2) #L2 norm of second derivative
        return smoothness

    def obstacle_inequality(self):
        """Create inequality constraints based on obstacle avoidance."""
        def constraint(traj):
            traj = traj.reshape(-1, 3)
            distances, _ = self.obstacle_tree.query(traj)
            return np.min(distances) - self.obstacle_margin #For faster optimization
            #return distances - self.obstacle_margin
        return {'type': 'ineq', 'fun': constraint}
    
    def gate_inequality(self):
        
        def constraint(traj):
            traj = traj.reshape(-1, 3)
            distances, _ = self.gate_tree.query(traj)
            return np.min(distances) - self.obstacle_margin_gate #For faster optimization
            #return distances - self.obstacle_margin_gate
        
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
    scaling_factor = 100
    grid_size = (100, 100, 100)
    
    x, y, z = np.indices(grid_size)
    waypoints = np.array([[1, 1, 1], [50, 30, 20], [60, 85, 10], [90, 60, 30], [99, 99, 20]])
    t2 = np.linspace(0, 1, waypoints.shape[0])
    cylinder_radius = 3

    obstacle1 = (x < 2) & (y < 3) & (3 < z) & (z < 5)
    cylinder1 = create_cylinder((100, 100, 100), center=(20, 20, 0), radius=3, height=40, axis='z')
    cylinder2 = create_cylinder((100, 100, 100), center=(50, 50, 0), radius=3, height=40, axis='z')
    cylinder3 = create_cylinder((100, 100, 100), center=(80, 80, 0), radius=3, height=40, axis='z')
    obstacles = [cylinder1, cylinder2, cylinder3]

    traj_gen = TrajGen(waypoints, obstacles, t2, duration=10, ctrl_freq=30, obstacle_margin=cylinder_radius*2, max_iterations=500,alpha=0.15,scaling_factor=1)
    
    run_optimization = True
    
    if run_optimization:
        optimized_trajectory = traj_gen.optimize_trajectory()
        #print intermediate trajectories
        traj_gen.plot_intermediate_trajectories()
        #Print final trajectory
        print(traj_gen.intermediate_trajectories[f"Iteration{traj_gen.optimization_iterations}"])
        #Print shape of final trajectory
        print(traj_gen.intermediate_trajectories[f"Iteration{traj_gen.optimization_iterations}"].shape)
        #Save final trajectory
        traj_gen.save_trajectory('optimized_trajectory.csv')
        
    else:
        traj_gen.plot_from_csv('bs_testing/combined_1500.csv')
if __name__ == '__main__':
    main()
