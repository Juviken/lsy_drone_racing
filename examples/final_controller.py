from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import math

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
from lsy_drone_racing.optimization_utils import TrajGen
from lsy_drone_racing.planning_utils import Cylinder, gate_obstacle, waypoint_magic
from lsy_drone_racing.PID import PIDController


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = True,
    ):
        """Initialization of the controller."""
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose  
        self.BUFFER_SIZE = buffer_size

        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
                
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]

        self.reset()
        self.episode_reset()
        
        # Define PID controllers
        self.initialize_pid_controllers()
        
        self.target_yaw = np.pi/8  # Target yaw angle
        
        # Generate and optimize trajectory
        self.waypoints, self.ref_x, self.ref_y, self.ref_z = self.generate_trajectory(initial_info, z_high, z_low)
        
        if self.VERBOSE:
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False

    def initialize_pid_controllers(self):
        """Initialize the PID controllers."""
        dt = 1.0 / self.CTRL_FREQ  # Time step
        
        # Define PID gains for position to attitude conversion
        self.kp_pos = 0.3
        self.ki_pos = 0.001
        self.kd_pos = 0.01
        
        self.pos_pid = PIDController(np.array([self.kp_pos]), np.array([self.ki_pos]*3), np.array([self.kd_pos]*3), dt)
        
        # Define PID gains for attitude control
        self.kp_att = 0.1
        self.kd_att = 0.05
        self.att_pid = PIDController(np.array([self.kp_att]*3), np.array([0.0]*3), np.array([self.kd_att]*3), dt)

    def generate_trajectory(self, initial_info, z_high, z_low):
        """Generate the trajectory for the drone."""
        gates = self.NOMINAL_GATES
        
        start = np.array([[self.initial_obs[0], self.initial_obs[2], 0.3]])  # Add the starting position of the drone
        
        gatepoints = [[g[0], g[1], z_high if g[-1] == 0 else z_low, g[-2]] for g in gates]
        
        waypoints = waypoint_magic(np.array(gatepoints), buffer_distance=0.35)
        waypoints = np.concatenate((start, waypoints), axis=0)
        
        last_point = [initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]]
        waypoints = np.concatenate((waypoints, [last_point]), axis=0)
        waypoints = np.array(waypoints)
    
        obstacle_height = 0.9
        obstacle_radius = 0.15

        obstacles = [Cylinder(obstacle_radius, obstacle_height, pos[:3]) for pos in self.NOMINAL_OBSTACLES]
        obstacles.extend([gate_obstacle(g, z_high if g[-1] == 0 else z_low) for g in gates])
        
        t2 = np.linspace(0, 1, waypoints.shape[0])
        duration = 10
        
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        trajectory = np.array(interpolate.splev(t, tck)).T
        
        initial_traj = np.loadtxt("trajectory/success_10sec.csv", delimiter=',')
        
        traj_gen = TrajGen(
            waypoints,
            obstacles,
            t2,
            initial_guess=trajectory,
            duration=duration,
            ctrl_freq=30,
            obstacle_margin=0.3,
            obstacle_margin_gate=0.2,
            max_iterations=20,
            alpha=0.01,
            use_initial=False
        )
        
        self.run_opt = False
        
        if self.run_opt:
            optimized_trajectory = traj_gen.optimize_trajectory()
            traj_gen.save_trajectory('trajectory/optimized_trajectory.csv')
            current_traj = traj_gen.give_current()
        else:
            filename = "trajectory/success_10sec.csv"
            optimized_trajectory = np.loadtxt(filename, delimiter=',')
            current_traj = optimized_trajectory
        
        ref_x, ref_y, ref_z = current_traj[:, 0], current_traj[:, 1], current_traj[:, 2]
        assert max(ref_z) < 2.5, "Drone must stay below the ceiling"
        
        return waypoints, ref_x, ref_y, ref_z

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        iteration = int(ep_time * self.CTRL_FREQ)
        drone_position = obs[0:3]
        drone_velocity = obs[3:6]
        drone_attitude = obs[6:9]

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True
        else:
            step = iteration - 2 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - 2 > 0 and step < len(self.ref_x):
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                
                # Compute position error
                pos_error = target_pos - drone_position
                
                # Compute desired roll and pitch from position error
                desired_att = self.pos_pid.compute(pos_error)
                desired_roll = desired_att[1]  # Assuming roll is affected by y-error
                desired_pitch = desired_att[0]  # Assuming pitch is affected by x-error
                desired_yaw = self.target_yaw  # Assuming we want to keep yaw constant

                # Compute attitude error
                att_error = np.array([desired_roll, desired_pitch, desired_yaw]) - drone_attitude
                
                # Compute desired roll, pitch, and yaw rates
                rpy_rate_command = self.att_pid.compute(att_error)
                
                command_type = Command.FULLSTATE
                args = [target_pos, np.zeros(3), np.zeros(3), rpy_rate_command[2], rpy_rate_command, ep_time]
            elif step >= len(self.ref_x) and not self._setpoint_land:
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self._setpoint_land = True
            elif step >= len(self.ref_x) and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []

        return command_type, args

    def step_learn(self, action: list, obs: np.ndarray, reward: float | None = None, done: bool | None = None, info: dict | None = None):
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

    def episode_learn(self):
        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer
