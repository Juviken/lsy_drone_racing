
from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
#from lsy_drone_racing.wrapper import DroneRacingObservationWrapper
from lsy_drone_racing.optimization_utils import TrajGen
from lsy_drone_racing.planning_utils import Cylinder,gate_obstacle, waypoint_magic
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

import math
class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = True,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`.
            Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
                
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ## 
        #########################
        
        # Improved PID parameters
        self.kp = np.array([0.1, 0.1, 0.1])  # Increased Proportional gains for x, y, z
        self.ki = np.array([0.001, 0.001, 0.001])  # Increased Integral gains for x, y, z
        self.kd = np.array([0.03, 0.03, 0.03])  # Increased Derivative gains for x, y, z
        
 
        


        # Error accumulators
        self.target_velocity = np.zeros(3)
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

       
        gates = self.NOMINAL_GATES
        
        waypoints = []
        
        first = [self.initial_obs[0], self.initial_obs[2], 0.3] #Start position
        second = [1,0,z_low]    
        start = np.array([first])

        gatepoints = []
        for i in range(len(gates)):
            gatepoints.append([gates[i][0], gates[i][1], z_high if gates[i][-1] == 0 else z_low, gates[i][-2]]) #Add gate waypoints
        
        #Use waypoint magic - args: gatepoints, buffer_distance
        waypoints = waypoint_magic(np.array(gatepoints),buffer_distance=0.35)    #Add waypoints before and after gate to force straight passage
    
        #Add start to waypoints
        waypoints = np.concatenate((start,waypoints),axis=0)
        
        last_point = [initial_info["x_reference"][0],initial_info["x_reference"][2],initial_info["x_reference"][4]] #End position
        waypoints = np.concatenate((waypoints,[last_point]),axis=0)
        waypoints = np.array(waypoints)
    
        #Define obstacle dimensions
        obstacle_height = 0.9
        obstacle_radius = 0.15

        obstacles = []
        
        #Create obstacle models, args: radius, height, position
        for i in range(len(self.NOMINAL_OBSTACLES)):
            obstacles.append(Cylinder(obstacle_radius, obstacle_height, self.NOMINAL_OBSTACLES[i][0:3]))
        
        #Create gate obstacles
        for i in range(len(gates)):
            obstacles.append(gate_obstacle(gates[i],z_high if gates[i][-1] == 0 else z_low))
        
        
        t2 = np.linspace(0, 1, waypoints.shape[0])  # Time vector for each waypoint
        duration = 10  #Duration of the trajectory
        
        #Create an initial trajectory - optional
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ)) # Time vector for the trajectory
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        trajectory = interpolate.splev(t, tck)
        trajectory = np.array(trajectory).T
        
        #Load trajectory from file - optional
        initial_traj = np.loadtxt("trajectory/success_10sec.csv", delimiter=',')
        
        #Create trajectory generator
        traj_gen = TrajGen(
            waypoints,                          # Waypoints
            obstacles,                          # Obstacles
            t2,
            initial_guess=trajectory,         # Trajectory for initial guess
            duration=duration,                  # Duration of the trajectory
            ctrl_freq=30,
            obstacle_margin=0.3,
            obstacle_margin_gate=0.2, 
            max_iterations=20,
            alpha=0.01,
            use_initial=False)
        print("Trajectory object created")
        
        self.run_opt = False
        
        if self.run_opt:
            optimized_trajectory = traj_gen.optimize_trajectory() #Optimize the trajectory
            print("Optimized trajectory")
            
            traj_gen.save_trajectory('trajectory/optimized_trajectory.csv')
            print("Trajectory saved")
            current_traj = traj_gen.give_current()
            self.ref_x, self.ref_y, self.ref_z = current_traj[:, 0],current_traj[:, 1],current_traj[:, 2]
        
        else:
            print("Plotting from file...")
            filename = "trajectory/optimized_trajectory_test.csv"
            optimized_trajectory = np.loadtxt(filename, delimiter=',')
            current_traj = optimized_trajectory
            self.ref_x, self.ref_y, self.ref_z = current_traj[:, 0],current_traj[:, 1],current_traj[:, 2]
           
        self.waypoints = waypoints

        
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False


    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        iteration = int(ep_time * self.CTRL_FREQ)
        drone_position = obs[0:3]  # x, y, z position
        drone_velocity = obs[3:6]  # x, y, z acceleration
        drone_attitude = obs[6:9]  # phi (roll), theta (pitch), psi (yaw)
        #print("Viktigt",info.keys())

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True
        else:
            step = iteration - 2 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - 2 > 0 and step < len(self.ref_x):
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                error = target_pos - drone_position
                self.integral_error += error * self.CTRL_TIMESTEP
                derivative_error = (error - self.prev_error) / self.CTRL_TIMESTEP
                vel_der_error = (self.target_velocity - drone_velocity) 

                # PID output
                self.target_velocity = -self.kp * error  - self.kd * vel_der_error
                self.prev_error = error

                target_vel = self.target_velocity  # Assuming direct control of velocity
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)    
                target_yaw = 0.0             
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
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

