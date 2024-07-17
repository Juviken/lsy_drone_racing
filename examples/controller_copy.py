"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
#from lsy_drone_racing.wrapper import DroneRacingObservationWrapper
from lsy_drone_racing.normalized_class import TrajGen
from lsy_drone_racing.planning_utils import Cylinder,gate_obstacle, waypoint_magic
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from IPython import display





class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
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
        
        #Print the keys of initial_info
        print("Keys of initial_info: ", initial_info.keys())

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ## 
        #########################
        self.run_opt = False
        self.plot_env = True
        
        gates = self.NOMINAL_GATES
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]
        
        
        waypoints = []
        
        first = [self.initial_obs[0], self.initial_obs[2], 0.3]
        second = [1,0,z_low]
        start = np.array([first,second])
        
        
        #Print start position

        gates = self.NOMINAL_GATES
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]
        #Print gates
        print("Gates")
        print(gates [0][-1])
        print(gates [1][-1])
        print(gates [2][-1])
        print(gates [3][-1])
        #waypoints.append([1, 0, z_low])
        gatepoints = []
        gatepoints.append([gates[0][0] , gates[0][1], z_high if gates[0][-1] == 0 else z_low, gates[0][-2]])
        gatepoints.append([gates[1][0], gates[1][1], z_high if gates[1][-1] == 0 else z_low, gates[1][-2]])
        gatepoints.append([gates[2][0], gates[2][1], z_high if gates[2][-1] == 0 else z_low, gates[2][-2]])
        gatepoints.append([gates[3][0], gates[3][1] , z_high if gates[3][-1] == 0 else z_low, gates[3][-2]])
        
        waypoints.append([gates[0][0] , gates[0][1], z_low])
        waypoints.append([gates[1][0], gates[1][1], z_high])
        waypoints.append([gates[2][0], gates[2][1], z_low])

        waypoints.append([gates[3][0], gates[3][1] , z_high])
        #Use waypoint magic
        waypoints = waypoint_magic(np.array(gatepoints),buffer_distance=0.35)
        print("Waypoints",waypoints)
        #Add start to waypoints
        waypoints = np.concatenate((start,waypoints),axis=0)
        
        last_point = [initial_info["x_reference"][0],initial_info["x_reference"][2],initial_info["x_reference"][4]]
        waypoints = np.concatenate((waypoints,[last_point]),axis=0)
        #Print end position

        waypoints = np.array(waypoints)
    
        #Create obstacles
        obstacle_height = 0.9
        obstacle_radius = 0.15
        obstacle_margin = 0.3
        obstacle_positions = [self.NOMINAL_OBSTACLES[i][0:3] for i in range(4)]
        cyl1 = Cylinder(obstacle_radius, obstacle_height, self.NOMINAL_OBSTACLES[0][0:3])
        cyl2 = Cylinder(obstacle_radius, obstacle_height, self.NOMINAL_OBSTACLES[1][0:3])
        cyl3 = Cylinder(obstacle_radius, obstacle_height, self.NOMINAL_OBSTACLES[2][0:3])
        cyl4 = Cylinder(obstacle_radius, obstacle_height, self.NOMINAL_OBSTACLES[3][0:3])
        #Gate obstacles
        gate_obstacle1 = gate_obstacle(gates[0],z_high if gates [0][-1] == 1 else z_low)
        gate_obstacle2 = gate_obstacle(gates[1],z_high)
        gate_obstacle3 = gate_obstacle(gates[2],z_low)
        gate_obstacle4 = gate_obstacle(gates[3],z_high)
        obstacles = [cyl1,cyl2,cyl3,cyl4,gate_obstacle1,gate_obstacle2,gate_obstacle3,gate_obstacle4]
        
        
        t2 = np.linspace(0, 1, waypoints.shape[0])  # Time vector for each waypoint
        duration = 14
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ)) # Time vector for the trajectory
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        trajectory = interpolate.splev(t, tck)
        traj_gen = TrajGen(waypoints, obstacles,t2,trajectory,duration,ctrl_freq=30,obstacle_margin=0.35,obstacle_margin_gate=0.2, max_iterations=500,alpha=0.3,use_initial=False)
        print("Trajectory object created")
        

        
        self.run_opt = False
        if self.run_opt:
            optimized_trajectory = traj_gen.optimize_trajectory(plot_val=False)
            #print(optimized_trajectory)
            #Save final trajectory
            print("Hej")
            traj_gen.save_trajectory('optimized_trajectory.csv')
            current_traj = traj_gen.give_current()
            self.ref_x, self.ref_y, self.ref_z = current_traj[:, 0],current_traj[:, 1],current_traj[:, 2]
        
        else:
            print("Plotting from file...")
            filename = "optimized_trajectory.csv"
            optimized_trajectory = np.loadtxt(filename, delimiter=',')
            current_traj = optimized_trajectory
            self.ref_x, self.ref_y, self.ref_z = current_traj[:, 0],current_traj[:, 1],current_traj[:, 2]
   
        print("Here is info: ", initial_info.keys, self.NOMINAL_GATES)
        print("DIMS: ", initial_info["gate_dimensions"])
        #DOVUL!!!!


        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        

        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        self.waypoints = waypoints

        
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        #########################
        # REPLACE THIS (END) ####
        #########################

    
    

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)
        #Print the keys of info
        #print("Keys of info: ", info.keys())
        #Keys of interest: gates_pose, gates_in_range, obstacles_pose, obstacles_in_range, gate_id
        #Print the obstacle positions
        #print("Obstacle positions: ", info["gates_pose"])

        #########################
        # REPLACE THIS (START) ##
        #########################

        #DOVUL OBS 
       
        # DOVUL OBS END



        # Handcrafted solution for getting_stated scenario.

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            step = iteration - 2 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - 2 > 0 and step < len(self.ref_x):
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif step >= len(self.ref_x) and not self._setpoint_land:
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self._setpoint_land = True
            elif step >= len(self.ref_x) and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True  # Send landing command only once
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # Implement some learning algorithm here if needed

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################
