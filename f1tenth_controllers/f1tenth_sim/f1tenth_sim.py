

from f1tenth_controllers.f1tenth_sim.dynamics_simulator import DynamicsSimulator
from f1tenth_controllers.f1tenth_sim.laser_models import ScanSimulator2D
from f1tenth_controllers.f1tenth_sim.utils import CenterLine, SimulatorHistory

import numpy as np

'''
    params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
    mu: surface friction coefficient
    C_Sf: Cornering stiffness coefficient, front
    C_Sr: Cornering stiffness coefficient, rear
    lf: Distance from center of gravity to front axle
    lr: Distance from center of gravity to rear axle
    h: Height of center of gravity
    m: Total mass of the vehicle
    I: Moment of inertial of the entire vehicle about the z axis
    s_min: Minimum steering angle constraint
    s_max: Maximum steering angle constraint
    sv_min: Minimum steering velocity constraint
    sv_max: Maximum steering velocity constraint
    v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
    a_max: Maximum longitudinal acceleration
    v_min: Minimum longitudinal velocity
    v_max: Maximum longitudinal velocity
    width: width of the vehicle in meters
    length: length of the vehicle in meters
'''


class F1TenthSim:
    """
            seed (int, default=12345): seed for random state and reproducibility
    """
    def __init__(self, map_name, run_dict, save_history=False, run_name=None):
        self.run_dict = run_dict
        self.map_name = map_name
        self.timestep = self.run_dict.timestep

        self.scan_simulator = ScanSimulator2D(self.run_dict.num_beams, self.run_dict.fov)
        self.scan_simulator.set_map(self.map_name)
        self.dynamics_simulator = DynamicsSimulator(self.run_dict.random_seed, self.timestep)
        self.scan_rng = np.random.default_rng(seed=self.run_dict.random_seed)
        self.center_line = CenterLine(map_name)

        self.current_time = 0.0
        self.current_state = np.zeros((7, ))
        self.lap_number = -1

        self.history = None
        if save_history:
            self.history = SimulatorHistory(run_name)
            self.history.set_path(self.map_name)

    def step(self, action):
        if self.history is not None:
            self.history.add_memory_entry(self.current_state, action)

        mini_i = self.run_dict.n_sim_steps
        while mini_i > 0:
            vehicle_state = self.dynamics_simulator.update_pose(action[0], action[1])
            self.current_time = self.current_time + self.timestep
            mini_i -= 1
        
        pose = np.append(vehicle_state[0:2], vehicle_state[4])
        scan = self.scan_simulator.scan(np.append(vehicle_state[0:2], vehicle_state[4]), self.scan_rng)

        self.current_state = vehicle_state
        self.collision = self.check_vehicle_collision(pose)
        self.lap_complete, progress = self.check_lap_complete(pose)

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        observation = {"scan": scan,
                        "vehicle_state": self.dynamics_simulator.state,
                        "collision": self.collision,
                        "lap_complete": self.lap_complete,
                        "laptime": self.current_time,
                        "progress": progress}
        
        done = self.collision or self.lap_complete
        if done: 
            self.history.save_history()

        if self.collision:
            print(f"{self.lap_number} COLLISION: Time: {self.current_time:.2f}, Progress: {100*progress:.1f}")
        elif self.lap_complete:
            print(f"{self.lap_number} LAP COMPLETE: Time: {self.current_time:.2f}, Progress: {(100*progress):.1f}")


        return observation, done

    def check_lap_complete(self, pose):
        progress = self.center_line.calculate_pose_progress(pose)
        
        done = False
        if progress > 0.99 and self.current_time > 5: done = True
        if self.current_time > 150: 
            print("Time limit reached")
            done = True

        return done, progress
        

    def check_vehicle_collision(self, pose):
        rotation_mtx = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])

        pts = np.array([[self.run_dict.vehicle_length/2, self.run_dict.vehicle_width/2], 
                        [self.run_dict.vehicle_length, -self.run_dict.vehicle_width/2], 
                        [-self.run_dict.vehicle_length, self.run_dict.vehicle_width/2], 
                        [-self.run_dict.vehicle_length, -self.run_dict.vehicle_width/2]])
        pts = np.matmul(pts, rotation_mtx.T) + pose[0:2]

        for i in range(4):
            if self.scan_simulator.check_location(pts[i, :]):
                return True

        return False
    

    def reset(self, poses):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self.current_time = 0.0

        self.dynamics_simulator.reset(poses)

        # get no input observations
        action = np.zeros(2)
        obs, done = self.step(action)

        self.lap_number += 1
        
        return obs, done


    
