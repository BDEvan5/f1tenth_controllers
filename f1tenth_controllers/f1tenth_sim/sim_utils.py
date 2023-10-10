import numpy as np
from scipy import interpolate
import datetime 
import yaml
import os


class CenterLine:
    def __init__(self, map_name) -> None:
        center_line = np.loadtxt("maps/" + map_name + "_centerline.csv", delimiter=',')[:, :2]
        el_lengths = np.linalg.norm(np.diff(center_line, axis=0), axis=1)
        old_s_track = np.insert(np.cumsum(el_lengths), 0, 0)
        self.s_track = np.arange(0, old_s_track[-1], 0.01) # cm level resolution
        tck = interpolate.splprep([center_line[:, 0], center_line[:, 1]], u=old_s_track, k=3, s=0)[0]
        self.center_line = np.array(interpolate.splev(self.s_track, tck, ext=3)).T

    def calculate_pose_progress(self, pose):
        dists = np.linalg.norm(pose[:2] - self.center_line[:, :2], axis=1) # last 20 points.
        progress = self.s_track[np.argmin(dists)] / self.s_track[-1]

        return progress
    
    def get_start_pose(self):
        pose = np.zeros(3)
        pose[:2] = self.center_line[0, :2]
        pose[2] = np.arctan2(self.center_line[1, 1] - self.center_line[0, 1],
                            self.center_line[1, 0] - self.center_line[0, 0])
        
        return pose

#TODO: change this to use arrays not lists.
class SimulatorHistory:
    def __init__(self, run_name=None, test_id="test"):
        dt = datetime.datetime.now().strftime("%y%m%d-%H%M%S") 
        if run_name is not None:
            dt = run_name 
        self.path = f"Logs/{dt}/"
        if os.path.exists(self.path) == False:
            os.mkdir(self.path)

        self.test_id = test_id
        self.current_path = None
        self.map_name = None
        self.states = []
        self.actions = []
    
        self.lap_n = 0

    def set_path(self, map_name):
        # self.current_path = self.path + map_name.upper() + "/"
        self.map_name = map_name
        self.current_path = self.path
    
    def add_memory_entry(self, state, action):
        self.states.append(state)
        self.actions.append(action)
    
    def save_history(self):
        states = np.array(self.states)
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)
        
        np.save(self.current_path + f"SimLog_{self.map_name}_{self.test_id}_{self.lap_n}.npy", lap_history)

        self.states = []
        self.actions = []
        self.lap_n += 1
        
    def save_run_dict(self, run_dict, computation_time):
        run_dict = vars(run_dict)
        run_dict["computation_time"] = computation_time

        with open(self.current_path + f"RunDict_{self.map_name}_{self.test_id}.yaml", "w") as f:
            yaml.dump(run_dict, f)



