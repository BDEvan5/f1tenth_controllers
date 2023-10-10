import csv
import numpy as np
from numba import njit
from scipy.interpolate import splev, splprep
from scipy.optimize import fmin
from scipy.spatial import distance 
import glob, os


class TrackingAccuracy:
    def __init__(self, map_name) -> None:
        self.map_name = map_name
        self.wpts = None
        self.widths = None
        self.load_centerline()
        
        self.el_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.total_s = self.s_track[-1]

        self.tck = splprep([self.wpts[:, 0], self.wpts[:, 1]], k=3, s=0)[0]


    def load_centerline(self):
        filename = 'maps/' + self.map_name + '_centerline.csv'
        xs, ys, w_rs, w_ls = [], [], [], []
        with open(filename, 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))
                w_rs.append(float(lines[2]))
                w_ls.append(float(lines[3]))
        xs[-1] = 0
        ys[-1] = 0
        self.xs = np.array(xs)[:, None]
        self.ys = np.array(ys)[:, None]
        self.centre_length = len(xs)

        self.wpts = np.vstack((xs, ys)).T
        self.ws = np.vstack((w_rs, w_ls)).T

    def calculate_progress_percent(self, point):
        return self.calculate_s(point)/self.total_s

    def calculate_tracking_accuracy(self, points):
        s_points = np.zeros(len(points))
        for i in range(len(points)):
            s_points[i] = self.calculate_s(points[i])

        closest_pts = np.array(splev(s_points, self.tck, ext=3)).T
        # if len(closest_pts.shape) > 2: closest_pts = closest_pts[:, 0]
        cross_track_errors = np.linalg.norm(points - closest_pts, axis=1)

        return s_points, cross_track_errors
    
    def calculate_cross_track_error(self, point):
        s_point = self.calculate_s(point)

        closest_pt = np.array(splev(s_point, self.tck, ext=3)).T
        if len(closest_pt.shape) > 1: closest_pt = closest_pt[0]
        cross_track_error = np.linalg.norm(point - closest_pt)

        return cross_track_error
        
    def calculate_s(self, point):
        if self.tck == None:
            return point, 0
        dists = np.linalg.norm(point - self.wpts[:, :2], axis=1)
        t_guess = self.s_track[np.argmin(dists)] / self.s_track[-1]

        s_point = fmin(dist_to_p, x0=t_guess, args=(self.tck, point), disp=False)

        return s_point
    


# @jit(cache=True)
def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = splev(t_glob, path, ext=3)
    s = np.concatenate(s)
    return distance.euclidean(p, s)



def load_agent_test_data(file_name):
    data = np.load(file_name)
    
    return data[:, :7], data[:, 7:]


def calculate_tracking_accuracy(vehicle_name):
    agent_path = f"Logs/{vehicle_name}/"
    print(f"Vehicle name: {vehicle_name}")
    
    testing_logs = glob.glob(f"{agent_path}*.npy")
    if len(testing_logs) == 0: raise ValueError("No logs found")
    for test_log in testing_logs:
        test_folder_name = test_log.split("/")[-1]
        test_log_key = "_".join(test_folder_name.split(".")[0].split("_")[1:])
        file_name = f"{agent_path}TrackingAccuracy_{test_log_key}.npy"
        if os.path.exists(file_name): continue

        print(f"Analysing log: {test_folder_name}")

        testing_map = test_folder_name.split("_")[1]
        std_track = TrackingAccuracy(testing_map)
        states, actions = load_agent_test_data(test_log)

        progresses, cross_track = std_track.calculate_tracking_accuracy(states[:, 0:2]) 

        save_data = np.column_stack((progresses, cross_track))
        np.save(file_name, save_data)



if __name__ == "__main__":
    calculate_tracking_accuracy("TunePointsMPCC2")

