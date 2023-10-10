from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os
import math, cmath

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from f1tenth_controllers.map_utils.MapData import MapData
from f1tenth_controllers.map_utils.Track import Track 
from f1tenth_controllers.analysis.plotting_utils import *
from matplotlib.ticker import MultipleLocator
import trajectory_planning_helpers as tph

SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class TrajectoryPlotter:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.lap_n = 0
        
        self.track_progresses = None

    def explore_folder(self, path):
        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        set = 1
        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
                
            self.process_folder(folder)

    def process_folder(self, folder):
        self.path = folder
        self.test_folder = folder

        self.vehicle_name = self.path.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        
        testing_logs = glob.glob(f"{folder}*.npy")
        for test_log in testing_logs:
            self.test_log_key = test_log.split("/")[-1].split(".")[0].split("_")[1:]
            self.test_log_key = "_".join(self.test_log_key)
            test_folder_name = test_log.split("/")[-1]
            self.map_name = test_folder_name.split("_")[1]
        
            self.map_data = MapData(self.map_name)
            self.std_track = Track(self.map_name)

            if not self.load_lap_data(test_log): break # no more laps
            self.calculate_state_progress()
            
            self.plot_analysis()
            self.plot_tracking_accuracy()
            # self.plot_steering_profile()    
            self.plot_trajectory()

    def load_lap_data(self, test_log):
        try:
            data = np.load(test_log)
        except Exception as e:
            print(f"No data for: " + test_log)
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]
        
        return 1 # to say success
    
    def calculate_state_progress(self):
        progresses = []
        n = len(self.states)
        for i in range(len(self.states)):
            p = self.std_track.calculate_progress_percent(self.states[i, 0:2])
            if i < 20 and p > 0.9:
                p = 0.0
            if i > n * 0.3 and p < 0.1:
                p = 1.0
            progresses.append(p)
            
        self.track_progresses = np.array(progresses) * 100

    def plot_analysis(self):
        fig = plt.figure(figsize=(8, 6))
        a1 = plt.subplot(311)
        a2 = plt.subplot(312, sharex=a1)
        a3 = plt.subplot(313, sharex=a1)
        plt.rcParams['lines.linewidth'] = 2

        a1.plot(self.track_progresses[:-1], self.actions[:-1, 1], label="Actions", alpha=0.6, color=sunset_orange)
        a1.plot(self.track_progresses[:-1], self.states[:-1, 3], label="State", color=periwinkle)
        a1.set_ylabel("Speed (m/s)")
        a1.grid(True)

        a2.plot(self.track_progresses[:-1], self.states[:-1, 2], label="State", color=periwinkle)
        a2.plot(self.track_progresses[:-1], self.actions[:-1, 0], label="Actions", color=sunset_orange)
        max_value = np.max(np.abs(self.actions[:-1, 0])) * 1.1
        a2.set_ylim(-max_value, max_value)
        a2.grid(True)
        a2.legend()
        a2.set_ylabel("Steering (rad)")
    
        a3.plot(self.track_progresses[:-1], self.states[:-1, 6], label="State", color=periwinkle)
        max_value = np.max(np.abs(self.states[:-1, 6])) * 1.1
        a3.set_ylim(-max_value, max_value)
        a3.set_ylabel("Slip (rad)")
        a3.grid(True)
        a3.set_xlabel("Track progress (m)")

        plt.tight_layout()
        plt.savefig(f"{self.test_folder}Analysis_{self.test_log_key}.svg", bbox_inches='tight', pad_inches=0)  
    
    def plot_trajectory(self): 
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.tracking_accuracy)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        cbar.ax.tick_params(labelsize=25)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.axis('off')
        
        name = self.test_folder + f"Trajectory_{self.test_log_key}"
        # std_img_saving(name)
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)
        plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)

    def plot_steering_profile(self):
        plt.figure(figsize=(7, 2))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.states[:-1, 2], label="State", color=periwinkle)
        plt.plot(self.track_progresses[:-1], self.actions[:-1, 0], label="Actions", color=sunset_orange)
        max_value = np.max(np.abs(self.actions[:-1, 0])) * 1.1
        plt.ylim(-max_value, max_value)
        plt.grid(True)
        plt.legend()
        plt.ylabel("Steering (rad)")

        plt.savefig(f"{self.test_folder}Steering_{self.test_log_key}.png", bbox_inches='tight', pad_inches=0)

    def plot_tracking_accuracy(self):
        pts = self.states[:, 0:2]
        thetas = self.states[:, 4]
        racing_cross_track = []
        racing_heading_error = []
        for i in range(len(pts)):
            track_heading, deviation = self.std_track.get_cross_track_heading(pts[i])
            racing_cross_track.append(deviation)
        racing_cross_track = np.array(racing_cross_track) * 100

        self.tracking_accuracy = racing_cross_track
            
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, racing_cross_track)
        
        plt.title("Tracking Accuracy (cm)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}Tracking_{self.test_log_key}.svg", bbox_inches='tight', pad_inches=0)
        # plt.savefig(f"{self.test_folder}Tracking_{self.test_log_key}.pdf", bbox_inches='tight', pad_inches=0)

            
        plt.figure(1, figsize=(5, 4))
        plt.clf()
        plt.hist(racing_cross_track, bins=20)
        
        plt.xlabel("Tracking Accuracy (cm)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}TrackingHist_{self.test_log_key}.svg", bbox_inches='tight', pad_inches=0)
        # plt.savefig(f"{self.test_folder}TrackingHist_{self.map_name}_{self.lap_n}.png", bbox_inches='tight', pad_inches=0)

        

def plot_analysis(vehicle_name):
    TestData = TrajectoryPlotter()

    TestData.process_folder(f"Logs/{vehicle_name}/")



def analyse_folder():
    
    TestData = TrajectoryPlotter()
    # TestData.explore_folder("Data/")

    # TestData.process_folder("Logs/TestMPCC/")
    TestData.process_folder("Logs/TunePointsMPCC/")


if __name__ == '__main__':
    analyse_folder()
