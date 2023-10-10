import numpy as np
import glob
import os
import pandas as pd


from f1tenth_controllers.map_utils.Track import Track 
from f1tenth_controllers.map_utils.TrackingAccuracy import TrackingAccuracy


SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_agent_test_data(file_name):
    try:
        data = np.load(file_name)
    except Exception as e:
        print(f"No data for: " + file_name)
        return None, None
    
    states = data[:, :7]
    actions = data[:, 7:]

    return states, actions


def calculate_tracking_accuracy(agent_path):
    vehicle_name = agent_path.split("/")[-2]
    print(f"Vehicle name: {vehicle_name}")
    
    testing_logs = glob.glob(f"{agent_path}*.npy")
    for test_log in testing_logs:
        test_folder_name = test_log.split("/")[-1]
        test_log_key = "_".join(test_folder_name.split(".")[0].split("_")[1:])
        file_name = f"{agent_path}TrackingAccuracy_{test_log_key}.npy"
        if os.path.exists(file_name): continue

        print(f"Analysing log: {test_folder_name}")

        testing_map = test_folder_name.split("_")[1]
        std_track = TrackingAccuracy(testing_map)

        states, actions = load_agent_test_data(test_log)
        if states is None: raise IOError

        progresses, cross_track = std_track.calculate_tracking_accuracy(states[:, 0:2]) 

        save_data = np.column_stack((progresses, cross_track))
        np.save(file_name, save_data)


def calculate_and_save_accuracies(test_name):
    path = f'Logs/{test_name}/'
    calculate_tracking_accuracy(path)


if __name__ == '__main__':
    calculate_and_save_accuracies("TunePointsMPCC")



