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


def create_main_agent_df(agent_path, test_laps=20):
    agent_data = []
    vehicle_name = agent_path.split("/")[-2]
    print(f"Vehicle name: {vehicle_name}")
    
    testing_logs = glob.glob(f"{agent_path}*.npy")
    for test_log in testing_logs:
        test_folder_name = test_log.split("/")[-1]
        if test_folder_name[:6] != "SimLog": continue
        print(f"Analysing log: {test_folder_name}")
        test_name_list = test_folder_name.split("_")

        testing_map = test_name_list[1]
        test_id = test_name_list[2]
        lap_number = test_name_list[3].split(".")[0]
    
        states, actions = load_agent_test_data(test_log)
        if states is None: break

        time = len(states) /20 # problem here due to frequency
        ss = np.linalg.norm(np.diff(states[:, 0:2], axis=0), axis=1)
        total_distance = np.sum(ss)

        accuracy_data = np.load(f"{agent_path}TrackingAccuracy_{testing_map}_{test_id}_{lap_number}.npy")
        progress = np.max(accuracy_data[:, 0])
        racing_cross_track = accuracy_data[:, 1] * 100

        agent_data.append({"Lap": lap_number, "TestMap": testing_map, "TestID": test_id, "Distance": total_distance, "Progress": progress, "Time": time, "TA_mean:": np.mean(racing_cross_track), "TA_q1": np.percentile(racing_cross_track, 25), "TA_q3": np.percentile(racing_cross_track, 75), "TA_std": np.std(racing_cross_track), "TA_max": np.max(racing_cross_track)})

    agent_df = pd.DataFrame(agent_data)
    agent_df = agent_df.sort_values(by=["TestMap", "TestID", "Lap"])
    agent_df.to_csv(agent_path + "PlannerResults.csv", index=False)





def main():
    p = "Logs/"
    
    path = p + f"TunePointsMPCC/"

    vehicle_folders = glob.glob(f"{path}")

    for j, path in enumerate(vehicle_folders):
        if path.split("/")[-2] == "Imgs": continue

        create_main_agent_df(path)


if __name__ == '__main__':
    main()



