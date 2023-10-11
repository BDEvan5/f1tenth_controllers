import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_table(vehicle_name="MapsMPCC"):
    pp_vehicle = "MapsPP"
    map_info = pd.read_csv(f"maps/MapInfo.csv")
    mpcc_results = pd.read_csv(f"Logs/{vehicle_name}/PlannerResults_{vehicle_name}.csv")
    pp_results = pd.read_csv(f"Logs/{pp_vehicle}/PlannerResults_{pp_vehicle}.csv")

    data = []

    for map_name in mpcc_results["TestMap"]:
        map_data = map_info.loc[map_info["MapName"] == map_name]
        mpcc_data = mpcc_results.loc[mpcc_results["TestMap"] == map_name]
        pp_data = pp_results.loc[pp_results["TestMap"] == map_name]
        data.append({"MapName": map_name, "TrackLength": map_data["TrackLength"].values[0], "TotalCurvature": map_data["TotalCurvature"].values[0], 
                     "MPCC_TrackingAccuracy": mpcc_data["TA_mean"].values[0]*10, "MPCC_MaxError":  mpcc_data["TA_max"].values[0]*10, 
                     "PP_TrackingAccuracy": pp_data["TA_mean"].values[0]*10, "PP_MaxError":  pp_data["TA_max"].values[0]*10})

    data = pd.DataFrame(data)
    print(data)

    data.to_latex(f"Logs/ResultsTable_{vehicle_name}.tex", index=False, float_format='%.2f')

make_table()

