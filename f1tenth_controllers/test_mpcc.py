from f1tenth_controllers.f1tenth_sim.f1tenth_sim import F1TenthSim 
from f1tenth_controllers.mpcc.ConstantMPCC2 import ConstantMPCC2
from f1tenth_controllers.mpcc.ConstantMPCC import ConstantMPCC
from f1tenth_controllers.path_following.PurePursuit import PurePursuit

import numpy as np
import yaml 
import time, os
from argparse import Namespace
from f1tenth_controllers.analysis.plot_trajectory import plot_analysis
from f1tenth_controllers.analysis.BuildPlannerDfs import build_planner_df
from f1tenth_controllers.analysis.TrackingAccuracy import calculate_tracking_accuracy




def load_configuration(config_name):
    with open(f"configurations/{config_name}.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run_dict = Namespace(**config)
    return run_dict 


def run_simulation_loop_laps(env, planner):
    for lap in range(env.run_dict.number_test_laps):
        observation, done = env.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    

def setup_run_list(experiment_file):
    full_path =  "configurations/" + experiment_file + '.yaml'
    with open(full_path) as file:
        experiment_dict = yaml.load(file, Loader=yaml.FullLoader)
        
    run_list = []
    for run in experiment_dict['runs']:
        for key in experiment_dict.keys():
            if key not in run.keys() and key != "runs":
                run[key] = experiment_dict[key]

        run['run_name'] = f"{run['map_name']}_{run['test_id']}"

        run_list.append(Namespace(**run))

    return run_list


def run_profiling(function, name):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    with open(f"Logs/profile_{name}.txt", "w") as f:
        ps.print_stats()
        f.write(s.getvalue())


def run_mpcc_experiment2():
    run_list = setup_run_list("test_maps_config")
    vehicle_name = "MapsMPCC2"

    for run_dict in run_list:

        simulator = F1TenthSim(run_dict, True, vehicle_name)
        planner = ConstantMPCC2(run_dict)

        run_simulation_loop_laps(simulator, planner)

    calculate_tracking_accuracy(vehicle_name)
    build_planner_df(vehicle_name)
    plot_analysis(vehicle_name)


def run_mpcc_experiment():
    # run_list = setup_run_list("tune_points_config")
    # vehicle_name = "TunePointsMPCC3"

    # run_list = setup_run_list("tune_frequency_config")
    # vehicle_name = "TuneFrequencyMPCC3"

    # run_list = setup_run_list("tuning_config")
    # vehicle_name = "TuneMPCC"

    run_list = setup_run_list("test_maps_config")
    vehicle_name = "MapsMPCC"

    for run_dict in run_list:

        simulator = F1TenthSim(run_dict, True, vehicle_name)
        planner = ConstantMPCC(run_dict)

        run_simulation_loop_laps(simulator, planner)

    calculate_tracking_accuracy(vehicle_name)
    build_planner_df(vehicle_name)
    plot_analysis(vehicle_name)


def run_pp_experiment():
    run_list = setup_run_list("test_maps_config")
    vehicle_name = "MapsPP"

    for run_dict in run_list:

        simulator = F1TenthSim(run_dict, True, vehicle_name)
        planner = PurePursuit(run_dict)

        run_simulation_loop_laps(simulator, planner)

    calculate_tracking_accuracy(vehicle_name)
    build_planner_df(vehicle_name)
    # plot_analysis(vehicle_name)


if __name__ == "__main__":
    # run_test()
    # run_profiling(run_test, "mpcc")
    # run_test()
    # test_all_maps()
    run_mpcc_experiment()
    # run_mpcc_experiment2()
    # run_pp_experiment()



