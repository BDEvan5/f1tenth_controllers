from f1tenth_controllers.f1tenth_sim.f1tenth_sim import F1TenthSim 
from f1tenth_controllers.mpcc.ConstantMPCC import ConstantMPCC
import numpy as np
import yaml 
import time
from argparse import Namespace
from f1tenth_controllers.analysis.plot_trajectory import plot_analysis

map_list = ["Austin",
                 "Catalunya",
                 "Monza",
                 "Sakhir",
                 "SaoPaulo",
                 "Sepang",
                 "Silverstone",
                 "Spielberg",
                 "Zandvoort",
                 "Oschersleben"]

mini_map_list = ["Sepang",
                 "Zandvoort",
                 "Oschersleben"]


def load_configuration(config_name):
    with open(f"configurations/{config_name}.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run_dict = Namespace(**config)
    return run_dict 


def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done = env.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    

def run_test():
    # map_name = "aut"
    map_name = "Monza"
    map_name = "Spielberg"
    # map_name = "Catalunya"

    print(f"Testing....")
    std_config = load_configuration("std_config")

    vehicle_name = "TestMPCC1"
    simulator = F1TenthSim(map_name, std_config, True, vehicle_name)
    planner = ConstantMPCC(simulator.map_name)

    run_simulation_loop_laps(simulator, planner, 1)
    plot_analysis(vehicle_name)

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

def test_all_maps():
    std_config = load_configuration("std_config")
    vehicle_name = "TestMPCC4"

    start_time = time.time()
    for map_name in map_list:
    # for map_name in mini_map_list:
        map_start_time = time.time()
        print(f"Testing on map {map_name}")

        simulator = F1TenthSim(map_name, std_config, True, vehicle_name)
        planner = ConstantMPCC(simulator.map_name)

        run_simulation_loop_laps(simulator, planner, 1)

        print(f"Time taken for map {map_name}: {(time.time() - map_start_time):.4f}")
        print(f"")

    print(f"Total time taken: {(time.time() - start_time):.4f}")
    plot_analysis(vehicle_name)


if __name__ == "__main__":
    # run_test()
    # run_profiling(run_test, "mpcc")
    # run_test()
    test_all_maps()




