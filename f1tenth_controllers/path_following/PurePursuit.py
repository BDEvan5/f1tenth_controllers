
import numpy as np
from f1tenth_controllers.path_following.planner_utils import RaceTrack, get_actuation
from f1tenth_controllers.path_following.NavigationTrack import NavigationTrack



class PurePursuit:
    def __init__(self, run_dict):
        self.racetrack = NavigationTrack(run_dict.map_name)
        # self.racetrack = RaceTrack(run_dict.map_name)
        self.speed = run_dict.vehicle_speed
        self.max_steer = run_dict.delta_max
        self.wheelbase = run_dict.wheelbase
        self.counter = 0

    def plan(self, obs):
        state = obs["vehicle_state"]

        lookahead_distance = 0.3 + state[3] * 0.18
        lookahead_point = self.racetrack.get_lookahead_point(state[:2], lookahead_distance)

        if state[3] < 1:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(state[4], lookahead_point, state[:2], lookahead_distance, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
            
        action = np.array([steering_angle, self.speed])

        return action




