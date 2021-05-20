import fluids
import pygame
import numpy as np
import gym
from gym import spaces


# OBS_W = 500
# TIME_LIMIT = 100

# FLUIDS_ARGS = {
#     "visualization_level": 4,
#     "fps": 30,
#     "obs_args": {"obs_dim": OBS_W},
#     "obs_space": fluids.OBS_BIRDSEYE,
#     "background_control": fluids.BACKGROUND_CSP,
# }

# STATE_ARGS = {
#     "layout": fluids.STATE_CITY,
#     "background_cars": 10,
#     "controlled_cars": 1,
#     "background_peds": 0,
#     "waypoint_width": 5,
#     "use_traffic_lights": False,
#     "use_ped_lights": False,
#     "vis_level":1
# }


class FluidsEnv(gym.Env):
    def __init__(self, fluids_args, obs_w, state_args, time_limit):
        self.fluidsim = fluids.FluidSim(**fluids_args)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_w, obs_w, 3), dtype=np.uint8
        )
        self.fluids_action_type = fluids.SteeringAccAction
        self.state_args = state_args
        self.time_limit = time_limit

    def reset(self):
        self.fluidsim.set_state(fluids.State(**self.state_args))
        car_keys = list(self.fluidsim.get_control_keys())
        assert len(car_keys) == 1
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()

    def fluids_action_maker(self, action):
        return fluids.SteeringAccAction(action[0], action[1])

    def step(self, action):
        car_keys = list(self.fluidsim.get_control_keys())
        assert len(car_keys) == 1
        actions = {car_keys[0]: self.fluids_action_maker(action)}
        reward_step = self.fluidsim.step(actions)
        obs = self.fluidsim.get_observations(car_keys)
        done = self.fluidsim.run_time() > self.time_limit
        obs = obs[car_keys[0]].get_array()
        info_dict = {
            "supervisor_action": self.fluidsim.get_supervisor_actions(
                keys=car_keys, action_type=self.fluids_action_type
            )[car_keys[0]].get_array()
        }
        return obs, reward_step, done, info_dict

    def render(self, mode="human"):
        self.fluidsim.render()


class FluidsVelEnv(FluidsEnv):
    def __init__(self):
        super(FluidsVelEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32)
        self.fluids_action_type = fluids.VelocityAction

    def fluids_action_maker(self, action):
        return fluids.VelocityAction(action[0])
