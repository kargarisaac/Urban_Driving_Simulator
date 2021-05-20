import gym
import gym_fluids
import matplotlib.pyplot as plt
import numpy as np
import fluids
from gym_fluids.envs.fluids_env import FluidsEnv, FluidsVelEnv
from fluids.assets import * 
import cv2


obs_w = 500
time_limit = 500

fluids_args = {
    "visualization_level": 4,
    "fps": 10,
    "obs_args": {"obs_dim": obs_w},
    "obs_space": fluids.OBS_BIRDSEYE,
    "background_control": fluids.BACKGROUND_CSP,
}

state_args = {
    "layout": fluids.STATE_CITY,
    "background_cars": 10,
    "controlled_cars": 1,
    "background_peds": 0,
    "waypoint_width": 5,
    "use_traffic_lights": False,
    "use_ped_lights": False,
    "vis_level":1
}

# env = gym.make("fluids-v2")
env = FluidsEnv(fluids_args, obs_w, state_args, time_limit)

env.reset()

action = [0, 0]  # [steer, acc]
reward = 0
t = 0
while True:
    obs, rew, done, info = env.step(action)
    obs = np.transpose(obs, (1, 0, 2))
    if t % 10 == 0:
        resized = cv2.resize(obs, (64, 64), interpolation = cv2.INTER_AREA)
        plt.imshow(resized)
        plt.show()

    reward += rew
    env.render()
    # print(rew, action)
    action = gym_fluids.agents.fluids_supervisor(obs, info)
    t += 1
    if done: 
        break

"""
to get dynamic objects' info use:
dynamic objects are CrossWalkLight, CrossWalkLight, Pedestrian.   
check if type(env.fluidsim.state.dynamic_objects[113]) == Car
then get values you want using env.fluidsim.state.dynamic_objects[id].x or .y or .vel 
"""