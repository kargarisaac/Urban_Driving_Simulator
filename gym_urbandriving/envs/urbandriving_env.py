import gym
from copy import deepcopy
from gym_urbandriving.agents import *
import gym_urbandriving as uds
import numpy as np
import IPython
from gym_urbandriving.assets import Car, TrafficLight, CrosswalkLight

class UrbanDrivingEnv(gym.Env):
    """
    This class controls the evolution of a world state.
    While the :class:`PositionState` represents the layout of objects in the scene,
    the :class:`UrbanDrivingEnv` controls the evolution of the scene, and manages background
    actors.

    Note
    -----
        This class is used both to represent the true, global state of the world, and
        as a search agent's internal view of the world.

    Parameters
    ----------
    init_state : PositionState
        The starting state for this environment
    visualizer :
        A visualizer object, if provided, which this env will use to display the scene
    reward_fn :
        A function which takes in one parameter, the :class:`PositionState`,
        and returns the reward for the 0th controllable object in that scene
    max_time : int
        If set > 1, the environment will return "done" after this many ticks
    randomize : bool
        Sets whether :func:`env._reset()` returns the environment to the initial state,
        or if it returns it to a random state generated by the state's :func:`randomize()`
        function
    use_ray : bool
        Sets whether we parallelize agent planning using Ray
    agent_mappings : dict
        Specify the types of agents this environment will use to control background objects
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 config_data=None,
                 init_state = None,
                 visualizer=None,
                 reward_fn=lambda x: 0,
                 max_time=500,
                 randomize=False,
                 agent_mappings={}):
    
        
        self.reward_fn = reward_fn


        if config_data:
            if config_data['enviroment']['state'] == 'four_way_intersection':
                self.init_state = uds.state.SimpleIntersectionState(config_data)
            if config_data['enviroment']['visualize']:
                self.visualizer = uds.PyGameVisualizer((800, 800))
        else:
            self.init_state = init_state
            self.visualizer = None



        self.max_time = max_time
        self.randomize = randomize
        self.statics_rendered = False

        self.dynamic_collisions, self.static_collisions, self.last_col = [], [], -1
        if (self.init_state):
            self._reset()


    def _step(self, action,background_simplified = False):
        """
        The step function accepts a control for the 0th agent in the scene. Then, it queries
        all the background agents to determine their actions. Then, it updates the scene
        and returns.
        
        Parameters
        ----------
        action :
            An action for the agentnum object in the scene.
        agentnum : int
            The index for the object which the action is applied for.

        Returns
        -------
        PositionState
            State of the world after this step;
        float
            Reward calculated by :func:`self.reward_fn`,
        bool
            Whether we have timed out, or there is a collision)
        dict

        """
        assert(self.current_state is not None)
        # Get actions for all objects
        background_car_actions = []
        controlled_car_actions = []
        background_traffic_actions = []
     
        ### GET ALL ACTIONS ####
    
        for agent in self.current_state.bg_agents['background_cars']:
            if background_simplified:
                background_car_actions.append(agent.eval_policy(self.current_state,simplified=True))
            else:
                background_car_actions.append(agent.eval_policy(self.current_state))

        for agent in self.current_state.bg_agents['traffic_lights']:
            background_traffic_actions.append(agent.eval_policy(self.current_state))

        if not background_simplified:
            for i,agent in enumerate(self.current_state.bg_agents['controlled_cars']):
                controlled_car_actions.append(agent.eval_policy(action[i],self.current_state))
        

        ####UPDATE ALL POLICIES#########
        for index, dobj in self.current_state.dynamic_objects['background_cars'].items():
            dobj.step(background_car_actions[int(index)])

        for index, dobj in self.current_state.dynamic_objects['traffic_lights'].items():
            dobj.step(background_traffic_actions[int(index)])

        if not background_simplified:
            for i, dobj in self.current_state.dynamic_objects['controlled_cars'].items():
                dobj.step(controlled_car_actions[int(i)])



        self.current_state.time += 1
        dynamic_coll, static_coll = self.current_state.get_collisions()
        state = self.current_state
        reward = self.reward_fn(self.current_state)

        done = (self.current_state.time == self.max_time) or len(dynamic_coll) or len(static_coll)

        info_dict = {"saved_actions": action}

        return state, reward, done, info_dict


    def _reset(self, new_state=None):
        """
        Resets the environment to its initial state, or a new state
        
        Parameters
        ----------
        new_state : PositionState
            If specified, the environment will reset to this state
        """
        self.last_col = -1
        if new_state:
            self.init_state = new_state
            self.statics_rendered = False
        if self.randomize:
            self.init_state.randomize()
        self.current_state = deepcopy(self.init_state)

        

        return

    def _render(self, mode='human', close=False, waypoints=[],traffic_trajectories = [],transparent_surface = None):
        """
        If the renderer was specifed at construction, renders the current state of the world
        
        Parameters
        ----------
        mode : str
            For OpenAI Gym compatibility
        waypoints :
            Extra points you would like to render over top of the the scene, for debugging
        """
        if not config_data['enviroment']['visualize']:
            return

        if close:
            return
        if self.visualizer:
            
            window = [0, self.current_state.dimensions[0],
                      0, self.current_state.dimensions[1]]
            self.visualizer.render(self.current_state, window,
                                   rerender_statics=not self.statics_rendered,
                                   waypoints=waypoints,
                                   traffic_trajectories = traffic_trajectories,
                                   transparent_surface = transparent_surface)
            self.statics_rendered = True

    def get_state_copy(self):
        return deepcopy(self.current_state)
