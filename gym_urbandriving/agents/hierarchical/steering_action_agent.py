import numpy as np
from gym_urbandriving.utils.PID import PIDController
from gym_urbandriving.agents import PursuitAgent
from gym_urbandriving.planning import VelocityMPCPlanner,GeometricPlanner
from copy import deepcopy
import gym_urbandriving as uds

class SteeringActionAgent(PursuitAgent):
    """
    Hierarichal agent which does not include any planning stack and only requires 
    specifiying the steering agent.  

    Attributes
    ----------
    agent_num : int
        Index of this agent in the world.
        Used to access its object in state.dynamic_objects

    """

    def __init__(self, agent_num=0):
        """
        Initializes the PlanningPursuitAgent Class

        Parameters
        ----------
        agent_num: int
            The number which specifies the agent in the dictionary state.dynamic_objects['controlled_cars']

        """
        self.agent_num = agent_num
        self.PID_acc = PIDController(1.0, 0, 0)
        self.PID_steer = PIDController(2.0, 0, 0)
        self.not_initiliazed = True
        

        
    def eval_policy(self, action,state):
        """
        Returns action based next state in trajectory. 

        Parameters
        ----------
        state : PositionState
            State of the world, unused
        action : float
            Target velocity for car to travel at 

        Returns
        -------
        tuple with floats (steering,acceleration)
        """
        
        if (not type(action) == np.array):
            if not action.shape[0] == 2:
                raise Exception('Steering Action is not a numpy array of shape (2,)')

        else:
            raise Exception('Steering Action is not a numpy array of shape (2,)')

        return action

