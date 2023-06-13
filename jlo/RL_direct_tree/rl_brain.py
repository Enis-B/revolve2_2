"""
Neural Network brain for Reinforcement Learning.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
from jlo.RL_direct_tree.rl_controller import RLcontroller
from jlo.RL_direct_tree.actor_critic_network import Actor, ActorCritic
from jlo.RL_direct_tree.config import NUM_OBS_TIMES, OBS_DIM


class RLbrain(Brain, ABC):
    """
    Brain of an agent
    """
    def __init__(self, from_checkpoint: bool = False) -> None:
        super().__init__()
        self._from_checkpoint = from_checkpoint

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Initialize neural network used as controller and return the brain
        """
        active_hinges = body.find_active_hinges()
        num_hinges = len(active_hinges)
        actor_critic = ActorCritic(OBS_DIM, num_hinges)

        return RLcontroller(
            actor_critic,
            np.array([active_hinge.RANGE for active_hinge in active_hinges]),
            from_checkpoint=self._from_checkpoint,
        )