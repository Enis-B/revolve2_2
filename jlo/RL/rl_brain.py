"""
Neural Network brain for Reinforcement Learning.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from revolve2.actor_controllers.cpg import Cpg as ControllerCpg

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
from RL.rl_controller import RLcontroller
from .actor_critic_network import Actor, ActorCritic


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
        actor_critic = ActorCritic([num_hinges, num_hinges], num_hinges)

        return RLcontroller(
            actor_critic,
            np.array([active_hinge.RANGE for active_hinge in active_hinges]),
            from_checkpoint=self._from_checkpoint,
        )