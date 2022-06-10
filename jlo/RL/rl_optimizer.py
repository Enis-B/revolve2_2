import math
import pickle
from random import Random
from typing import List, Tuple

import sqlalchemy
import torch
from RL.actor_critic_network import ActorCritic
from RL.rl_brain import RLbrain
from RL.rl_agent import Agent, develop
from pyrr import Quaternion, Vector3
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
from revolve2.actor_controller import ActorController
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from RL.rl_runner_train import LocalRunner


class RLOptimizer():

    _runner: Runner

    _controller: ActorController

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    def __init__(
        self,
        rng: Random,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
    ) -> None:
        
        self._init_runner()
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

    def _init_runner(self) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=False)

    def _control(self, dt: float, control: ActorControl, observations): # TODO what is td?
        num_agents = observations.shape[1]
        num_obs = observations.shape[0]
        num_act = observations.shape[2]
        actions = torch.zeros(num_agents,num_act)
        values = torch.zeros(num_agents)
        logps = torch.zeros(num_agents)

        # for each agent in the simulation make a step
        for control_i in range(num_agents):
            action, value, logp = self._controller.get_dof_targets(observations[:,control_i,:])
            control.set_dof_targets(control_i, 0, action)
            actions[control_i] = action
            values[control_i] = value
            logps[control_i] = logp
        return actions, values, logps

    async def train(self, agents: List[Agent], from_checkpoint: bool = False):
        """
        Create the agents, insert them in the simulation and run it
        args:
            agents: list of agents to simulate
            from_checkpoint: if True resumes training from the last checkpoint
        """
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        # we create only one brain as it is shared across all the agents
        brain = RLbrain(from_checkpoint=from_checkpoint)

        # insert agents in the simulation environment
        for agent_idx, agent in enumerate(agents):
            agent.brain = brain
            actor, controller = develop(agent).make_actor_and_controller()
            if agent_idx == 0:
                self._controller = controller
            bounding_box = actor.calc_aabb()
            env = Environment()
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                )
            )
            batch.environments.append(env)
        
        # run the simulation
        await self._runner.run_batch(batch, self._controller, len(agents))

        
        return 