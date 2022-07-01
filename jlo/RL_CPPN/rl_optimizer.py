import math
import pickle
from random import Random
from typing import List, Tuple

import sqlalchemy
import torch
from jlo.RL_CPPN.actor_critic_network import ActorCritic
from jlo.RL_CPPN.rl_brain import RLbrain
from jlo.RL_CPPN.rl_agent import Genotype, develop
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
from jlo.RL_CPPN.rl_runner_train import LocalRunner
from revolve2.core.modular_robot import ModularRobot


class RLOptimizer():
    _runner: Runner

    _controller: ActorController

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float
    _visualize: bool

    def __init__(
            self,
            rng: Random,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            visualize: bool,
    ) -> None:

        self._visualize = visualize
        self._init_runner()
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency

    def _init_runner(self) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=(not self._visualize))

    def _control(self, dt: float, control: ActorControl, observations):  # TODO what is td?
        num_agents = observations[0].shape[0]
        actions = []
        values = []
        logps = []

        # for each agent in the simulation make a step
        for control_i in range(num_agents):
            agent_obs = [[] for _ in range(NUM_OBSERVATIONS)]
            for i, obs in enumerate(observations):
                agent_obs[i] = obs[control_i]
            action, value, logp = self._controller.get_dof_targets(agent_obs)
            control.set_dof_targets(control_i, 0, torch.clip(action, -0.7, 0.7))
            # control.set_dof_targets(control_i, 0, (action*2 - 1))
            actions.append(action.tolist())
            values.append(value.tolist())
            logps.append(logp.tolist())
        return actions, values, logps

    async def train(self, agents: List[Genotype], from_checkpoint: bool = False):
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
                    [0.0 for _ in range(8)]
                )
            )
            batch.environments.append(env)

        # run the simulation
        await self._runner.run_batch(batch, self._controller, len(agents))

        return