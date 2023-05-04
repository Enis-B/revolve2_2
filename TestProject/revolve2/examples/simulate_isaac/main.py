"""
Visualize and run a modular robot using Isaac Gym.
"""

import math
from random import Random
from typing import List
from pyrr import Quaternion, Vector3

import pprint

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbourRandom
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.isaacgym import LocalRunner

import numpy as np

from tensorforce.environments import Environment as tfenv
from tensorforce.agents import Agent as ag
from tensorforce.execution import Runner as rn

class Simulator:
    _controller: ActorController

    async def simulate(self, robot: ModularRobot, control_frequency: float) -> List:
        batch = Batch(
            simulation_time=1,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
            control=self._control,
        )

        actor, self._controller = robot.make_actor_and_controller()

        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, 0.1]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(LocalRunner.SimParams())
        return await runner.run_batch(batch)
        #print(batch)

    def _control(self, dt: float, control: ActorControl) -> None:
        self._controller.step(dt)
        control.set_dof_targets(0, 0, self._controller.get_dof_targets())
        #pprint.pprint(self._controller.get_dof_targets())

class CustomEnvironment(tfenv):

    def __init__(self):
        super().__init__()

    def states(self):
        #return sim._controller.get_dof_targets()
        return dict(type='float', shape=(8,))

    def actions(self):
        #return sim._controller.set_dof_targets()
        return dict(type='float', num_values=8)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = False  # Always False if no "natural" terminal state
        reward = np.random.random()
        return next_state, terminal, reward

async def main() -> None:
    rng = Random()
    rng.seed(5)
    body = Body()
    ## Default
    '''
    body.core.left = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.front = Brick(0.0)
    body.core.front.front = Brick(0.0)
    body.core.front.front = ActiveHinge(math.pi / 2.0)
    body.core.front.left = Brick(0.0)
    body.core.front.right = Brick(0.0)
    body.core.right = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment.attachment = Brick(0.0)
    '''
    ## Spider
    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)

    body.core.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment = Brick(-np.pi / 2.0)
    body.core.front.attachment.front = ActiveHinge(0.0)
    body.core.front.attachment.front.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)

    body.finalize()
    brain = BrainCpgNetworkNeighbourRandom(rng)
    robot = ModularRobot(body, brain)

    sim = Simulator()
    states = await sim.simulate(robot, 10)
    pprint.pprint(states)
    ### TF

    environment = tfenv.create(environment=CustomEnvironment, max_episode_timesteps=100)

    agent = ag.create(
        agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
    )

    runner = rn(
        agent=agent,
        environment=environment
    )

    #runner.run(num_episodes=200)

    #runner.run(num_episodes=100, evaluation=True)

    runner.close()

    ###


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
