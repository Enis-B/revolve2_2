"""
Visualize and run a modular robot using Isaac Gym.
"""
import math
from random import Random
from typing import List, Optional, Tuple
from pyrr import Quaternion, Vector3
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pickle

from dataclasses import dataclass
from isaacgym import gymapi
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import *
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.isaacgym import LocalRunner
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf

from jlo.RL.rl_brain import RLbrain

from tensorforce.environments import Environment as tfenv
from tensorforce.agents import Agent as ag
from tensorforce.execution import Runner as rn

class CustomEnvironment(tfenv):
    _controller: ActorController
    async def simulate(self, robot: ModularRobot, control_frequency: float, headless = True):
        batch = Batch(
            simulation_time=30,
            sampling_frequency=10,
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
        runner = LocalRunner(LocalRunner.SimParams(),headless= headless)
        #print("okay 1")
        states = await runner.run_batch(batch)
        #print("okay 2")
        return states

    def _control(self, dt: float, control: ActorControl) -> None:
        self._controller.step(dt)
        control.set_dof_targets(0, 0, self._controller.get_dof_targets()) # ACTION
        #control.set_dof_targets(0, 0, action)
        #print("DOF STATES: ",self._controller.get_dof_targets())

    def init_pos(self):
        action = np.random.uniform(low=-1, high=1, size=8).astype(np.float32)
        return action

    @staticmethod
    def _calculate_velocity(begin_state, end_state) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        )

    def __init__(self):
        super().__init__()
    def states(self):
        #return sim._controller.get_dof_targets()
        return dict(type='float', shape=(8,))

    def actions(self):
        #return sim._controller.set_dof_targets()
        return dict(type='float', shape=(8,))

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
    def make_robot(self):
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
        rng = Random()
        #rng.seed(3)
        brain = BrainCpgNetworkNeighbourRandom(rng)
        #brain = RLbrain(from_checkpoint=False)
        robot = ModularRobot(body, brain)
        return robot
    async def train(self,robot):
        print("Training...")
        states = await self.simulate(robot, 10)
        print("ALL STATES: ","\n", states)
        reward = self._calculate_velocity(states[0].envs[0].actor_states[0],
                                          states[-1].envs[0].actor_states[0])
        print("REWARD: ",reward)
        next_state = states[-1].envs[0].actor_states[0]
        print("NEXT STATE: ","\n",next_state)
        return next_state,reward
    async def execute(self, actions):
        next_state,reward = await self.train(self.make_robot())
        terminal = False  # Always False if no "natural" terminal state
        return next_state, terminal, reward

async def main() -> None:
    environment = CustomEnvironment()
    #state = await environment.simulate(environment.make_robot(),10)
    #print(state)
    agent = ag.create(
        agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3, max_episode_timesteps=100
    )
    async def run(environment, agent, n_episodes, max_episode_timesteps):
        """
        Train agent for n_episodes
        """
        # Loop over episodes
        rewards = []
        controllers = []
        cnt = []
        for i in range(n_episodes):
            print("EPISODE ",i)
            states = environment.reset()
            # Run episode
            actions = agent.act(states=states)
            print("ACTIONS :","\n",actions)
            states, terminal, reward = await environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            rewards.append(reward)
            cnt.append(i)
            controllers.append(states)
        plt.plot(cnt,rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.show()
        sum_rewards = sum(rewards)
        print("Sum rewards: ", sum_rewards)

        best_controller = []
        best_reward = rewards.index(max(rewards))
        for i in range(len(controllers)):
            if i == best_reward:
                best_controller = controllers[i]

        with open('best_controller.pickle', 'wb') as f:
            pickle.dump(best_controller, f)


    async def runner(
            environment,
            agent,
            max_episode_timesteps):
        # Train agent
        await run(environment, agent, n_episodes=100, max_episode_timesteps=100)
        # Terminate the agent and the environment
        agent.close()
        environment.close()

    #await runner(
    #    environment,
    #    agent,
    #    max_episode_timesteps=100)

    with open('best_controller.pickle', 'rb') as f:
        best_controller = pickle.load(f)
    state = await environment.simulate(environment.make_robot(),10, headless=False)
    print(state)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())