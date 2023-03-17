import math
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional
from sklearn import preprocessing
import numpy as np
from isaacgym import gymapi
from pyrr import Quaternion, Vector3

from revolve2.core.physics.actor import Actor
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    EnvironmentState,
    Runner,
    RunnerState,
)
import math
from random import Random
from typing import List, Optional, Tuple
from pyrr import Quaternion, Vector3
import os
import tempfile
import matplotlib.pyplot as plt
import pickle

from dataclasses import dataclass
from isaacgym import gymapi
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import *
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
#from revolve2.runners.isaacgym import LocalRunner
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf

from jlo.RL.rl_brain import RLbrain

from tensorforce.environments import Environment as tfenv
from tensorforce.agents import Agent as ag
from tensorforce.execution import Runner as rn

class LocalRunner(Runner):
    class _Simulator(tfenv): ##tfenv
        ENV_SIZE = 0.5

        @dataclass
        class GymEnv:
            env: gymapi.Env  # environment handle
            actors: List[
                int
            ]  # actor handles, in same order as provided by environment description

        _gym: gymapi.Gym
        _batch: Batch

        _sim: gymapi.Sim
        _viewer: Optional[gymapi.Viewer]
        _simulation_time: int
        _gymenvs: List[
            GymEnv
        ]  # environments, in same order as provided by batch description

        def __init__(
            self,
            batch: Batch,
            sim_params: gymapi.SimParams,
            headless: bool,
        ):
            super().__init__()
            self._gym = gymapi.acquire_gym()
            self._batch = batch

            self._sim = self._create_sim(sim_params)
            self._gymenvs = self._create_envs()

            if headless:
                self._viewer = None
            else:
                self._viewer = self._create_viewer()

            self._gym.prepare_sim(self._sim)
        
        def _create_sim(self, sim_params: gymapi.SimParams) -> gymapi.Sim:
            sim = self._gym.create_sim(type=gymapi.SIM_PHYSX, params=sim_params)

            if sim is None:
                raise RuntimeError()

            return sim

        def _create_envs(self) -> List[GymEnv]:
            gymenvs: List[LocalRunner._Simulator.GymEnv] = []

            # TODO this is only temporary. When we switch to the new isaac sim it should be easily possible to
            # let the user create static object, rendering the group plane redundant.
            # But for now we keep it because it's easy for our first test release.
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            plane_params.distance = 0
            plane_params.static_friction = 1.0
            plane_params.dynamic_friction = 1.0
            plane_params.restitution = 0
            self._gym.add_ground(self._sim, plane_params)

            num_per_row = int(math.sqrt(len(self._batch.environments)))

            for env_index, env_descr in enumerate(self._batch.environments):
                env = self._gym.create_env(
                    self._sim,
                    gymapi.Vec3(-self.ENV_SIZE, -self.ENV_SIZE, 0.0),
                    gymapi.Vec3(self.ENV_SIZE, self.ENV_SIZE, self.ENV_SIZE),
                    num_per_row,
                )

                gymenv = self.GymEnv(env, [])
                gymenvs.append(gymenv)

                for actor_index, posed_actor in enumerate(env_descr.actors):
                    # sadly isaac gym can only read robot descriptions from a file,
                    # so we create a temporary file.
                    botfile = tempfile.NamedTemporaryFile(
                        mode="r+", delete=False, suffix=".urdf"
                    )
                    botfile.writelines(
                        physbot_to_urdf(
                            posed_actor.actor,
                            f"robot_{actor_index}",
                            Vector3(),
                            Quaternion(),
                        )
                    )
                    botfile.close()
                    asset_root = os.path.dirname(botfile.name)
                    urdf_file = os.path.basename(botfile.name)
                    asset_options = gymapi.AssetOptions()
                    asset_options.angular_damping = 0.0
                    actor_asset = self._gym.load_urdf(
                        self._sim, asset_root, urdf_file, asset_options
                    )
                    os.remove(botfile.name)

                    if actor_asset is None:
                        raise RuntimeError()

                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(
                        posed_actor.position.x,
                        posed_actor.position.y,
                        posed_actor.position.z,
                    )
                    pose.r = gymapi.Quat(
                        posed_actor.orientation.x,
                        posed_actor.orientation.y,
                        posed_actor.orientation.z,
                        posed_actor.orientation.w,
                    )

                    # create an aggregate for this robot
                    # disabling self collision to both improve performance and improve stability
                    num_bodies = self._gym.get_asset_rigid_body_count(actor_asset)
                    num_shapes = self._gym.get_asset_rigid_shape_count(actor_asset)
                    enable_self_collision = False
                    self._gym.begin_aggregate(
                        env, num_bodies, num_shapes, enable_self_collision
                    )

                    actor_handle: int = self._gym.create_actor(
                        env,
                        actor_asset,
                        pose,
                        f"robot_{actor_index}",
                        env_index,
                        0,
                    )
                    gymenv.actors.append(actor_handle)

                    self._gym.end_aggregate(env)

                    # TODO make all this configurable.
                    props = self._gym.get_actor_dof_properties(env, actor_handle)
                    props["driveMode"].fill(gymapi.DOF_MODE_POS)
                    props["stiffness"].fill(1.0)
                    props["damping"].fill(0.05)
                    self._gym.set_actor_dof_properties(env, actor_handle, props)

                    all_rigid_props = self._gym.get_actor_rigid_shape_properties(
                        env, actor_handle
                    )

                    for body, rigid_props in zip(
                        posed_actor.actor.bodies,
                        all_rigid_props,
                    ):
                        rigid_props.friction = body.static_friction
                        rigid_props.rolling_friction = body.dynamic_friction

                    self._gym.set_actor_rigid_shape_properties(
                        env, actor_handle, all_rigid_props
                    )

                    self.set_actor_dof_position_targets(
                        env, actor_handle, posed_actor.actor, posed_actor.dof_states
                    )
                    self.set_actor_dof_positions(
                        env, actor_handle, posed_actor.actor, posed_actor.dof_states
                    )

            return gymenvs

        def _create_viewer(self) -> gymapi.Viewer:
            # TODO provide some sensible default and make configurable
            viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError()
            num_per_row = math.sqrt(len(self._batch.environments))
            cam_pos = gymapi.Vec3(
                num_per_row / 2.0 - 0.5, num_per_row / 2.0 + 0.5, num_per_row + 2
            )
            cam_target = gymapi.Vec3(
                num_per_row / 2.0 - 0.5, num_per_row / 2.0 + 0.5 - 1, 0.0
            )
            self._gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

            return viewer

        def run_v2(self,tfe) -> List[RunnerState]:
            states: List[RunnerState] = []

            control_step = 1 / self._batch.control_frequency
            sample_step = 1 / self._batch.sampling_frequency

            last_control_time = 0.0
            last_sample_time = 0.0

            # sample initial state
            states.append(self._get_state(0.0))
            old_position = self._get_state(0.0)
            reward = 0
            rewards = []
            cnt = []
            action = []

            #agent = ag.create(
            #    agent='ppo', states=self.states(), actions=self.actions(), batch_size=10, learning_rate=0.001,
            #    max_episode_timesteps=100000, memory=1100000, exploration=0.01
            #)

            agent = ag.load(directory="Agent5")
            internals = agent.initial_internals()
            state = tfe.reset()
            # with open('rewards.pickle', 'rb') as f:
            #    sum_rewards_list = list(pickle.load(f))
            i = 0
            tot_displacement = 0
            print("Testing...")
            while (
                    time := self._gym.get_sim_time(self._sim)
            ) < self._batch.simulation_time:
                # do control if it is time
                if time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()
                    if len(action) > 0:
                        #print("Action: ", action)
                        #print("Targets: ", targets)
                        action = np.clip(action, -1.0, 1.0)
                        action2 = action[:8]
                        #action3 = preprocessing.normalize([action2], norm='l2')
                        #action3 = action3.reshape(8,)
                        #for i in range(len(action3)):
                        #    action2[i] = action3[i]
                        self._batch.control(control_step, control, action2)
                        print("Action:", action2)
                    else:
                        self._batch.control(control_step, control, list(state[:8]))

                    #self._batch.control(control_step, control, targets)
                    for (env_index, actor_index, targets) in control._dof_targets:
                        env_handle = self._gymenvs[env_index].env
                        actor_handle = self._gymenvs[env_index].actors[actor_index]
                        actor = (
                            self._batch.environments[env_index]
                            .actors[actor_index]
                            .actor
                        )

                        #if len(action) > 0:
                        #    # print("Action: ", action)
                        #    # print("Targets: ", targets)
                        #    action = np.clip(action, -1.0, 1.0)
                        #    action2 = action[:8]
                        #    targets = action2
                        self.set_actor_dof_position_targets(
                            env_handle, actor_handle, actor, targets
                        )
                        #print("Targets:", targets)
                    # sample state if it is time
                if time >= last_sample_time + sample_step:
                    last_sample_time = int(time / sample_step) * sample_step
                    # print(time)
                    states.append(self._get_state(time))
                    new_position = self._get_state(time)
                    reward = self._calculate_velocity(old_position.envs[0].actor_states[0],
                                                      new_position.envs[0].actor_states[0],old_position.time_seconds,new_position.time_seconds)
                    tot_displacement += reward
                    #if int(time) % 100 == 0:
                    #    print("Distance traveled: ", tot_displacement)
                    #    break
                    print("Time Elapsed: ", time, " Reward: ", reward)

                    ## Getting inputs and rescaling
                    hinges_data = self._gym.get_actor_dof_states(self._gymenvs[0].env, 0, gymapi.STATE_ALL)
                    # print("Hinges data: ",hinges_data)
                    hinges_pos = np.array([[hinges_d[0] for hinges_d in hinges_data]])
                    hinges_pos = np.clip(hinges_pos, -1.0, 1.0)
                    hinges_vel = np.array([[hinges_d[1] for hinges_d in hinges_data]])
                    hinges_vel = np.clip(hinges_vel, -1.0, 1.0)
                    ## Normalization
                    #hinges_pos = preprocessing.normalize(hinges_pos, norm='l2')
                    #hinges_vel = preprocessing.normalize(hinges_vel, norm='l2')
                    '''
                    ## Standardization
                    scaler_pos = preprocessing.StandardScaler().fit(hinges_pos)
                    scaler_vel = preprocessing.StandardScaler().fit(hinges_vel)
                    hinges_pos = scaler_pos.transform(hinges_pos)
                    hinges_vel = scaler_vel.transform(hinges_vel)
                    '''
                    # print("Hinges pos: ", hinges_pos)
                    # print("Hinges vel: ", hinges_vel)
                    hinges = np.append(hinges_pos, hinges_vel)
                    state = list(hinges)
                    # print(hinges)
                    # train = self.train(hinges_pos,reward)
                    train = tfe.train(state, reward)
                    # print("okay 1")
                    # train = tfe.train(targets,reward)
                    # print("okay 2")
                    # print("TARGETS FOR ACT: ", targets)
                    #action = agent.act(states=state)
                    action, internals = agent.act(states=state, internals = internals, independent=True, deterministic = True)
                    # print("okay 3")
                    state, terminal, reward = tfe.execute(actions=action, train=train)
                    # next_state, terminal, reward = tfe.execute(actions=action, train=train)
                    # print("okay 4")
                    #agent.observe(terminal=terminal, reward=reward)
                    # print("okay 5")
                    #print(action)
                    rewards.append(reward)
                    cnt.append(i)
                    old_position = new_position
                # step simulation
                self._gym.simulate(self._sim)
                self._gym.fetch_results(self._sim, True)
                self._gym.step_graphics(self._sim)

                if self._viewer is not None:
                    self._gym.draw_viewer(self._viewer, self._sim, False)
                i = i + 1
            # sample one final time
            states.append(self._get_state(time))
            sum_rewards = sum(rewards)
            # sum_rewards_list.append(sum_rewards)
            print("Sum rewards: ", sum_rewards)
            # with open('rewards.pickle', 'wb') as f:
            #    pickle.dump(sum_rewards_list, f)
            '''
            plt.plot(cnt, rewards, label="rewards")
            plt.plot(cnt, [np.mean(rewards) for i in range(len(rewards))], label="mean")
            plt.plot(cnt, [np.std(rewards) for i in range(len(rewards))], label="std")
            plt.xlabel("Timesteps")
            plt.ylabel("Rewards")
            plt.legend()
            plt.show()

            rewards_added = rewards
            for ele in range(0, len(rewards)):
                if ele == 0:
                    rewards_added[ele] = rewards[ele]
                else:
                    rewards_added[ele] = rewards[ele] + rewards[ele - 1]

            plt.plot(cnt, rewards_added)
            plt.xlabel("Timesteps")
            plt.ylabel("Total Displacement on the X plane")
            plt.show()
            '''
            ag.save(self=agent, directory="Agent5")
            with open('rewards_test.pickle', 'wb') as f:
                pickle.dump(rewards, f)
            return states

        def run(self,tfe) -> List[RunnerState]:

            states: List[RunnerState] = []

            control_step = 1 / self._batch.control_frequency
            sample_step = 1 / self._batch.sampling_frequency

            last_control_time = 0.0
            last_sample_time = 0.0

            # sample initial state
            states.append(self._get_state(0.0))
            old_position = self._get_state(0.0)

            #agent = ag.create(
            #    agent='ppo', states=self.states(), actions=self.actions(), batch_size=10, learning_rate=0.01,
            #    max_episode_timesteps=100000, memory=1100000, exploration=0.01
            #)

            agent = ag.load(directory="Agent5")
            state = tfe.reset()
            internals = agent.initial_internals()

            episode_states = list()
            episode_internals = list()
            episode_actions = list()
            episode_terminal = list()
            episode_reward = list()

            reward = 0
            rewards = []
            cnt = []
            action = []
            test_count = 0
            #with open('rewards.pickle', 'rb') as f:
            #    sum_rewards_list = list(pickle.load(f))est = 0
            #                tot_displacement_test = 0

            i = 0
            tot_displacement = 0
            print("Training...")
            while (
                time := self._gym.get_sim_time(self._sim)
            ) < self._batch.simulation_time:
                # do control if it is time
                if time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()
                    if len(action) > 0:
                        #print("Action: ", action)
                        # print("Targets: ", targets)
                        action = np.clip(action, -1.0, 1.0)
                        action2 = action[:8]
                        #action3 = preprocessing.normalize([action2], norm='l2')
                        #action3 = action3.reshape(8, )
                        #for i in range(len(action3)):
                        #    action2[i] = action3[i]
                        self._batch.control(control_step, control, action2)
                        #print("Action:", action2)
                    else:
                        self._batch.control(control_step, control, list(state[:8]))

                    for (env_index, actor_index, targets) in control._dof_targets:
                        env_handle = self._gymenvs[env_index].env
                        actor_handle = self._gymenvs[env_index].actors[actor_index]
                        actor = (
                            self._batch.environments[env_index]
                            .actors[actor_index]
                            .actor
                        )

                        #if len(action) > 0:
                        #    #print("Action: ", action)
                        #    #print("Targets: ", targets)
                        #    action = np.clip(action,-1.0,1.0)
                        #    action2 = action[:8]
                        #    targets = action2
                        self.set_actor_dof_position_targets(
                            env_handle, actor_handle, actor, targets
                        )

                # sample state if it is time
                if time >= last_sample_time + sample_step:
                    last_sample_time = int(time / sample_step) * sample_step
                    #print(time)
                    states.append(self._get_state(time))
                    new_position = self._get_state(time)
                    reward = self._calculate_velocity(old_position.envs[0].actor_states[0],
                                                      new_position.envs[0].actor_states[0],old_position.time_seconds,new_position.time_seconds)
                    #reward = reward * pow(10,8)
                    tot_displacement += reward

                    print("Time Elapsed: ", time, " Reward: ", reward)
                    ## Getting inputs and rescaling
                    hinges_data = self._gym.get_actor_dof_states(self._gymenvs[0].env, 0, gymapi.STATE_ALL)
                    #print("Hinges data: ",hinges_data)
                    hinges_pos = np.array([[hinges_d[0] for hinges_d in hinges_data]])
                    hinges_pos = np.clip(hinges_pos, -1.0, 1.0)
                    hinges_vel = np.array([[hinges_d[1] for hinges_d in hinges_data]])
                    hinges_vel = np.clip(hinges_vel, -1.0, 1.0)
                    ## Normalization
                    #hinges_pos = preprocessing.normalize(hinges_pos, norm='l2')
                    #hinges_vel = preprocessing.normalize(hinges_vel, norm='l2')
                    '''
                    ## Standardization
                    hinges_pos = preprocessing.StandardScaler().fit_transform(hinges_pos)
                    hinges_vel = preprocessing.StandardScaler().fit_transform(hinges_vel)
                    '''
                    #print("Hinges pos: ", hinges_pos)
                    #print("Hinges vel: ", hinges_vel)
                    hinges = np.append(hinges_pos,hinges_vel)
                    #print(hinges)
                    #train = self.train(hinges_pos,reward)
                    state = list(hinges)
                    train = tfe.train(state, reward)
                    #episode_states.append(state)
                    #episode_internals.append(internals)
                    #action, internals = agent.act(states=state, internals=internals, independent=True, deterministic = False)
                    #print("okay 1")
                    #train = tfe.train(targets,reward)
                    #print("okay 2")
                    #print("TARGETS FOR ACT: ", targets)
                    action = agent.act(states=state)
                    #print("okay 3")
                    #episode_actions.append(action)
                    state, terminal, reward = tfe.execute(actions=action,train = train)
                   # episode_terminal.append(terminal)
                   # episode_reward.append(reward)
                    #next_state, terminal, reward = tfe.execute(actions=action,train=train)
                    #print(next_state)
                    #next_state, terminal, reward = tfe.execute(actions=action, train=train)
                    #print("okay 4")
                    agent.observe(terminal=terminal, reward=reward)
                    #print("okay 5")
                    rewards.append(reward)
                    cnt.append(i)
                    old_position = new_position

                # step simulation
                self._gym.simulate(self._sim)
                self._gym.fetch_results(self._sim, True)
                self._gym.step_graphics(self._sim)

                if self._viewer is not None:
                    self._gym.draw_viewer(self._viewer, self._sim, False)
                i = i + 1

            #episode_terminal[-1] = True
            #agent.experience(
            #    states=episode_states, internals=episode_internals, actions=episode_actions,
            #    terminal=episode_terminal, reward=episode_reward
            #)

            # Perform update
            #agent.update()

            # sample one final time
            states.append(self._get_state(time))
            sum_rewards = sum(rewards)
            # sum_rewards_list.append(sum_rewards)
            print("Sum rewards: ", sum_rewards)
            # with open('rewards.pickle', 'wb') as f:
            #    pickle.dump(sum_rewards_list, f)
            '''
            plt.plot(cnt, rewards, label="rewards")
            plt.plot(cnt, [np.mean(rewards) for i in range(len(rewards))], label="mean")
            plt.plot(cnt, [np.std(rewards) for i in range(len(rewards))], label="std")
            plt.xlabel("Timesteps")
            plt.ylabel("Rewards")
            plt.legend()
            plt.show()
            plt.clf()

            rewards_added = rewards
            for ele in range(0,len(rewards)):
                if ele == 0:
                    rewards_added[ele] = rewards[ele]
                else:
                    rewards_added[ele] = rewards[ele] + rewards[ele-1]

            plt.plot(cnt, rewards_added)
            plt.xlabel("Timesteps")
            plt.ylabel("Total Displacement on the XY plane")
            plt.show()
            plt.clf()
            '''
            #ag.save(self=agent, directory="Agent5")
            with open('rewards.pickle', 'wb') as f:
                pickle.dump(rewards, f)

            return states
        '''
        def states(self):
            # return sim._controller.get_dof_targets()
            min_value = -1.0
            max_value = 1.0
            return dict(type='float', shape=(8,), min_value = min_value, max_value = max_value)

        def actions(self):
            # return sim._controller.set_dof_targets()
            min_value = -1.0
            max_value = 1.0
            return dict(type='float', shape=(8,), min_value=min_value, max_value=max_value)

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
            return stateprogram stuck at tensorforce.act

        def train(self, next_state, reward):
            return next_state, reward

        def execute(self, actions, train):
            next_state, reward = train
            terminal = False  # Always False if no "natural" terminal state
            return next_state, terminal, reward
        '''
        @staticmethod
        def _calculate_velocity(begin_state, end_state, time1, time2) -> float:
            # TODO simulation can continue slightly passed the defined sim time.
            #print(begin_state.position[0])
            #print(end_state.position[0])
            #print(begin_state.position[1])
            #print(end_state.position[1])

            #xy = [[begin_state.position[0],end_state.position[0],
            #      begin_state.position[1],end_state.position[1]]]
            #t = [[time1,time2]]

            '''
            ##Standardization
            scaler_xy = preprocessing.StandardScaler().fit(xy)
            scaler_t = preprocessing.StandardScaler().fit(t)
            xy_normalized = scaler_xy.transform(xy)
            t_normalized = scaler_t.transform(t)
            '''

            ## Normalization
            #xy_normalized = preprocessing.normalize(xy, norm='l2')
            #t_normalized = preprocessing.normalize(t, norm='l2')

            #begin_state.position[0] = xy_normalized[0][0]
            #end_state.position[0] = xy_normalized[0][1]
            #begin_state.position[1] = xy_normalized[0][2]
            #end_state.position[1] = xy_normalized[0][3]

            #time1 = t_normalized[0][0]
            #time2 = t_normalized[0][1]
            #print(time1,time2, t_normalized, xy_normalized)

            # displacement in one axis (x)
            displacement_x = 100 * float((end_state.position[0] - begin_state.position[0])) #cm

            ## velocity x
            velocity_x = displacement_x/(time2 - time1)

            # distance traveled on the xy plane

            displacement_xy = float(
                math.sqrt(
                    (begin_state.position[0] - end_state.position[0]) ** 2
                    + ((begin_state.position[1] - end_state.position[1]) ** 2)
                )
            )
            velocity_xy = displacement_xy/(time2 - time1)

            return displacement_x



        def set_actor_dof_position_targets(
            self,
            env_handle: gymapi.Env,
            actor_handle: int,
            actor: Actor,
            targets: List[float],
        ) -> None:
            if len(targets) != len(actor.joints):
                raise RuntimeError("Need to set a target for every dof")

            if not all(
                [
                    target >= -joint.range and target <= joint.range
                    for target, joint in zip(
                        targets,
                        actor.joints,
                    )
                ]
            ):
                raise RuntimeError("Dof targets must lie within the joints range.")

            self._gym.set_actor_dof_position_targets(
                env_handle,
                actor_handle,
                targets,
            )

            self._gym.set_light_parameters(self._sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, 2, 3))

        def set_actor_dof_positions(
            self,
            env_handle: gymapi.Env,
            actor_handle: int,
            actor: Actor,
            positions: List[float],
        ) -> None:
            num_dofs = len(actor.joints)

            if len(positions) != num_dofs:
                raise RuntimeError("Need to set a position for every dof")

            if num_dofs != 0:  # isaac gym does not understand zero length arrays...
                dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
                dof_positions = dof_states["pos"]

                for i in range(len(dof_positions)):
                    dof_positions[i] = positions[i]
                self._gym.set_actor_dof_states(
                    env_handle, actor_handle, dof_states, gymapi.STATE_POS
                )

        def cleanup(self) -> None:
            if self._viewer is not None:
                self._gym.destroy_viewer(self._viewer)
            self._gym.destroy_sim(self._sim)

        def _get_state(self, time: float) -> RunnerState:
            state = RunnerState(time, [])

            for gymenv in self._gymenvs:
                env_state = EnvironmentState([])
                for actor_handle in gymenv.actors:
                    pose = self._gym.get_actor_rigid_body_states(
                        gymenv.env, actor_handle, gymapi.STATE_POS
                    )["pose"]
                    position = pose["p"][0]  # [0] is center of root element
                    #for i in range(0,len(pose["p"])):
                    #    print("Position: ",i ," ", pose["p"][i])
                    orientation = pose["r"][0]
                    #for i in range(0,len(pose["r"])):
                    #    print("Orientation: ",i ," ", pose["r"][i])
                    #print("Orientation: ", orientation)
                    env_state.actor_states.append(
                        ActorState(
                            Vector3([position[0], position[1], position[2]]),
                            Quaternion(
                                [
                                    orientation[0],
                                    orientation[1],
                                    orientation[2],
                                    orientation[3],
                                ]
                            ),
                        )
                    )
                state.envs.append(env_state)

            return state

    _sim_params: gymapi.SimParams
    _headless: bool

    def __init__(self, sim_params: gymapi.SimParams, headless: bool = False):
        self._sim_params = sim_params
        self._headless = headless

    @staticmethod
    def SimParams() -> gymapi.SimParams:
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.02    # step size
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 1
        sim_params.physx.use_gpu = True

        return sim_params

    async def run_batch(self,tfe,  batch: Batch) -> List[RunnerState]:
        # sadly we must run Isaac Gym in a subprocess, because it has some big memory leaks.
        result_queue: mp.Queue = mp.Queue()  # type: ignore # TODO
        process = mp.Process(
            target=self._run_batch_impl,
            args=(tfe, result_queue, batch, self._sim_params, self._headless),
        )
        process.start()
        states = []
        # states are sent state by state(every sample)
        # because sending all at once is too big for the queue.
        # should be good enough for now.
        # if the program hangs here in the future,
        # improve the way the results are passed back to the parent program.
        while (state := result_queue.get()) is not None:
            states.append(state)
        process.join()
        return states

    @classmethod
    def _run_batch_impl(
        cls,
        tfe,
        result_queue: mp.Queue,  # type: ignore # TODO
        batch: Batch,
        sim_params: gymapi.SimParams,
        headless: bool
    ) -> None:
        _Simulator = cls._Simulator(batch, sim_params, headless)
        states = _Simulator.run(tfe)
        _Simulator.cleanup()
        for state in states:
            result_queue.put(state)
        result_queue.put(None)

    async def run_batch_v2(self,tfe,  batch: Batch) -> List[RunnerState]:
        # sadly we must run Isaac Gym in a subprocess, because it has some big memory leaks.
        result_queue: mp.Queue = mp.Queue()  # type: ignore # TODO
        process = mp.Process(
            target=self._run_batch_impl_v2,
            args=(tfe, result_queue, batch, self._sim_params, self._headless),
        )
        process.start()
        states = []
        # states are sent state by state(every sample)
        # because sending all at once is too big for the queue.
        # should be good enough for now.
        # if the program hangs here in the future,
        # improve the way the results are passed back to the parent program.
        while (state := result_queue.get()) is not None:
            states.append(state)
        process.join()
        return states

    @classmethod
    def _run_batch_impl_v2(
        cls,
        tfe,
        result_queue: mp.Queue,  # type: ignore # TODO
        batch: Batch,
        sim_params: gymapi.SimParams,
        headless: bool
    ) -> None:
        _Simulator = cls._Simulator(batch, sim_params, headless)
        states = _Simulator.run_v2(tfe)
        _Simulator.cleanup()
        for state in states:
            result_queue.put(state)
        result_queue.put(None)

class CustomEnvironment(tfenv):

    def __init__(self):
        super().__init__()

    def states(self):
        # return sim._controller.get_dof_targets()
        min_value = -1.0
        max_value = 1.0
        return dict(type='float', shape=(16,), min_value=min_value, max_value=max_value)

    def actions(self):
        # return sim._controller.set_dof_targets()
        min_value = -1.0
        max_value = 1.0
        return dict(type='float', shape=(16,), min_value=min_value, max_value=max_value)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(16,))
        return state

    def train(self, next_state, reward):
        return next_state, reward

    def execute(self, actions, train):
        next_state, reward = train
        terminal = False  # Always False if no "natural" terminal state
        return next_state, terminal, reward

    _controller: ActorController

    async def simulate(self,tfe, robot: ModularRobot, control_frequency: float, headless = True ):
        batch = Batch(
            simulation_time=30,
            sampling_frequency=8,
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
        runner = LocalRunner(LocalRunner.SimParams(),headless=headless)
        #print("okay 1")
        states = await runner.run_batch(tfe, batch)
        #print("okay 2")
        return states

    async def simulate_v2(self, tfe, robot: ModularRobot, control_frequency: float, headless=True):
        batch = Batch(
            simulation_time=30,
            sampling_frequency=8,
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
        states = await runner.run_batch_v2(tfe, batch)
        #print("okay 2")
        return states

    def _control(self, dt: float, control: ActorControl, action) -> None:
        self._controller.step(dt)
        #control.set_dof_targets(0, 0, self._controller.get_dof_targets()) # ACTION
        control.set_dof_targets(0, 0, action)
        #print(self._controller.get_dof_targets())
        #print(action)

    def make_robot(self):
        body = Body()
        '''
        ## Default
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
        rng.seed(3)
        brain = BrainCpgNetworkNeighbourRandom(rng)
        # brain = RLbrain(from_checkpoint=False)
        robot = ModularRobot(body, brain)
        return robot


async def main() -> None:
    episodes_train = []
    episodes_test = []
    rewards_train = []
    rewards_test = []

    tfe = CustomEnvironment()

    '''
    ## Make agent
    agent = ag.create(
        agent='ddpg', states=tfe.states(), actions=tfe.actions(), batch_size=1, learning_rate=0.01,
        max_episode_timesteps=1000, memory=11000, exploration=0.2, summarizer = "Agent5summary"
    )
    ag.save(agent,directory="Agent5")
    print("Agent created!")
    '''

    robot = tfe.make_robot()

    ## Training X Validation
    for i in range(500):
        if (i+1) % 100 == 0:
            states_test = await tfe.simulate_v2(tfe, robot, control_frequency=8, headless=False)
            episodes_test.append(i)
            with open('rewards_test.pickle', 'rb') as f:
                rewards = pickle.load(f)
            rewards_test.append(np.sum(rewards))
        print("EPISODE: ", i, "\n")
        states_train = await tfe.simulate(tfe, robot, control_frequency=8, headless=True)
        episodes_train.append(i)
        with open('rewards.pickle', 'rb') as f:
            rewards = pickle.load(f)
        rewards_train.append(np.sum(rewards))
    
    plt.plot(np.array(episodes_train), np.array(rewards_train), label="Sum rewards")
    plt.plot(np.array(episodes_train), [np.mean(rewards_train) for _ in range(len(rewards_train))], label="Mean of Sum")
    plt.plot(np.array(episodes_train), [np.std(rewards_train) for _ in range(len(rewards_train))], label="STD of Sum")
    plt.xlabel("Episodes")
    plt.ylabel("Sum rewards / episode")
    plt.legend()
    plt.show()
    plt.clf()

    '''
    for i in range(5):
        states_test = await tfe.simulate_v2(tfe, robot, control_frequency=8, headless=False)
        episodes_test.append(i)
        with open('rewards_test.pickle',                                                                                                                                                                                                                                                                                        'rb') as f:
            rewards = pickle.load(f)
        rewards_test.append(np.sum(rewards))
    '''

    plt.plot(np.array(episodes_test), np.array(rewards_test), label="Sum rewards")
    plt.plot(np.array(episodes_test), [np.mean(rewards_test) for _ in range(len(rewards_test))],
             label="Mean of Sum")
    plt.plot(np.array(episodes_test), [np.std(rewards_test) for _ in range(len(rewards_test))],
             label="STD of Sum")
    plt.xlabel("Episodes")
    plt.ylabel("Sum rewards / episode")
    plt.legend()
    plt.show()
    plt.clf()

    agent = ag.load(directory="Agent5")

    agent.close()
    tfe.close()

if __name__ == "__main__":

    import asyncio

    asyncio.run(main())