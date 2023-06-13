import math
import multiprocessing as mp
import os
import tempfile
import time
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
from revolve2.core.modular_robot.render.render import Render
#from jlo.RL.rl_brain import RLbrain

from tensorforce.environments import Environment as tfenv
from tensorforce.agents import Agent as ag
from tensorforce.execution import Runner as rn
import neat
import visualize

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

        def run(self,genomes,config) -> List[RunnerState]:
            states: List[RunnerState] = []
            for genome_id, genome in genomes:

                net = neat.nn.FeedForwardNetwork.create(genome, config)
                fitness = 0
                outputs = []
                control_step = 1 / self._batch.control_frequency
                sample_step = 1 / self._batch.sampling_frequency

                last_control_time = self._gym.get_sim_time(self._sim)
                last_sample_time = self._gym.get_sim_time(self._sim)

                control = ActorControl()
                self._batch.control(control_step, control, [0,0,0,0,0,0,0,0])
                for (env_index, actor_index, targets) in control._dof_targets:
                    env_handle = self._gymenvs[env_index].env
                    actor_handle = self._gymenvs[env_index].actors[actor_index]
                    actor = (
                        self._batch.environments[env_index]
                        .actors[actor_index]
                        .actor
                    )
                    self.set_actor_dof_position_targets(
                        env_handle, actor_handle, actor, targets
                    )
                    self.set_actor_dof_positions(env_handle, actor_handle, actor,[0,0,0,0,0,0,0,0])
                #print(self._gym.get_actor_dof_states(self._gymenvs[0].env, 0, gymapi.STATE_ALL))

                # sample initial state
                states.append(self._get_state(self._gym.get_sim_time(self._sim)))
                old_position = self._get_state(self._gym.get_sim_time(self._sim))
                #print(round(time := self._gym.get_sim_time(self._sim),2))

                if int(time := self._gym.get_sim_time(self._sim)) != 0:
                    while (round(time := self._gym.get_sim_time(self._sim),2) % 30 == 0):
                        # step simulation
                        self._gym.simulate(self._sim)
                        self._gym.fetch_results(self._sim, True)
                        self._gym.step_graphics(self._sim)

                        if self._viewer is not None:
                            self._gym.draw_viewer(self._viewer, self._sim, False)
                while (
                        time := self._gym.get_sim_time(self._sim)
                 < self._batch.simulation_time ) and (
                        (round(time := self._gym.get_sim_time(self._sim),2) % 30 != 0)
                        or (int(time := self._gym.get_sim_time(self._sim)) == 0 )):

                    #print(time := self._gym.get_sim_time(self._sim))
                    # do control if it is time
                    if time >= last_control_time + control_step:
                        last_control_time = math.floor(time / control_step) * control_step
                        control = ActorControl()
                        if len(outputs) > 0:
                            #print("Output: ", outputs)
                            # print("Targets: ", targets)
                            self._batch.control(control_step, control, outputs)
                        for (env_index, actor_index, targets) in control._dof_targets:
                            env_handle = self._gymenvs[env_index].env
                            actor_handle = self._gymenvs[env_index].actors[actor_index]
                            actor = (
                                self._batch.environments[env_index]
                                .actors[actor_index]
                                .actor
                            )
                            self.set_actor_dof_position_targets(
                                env_handle, actor_handle, actor, targets
                            )
                    # sample state if it is time
                    if time >= last_sample_time + sample_step:
                        last_sample_time = int(time / sample_step) * sample_step
                        #print(time)
                        states.append(self._get_state(time))
                        new_position = self._get_state(time)
                        fitness += self._calculate_velocity(old_position.envs[0].actor_states[0],
                                                          new_position.envs[0].actor_states[0],
                                                          old_position.time_seconds, new_position.time_seconds)
                        #print("fitness: ", fitness)
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
                        hinges = np.append(hinges_pos,hinges_vel)
                        outputs = net.activate(hinges[:8])
                        old_position = new_position

                    # step simulation
                    self._gym.simulate(self._sim)
                    self._gym.fetch_results(self._sim, True)
                    self._gym.step_graphics(self._sim)

                    if self._viewer is not None:
                        self._gym.draw_viewer(self._viewer, self._sim, False)

                # sample one final time
                states.append(self._get_state(time))
                genome.fitness = fitness
                print("Genome fitness: ", genome.fitness)
            return states

        def run_v2(self,genome,config) -> List[RunnerState]:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = 0
            outputs = []
            states: List[RunnerState] = []
            control_step = 1 / self._batch.control_frequency
            sample_step = 1 / self._batch.sampling_frequency

            last_control_time = self._gym.get_sim_time(self._sim)
            last_sample_time = self._gym.get_sim_time(self._sim)

            # sample initial state
            states.append(self._get_state(self._gym.get_sim_time(self._sim)))
            old_position = self._get_state(self._gym.get_sim_time(self._sim))
            #print(round(time := self._gym.get_sim_time(self._sim),2))

            if int(time := self._gym.get_sim_time(self._sim)) != 0:
                while (round(time := self._gym.get_sim_time(self._sim),2) % 30 == 0):
                    # step simulation
                    self._gym.simulate(self._sim)
                    self._gym.fetch_results(self._sim, True)
                    self._gym.step_graphics(self._sim)

                    if self._viewer is not None:
                        self._gym.draw_viewer(self._viewer, self._sim, False)
            while (
                    time := self._gym.get_sim_time(self._sim)
             < self._batch.simulation_time ) and (
                    (round(time := self._gym.get_sim_time(self._sim),2) % 30 != 0)
                    or (int(time := self._gym.get_sim_time(self._sim)) == 0 )):

                #print(time := self._gym.get_sim_time(self._sim))
                # do control if it is time
                if time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()
                    if len(outputs) > 0:
                        #print("Output: ", outputs)
                        # print("Targets: ", targets)
                        self._batch.control(control_step, control, outputs)
                    for (env_index, actor_index, targets) in control._dof_targets:
                        env_handle = self._gymenvs[env_index].env
                        actor_handle = self._gymenvs[env_index].actors[actor_index]
                        actor = (
                            self._batch.environments[env_index]
                            .actors[actor_index]
                            .actor
                        )
                        self.set_actor_dof_position_targets(
                            env_handle, actor_handle, actor, targets
                        )
                # sample state if it is time
                if time >= last_sample_time + sample_step:
                    last_sample_time = int(time / sample_step) * sample_step
                    #print(time)
                    states.append(self._get_state(time))
                    new_position = self._get_state(time)
                    #fitness += self._calculate_velocity(old_position.envs[0].actor_states[0],
                    #                                  new_position.envs[0].actor_states[0],
                    #                                  old_position.time_seconds, new_position.time_seconds)
                    #print("fitness: ", fitness)
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
                    hinges = np.append(hinges_pos,hinges_vel)
                    outputs = net.activate(hinges[:8])
                    old_position = new_position

                # step simulation
                self._gym.simulate(self._sim)
                self._gym.fetch_results(self._sim, True)
                self._gym.step_graphics(self._sim)

                if self._viewer is not None:
                    self._gym.draw_viewer(self._viewer, self._sim, False)

            # sample one final time
            states.append(self._get_state(time))
            fitness = self._calculate_velocity(states[0].envs[0].actor_states[0],
                                               self._get_state(time).envs[0].actor_states[0],
                                               0.0, self._get_state(time).time_seconds)
            genome.fitness = fitness
            print("Genome fitness: ", genome.fitness)
            return states

        def run_v3(self,net):
            fitness = 0
            reps = 0
            outputs = []
            outputs_sum = []
            states: List[RunnerState] = []
            control_step = 1 / self._batch.control_frequency
            sample_step = 1 / self._batch.sampling_frequency

            last_control_time = self._gym.get_sim_time(self._sim)
            last_sample_time = self._gym.get_sim_time(self._sim)

            # sample initial state
            states.append(self._get_state(self._gym.get_sim_time(self._sim)))
            old_position = self._get_state(self._gym.get_sim_time(self._sim))
            #print(round(time := self._gym.get_sim_time(self._sim),2))
            #print(time)
            while (time:=self._gym.get_sim_time(self._sim)) < self._batch.simulation_time:
                # do control if it is time
                if time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()
                    if len(outputs) > 0:
                        #print("Output: ", outputs)
                        # print("Targets: ", targets)
                        #outputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                        self._batch.control(control_step, control, outputs)
                    for (env_index, actor_index, targets) in control._dof_targets:
                        env_handle = self._gymenvs[env_index].env
                        actor_handle = self._gymenvs[env_index].actors[actor_index]
                        actor = (
                            self._batch.environments[env_index]
                            .actors[actor_index]
                            .actor
                        )
                        self.set_actor_dof_position_targets(
                            env_handle, actor_handle, actor, targets
                        )
                # sample state if it is time
                if time >= last_sample_time + sample_step:
                    last_sample_time = int(time / sample_step) * sample_step
                    #print(time)
                    states.append(self._get_state(time))
                    new_position = self._get_state(time)
                    #fitness += (self._calculate_velocity(old_position.envs[0].actor_states[0],
                    #                                  new_position.envs[0].actor_states[0],
                    #                                  old_position.time_seconds, new_position.time_seconds))
                    #print("fitness: ", fitness)
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
                    hinges = np.append(hinges_pos,hinges_vel)
                    inputs = np.ndarray.flatten(hinges_pos)
                    # 1,3,5,7 (outer) 0,2,4,6 (inner) - spider / 2,3,4,5 (outer) 0,1 (inner) - gecko / 1,3,8,10 (outer) 0,2,7,9 (inner) - supergecko / 0,3,6,9 (inner) 2,5,8,11 (outer) - superspider
                    #reduced_inputs = []
                    #for cnt in range(0,len(inputs)):
                    #    if cnt not in [1,3,5,7]:
                    #        reduced_inputs.append(inputs[cnt])
                    #outputs = net.activate(reduced_inputs)
                    outputs = net.activate(inputs)
                    #abs_outputs = [abs(ele) for ele in outputs]
                    #print(outputs)
                    if outputs_sum == [] :
                        outputs_sum = outputs
                        reps+=1
                    else:
                        outputs_sum = [sum(i) for i in zip(outputs,outputs_sum)]
                        reps+=1
                    old_position = new_position

                # step simulation
                self._gym.simulate(self._sim)
                self._gym.fetch_results(self._sim, True)
                self._gym.step_graphics(self._sim)

                if self._viewer is not None:
                    self._gym.draw_viewer(self._viewer, self._sim, False)

            # sample one final time
            states.append(self._get_state(time))

            fitness = self._calculate_velocity(states[0].envs[0].actor_states[0], self._get_state(time).envs[0].actor_states[0],
                                              0.0, self._get_state(time).time_seconds)
            with open('genome_fitness', 'wb') as fp:
                pickle.dump(fitness, fp)
            avg_output = [x / reps for x in outputs_sum]
            with open('outputs','wb') as gp:
                pickle.dump(avg_output, gp)

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
            displacement_x = 100 * float((end_state.position[0] - begin_state.position[0])) # cm

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

            return displacement_xy

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

    async def run_batch(self, batch: Batch) -> List[RunnerState]:
        # sadly we must run Isaac Gym in a subprocess, because it has some big memory leaks.
        result_queue: mp.Queue = mp.Queue()  # type: ignore # TODO
        process = mp.Process(
            target=self._run_batch_impl,
            args=(result_queue, batch, self._sim_params, self._headless),
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

    async def run_batch_v2(self, batch: Batch) -> List[RunnerState]:
        # sadly we must run Isaac Gym in a subprocess, because it has some big memory leaks.
        result_queue: mp.Queue = mp.Queue()  # type: ignore # TODO
        process = mp.Process(
            target=self._run_batch_impl_v2,
            args=(result_queue, batch, self._sim_params, self._headless),
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
        result_queue: mp.Queue,  # type: ignore # TODO
        batch: Batch,
        sim_params: gymapi.SimParams,
        headless: bool
    ) -> None:
        _Simulator = cls._Simulator(batch, sim_params, headless)

        #states = _Simulator.run()
        #for state in states:
        #    result_queue.put(state)
        #result_queue.put(None)

        config_file = 'neat_config'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-99')

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to x generations.
        winner = p.run(states := _Simulator.run, 50)

        # Display the winning genome (in population of last generation)

        print('\nBest genome:\n{!s}'.format(winner))

        ## Overall best genome
        real_winner = stats.best_genome()

        # Plots for fitness, network and speciation
        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

        ##
        with open('best_genome', 'wb') as fp:
            pickle.dump(real_winner, fp)
        with open('winner', 'wb') as fp:
            pickle.dump(winner, fp)
        # with open ('best_genome', 'rb') as fp:
        #    real_winner = pickle.load(fp)
        result_queue.put(None)

    def _run_batch_impl_v2(
        cls,
        result_queue: mp.Queue,  # type: ignore # TODO
        batch: Batch,
        sim_params: gymapi.SimParams,
        headless: bool
    ) -> None:
        _Simulator = cls._Simulator(batch, sim_params, headless)
        with open('/revolve2/runners/isaacgym/neat/neat_test9xy/best_genome', 'rb') as fp:
            real_winner = pickle.load(fp)
        config_file = 'neat_config'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        states = _Simulator.run_v2(real_winner,config)
        for state in states:
            result_queue.put(state)
        result_queue.put(None)


    def run_batch_v3(self,net, batch: Batch):
        # sadly we must run Isaac Gym in a subprocess, because it has some big memory leaks.
        result_queue: mp.Queue = mp.Queue()  # type: ignore # TODO
        process = mp.Process(
            target=self._run_batch_impl_v3,
            args=(net, result_queue, batch, self._sim_params, self._headless),
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
    def _run_batch_impl_v3(
        cls,
        net,
        result_queue: mp.Queue,  # type: ignore # TODO
        batch: Batch,
        sim_params: gymapi.SimParams,
        headless: bool
    ) -> None:
        _Simulator = cls._Simulator(batch, sim_params, headless)
        states = _Simulator.run_v3(net)
        for state in states:
            result_queue.put(state)
        result_queue.put(None)


class CustomEnvironment(tfenv):

    def __init__(self):
        super().__init__()
        self.head_balance = []
        self.outputs = []

    def get_head_balance(self):
        return self.head_balance
    def set_head_balance(self, head_balance):
        self.head_balance = head_balance
    def get_outputs(self):
        return self.outputs
    def set_outputs(self, outputs):
        self.outputs = outputs
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

    async def simulate(self, robot: ModularRobot, control_frequency: float, headless = True ):
        batch = Batch(
            simulation_time=50000,
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
        states = await runner.run_batch(batch)
        #print("okay 2")
        return states

    async def simulate_v2(self, robot: ModularRobot, control_frequency: float, headless=True):
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
        runner = LocalRunner(LocalRunner.SimParams(), headless=headless)
        # print("okay 1")
        states = await runner.run_batch_v2(batch)
        # print("okay 2")
        return states

    def simulate_v3(self, genomes, config):

        control_frequency = 8
        robot = self.make_robot()
        headless = True
        head_balance = []
        avg_outputs = []
        reps = 0
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            batch = Batch(
                simulation_time=20,
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
            states = runner.run_batch_v3(net,batch)
            #print("okay 2")
            with open('genome_fitness','rb') as fp:
                fitness = pickle.load(fp)
            genome.fitness = fitness
            print("Genome Fitness: ", genome.fitness, '\n')
            with open('outputs', 'rb') as gp:
                outputs = pickle.load(gp)
            if avg_outputs == []:
                avg_outputs = outputs
                reps +=1
            else:
                avg_outputs = [sum(i) for i in zip(outputs, avg_outputs)]
                reps+=1
            balance = self._head_balance(states)
            head_balance.append(balance)
        avg_outputs2 = [x / reps for x in avg_outputs]
        self.head_balance.append(head_balance)
        self.outputs.append(avg_outputs2)


    def simulate_v4(self, genome, config):
        control_frequency = 8
        robot = self.make_robot()
        headless = False
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        batch = Batch(
            simulation_time=20,
            sampling_frequency=8,
            control_frequency=control_frequency,
            control=self._control,
        )
        actor, self._controller = robot.make_actor_and_controller()
        #print(actor.joints)
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
        states = runner.run_batch_v3(net,batch)
        #print("okay 2")
        with open('genome_fitness','rb') as fp:
            fitness = pickle.load(fp)
        return fitness, states

    def _head_balance(self, _states):
        """
                Returns the inverse of the average rotation of the head in the roll and pitch dimensions.
                The closest to 1 the most balanced.
                :return:
                """

        if _states is None:
            return -math.inf

        _orientations = []
        #print(_states)
        #print(len(_states))
        for idx_state in range(0, len(_states)):
            _orientation = _states[idx_state].envs[0].actor_states[0].serialize()['orientation']
            # w, x, y, z
            #qua = Quaternion(value=(_orientations[0], _orientations[1], _orientations[2], _orientations[3]))
            r,p,y = self.euler_from_quaternion(_orientation[1], _orientation[2], _orientation[3],
                                                      _orientation[0])
            eulers = [r,p,y]  # roll / pitch / yaw
            #print(_orientation, '\n', eulers)
            _orientations.append(eulers)

        roll = 0
        pitch = 0
        instants = len(_states)
        # print(_orientations)

        for o in _orientations:
            roll = roll + abs(o[0]) * 180 / math.pi
            pitch = pitch + abs(o[1]) * 180 / math.pi

        #  accumulated angles for each type of rotation
        #  divided by iterations * maximum angle * each type of rotation
        if instants == 0:
            balance = None
        else:
            balance = (roll + pitch) / (instants * 180 * 2)
            # turns imbalance to balance
            balance = 1 - balance

        return balance

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians


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
        '''
        ## Ant 
        body.core.right = ActiveHinge(0.0)
        body.core.right.attachment = Brick(0.0)
        
        body.core.left = ActiveHinge(0.0)
        body.core.left.attachment = Brick(0.0)
        
        body.core.back = ActiveHinge(math.pi / 2)
        body.core.back.attachment = Brick(math.pi / 2)
        body.core.back.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.right.attachment = Brick(0.0)
        body.core.back.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)
        '''
        '''
        ## Super Ant
        body.core.right = ActiveHinge(math.pi / 2)
        body.core.right.attachment = Brick(math.pi / 2)
        body.core.right.attachment.front = ActiveHinge(0.0)
        body.core.right.attachment.front.attachment = Brick(0.0)

        body.core.left = ActiveHinge(math.pi / 2)
        body.core.left.attachment = Brick(math.pi / 2)
        body.core.left.attachment.front = ActiveHinge(0.0)
        body.core.left.attachment.front.attachment = Brick(0.0)

        body.core.back = ActiveHinge(math.pi / 2)
        body.core.back.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment = Brick(math.pi / 2)

        body.core.back.attachment.front.attachment.left = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.left.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.left.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.left.attachment.front.attachment = Brick(0.0)
        
        body.core.back.attachment.front.attachment.right = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.right.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.right.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.right.attachment.front.attachment = Brick(0.0)
        
        body.core.back.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(math.pi / 2)
        
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.left = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment.front.attachment = Brick(0.0)
        
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.right = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment.front.attachment = Brick(0.0)
        '''

        ## Gecko
        body.core.left = ActiveHinge(0.0)
        body.core.left.attachment = Brick(0.0)

        body.core.right = ActiveHinge(0.0)
        body.core.right.attachment = Brick(0.0)

        body.core.back = ActiveHinge(math.pi / 2)
        body.core.back.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

        '''
        ## Super Gecko
        body.core.left = ActiveHinge(math.pi / 2)
        body.core.left.attachment = Brick(math.pi / 2)
        body.core.left.attachment.front = ActiveHinge(0.0)
        body.core.left.attachment.front.attachment = Brick(0.0)

        body.core.right = ActiveHinge(math.pi / 2)
        body.core.right.attachment = Brick(math.pi / 2)
        body.core.right.attachment.front = ActiveHinge(0.0)
        body.core.right.attachment.front.attachment = Brick(0.0)

        body.core.back = ActiveHinge(math.pi / 2)
        body.core.back.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment = Brick(math.pi / 2)

        body.core.back.attachment.front.attachment.front.attachment.left = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.left.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.left.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.front.attachment.left.attachment.front.attachment = Brick(0.0)
        
        body.core.back.attachment.front.attachment.front.attachment.right = ActiveHinge(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.right.attachment = Brick(math.pi / 2)
        body.core.back.attachment.front.attachment.front.attachment.right.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.front.attachment.right.attachment.front.attachment = Brick(0.0)
        '''
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
        '''
        '''
        ## Super Spider
        body.core.left = ActiveHinge(np.pi / 2.0)
        body.core.left.attachment = Brick(-np.pi / 2.0)
        body.core.left.attachment.front = ActiveHinge(np.pi / 2)
        body.core.left.attachment.front.attachment = Brick(-np.pi / 2)
        body.core.left.attachment.front.attachment.front = ActiveHinge(0.0)
        body.core.left.attachment.front.attachment.front.attachment = Brick(0.0)

        body.core.right = ActiveHinge(np.pi / 2.0)
        body.core.right.attachment = Brick(-np.pi / 2.0)
        body.core.right.attachment.front = ActiveHinge(np.pi / 2)
        body.core.right.attachment.front.attachment = Brick(-np.pi / 2)
        body.core.right.attachment.front.attachment.front = ActiveHinge(0.0)
        body.core.right.attachment.front.attachment.front.attachment = Brick(0.0)

        body.core.front = ActiveHinge(np.pi / 2.0)
        body.core.front.attachment = Brick(-np.pi / 2.0)
        body.core.front.attachment.front = ActiveHinge(np.pi / 2)
        body.core.front.attachment.front.attachment = Brick(-np.pi / 2)
        body.core.front.attachment.front.attachment.front = ActiveHinge(0.0)
        body.core.front.attachment.front.attachment.front.attachment = Brick(0.0)

        body.core.back = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front = ActiveHinge(np.pi / 2)
        body.core.back.attachment.front.attachment = Brick(-np.pi / 2)
        body.core.back.attachment.front.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.front.attachment = Brick(0.0)
        '''
        body.finalize()
        rng = Random()
        rng.seed(3)
        brain = BrainCpgNetworkNeighbourRandom(rng)
        # brain = RLbrain(from_checkpoint=False)
        robot = ModularRobot(body, brain)
        return robot
'''
async def main() -> None:
    tfe = CustomEnvironment()
    robot = tfe.make_robot()
    #states_train = await tfe.simulate(robot, control_frequency=8, headless=False)
    states_test = await tfe.simulate_v2(robot, control_frequency=8, headless=False)

if __name__ == "__main__":

    import asyncio

    asyncio.run(main())

'''
def main() -> None:
    mode = 'test'
    avg2_fitness = []
    std2_fitness = []
    best2_fitness = []
    median2_fitness = []
    stat25_2 = []
    stat75_2 = []
    mean2_lst = []
    max2_lst = []
    std2_lst = []
    avg_outputs = []
    ini_total = time.time()
    if mode == 'train':
        for i in range(10):
            print("-----RUN "+ str(i)+'-----')
            ini = time.time()
            tfe = CustomEnvironment()
            config_file = 'neat_config'
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            #p = neat.Checkpointer.restore_checkpoint('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat_test9xy/neat-checkpoint-49')

            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(generation_interval=40,time_interval_seconds=None, filename_prefix='checkpoint_'+str(i)+'_'))
            #genomes = []
            #for i in p.population:
            #    k = (i, p.population[i])
            #    genomes.append(k)
            # Run for up to x generations.
            #print(genomes)

            # Run for up to x generations in parallel.
            #pe = neat.ParallelEvaluator(3, tfe.simulate_v4)
            #winner = p.run(pe.evaluate, 5)

            winner = p.run(tfe.simulate_v3, 40)
            # Display the winning genome (in population of last generation)
            print('\nBest genome:\n{!s}'.format(winner))

            ## Overall best genome
            real_winner = stats.best_genome()
            head_balance = tfe.get_head_balance()
            outputs = tfe.get_outputs()

            ##
            with open('stats_' + str(i), 'wb') as fp:
                pickle.dump(stats, fp)

            with open('avg_outputs_' + str(i), 'wb') as fp:
                pickle.dump(outputs, fp)

            # Plots for fitness, network and speciation
            visualize.plot_stats(stats,cnt=i, ylog=False, view=False)
            visualize.plot_stats2(stats,cnt=i, ylog=False, view=False)
            visualize.plot_head_balance(stats, head_balance,cnt=i, ylog=False, view=False)
            visualize.plot_outputs(stats,outputs,cnt=i,ylog=False,view=False)

            visualize.plot_stats_nobest(stats, cnt=i, ylog=False, view=False)
            visualize.plot_stats2_nobest(stats, cnt=i, ylog=False, view=False)
            visualize.plot_head_balance_nobest(stats, head_balance, cnt=i, ylog=False, view=False)

            visualize.draw_net(config, real_winner,cnt=i, view=False)
            visualize.plot_species(stats,cnt=i, view=False)
            ##
            with open('best_genome_'+ str(i), 'wb') as fp:
                pickle.dump(real_winner, fp)

            with open('winner_'+str(i), 'wb') as fp:
                pickle.dump(winner, fp)

            best_fitness = [c.fitness for c in stats.most_fit_genomes]
            avg_fitness = stats.get_fitness_mean()
            stdev_fitness = stats.get_fitness_stdev()
            median_fitness = stats.get_fitness_median()
            stat25 = []
            stat75 = []
            for st in stats.generation_statistics:
                scores = []
                for species_stats in st.values():
                    scores.extend(species_stats.values())
                stat25.append(np.percentile(scores, 25))
                stat75.append(np.percentile(scores, 75))

            mean_lst = []
            max_lst = []
            std_lst = []
            for lst in head_balance:
                mean_lst.append(np.mean(lst))
                max_lst.append(np.max(lst))
                std_lst.append(np.std(lst))

            if i > 0:
                avg2_fitness = [ele1 + ele2 for ele1, ele2 in zip(avg2_fitness, avg_fitness)]
                std2_fitness = [ele1 + ele2 for ele1, ele2 in zip(std2_fitness, stdev_fitness)]
                best2_fitness = [ele1 + ele2 for ele1, ele2 in zip(best2_fitness, best_fitness)]
                median2_fitness = [ele1 + ele2 for ele1, ele2 in zip(median2_fitness, median_fitness)]
                stat25_2 = [ele1 + ele2 for ele1, ele2 in zip(stat25_2, stat25)]
                stat75_2 = [ele1 + ele2 for ele1, ele2 in zip(stat75_2, stat75)]
                mean2_lst = [ele1 + ele2 for ele1, ele2 in zip(mean2_lst, mean_lst)]
                max2_lst = [ele1 + ele2 for ele1, ele2 in zip(max2_lst, max_lst)]
                std2_lst = [ele1 + ele2 for ele1, ele2 in zip(std2_lst, std_lst)]
                for cnt in range(0,len(avg_outputs)):
                    avg_outputs[cnt] = [ele1 + ele2 for ele1,ele2 in zip(avg_outputs[cnt],outputs[cnt])]
            else:
                avg2_fitness = avg_fitness
                std2_fitness = stdev_fitness
                best2_fitness = best_fitness
                median2_fitness = median_fitness
                stat25_2 = stat25
                stat75_2 = stat75
                mean2_lst = mean_lst
                max2_lst = max_lst
                std2_lst = std_lst
                avg_outputs = outputs

            tfe.set_head_balance([])
            tfe.set_outputs([])
            fim = time.time()  # prints execution time
            print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

        avg2_fitness = [number / 10 for number in avg2_fitness]
        std2_fitness = [number / 10 for number in std2_fitness]
        best2_fitness = [number / 10 for number in best2_fitness]
        median2_fitness = [number / 10 for number in median2_fitness]
        stat25_2 = [number / 10 for number in stat25_2]
        stat75_2 = [number / 10 for number in stat75_2]
        mean2_lst = [number / 10 for number in mean2_lst]
        max2_lst = [number / 10 for number in max2_lst]
        std2_lst = [number / 10 for number in std2_lst]
        for cnt in range(0,len(avg_outputs)):
            avg_outputs[cnt] = [number/ 10 for number in avg_outputs[cnt]]

        gens = 40

        # Plots for fitness, network and speciation
        visualize.plot_stats_avg(best2_fitness,avg2_fitness,std2_fitness,gens, ylog=False, view=False)
        visualize.plot_stats2_avg(median2_fitness,stat25_2,stat75_2,best2_fitness,gens, ylog=False, view=False)
        visualize.plot_head_balance_avg(mean2_lst,max2_lst,std2_lst,gens, ylog=False, view=False)
        visualize.plot_avg_outputs(avg_outputs,gens,ylog=False,view=False)

        visualize.plot_stats_avg_nobest(best2_fitness, avg2_fitness, std2_fitness, gens, ylog=False, view=False)
        visualize.plot_stats2_avg_nobest(median2_fitness, stat25_2, stat75_2, best2_fitness, gens, ylog=False, view=False)
        visualize.plot_head_balance_avg_nobest(mean2_lst, max2_lst, std2_lst, gens, ylog=False, view=False)

        fim_total = time.time()  # prints total execution time
        print('\nTotal Execution time: ' + str(round((fim_total - ini_total) / 60)) + ' minutes \n')
    elif mode == 'test':
        tfe = CustomEnvironment()

        #robot = tfe.make_robot()
        #render = Render()
        #img_path = 'robot.png'
        #render.render_robot(robot.body.core, img_path)

        with open('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/gecko_all_10/gecko_all_1/best_genome_1', 'rb') as fp:
            real_winner = pickle.load(fp)
        config_file = 'neat_config'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        fitness,states = tfe.simulate_v4(real_winner,config)
        print("Genome fitness: ", fitness)
        balance = tfe._head_balance(states)
        print("Head balance: ", balance)

    elif mode == 'train1':
        i = 0
        print("-----RUN " + str(i) + '-----')
        ini = time.time()
        tfe = CustomEnvironment()
        config_file = 'neat_config'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        #p = neat.Checkpointer.restore_checkpoint('neat/neat_test12xy11/checkpoint_0_49')

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=40, time_interval_seconds=None,
                                         filename_prefix='checkpoint_' + str(i) + '_'))
        # genomes = []
        # for i in p.population:
        #    k = (i, p.population[i])
        #    genomes.append(k)
        # Run for up to x generations.
        # print(genomes)

        # Run for up to x generations in parallel.
        # pe = neat.ParallelEvaluator(3, tfe.simulate_v4)
        # winner = p.run(pe.evaluate, 5)

        winner = p.run(tfe.simulate_v3, 40)
        # Display the winning genome (in population of last generation)
        print('\nBest genome:\n{!s}'.format(winner))

        ## Overall best genome
        real_winner = stats.best_genome()
        head_balance = tfe.get_head_balance()
        outputs = tfe.get_outputs()

        # Plots for fitness, network and speciation
        visualize.plot_stats(stats, cnt=i, ylog=False, view=False)
        visualize.plot_stats2(stats, cnt=i, ylog=False, view=False)
        visualize.plot_head_balance(stats, head_balance, cnt=i, ylog=False, view=False)
        visualize.plot_outputs(stats,outputs,cnt = i, ylog=False, view=False)

        visualize.plot_stats_nobest(stats, cnt=i, ylog=False, view=False)
        visualize.plot_stats2_nobest(stats, cnt=i, ylog=False, view=False)
        visualize.plot_head_balance_nobest(stats, head_balance, cnt=i, ylog=False, view=False)

        visualize.draw_net(config, real_winner, cnt=i, view=False)
        visualize.plot_species(stats, cnt=i, view=False)
        ##
        with open('best_genome_' + str(i), 'wb') as fp:
            pickle.dump(real_winner, fp)

        with open('winner_' + str(i), 'wb') as fp:
            pickle.dump(winner, fp)

        fim = time.time()  # prints execution time
        print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

if __name__ == "__main__":
    main()
