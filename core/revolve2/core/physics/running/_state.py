"""
State class and subclasses used by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pyrr import Quaternion, Vector3


@dataclass
class ActorState:
    """
    State of an actor.
    """

    position: Vector3
    orientation: Quaternion

    def serialize(self) -> StaticData:
        return {
            "position": [
                float(self.position.x),
                float(self.position.y),
                float(self.position.z),
            ],
            "orientation": [
                float(self.orientation.x),
                float(self.orientation.y),
                float(self.orientation.z),
                float(self.orientation.w),
            ],
        }

    @classmethod
    def deserialize(cls, data: StaticData) -> ActorState:
        raise NotImplementedError()

@dataclass
class EnvironmentState:
    """
    State of an environment.
    """

    actor_states: List[ActorState]


@dataclass
class RunnerState:
    """
    State of a runner.
    """

    time_seconds: float
    envs: List[EnvironmentState]
