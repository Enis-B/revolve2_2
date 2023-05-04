from __future__ import annotations

from typing import List, Optional, Tuple

from ._not_finalized_error import NotFinalizedError


class Module:
    _children: List[Optional[Module]]
    # the rotation relative to the parent
    _rotation: float
    # the rotation regarding the ground: for differentiating horizontal or vertical joints
    _absolute_rotation: int

    # The following members are initialized by the ModularRobot finalize function:
    _id: Optional[int]
    _parent: Optional[Module]
    _parent_child_index: Optional[int]

    def __init__(self, num_children: int, rotation: float):
        self._children = [None] * num_children
        self._rotation = rotation
        self.rgb = None
        self.turtle_direction = None
        self.substrate_coordinates = None

        self._id = None
        self._parent = None
        self._parent_child_index = None

    @property
    def children(self) -> List[Optional[Module]]:
        return self._children

    def has_children(self) -> bool:
        """
        Check wheter module has children
        :return: True if module has children
        """
        has_children = False
        if self._children == {1: None}: return False

        for i, child in enumerate(self._children):
            if child is not None:
                has_children = True

        return has_children

    @property
    def rotation(self) -> float:
        return self._rotation

    @property
    def id(self) -> int:
        if self._id is None:
            raise NotFinalizedError()
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        if self._id is not None:
            raise RuntimeError("Cannot set id twice.")
        self._id = id

    def neighbours(self, within_range: int) -> List[Module]:
        if self._id is None:
            raise NotFinalizedError()

        out_neighbours: List[Module] = []

        open_nodes: List[Tuple[Module, Optional[Module]]] = [
            (self, None)
        ]  # (module, came_from)

        for _ in range(within_range):
            new_open_nodes: List[Tuple[Module, Optional[Module]]] = []
            for (open_node, came_from) in open_nodes:
                neighbours = [
                    mod
                    for mod in open_node.children + [open_node._parent]
                    if mod is not None
                    and (came_from is None or mod.id is not came_from.id)
                ]
                out_neighbours += neighbours
                new_open_nodes += list(zip(neighbours, [open_node] * len(neighbours)))
            open_nodes = new_open_nodes

        return out_neighbours
