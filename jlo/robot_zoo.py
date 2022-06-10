
from revolve2.core.modular_robot import Core, Body, Module, Brick, ActiveHinge
from direct_tree.direct_tree_genotype import DirectTreeGenotype
import math

"""
All the robots bodies are stored here
Use one of the functions to create your robot
"""


def make_ant_body() -> DirectTreeGenotype:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)
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

    return DirectTreeGenotype(body)

def make_gecko_body() -> DirectTreeGenotype:
    body = Body()
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

    return DirectTreeGenotype(body)

def make_super_ant_body() -> DirectTreeGenotype:
    body = Body()
    # head and first set of arms
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2)
    body.core.right.attachment.attachment = Brick(0.0)

    # second part of the body and arms
    body.core.back = ActiveHinge(math.pi / 2)
    body.core.back.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.left.attachment.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.right.attachment.attachment = Brick(0.0)

    # third part of the body and arms    
    body.core.back.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment.attachment = Brick(math.pi / 2)

    return DirectTreeGenotype(body)