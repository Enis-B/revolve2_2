import random
import sys
import numpy as np
from typing import List, Tuple
#
# from .array_genotype_config import ArrayGenotypeConfig, ArrayCrossoverConfig
# from .array_genotype_utils import *
from .array_genotype import ArrayGenotype
from revolve2.core.modular_robot import Module, Core

def generate_child_genotype(parent_a: ArrayGenotype,
                            parent_b: ArrayGenotype,
                            P):
    """
    To implement the uniform crossover, the following python code can be used.
    A uniform_crossover function is defined where incoming arguments A & B represent the parents,
    P denotes the probability matrix, and returning A & B represent the children.
    It can be observed that the information between parents is exchanged at the indexes where probability is less than the threshold (0.5) to form children.
    https://medium.com/@samiran.bera/crossover-operator-the-heart-of-genetic-algorithm-6c0fdcb405c0
    """
    for i in range(len(P)):
        if P[i] < 0.5:
            temp = parent_a[i]
            parent_a[i] = parent_b[i]
            parent_b[i] = temp
    return parent_b  # or parent_a as the new genotype


def crossover(parents: List[ArrayGenotype]):
    """
       Creates an child (new brain) through crossover with two parents using uniform crossover
       :param parents: genotypes of the parents to be used for crossover
       :return: genotype result of the crossover
       """
    assert len(parents) == 2
    parent_a = parents[0]
    parent_b = parents[1]
    P = np.random.rand(10)
    new_genotype = generate_child_genotype(parent_a, parent_b, P)
    return new_genotype


# TODO: add config to fine tune the xover rate
# def crossover(parent_a: ArrayGenotype,
#               parent_b: ArrayGenotype,
#               conf: ArrayGenotypeConfig) \
#         -> ArrayGenotype:
#     """
#     Performs actual crossover between two brains-weights exchange, parent_a and parent_b.
#     :return: New genotype
#     """
