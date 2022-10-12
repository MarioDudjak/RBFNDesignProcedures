import random as rnd
import numpy as np

from abc import ABCMeta, abstractmethod


class Crossover(metaclass=ABCMeta):
    """
    The Strategy interface for crossover operator implementations. The interface declares operations common to all supported crossover versions.

    The bio-inspired optimizer uses this interface to call the algorithm defined by the concrete crossover implementations.
    """

    @abstractmethod
    def mate(self, parents):
        pass


class SinglePointCrossover(Crossover):

    def __init__(self):
        self.name = "SinglePointCrossover"

    def mate(self, parents):
        parent_a, parent_b = parents

        crossover_point = rnd.randrange(1, len(parent_a) - 1)

        child_a = np.append(parent_a[:crossover_point], parent_b[crossover_point:])
        child_b = np.append(parent_b[:crossover_point], parent_a[crossover_point:])

        return [child_a, child_b]
