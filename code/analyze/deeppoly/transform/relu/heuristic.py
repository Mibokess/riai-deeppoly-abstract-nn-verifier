from abc import ABC, abstractmethod
import torch


class Heuristic(ABC):

    def init(self, net, inputs):
        pass

    def next(self, final_lower_bounds):
        """
        :param final_lower_bounds: the robustness property lower bounds
        :returns False when heuristics does not have any further step
        """
        return False
    
    @abstractmethod
    def compute_lambda(self, lower_bounds, upper_bounds):
        pass


class Zero(Heuristic):

    def compute_lambda(self, lower_bounds, upper_bounds):
        return torch.zeros_like(lower_bounds)


class MinimizeArea(Heuristic):

    def compute_lambda(self, lower_bounds, upper_bounds):
        l = upper_bounds > -lower_bounds
        return l.type(torch.FloatTensor)


class Zonotope(Heuristic):

    def compute_lambda(self, lower_bounds, upper_bounds):
        l = upper_bounds / (upper_bounds - lower_bounds)
        return l


class Random(Heuristic):

    def next(self, lower_bounds):
        return True

    def compute_lambda(self, lower_bounds, upper_bounds):
        return torch.rand_like(lower_bounds)


class Constant(Heuristic):

    def __init__(self, constants):
        """
        :param a list of values for lambda
        """
        self._c = constants
        self._i = 0

    def init(self, net, inputs):
        self._i = 0

    def next(self, final_lower_bounds):
        self._i += 1
        return self._i < len(self._c)

    def compute_lambda(self, lower_bounds, upper_bounds):
        return torch.ones_like(lower_bounds) * self._c[self._i]


class Ensemble(Heuristic):
    
    def __init__(self, *heuristics):
        self._heuristics = list(heuristics)
        self._i = 0

    def init(self, net, inputs):
        self._i = 0

    def next(self, lower_bounds):
        self._i += 1
        return self._i < len(self._heuristics)

    def compute_lambda(self, lower_bounds, upper_bounds):
        return self._heuristics[self._i].compute_lambda(lower_bounds, upper_bounds)

