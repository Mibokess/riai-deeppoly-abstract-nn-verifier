from abc import ABC, abstractmethod
import torch


class Heuristic(ABC):

    def is_done(self, lower_bounds):
        """
        :param lower_bounds: the robustness property lower bounds
        """
        return True


    @abstractmethod
    def compute_lambda(self, lower_bounds, upper_bounds):
        pass



class Zero(Heuristic):

    def compute_lambda(self, lower_bounds, upper_bounds):
        return torch.zeros_like(lower_bounds)



class MinimizeArea(Heuristic):

    def compute_lambda(self, lower_bounds, upper_bounds):
        return upper_bounds > -lower_bounds



class Random(Heuristic):

    def is_done(self, lower_bounds):
        return False

    def compute_lambda(self, lower_bounds, upper_bounds):
        return torch.rand_like(lower_bounds)



class NetAndInputSpecific(Heuristic):

    def __init__(self, net, inputs):
        self._net = net
        self._inputs = inputs

    def compute_lambda(self, lower_bounds, upper_bound):
        raise NotImplementedError()



class HeuristicFactory:

    @staticmethod
    def default():
        return Zero()

    @staticmethod
    def create(heuristic, net, inputs):

        if heuristic is None:
            return HeuristicFactory.default()

        if heuristic is NetAndInputSpecific:
            return NetAndInputSpecific(net, inputs)

        return heuristic()

