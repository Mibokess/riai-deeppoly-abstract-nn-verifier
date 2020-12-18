from abc import ABC, abstractmethod
import torch
from torch.nn import *
import numpy as np


class LambdaCalculator(ABC):

    @abstractmethod
    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        pass


class Zero(LambdaCalculator):

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        return torch.zeros_like(lower_bounds)


class MinimizeArea(LambdaCalculator):

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        l = upper_bounds > -lower_bounds
        return l.type(torch.FloatTensor)


class Zonotope(LambdaCalculator):

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        l = upper_bounds / (upper_bounds - lower_bounds)
        return l


class Random(LambdaCalculator):

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        return torch.rand_like(lower_bounds)


class Constant(LambdaCalculator):

    def __init__(self, constant):
        self.c = constant

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        return torch.ones_like(lower_bounds) * self.c


class Matrix(LambdaCalculator):

    def __init__(self, lambdas, num_lambdas, changed=True):
        self.lambdas = lambdas
        self._num_lambdas = num_lambdas
        self.changed_lambdas_index = []
        self.changed = changed

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        layer_index = sum(self._num_lambdas[:layer_id])
        lambdas = self.lambdas[layer_index:layer_index + self._num_lambdas[layer_id]]

        if len(self.changed_lambdas_index) == len(self._num_lambdas):
            if mask.sum() != len(self.changed_lambdas_index[layer_id]):
                self.changed = True
        else:
            self.changed_lambdas_index.append(torch.nonzero(mask) + layer_index)

        return torch.cat(lambdas)[mask]
