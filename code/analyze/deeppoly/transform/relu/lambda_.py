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

    def __init__(self, lambdas):
        self._lambdas = lambdas

    def compute_lambda(self, layer_id, mask, lower_bounds, upper_bounds):
        return torch.clamp(self._lambdas[layer_id][mask], 0.0, 1.0)