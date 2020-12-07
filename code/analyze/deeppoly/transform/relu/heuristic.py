from abc import ABC, abstractmethod
import torch
import numpy as np


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
        self.__c = constants
        self.__i = 0

    def init(self, net, inputs):
        self.__i = 0

    def next(self, final_lower_bounds):
        self.__i += 1
        return self.__i < len(self.__c)

    def compute_lambda(self, lower_bounds, upper_bounds):
        lambda_low = self.__c[self.__i]
        return torch.ones_like(lower_bounds) * lambda_low


class Ensemble(Heuristic):
    
    def __init__(self, *heuristics):
        self._heuristics = list(heuristics)
        self.__i = 0

    def init(self, net, inputs):
        for h in self._heuristics:
            h.init(net, inputs)
        self.__i = 0

    def next(self, lower_bounds):
        if not self._heuristics[self.__i].next(lower_bounds):
            self.__i += 1
            return self.__i < len(self._heuristics)
        return True

    def compute_lambda(self, lower_bounds, upper_bounds):
        return self._heuristics[self.__i].compute_lambda(lower_bounds, upper_bounds)



class OptimizeConstantLambdaPerLayer(Heuristic):

    def __init__(self, max_nb_iterations=100, epsilon=1e-3, step_size=0.1, debug=False):
        """ Optimize one lambda lower bound per relu layer via Gradient ascent.
        :param max_nb_iterations: max nb of iteration per layer
        :param epsilon: stops iterating for the current layer when the gradient is lower than epsilon.
        """
        self._max_iter = max_nb_iterations
        self._epsilon = epsilon
        self._step = step_size
        self._debug = debug


    def init(self, net, inputs):
        self.relu_layer_indices = [i for i, l in enumerate(net.layers) if type(l) == torch.nn.ReLU]
        self._nb_relus = len(self.relu_layer_indices)
        self.relu_input_dims = [net.layers[i-1].out_features for i in self.relu_layer_indices]
        self._lambdas = np.random.rand(self._nb_relus)
        self._optimized_layer_idx = 0
        self._computed_layer_idx = 0
        self._iter = 0
        self._cost = -np.inf
        self._logs = []


    def next(self, lower_bounds):
        self._computed_layer_idx = 0

        cost = self._cost_function(lower_bounds)

        if self._iter == 0:
            self._cost = cost
            new_lambda = self._lambdas[self._optimized_layer_idx] + self._step
            self._lambdas[self._optimized_layer_idx] = new_lambda
            self._iter += 1
            return True

        diff = (cost - self._cost)
        grad = diff / self._lambdas[self._optimized_layer_idx]

        if (grad.abs() >= self._epsilon) and (self._iter < self._max_iter):
            new_lambda = self._lambdas[self._optimized_layer_idx] + grad * self._step
            self._lambdas[self._optimized_layer_idx] = new_lambda
            self._iter += 1
            self._cost = cost
            return True

        # done with optimizing current layer
        self._optimized_layer_idx += 1
        self._iter = 0
        self._cost = -np.inf
        return self._optimized_layer_idx < self._nb_relus


    def compute_lambda(self, lower_bounds, upper_bounds):
        lambdas = torch.ones_like(lower_bounds) * self._lambdas[self._computed_layer_idx]
        self._computed_layer_idx += 1
        return lambdas


    def _cost_function(self, lower_bounds):
        cost = (lower_bounds[lower_bounds <= 0]).square().sum()
        return cost

