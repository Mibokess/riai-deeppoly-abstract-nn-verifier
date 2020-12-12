import itertools
from abc import ABC, abstractmethod
import torch
import numpy as np
import time

from analyze.deeppoly.transform.relu.lambda_ import LambdaCalculator
from analyze.deeppoly.transform.relu.lambda_ import Matrix


class Heuristic(ABC):

    @abstractmethod
    def run(self, deeppoly, elapsed_time):
        pass


class Sequential(Heuristic):

    def __init__(self, heuristics, timeout, check_intersection=False):
        self._check_intersection = check_intersection
        self._timeout = timeout
        self._heuristics = heuristics

    def _set_timeout(self, timeout):
        self._timeout = timeout

    def run(self, deeppoly, elapsed_time=0):
        verified = False
        start_time = time.time()

        def elapsed_time():
            return time.time() - start_time

        ad_intersect = None

        for h in self._heuristics:

            heuristic = Implicit.convert(h)
            elapsed = elapsed_time()
            self._update_timeout(heuristic, elapsed)
            verified, ads, steps = heuristic.run(deeppoly, elapsed_time=elapsed)

            if not verified and self._check_intersection:
                robustness_ad = ads[-1]
                ad_intersect = robustness_ad if ad_intersect is None else ad_intersect.intersection(robustness_ad)
                verified = deeppoly.robustness.verify_bounds(
                    ad_intersect.lower_bounds) if not verified else verified

            timeout = elapsed_time() > self._timeout
            if verified or timeout:
                break

        return verified, ads, steps

    def _update_timeout(self, heuristic, elapsed_time):
        if isinstance(heuristic, Sequential):
            if heuristic._timeout is None:
                heuristic._set_timeout(self._timeout)
            else:
                timeout = min(heuristic._timeout, self._timeout - elapsed_time)
                heuristic._set_timeout(timeout)


class Implicit(Heuristic):

    def __init__(self, lambda_calculator):
        self._lamba_calculator = lambda_calculator

    def run(self, deeppoly, elapsed_time=0):
        deeppoly.set_lambda_calculator(self._lamba_calculator)
        return deeppoly.forward()

    @staticmethod
    def convert(h):
        heuristic = Implicit(h) if isinstance(h, LambdaCalculator) else h
        return heuristic


class IterateOverArgs(Sequential):

    def __init__(self, lambda_calculator, args, timeout=None):
        h = (Implicit(lambda_calculator(a)) for a in args)
        super().__init__(h, timeout)


class Loop(Sequential):

    def __init__(self, lambda_calculator, n_repeat=None, timeout=None):
        if n_repeat is None:
            h = itertools.repeat(Implicit(lambda_calculator()))
        else:
            h = itertools.repeat(Implicit(lambda_calculator()), times=n_repeat)
        super().__init__(h, timeout)


class Optimize(Heuristic):
    def __init__(self, net, true_label, false_label=None, timeout=180):
        self.net = net
        self.timeout = timeout

        self.labels = torch.zeros(net.layers[-1].out_features, dtype=torch.bool)
        self.labels[true_label] = True

        self.false_label = None
        if false_label:
            self.false_label = false_label
            if false_label > true_label:
                self.false_label -= 1

        num_lambdas = []

        for i, layer in enumerate(net.layers):
            if i > 0 and isinstance(net.layers[i - 1], torch.nn.ReLU):
                num_lambdas.append(layer.in_features)

        self.lambdas = list(
            map(lambda num_lambdas_layer: torch.rand(num_lambdas_layer, requires_grad=True), num_lambdas))
        self.num_lambdas = num_lambdas

    def run(self, deeppoly, elapsed_time=0):
        verified = False
        start_time = time.time()

        def elapsed_time():
            return time.time() - start_time

        optimizer = torch.optim.SGD(self.lambdas, lr=0.01)
        matrix = Matrix(list(map(lambda tensor: tensor.reshape(len(tensor), 1), self.lambdas)))

        while elapsed_time() < self.timeout:
            optimizer.zero_grad()

            deeppoly.set_lambda_calculator(matrix)

            verified, ads, steps = deeppoly.forward()
            if verified:
                return verified, ads, steps

            if not self.false_label:
                loss = loss_mean(ads[-1].lower_bounds)
            else:
                loss = loss_false_label(ads[-1].lower_bounds, self.false_label)

            loss.backward()
            optimizer.step()

        return verified, ads, steps


def loss_mean(lower_bounds):
    return -(lower_bounds[lower_bounds < 0.0].mean())


def loss_squared(lower_bounds):
    return torch.square(lower_bounds[lower_bounds < 0.0]).mean()


def loss_max(lower_bounds):
    return -torch.min(lower_bounds[lower_bounds < 0.0])


def loss_false_label(lower_bounds, false_label):
    return -lower_bounds[false_label]