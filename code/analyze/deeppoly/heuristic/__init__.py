from abc import ABC, abstractmethod
import torch
import numpy as np
import time

from analyze.deeppoly.transform.relu.lambda_ import LambdaCalculator


class Heuristic(ABC):

    @abstractmethod
    def run(self, deeppoly, elapsed_time):
        pass


class Sequential(Heuristic):

    def __init__(self, heuristics, check_intersection=False, timeout=180):
        self._check_intersection = check_intersection
        self._timeout = timeout
        self._heuristics = heuristics

    def run(self, deeppoly, elapsed_time=0):
        verified = False
        start_time = time.time()
        def elapsed_time(): return time.time() - start_time
        ad_intersect = None

        for h in self._heuristics:

            heuristic = Implicit.convert(h)
            verified, ads, steps = heuristic.run(deeppoly, elapsed_time=elapsed_time())

            if not verified and self._check_intersection:
                robustness_ad = ads[-1]
                ad_intersect = robustness_ad if ad_intersect is None else ad_intersect.intersection(robustness_ad)
                verified = deeppoly.robustness.verify_bounds(
                    ad_intersect.lower_bounds) if not verified else verified

            timeout = elapsed_time() > self._timeout
            if verified or timeout:
                break

        return verified, ads, steps


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

    def __init__(self, lambda_calculator, args):
        h = (Implicit(lambda_calculator(a)) for a in args)
        super().__init__(h)


class Loop(Sequential):

    def __init__(self, lambda_calculator, n_repeat=None):
        if n_repeat is not None:
            h = (Implicit(lambda_calculator()) for _ in range(0, n_repeat))
        else:
            h = iter(int, Implicit(lambda_calculator()))
        super().__init__(h)


