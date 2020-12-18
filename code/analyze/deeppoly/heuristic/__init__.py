import itertools
from abc import ABC, abstractmethod
import torch
import numpy as np
import time

from analyze.deeppoly.transform.relu.lambda_ import LambdaCalculator
from analyze.deeppoly.transform.relu.lambda_ import Matrix
from analyze.utils import Timer


class Heuristic(ABC):

    def __init__(self, timeout=None):
        self._timeout = timeout
        self._timer = Timer()

    def update_timeout(self, remaining_time):
        if self._timeout is None:
            self._timeout = remaining_time
        elif remaining_time is not None:
            self._timeout = min(self._timeout, remaining_time)

    def remaining_time(self):
        if self._timeout is None:
            return None
        return self._timeout - self._timer.elapsed_time()

    def timeout(self):
        return False if self._timeout is None else self._timer.elapsed_time() > self._timeout

    def run(self, deeppoly):
        self._timer.reset()
        return self._run(deeppoly)

    @abstractmethod
    def _run(self, deeppoly):
        """ :returns verified, ads, steps """
        pass


class Sequential(Heuristic):

    def __init__(self, heuristics, timeout, check_intersection=False):
        super().__init__(timeout)
        self._check_intersection = check_intersection
        self.heuristics = heuristics

    def _run(self, deeppoly):

        ad_intersect = None

        for h in self.heuristics:
            heuristic = Implicit.convert(h)
            heuristic.update_timeout(self.remaining_time())
            verified, ads, steps = heuristic.run(deeppoly)

            if not verified and self._check_intersection:
                robustness_ad = ads[-1]
                ad_intersect = robustness_ad if ad_intersect is None else ad_intersect.intersection(robustness_ad)
                verified = deeppoly.robustness.verify_bounds(ad_intersect.lower_bounds) if not verified else verified

            if verified or self.timeout():
                break

        return verified, ads, steps



class Implicit(Heuristic):

    def __init__(self, lambda_calculator):
        super(Implicit, self).__init__()
        self._lamba_calculator = lambda_calculator

    def _run(self, deeppoly):
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



class AdHocBackSubstitution(Heuristic):

    def __init__(self, heuristic, layer_ids=None, timeout=None, debug=False):
        """
            runs deeppoly with backsubstitution to each given layer_id
            :param layer_ids when set to None, the layers before the relu layers will be used
        """
        super().__init__(timeout)
        self._layer_ids = layer_ids
        self._heuristic = heuristic
        self.debug = debug

    def _run(self, deeppoly):
        verified, ads, steps = False, [], []
        ids = self._layer_ids
        if ids is None:
            relu_ids = deeppoly.get_relu_layer_ids()
            ids = [0] + list(map(lambda id: id - 1, relu_ids))

        for id in ids:
            if self.debug:
                print(f"running deep poly with backsubstitution to layer {id}")
            if id < deeppoly.get_nb_layers():
                deeppoly.set_backsubstitution_to(id)
                self._heuristic.update_timeout(self.remaining_time())
                verified, ads, steps = self._heuristic.run(deeppoly)
                if verified or self.timeout():
                    break

        return verified, ads, steps

