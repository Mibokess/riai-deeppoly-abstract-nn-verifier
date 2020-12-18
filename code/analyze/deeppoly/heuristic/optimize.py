import itertools
from abc import ABC, abstractmethod
import torch
import numpy as np
import time

from analyze.deeppoly.transform.relu.lambda_ import Matrix
from analyze.deeppoly.heuristic import Heuristic


class LossFunctions:

    @staticmethod
    def loss_mean(lower_bounds):
        return -lower_bounds[lower_bounds < 0.0].mean()

    @staticmethod
    def loss_sum(lower_bounds):
        return -lower_bounds[lower_bounds < 0.0].sum()

    @staticmethod
    def loss_diff(lower_bounds):
        up_pos = lower_bounds[lower_bounds > 0.0]
        up_mean = 0
        if up_pos.shape[0] > 0:
            up_mean = up_pos.mean()
        return -lower_bounds[lower_bounds < 0.0].sum() - up_mean

    @staticmethod
    def loss_squared(lower_bounds):
        return torch.square(lower_bounds[lower_bounds < 0.0]).sum()

    @staticmethod
    def loss_max(lower_bounds):
        return -torch.min(lower_bounds)

    @staticmethod
    def loss_false_label(lower_bounds, false_label):
        return -lower_bounds[false_label]



class Optimize(Heuristic):

    def __init__(self, num_lambdas, loss_fn, timeout, debug=False):
        super().__init__(timeout)
        self.num_lambdas = num_lambdas
        self.loss_fn = loss_fn
        self.debug = debug

    def _run(self, deeppoly):
        verified = False
        ads, steps = None, None
        self.lambdas = list(map(lambda n: torch.rand((n, 1), requires_grad=True), self.num_lambdas))
        optimizer = torch.optim.SGD(self.lambdas, lr=0.5, momentum=0.3, weight_decay=0.0, nesterov=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=self.debug, patience=5, cooldown=10)
        matrix = Matrix(self.lambdas)
        deeppoly.set_lambda_calculator(matrix)

        best_loss = float("inf")
        steps_since_improvement = 0

        while not self.timeout():
            optimizer.zero_grad()

            verified, ads, steps = deeppoly.forward()

            if verified:
                return verified, ads, steps

            loss = self.loss_fn(ads[-1].lower_bounds)
            self.log(loss, ads[-1].lower_bounds)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            for lambdas_layer in self.lambdas:
                lambdas_layer.data = torch.clamp(lambdas_layer.data, 0.0, 1.0)

            if loss >= best_loss:
                steps_since_improvement += 1
            else:
                best_loss = loss
                steps_since_improvement = 0

            if steps_since_improvement > 10000:
                return verified, ads, steps

        return verified, ads, steps


    def log(self, loss, lower_bounds):
        if self.debug:
            print(f"loss={loss:.6f}, lower_bounds #_neg={len(lower_bounds[lower_bounds<0])}, min_neg={lower_bounds[lower_bounds < 0].min():.6f}, max_neg={lower_bounds[lower_bounds < 0].max():.6f}", end="")
            if len(lower_bounds[lower_bounds > 0]) > 0:
                print(f", min_pos={lower_bounds[lower_bounds > 0].min():.6f}", end=" - ")
            else:
                print(end=" - ")

            for i, lambdas in enumerate(self.lambdas):
                nb_to_0 = lambdas[(lambdas <= 0)].flatten().size()[0] / lambdas.flatten().size()[0]
                nb_to_1 = lambdas[(lambdas >= 1)].flatten().size()[0] / lambdas.flatten().size()[0]
                print(f"layer {i} lambdas<= 0: {nb_to_0 * 100:.0f}% - lambdas>=1: {nb_to_1 * 100:.0f}%", end=" - ")

            print("")




class ReduceParams(Heuristic):

    def __init__(self, num_lambdas, num_parameters, mapping_type, loss_fn, timeout, debug=False, epsilon=1e-6):
        super().__init__(timeout)

        self.loss_fn = loss_fn
        self.debug = debug
        self.epsilon = epsilon

        if num_parameters is None:
            self.num_parameters = num_lambdas
            self.lambda_calc_factory = lambda params: Matrix(params)
        else:
            self.num_parameters = num_parameters

            mapping = []
            for i, (expected_dim, opti_dim) in enumerate(zip(num_lambdas, num_parameters)):
                if (type(mapping_type) is list and mapping_type[i] == "random") or (mapping_type == "random"):
                    idx = torch.randint(0, opti_dim, (expected_dim,))
                else:
                    nb_repeats = expected_dim // opti_dim
                    idx = torch.arange(0, opti_dim).repeat_interleave(nb_repeats)
                mapping.append(idx)

            self.lambda_calc_factory = lambda params: Map(params, mapping)


    def _run(self, deeppoly):
        verified = False
        ads, steps = None, None
        self.params = list(map(lambda n: torch.rand((n, 1), requires_grad=True), self.num_parameters))
        optimizer = torch.optim.SGD(self.params, lr=0.1, momentum=0.0, weight_decay=0.0, nesterov=False)

        lambda_calc = self.lambda_calc_factory(self.params)
        deeppoly.set_lambda_calculator(lambda_calc)
        delta_loss = np.inf
        prev_loss = 0.0

        while not self.timeout() and (delta_loss > self.epsilon):
            optimizer.zero_grad()

            verified, ads, steps = deeppoly.forward()

            if verified:
                return verified, ads, steps

            loss = self.loss_fn(ads[-1].lower_bounds)
            self.log(loss, ads[-1].lower_bounds)
            loss.backward()
            optimizer.step()

            delta_loss = torch.abs(loss - prev_loss)/torch.abs(loss)
            prev_loss = loss

        return verified, ads, steps


    def log(self, loss, lower_bounds):
        if self.debug:
            print(f"loss={loss:.6f}, lower_bounds #_neg={len(lower_bounds[lower_bounds<0])}, min_neg={lower_bounds[lower_bounds < 0].min():.6f}, max_neg={lower_bounds[lower_bounds < 0].max():.6f}", end="")
            if len(lower_bounds[lower_bounds > 0]) > 0:
                print(f", min_pos={lower_bounds[lower_bounds > 0].min():.6f}", end=" - ")
            else:
                print(end=" - ")

            for i, lambdas in enumerate(self.params):
                nb_to_0 = lambdas[(lambdas <= 0)].flatten().size()[0] / lambdas.flatten().size()[0]
                nb_to_1 = lambdas[(lambdas >= 1)].flatten().size()[0] / lambdas.flatten().size()[0]
                print(f"layer {i} lambdas<= 0: {nb_to_0 * 100:.0f}% - lambdas>=1: {nb_to_1 * 100:.0f}%", end=" - ")

            print("")




class RetriggerWhenTight(Heuristic):

    def __init__(self, optimize, n_repeats, threshold=0.10, timeout=None, debug=False):
        super().__init__(timeout)
        self.optimize = optimize
        self.n_repeats = n_repeats
        self.threshold = threshold
        self.debug = debug

    def _run(self, deeppoly):

        iter = 0
        while not self.timeout() and iter < self.n_repeats:

            self.optimize.update_timeout(self.remaining_time())

            verified, ads, steps = self.optimize.run(deeppoly)

            lower_bounds = ads[-1].lower_bounds
            if verified or lower_bounds[lower_bounds < 0].size()[0] > 1:
                return verified, ads, steps

            value = lower_bounds[lower_bounds < 0][0]
            if value < self.threshold:
                iter += 1
                if self.debug:
                    print(f"retriggering #{iter}")


        return verified, ads, steps


