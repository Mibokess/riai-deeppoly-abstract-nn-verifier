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
        return -lower_bounds[lower_bounds < 0.0].sum() - lower_bounds[lower_bounds > 0.0].sum()

    @staticmethod
    def loss_squared(lower_bounds):
        return torch.square(lower_bounds[lower_bounds < 0.0]).mean()

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

        self.lambdas = list(map(lambda _: torch.rand((1,), requires_grad=True), range(sum(self.num_lambdas))))
        #self.lambdas = list(map(lambda n: torch.rand((n, 1), requires_grad=True), self.num_lambdas))
        self.debug = debug

    def _run(self, deeppoly):
        verified = False
        ads, steps = None, None

        optimizer = torch.optim.SGD(self.lambdas, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)
        matrix = Matrix(self.lambdas, self.num_lambdas)
        deeppoly.set_lambda_calculator(matrix)

        deeppoly.forward()

        indices = torch.cat(matrix.changed_lambdas_index)
        optimizer = torch.optim.SGD([self.lambdas[i] for i in indices], lr=0.1, momentum=0.0, weight_decay=0.0, nesterov=False)
        matrix.changed = False

        best_loss = float("inf")
        rounds_not_improved = 0

        while not False:
            optimizer.zero_grad()

            for lambdas_layer in self.lambdas:
                lambdas_layer.data = torch.clamp(lambdas_layer, 0.0, 1.0)

            verified, ads, steps = deeppoly.forward()

            if verified:
                return verified, ads, steps

            loss = self.loss_fn(ads[-1].lower_bounds)
            self.log(loss, ads[-1].lower_bounds)
            print(loss)
            loss.backward()
            optimizer.step()

            if loss >= best_loss - 0.001:
                rounds_not_improved += 1
            else:
                rounds_not_improved = 0
                best_loss = loss

            if rounds_not_improved > 50:
                return verified, ads, steps

            if matrix.changed:
                indices = torch.cat(matrix.changed_lambdas_index)
                for l in self.lambdas: l.detach()
                optimizer = torch.optim.SGD(list(filter(lambda x: x.grad > 0.001, [self.lambdas[i].requires_grad_() for i in indices])), lr=0.1, momentum=0, weight_decay=0.0, nesterov=False)

                matrix.changed = False
                matrix.changed_lambdas_index = []
            else:
                optimizer = torch.optim.SGD(
                    list(filter(lambda x: x.grad > 0.001, self.lambdas)), lr=0.1,
                    momentum=0, weight_decay=0.0, nesterov=False)

        return verified, ads, steps

    def log(self, loss, lower_bounds):
        if self.debug:
            print(f"loss={loss:.4f}, lower_bounds #_neg={len(lower_bounds[lower_bounds<0])}, min_neg={lower_bounds[lower_bounds < 0].min():.2f}, max_neg={lower_bounds[lower_bounds < 0].max():.2f}", end="")
            if len(lower_bounds[lower_bounds > 0]) > 0:
                print(f", min_pos={lower_bounds[lower_bounds > 0].min():.2f}", end=" - ")
            else:
                print(end=" - ")

            for i, lambdas in enumerate(self.lambdas):
                nb_to_0 = lambdas[(lambdas <= 0)].flatten().size()[0] / lambdas.flatten().size()[0]
                nb_to_1 = lambdas[(lambdas >= 1)].flatten().size()[0] / lambdas.flatten().size()[0]
                print(f"layer {i} lambdas<= 0: {nb_to_0 * 100:.0f}% - lambdas>=1: {nb_to_1 * 100:.0f}%", end=" - ")

            print("")
