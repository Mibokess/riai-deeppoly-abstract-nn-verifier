import itertools
from abc import ABC, abstractmethod
import torch
import numpy as np
import time

from analyze.deeppoly.transform.relu.lambda_ import Matrix
from analyze.deeppoly.heuristic import Heuristic




def loss_mean(lower_bounds):
    return -(lower_bounds[lower_bounds < 0.0].mean())


def loss_squared(lower_bounds):
    return torch.square(lower_bounds[lower_bounds < 0.0]).mean()


def loss_max(lower_bounds):
    return -torch.min(lower_bounds[lower_bounds < 0.0])


def loss_false_label(lower_bounds, false_label):
    return -lower_bounds[false_label]




class Optimize(Heuristic):

    def __init__(self, net, true_label, false_label=None, timeout=10):
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
            if isinstance(net.layers[i - 1], torch.nn.ReLU):
                num_lambdas.append(layer.in_features)

        self.lambdas = list(map(lambda num_lambdas_layer: torch.rand(num_lambdas_layer, requires_grad=True), num_lambdas))
        self.num_lambdas = num_lambdas


    def run(self, deeppoly, elapsed_time=0):
        verified = False
        start_time = time.time()

        ads, steps = None, None

        def elapsed_time():
            return time.time() - start_time

        optimizer = torch.optim.SGD(self.lambdas, lr=0.01, momentum=0.3, weight_decay=0.1, nesterov=True)
        lambdas_reshaped = list(map(lambda tensor: tensor.reshape(len(tensor), 1), self.lambdas))
        matrix = Matrix(lambdas_reshaped)

        deeppoly.set_lambda_calculator(matrix)

        while elapsed_time() < self.timeout:
            optimizer.zero_grad()

            verified, ads, steps = deeppoly.forward()

            if verified:
                return verified, ads, steps

            if not self.false_label:
                loss = loss_mean(ads[-1].lower_bounds)
            else:
                loss = loss_false_label(ads[-1].lower_bounds, self.false_label)

            #print(loss)

            loss.backward()
            optimizer.step()
        return verified, ads, steps

