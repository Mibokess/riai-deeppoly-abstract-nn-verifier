from analyze.utils import NetworkInspector
from networks import FullyConnected, Conv
import analyze.deeppoly.heuristic as H
from analyze.deeppoly.heuristic.optimize import Optimize, LossFunctions as LF
import analyze.deeppoly.transform.relu.lambda_ as L
import numpy as np


class HeuristicFactory:

    def __init__(self, net, inputs, true_label):
        self.net = net
        self.inputs = inputs
        self.true_label = true_label


    def create(self):

        """
        heuristics = []
        timeout_global = 30.0

        timeout = timeout_global / net.layers[-1].out_features

        for i in range(net.layers[-1].out_features):
            if i != true_label:
                heuristics.append(H.Optimize(net, true_label, i, timeout))

        heuristic = H.Sequential(heuristics, timeout_global, True)
        """

        if isinstance(self.net, Conv):
            return self.optimize(loss_fn=LF.loss_sum, timeout=170)

        # return H.AdHocBackSubstitution(self.optimize(loss_fn=LF.loss_sum, timeout=30, debug=False))
        return self.optimize(loss_fn=LF.loss_sum, timeout=170)

    @property
    def simplest(self):
        return H.Sequential([
                    L.MinimizeArea(),
                    L.Zonotope(),
                    H.IterateOverArgs(L.Constant, np.linspace(0, 1, 10))
        ], timeout=170)


    @property
    def desperate(self):
        return H.Loop(L.Random, timeout=30)


    def optimize(self, loss_fn=LF.loss_sum, timeout=170, debug=False):
        inspector = NetworkInspector(self.net, self.inputs)
        return Optimize(inspector.get_relu_input_sizes(), loss_fn, timeout=timeout, debug=debug)
