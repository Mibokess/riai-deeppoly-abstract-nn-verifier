from analyze.utils import NetworkInspector
from networks import FullyConnected, Conv
import analyze.deeppoly.heuristic as H
import analyze.deeppoly.heuristic.optimize as O
from analyze.deeppoly.heuristic.optimize import LossFunctions as LF
import analyze.deeppoly.transform.relu.lambda_ as L
import numpy as np


class HeuristicFactory:

    def __init__(self, name, net, inputs, true_label):
        self.name = name
        self.net = net
        self.inputs = inputs
        self.true_label = true_label

    def create(self):
        inspector = NetworkInspector(self.net, self.inputs)

        if isinstance(self.net, Conv):
            return H.Sequential([
                    L.MinimizeArea(),
                    L.Zonotope(),
                    H.IterateOverArgs(L.Constant, np.linspace(0, 1, 10)),
                    # O.ReduceParams(inspector.get_relu_input_sizes(), num_params, mapping_type, LF.loss_sum, timeout=180, debug=False),
                    O.Optimize(inspector.get_relu_input_sizes(), LF.loss_sum, timeout=180, debug=False),
            ], timeout=180)


        return O.Optimize(inspector.get_relu_input_sizes(), LF.loss_sum, timeout=180, debug=False)

