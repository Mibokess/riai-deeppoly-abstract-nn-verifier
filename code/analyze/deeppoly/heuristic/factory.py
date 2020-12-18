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
            return self.optimize(loss_fn=LF.loss_sum, timeout=170)

        # return H.AdHocBackSubstitution(self.optimize(loss_fn=LF.loss_sum, timeout=30, debug=False))
        return self.optimize(loss_fn=LF.loss_sum, timeout=170)

    @property
    def simplest(self):
        return H.Sequential([
            return H.Sequential([
                    L.MinimizeArea(),
                    L.Zonotope(),
                    H.IterateOverArgs(L.Constant, np.linspace(0, 1, 10)),
                    # O.ReduceParams(inspector.get_relu_input_sizes(), num_params, mapping_type, LF.loss_sum, timeout=180, debug=False),
                    O.Optimize(inspector.get_relu_input_sizes(), LF.loss_sum, timeout=180, debug=False),
            ], timeout=180)
                    H.IterateOverArgs(L.Constant, np.linspace(0, 1, 10))
        ], timeout=170)


        return O.Optimize(inspector.get_relu_input_sizes(), LF.loss_sum, timeout=180, debug=False)


    def optimize(self, loss_fn=LF.loss_sum, timeout=170, debug=False):
        inspector = NetworkInspector(self.net, self.inputs)
        return Optimize(inspector.get_relu_input_sizes(), loss_fn, timeout=timeout, debug=debug)
