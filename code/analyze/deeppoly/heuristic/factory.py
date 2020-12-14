from networks import FullyConnected, Conv
import analyze.deeppoly.heuristic as H
from analyze.deeppoly.heuristic.optimize import Optimize
import analyze.deeppoly.transform.relu.lambda_ as L
import numpy as np


class HeuristicFactory:

    @staticmethod
    def create(net, true_label):

        simplest = HeuristicFactory.simplest()
        desperate = H.Loop(L.Random, timeout=30)

        """
        heuristics = []
        timeout_global = 30.0

        timeout = timeout_global / net.layers[-1].out_features

        for i in range(net.layers[-1].out_features):
            if i != true_label:
                heuristics.append(H.Optimize(net, true_label, i, timeout))

        heuristic = H.Sequential(heuristics, timeout_global, True)
        """

        if isinstance(net, Conv):
            # in case we need to put different heuristic per net...
            return H.Sequential(simplest.heuristics, timeout=180)

        optimize = Optimize(net, true_label, timeout=30)
        # return H.Sequential(simplest.heuristics + [optimize], timeout=180)
        return optimize


    @staticmethod
    def simplest():
        return H.Sequential([
                    L.MinimizeArea(),
                    L.Zonotope(),
                    H.IterateOverArgs(L.Constant, np.linspace(0, 1, 10))
        ], timeout=180)
