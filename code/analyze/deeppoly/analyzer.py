import torch
import networks
from analyze import Analyzer
from analyze.deeppoly.transform.factory import TransformerFactory
from analyze.deeppoly.transform.relu.heuristic import MinimizeArea
from analyze.deeppoly.domain import AbstractDomain
import numpy as np
import time



class DeepPoly(Analyzer):

    def __init__(self, relu_heuristics=MinimizeArea(), save_intermediate_steps=False, timeout=180):
        self._save_intermediate_steps = save_intermediate_steps
        self._heuristic = relu_heuristics
        self._timeout = timeout


    def verify(self, net, inputs, eps, true_label, domain_bounds=[0, 1], robustness_fn=torch.greater):
        ini = AbstractDomain.create(inputs, eps, domain_bounds=domain_bounds)
        self._heuristic.init(net, inputs)

        transformers = []
        for layer in net.layers:
            transformer = TransformerFactory.create(layer, self._heuristic)
            transformers.append(transformer)

        transformer = self.create_final_transformer(layer.out_features, true_label)
        transformers.append(transformer)

        return self._run(transformers, ini, robustness_fn)


    def _run(self, transformers, ad_input, robustness_fn):
        verified = False
        done = False
        timeout = False
        start_time = time.time()
        while not verified and not done and not timeout:
            verified, ads, steps = self._forward(transformers, ad_input, robustness_fn)
            lower_bounds = ads[-1].lower_bounds
            if not verified:
                done = not self._heuristic.next(lower_bounds)
            timeout = (time.time() - start_time) > self._timeout

        return verified, ads, steps


    def _forward(self, transformers, ad_input, robustness_fn):
        ads = [ad_input]
        steps = ["input"]
        for transformer in transformers:
            ad = transformer.transform(ads)
            if self._save_intermediate_steps:
                ads.append(ad)
            else:
                if not ad.has_constraints():
                    ini = ad
                    ads = [ad]
                else:
                    ads = [ini, ad]
            steps.append(transformer.__class__.__name__)

        verified = torch.all(robustness_fn(ad.lower_bounds, 0))
        return verified, ads, steps


    def create_final_transformer(self, prev_layer_size, true_label):
        layer = torch.nn.Linear(prev_layer_size, prev_layer_size-1, False)
        weights = torch.zeros((prev_layer_size-1, prev_layer_size))
        weights.fill_diagonal_(-1)
        weights[:, true_label] = 1
        layer.weight = torch.nn.Parameter(weights)
        transformer = TransformerFactory.create(layer)
        return transformer


