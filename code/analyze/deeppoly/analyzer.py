import torch
import networks
from analyze import Analyzer
from analyze.deeppoly.transform.factory import TransformerFactory
from analyze.deeppoly.transform.relu.heuristic import MinimizeArea
from analyze.deeppoly.domain import AbstractDomain
import numpy as np
import time



class RobustnessProperty:

    def __init__(self, nb_features, true_label, comparison_fn):
        self._compare = comparison_fn
        self._transformer = self._create_transformer(nb_features, true_label)

    def verify(self, ads):
        ad = self._transformer.transform(ads)
        verified = torch.all(self._compare(ad.lower_bounds, 0))
        return verified, ad

    @staticmethod
    def _create_transformer(nb_labels, true_label):
        layer = torch.nn.Linear(nb_labels, nb_labels - 1, False)
        weights = torch.zeros((nb_labels - 1, nb_labels))
        weights.fill_diagonal_(-1)
        weights[:, true_label] = 1
        layer.weight = torch.nn.Parameter(weights)
        transformer = TransformerFactory.create(layer)
        return transformer


class DeepPoly(Analyzer):

    def __init__(self,
                 relu_heuristics=MinimizeArea(),
                 check_domain_intersect=True,
                 save_intermediate_steps=False,
                 timeout=180):
        self._save_intermediate_steps = save_intermediate_steps
        self._heuristic = relu_heuristics
        self._timeout = timeout
        self._check_intersect = check_domain_intersect


    def verify(self, net, inputs, eps, true_label, domain_bounds=[0, 1], robustness_fn=torch.greater):
        ini = AbstractDomain.create(inputs, eps, domain_bounds=domain_bounds)
        self._heuristic.init(net, inputs)

        transformers = []
        for layer in net.layers:
            transformer = TransformerFactory.create(layer, self._heuristic)
            transformers.append(transformer)

        robustness = RobustnessProperty(layer.out_features, true_label, robustness_fn)
        return self._run(transformers, robustness, ini)


    def _run(self, transformers, robustness, ad_input):

        verified = False
        done = False
        timeout = False
        start_time = time.time()
        ad_intersect = None

        while not verified and not done and not timeout:
            ads, steps = self._forward(transformers, ad_input)
            verified, robustness_ad = robustness.verify(ads)

            if not verified and self._check_intersect:
                ad_last = ads[-1]
                ad_intersect = ad_last if ad_intersect is None else ad_intersect.intersection(ad_last)
                verified, robustness_ad_intersect = robustness.verify(ads[:-1] + [ad_intersect]) if not verified else verified

            if not verified:
                done = not self._heuristic.next(robustness_ad.lower_bounds)

            timeout = (time.time() - start_time) > self._timeout

        return verified, ads, steps


    def _forward(self, transformers, ad_input):
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

        return ads, steps



