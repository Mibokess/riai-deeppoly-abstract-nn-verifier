
from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain, RelationalConstraints, GreaterThanConstraints, LowerThanConstraints
from analyze.deeppoly.transform import Transformer
from analyze.utils import TensorUtils


class LinearTransformer(Transformer):

    def __init__(self, layer, backprop=False):
        self.backprop = backprop

        w = layer.weight.detach()
        self.weights = w
        self.weights_pos, self.weights_neg = TensorUtils.split_positive_negative(w)
        if layer.bias is not None:
            self.bias = layer.bias.detach().reshape((layer.out_features, 1))
        else:
            self.bias = torch.zeros((layer.out_features, 1))
        self.N = layer.out_features


    def _transform(self, ads):
        greater_than = GreaterThanConstraints(self.weights, self.bias)
        lower_than = LowerThanConstraints(self.weights, self.bias)

        if True or self.backprop:
            lower_bounds, upper_bounds = self.compute_bounds(ads, lower_than, greater_than)
        else:
            ad = ads[-1]
            upper_bounds = greater_than.compute_bounds(ad.lower_bounds, ad.upper_bounds)
            lower_bounds = lower_than.compute_bounds(ad.lower_bounds, ad.upper_bounds)

        ad_lin = AbstractDomain(lower_bounds, upper_bounds, lower_than, greater_than)
        return ad_lin




class Conv2dTransformer(Transformer):
    pass

