
from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain, RelationalConstraints, GreaterThanConstraints, LowerThanConstraints
from analyze.deeppoly.transform import Transformer
from analyze.utils import TensorUtils


class AffineTransformer(Transformer):

    def __init__(self, layer):
        w = layer.weight.detach()
        self.weights = w
        self.weights_pos, self.weights_neg = TensorUtils.split_positive_negative(w)
        if layer.bias is not None:
            self.bias = layer.bias.detach().reshape((layer.out_features, 1))
        else:
            self.bias = torch.zeros((layer.out_features, 1))
        self.N = layer.out_features


    def _transform(self, ad, input):

        A_gt = torch.matmul(self.weights_pos, ad.greater_than.A) + torch.matmul(self.weights_neg, ad.lower_than.A)
        v_gt = self.bias + torch.matmul(self.weights_pos, ad.greater_than.v) + torch.matmul(self.weights_neg, ad.lower_than.v)
        gt = GreaterThanConstraints(A_gt, v_gt)

        A_lt = torch.matmul(self.weights_pos, ad.lower_than.A) + torch.matmul(self.weights_neg, ad.greater_than.A)
        v_lt = self.bias + torch.matmul(self.weights_pos, ad.lower_than.v) + torch.matmul(self.weights_neg, ad.greater_than.v)
        lt = LowerThanConstraints(A_lt, v_lt)

        upper_bound = gt.compute_bounds(input.lower_bounds, input.upper_bounds)
        lower_bound = lt.compute_bounds(input.lower_bounds, input.upper_bounds)

        ad_lin = AbstractDomain(lower_bound, upper_bound, lt, gt)
        return ad_lin


#TODO

class LinearTransformer(AffineTransformer):
    pass


class Conv2dTransformer(AffineTransformer):
    pass



