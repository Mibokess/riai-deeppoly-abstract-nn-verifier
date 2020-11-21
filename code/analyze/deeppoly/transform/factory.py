from analyze.deeppoly.transform.affine import AffineTransformer
from analyze.deeppoly.transform.preprocess import NormalizeTransformer, FlattenTransformer
from analyze.deeppoly.transform.relu import ReluTransformer

import networks
import torch


class TransformerFactory:

    @staticmethod
    def create(layer, heuristic=None):
        if type(layer) is networks.Normalization:
            return NormalizeTransformer(layer)
        if type(layer) is torch.nn.modules.flatten.Flatten:
            return FlattenTransformer()
        if type(layer) is torch.nn.modules.linear.Linear:
            return AffineTransformer(layer)
        if type(layer) is torch.nn.modules.activation.ReLU:
            return ReluTransformer(heuristic)
        raise ValueError(f"unknown layer type {type(layer)}")


