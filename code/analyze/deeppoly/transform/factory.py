from analyze.deeppoly.transform.affine import LinearTransformer, Conv2dTransformer
from analyze.deeppoly.transform.preprocess import NormalizeTransformer, FlattenTransformer
from analyze.deeppoly.transform.relu import ReluTransformer

import networks
import torch


class TransformerFactory:

    @staticmethod
    def create(layer, backsubstitution, relu_id=None):
        if type(layer) is networks.Normalization:
            return NormalizeTransformer(layer)
        if type(layer) is torch.nn.modules.flatten.Flatten:
            return FlattenTransformer()
        if type(layer) is torch.nn.modules.linear.Linear:
            return LinearTransformer(layer, backsubstitution)
        if type(layer) is torch.nn.modules.conv.Conv2d:
            return Conv2dTransformer(layer, backsubstitution)
        if type(layer) is torch.nn.modules.activation.ReLU:
            return ReluTransformer(relu_id)
        raise ValueError(f"unknown layer type {type(layer)}")


