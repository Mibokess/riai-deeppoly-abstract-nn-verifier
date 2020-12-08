from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain, RelationalConstraints
from analyze.deeppoly.transform import Transformer


class Preprocessor(Transformer):

    def _transform(self, ads):
        raise NotImplementedError()


class NormalizeTransformer(Preprocessor):

    def __init__(self, norm_layer):
        assert type(norm_layer) is networks.Normalization, \
            f"expecting Normalization layer got {norm_layer.__class__.__name__}"
        self._l = norm_layer

    def transform(self, abstract_domains):
        return [AbstractDomain.preprocess(abstract_domains[-1], self._l.forward)]


class FlattenTransformer(Preprocessor):

    def transform(self, abstract_domains):
        def flatten(a):
            f = torch.flatten(a)
            return f.reshape(f.shape[0], 1)
        ad = AbstractDomain.preprocess(abstract_domains[-1], flatten)
        if abstract_domains[-1].has_constraints():
            # we are not in preprocessing step
            return abstract_domains + [ad]
        return [ad]


