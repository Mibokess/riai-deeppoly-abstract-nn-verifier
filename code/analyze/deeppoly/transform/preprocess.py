from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain
from analyze.deeppoly.transform import Transformer


class Preprocessor(Transformer):

    def transform(self, abstract_domains):
        ad = abstract_domains[-1].clone()
        ad = self._transform(ad)
        if ad.has_constraints():
            # we are not in preprocessing mode anymore
            return abstract_domains[:-1] + [ad]
        return [ad]



class NormalizeTransformer(Preprocessor):

    def __init__(self, norm_layer):
        assert type(norm_layer) is networks.Normalization, \
            f"expecting Normalization layer got {norm_layer.__class__.__name__}"
        self._l = norm_layer

    def _transform(self, ad):
        return AbstractDomain.preprocess(ad, self._l.forward)


class FlattenTransformer(Preprocessor):

    def _transform(self, ad):
        ad.output_shape = AbstractDomain.flatten(torch.zeros(ad.output_shape))
        return ad



