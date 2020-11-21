from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain, RelationalConstraints


class Transformer(ABC):

    def transform(self, abstract_domains):
        d0 = AbstractDomain.last_preprocessing_step(abstract_domains)
        d = abstract_domains[-1].init()
        return self._transform(d, d0)

    @abstractmethod
    def _transform(self, ad, input):
        pass




