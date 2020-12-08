from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain, RelationalConstraints, LowerThanConstraints, GreaterThanConstraints
from analyze.utils import TensorUtils


class Transformer(ABC):

    def transform(self, abstract_domains):
        ad = self._transform(abstract_domains)
        return abstract_domains + [ad]


    @abstractmethod
    def _transform(self, abstract_domains):
        pass


    @staticmethod
    def compute_bounds(ads, lower_than0, greater_than0):
        """ computes the upper and lower bounds with backpropagation """
        lower_than = lower_than0.clone()
        greater_than = greater_than0.clone()

        for ad in reversed(ads[1:]):
            greater_than_pos, greater_than_neg = TensorUtils.split_positive_negative(greater_than.A)

            A_gt = torch.matmul(greater_than_pos, ad.greater_than.A) + torch.matmul(greater_than_neg, ad.lower_than.A)
            v_gt = greater_than.v + torch.matmul(greater_than_pos, ad.greater_than.v) + torch.matmul(greater_than_neg, ad.lower_than.v)
            greater_than = GreaterThanConstraints(A_gt, v_gt)

            lower_than_pos, lower_than_neg = TensorUtils.split_positive_negative(lower_than.A)

            A_lt = torch.matmul(lower_than_pos, ad.lower_than.A) + torch.matmul(lower_than_neg, ad.greater_than.A)
            v_lt = lower_than.v + torch.matmul(lower_than_pos, ad.lower_than.v) + torch.matmul(lower_than_neg, ad.greater_than.v)
            lower_than = LowerThanConstraints(A_lt, v_lt)


        input = ads[0]
        upper_bound = greater_than.compute_bounds(input.lower_bounds, input.upper_bounds)
        lower_bound = lower_than.compute_bounds(input.lower_bounds, input.upper_bounds)

        return lower_bound, upper_bound

