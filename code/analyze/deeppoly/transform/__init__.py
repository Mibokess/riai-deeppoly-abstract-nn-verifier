from abc import ABC, abstractmethod
import torch
import networks
from analyze.deeppoly.domain import AbstractDomain, RelationalConstraints, LowerThanConstraints, GreaterThanConstraints
from analyze.utils import TensorUtils


class Transformer(ABC):

    def __init__(self):
        self._back_to = 0

    def transform(self, abstract_domains):
        ad = self._transform(abstract_domains)
        return abstract_domains + [ad]

    @abstractmethod
    def _transform(self, abstract_domains):
        pass

    def set_backsubstitution_to(self, id):
        self._back_to = id

    def compute_bounds(self, ads, lower_than0, greater_than0):
        """ computes the upper and lower bounds with backpropagation """
        id = self._back_to
        lower_than = lower_than0.clone()
        greater_than = greater_than0.clone()

        for ad in reversed(ads[(id+1):]):
            greater_than_pos, greater_than_neg = TensorUtils.split_positive_negative(greater_than.A)

            A_gt = torch.matmul(greater_than_pos, ad.greater_than.A) + torch.matmul(greater_than_neg, ad.lower_than.A)
            v_gt = greater_than.v + torch.matmul(greater_than_pos, ad.greater_than.v) + torch.matmul(greater_than_neg, ad.lower_than.v)
            greater_than = GreaterThanConstraints(A_gt, v_gt)

            lower_than_pos, lower_than_neg = TensorUtils.split_positive_negative(lower_than.A)

            A_lt = torch.matmul(lower_than_pos, ad.lower_than.A) + torch.matmul(lower_than_neg, ad.greater_than.A)
            v_lt = lower_than.v + torch.matmul(lower_than_pos, ad.lower_than.v) + torch.matmul(lower_than_neg, ad.greater_than.v)
            lower_than = LowerThanConstraints(A_lt, v_lt)


        input = ads[id] if id < len(ads) else ads[-1]
        upper_bound = greater_than.compute_bounds(input.lower_bounds, input.upper_bounds)
        lower_bound = lower_than.compute_bounds(input.lower_bounds, input.upper_bounds)

        return lower_bound, upper_bound

    def compute_bounds1(self, ads, lower_than_start, greater_than_start):
        lower_bounds = []
        upper_bounds = []

        for i, initial in enumerate(lower_than_start.A):
            lower_bound = self.backprop_lower(initial, ads)
            lower_bounds.append(lower_bound)

        for i, initial in enumerate(greater_than_start.A):
            upper_bound = self.backprop_upper(initial, ads)
            upper_bounds.append(upper_bound)

        return torch.cat(lower_bounds).reshape(len(lower_bounds), 1) + lower_than_start.v, torch.cat(upper_bounds).reshape(len(upper_bounds), 1) + greater_than_start.v

    def backprop_lower(self, initial, ads):
        initial_pos, initial_neg = TensorUtils.split_positive_negative(initial)

        if len(ads) == 1 or not ads[-1].lower_than or not ads[-1].greater_than:
            return torch.matmul(initial_pos, ads[-1].lower_bounds) + torch.matmul(initial_neg, ads[-1].upper_bounds)

        layer_prop_A = torch.matmul(initial_pos, ads[-1].lower_than.A) + torch.matmul(initial_neg, ads[-1].greater_than.A)
        layer_prop_v = torch.matmul(ads[-1].lower_than.v.T, initial_pos) + torch.matmul(ads[-1].greater_than.v.T, initial_neg)

        lower_bound = self.backprop_lower(layer_prop_A, ads[:-1])

        return lower_bound + layer_prop_v

    def backprop_upper(self, initial, ads):
        initial_pos, initial_neg = TensorUtils.split_positive_negative(initial)

        if len(ads) == 1 or not ads[-1].greater_than:
            return torch.matmul(initial_pos, ads[-1].upper_bounds) + torch.matmul(initial_neg, ads[-1].lower_bounds)

        layer_prop_A = torch.matmul(initial_pos, ads[-1].greater_than.A) + torch.matmul(initial_neg, ads[-1].lower_than.A)
        layer_prop_v = torch.matmul(ads[-1].greater_than.v.T, initial_pos) + torch.matmul(ads[-1].lower_than.v.T, initial_neg)
        
        upper_bound = self.backprop_upper(layer_prop_A, ads[:-1])

        return upper_bound + layer_prop_v




