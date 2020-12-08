from analyze.deeppoly.domain import LowerThanConstraints, GreaterThanConstraints, AbstractDomain
from analyze.deeppoly.transform import Transformer
import torch

class ReluTransformer(Transformer):

    def __init__(self, heuristic):
        self._heuristic = heuristic

    def _transform(self, ads):
        ad = ads[-1]
        relu_lower_than = LowerThanConstraints(torch.eye(len(ad.lower_bounds)), torch.zeros_like(ad.lower_bounds))
        relu_greater_than = GreaterThanConstraints(torch.eye(len(ad.lower_bounds)), torch.zeros_like(ad.lower_bounds))

        # 1. setting the equations to zero for nodes with upper_bound lower than zero
        set_to_zero = (ad.upper_bounds <= 0).flatten()
        relu_lower_than.A[set_to_zero, :] = 0.0
        relu_greater_than.A[set_to_zero, :] = 0.0

        # 2. for nodes where the lower_bound is bigger than 0 we don't do anything.
        # 3. for nodes with lower_bound < 0 and upper_bound > 0
        relu_mask = ((ad.lower_bounds < 0) & (ad.upper_bounds > 0)).flatten()

        # 3.1 we set the upper bounds and relational contraints:
        # x^L_j <= u_j/(u_j-l_j) * (x^(L-1)_j - l_j) where l_j <= x^(L-1) <= u_j

        delta = (ad.upper_bounds[relu_mask] - ad.lower_bounds[relu_mask])
        lambda_up = ad.upper_bounds[relu_mask] / delta
        c_up = - ad.lower_bounds[relu_mask] * ad.upper_bounds[relu_mask] / delta

        relu_greater_than.A[relu_mask, :] *= lambda_up
        relu_greater_than.v[relu_mask] += c_up

        # 3.2 for the lower bound, the heuristic..
        # x^L_j >= lambda * x^(L-1)_j

        lambda_low = self._heuristic.compute_lambda(ad.lower_bounds[relu_mask], ad.upper_bounds[relu_mask])
        relu_lower_than.A[relu_mask, :] *= lambda_low

        # ad_relu.lower_than.compute_bounds(input.lower_bounds, input.upper_bounds)
        # ad_relu.upper_bounds = ad_relu.greater_than.compute_bounds(input.lower_bounds, input.upper_bounds)

        lower_bounds, upper_bounds = self.compute_bounds(ads, relu_lower_than, relu_greater_than)
        ad_relu = AbstractDomain(lower_bounds, upper_bounds, relu_lower_than, relu_greater_than)

        return ad_relu


