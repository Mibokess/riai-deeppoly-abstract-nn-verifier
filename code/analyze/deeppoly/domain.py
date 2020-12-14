from dataclasses import dataclass, field
from abc import abstractmethod, ABC
import torch

from analyze.utils import TensorUtils


@dataclass
class RelationalConstraints(ABC):
    """
        The relational constraints for the set of N nodes x_j in a layer L
    """
    A: torch.Tensor
    v: torch.Tensor

    @abstractmethod
    def compute_bounds(self, lower_bounds, upper_bounds):
        pass



class GreaterThanConstraints(RelationalConstraints):
    """
        Describes "greater than" relational constraints.
        i.e. the constraints for the N nodes at layer L are given by A_{>} * x^{L-1} + c^L  >= x^L
    """

    def compute_bounds(self, input_lower_bounds, input_upper_bounds):
        """ Returns the UPPER bounds for the N nodes x_i in the layer L
            i.e. l={l_i, i=1,..,N}, such that x^L_i <= l_i
        """

        A_pos, A_neg = TensorUtils.split_positive_negative(self.A)
        upper_bounds = torch.matmul(A_pos, input_upper_bounds) + torch.matmul(A_neg, input_lower_bounds) + self.v
        return upper_bounds

    def clone(self):
        return GreaterThanConstraints(self.A.clone(), self.v.clone())



class LowerThanConstraints(RelationalConstraints):
    """
        describes "lower than" relational contraints
        i.e. constraints of the type A_{<} * x^0 + c^L <= x^L_i
    """

    def compute_bounds(self, input_lower_bounds, input_upper_bounds):
        """ Returns the LOWER bounds for the N nodes x_i in the layer L
            i.e. l={l_i, i=1,..,N}, such that x^L_i >= l_i
        """
        A_pos, A_neg = TensorUtils.split_positive_negative(self.A)
        lower_bounds = torch.matmul(A_pos, input_lower_bounds) + torch.matmul(A_neg, input_upper_bounds) + self.v
        return lower_bounds

    def clone(self):
        return LowerThanConstraints(self.A.clone(), self.v.clone())



@dataclass
class AbstractDomain:
    """
        The abstract domain after layer L. For each node x^L_i in layer L, we have:
        - lower_bounds[i] <= x^L_i <= upper_bounds[i]
        - x^L_i >= lower_than.A * x^{L-1} + lower_than.v[i]
        - x^L_i <= greater_than.A * x^{L-1} + greater_than.v[i]
          where x^{L-1} are the nodes of the previous layer
        - output_shape represents the real shape of x^L without the batch size, ie #channels x H x W
    """
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    lower_than: LowerThanConstraints
    greater_than: GreaterThanConstraints
    output_shape: torch.Size


    def preprocess(self, function):
        """ :param function - a preprocessing function that should not change the input dimension """
        l = AbstractDomain.flatten(function(self.lower_bounds))
        u = AbstractDomain.flatten(function(self.upper_bounds))
        assert self.lower_than is None and self.greater_than is None
        return AbstractDomain(l, u, None, None, self.output_shape)


    def has_constraints(self):
        return self.lower_than is not None and self.greater_than is not None


    @staticmethod
    def create(inputs, epsilon, norm="Linf", domain_bounds=[0, 1], clamp=True):
        assert norm.lower().strip() == "linf", f"{norm} norm is not supported"
        u = inputs + epsilon
        l = inputs - epsilon
        if clamp:
            u = torch.clamp(u, domain_bounds[0], domain_bounds[1])
            l = torch.clamp(l, domain_bounds[0], domain_bounds[1])

        l = AbstractDomain.flatten(l)
        u = AbstractDomain.flatten(u)
        return AbstractDomain(l, u, None, None, output_shape=inputs.shape[1:])


    def clone(self):
        lb = self.lower_bounds.clone()
        ub = self.upper_bounds.clone()
        lrc = self.lower_than.clone() if self.lower_than is not None else None
        urc = self.greater_than.clone() if self.greater_than is not None else None
        return AbstractDomain(lb, ub, lrc, urc, self.output_shape)


    def intersection(self, ad):
        # TODO
        inter = self.clone()
        lower = (self.lower_bounds < ad.lower_bounds).flatten()
        inter.lower_bounds[lower] = ad.lower_bounds[lower]
        inter.lower_than.A[lower, :] = inter.lower_than.A[lower, :]
        inter.lower_than.v[lower, :] = inter.lower_than.v[lower, :]
        upper = (self.upper_bounds > ad.upper_bounds).flatten()
        inter.upper_bounds[upper] = ad.upper_bounds[upper]
        inter.greater_than.A[upper, :] = inter.greater_than.A[upper, :]
        inter.greater_than.v[upper, :] = inter.greater_than.v[upper, :]
        return inter


    @staticmethod
    def flatten(a):
        f = torch.flatten(a)
        return f.reshape(f.shape[0], 1)
