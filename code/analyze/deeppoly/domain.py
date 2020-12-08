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

    A_no_sub: torch.Tensor
    v_no_sub: torch.Tensor

    @abstractmethod
    def compute_bounds(self, lower_bounds, upper_bounds):
        pass

    @abstractmethod
    def compute_bounds_backprop(self, ad, ads):
        pass



class GreaterThanConstraints(RelationalConstraints):
    """
        Describes "greater than" relational constraints.
        i.e. the constraints for the N nodes at layer L are given by A_{>} * x^0 + c^L  >= x^L
    """

    def compute_bounds(self, input_lower_bounds, input_upper_bounds):
        """ Returns the UPPER bounds for the N nodes x_i in the layer L
            i.e. l={l_i, i=1,..,N}, such that x^L_i <= l_i
        """
        A_pos, A_neg = TensorUtils.split_positive_negative(self.A)
        upper_bounds = torch.matmul(A_pos, input_upper_bounds) + torch.matmul(A_neg, input_lower_bounds) + self.v
        return upper_bounds

    def compute_bounds_backprop(self, ad, ads):
        upper_bounds = []

        for i, node_vals in enumerate(self.A_no_sub.T):
            upper_bounds.append([sum(self.backprop_layer(node_vals, ad, ads)) + self.v_no_sub.T[i]])

        return torch.tensor(upper_bounds)

    def backprop_layer(self, initial, ad, ads):
        initial_pos, initial_neg = TensorUtils.split_positive_negative(initial)

        if len(ads) == 1 or not ads[-1].greater_than:
            return torch.matmul(initial_pos, ads[-1].upper_bounds) + torch.matmul(initial_neg, ads[-1].lower_bounds)

        layer_prop_A = torch.matmul(ads[-1].greater_than.A_no_sub, initial_pos) + torch.matmul(ads[-1].lower_than.A_no_sub, initial_neg)
        layer_prop_v = torch.matmul(ads[-1].greater_than.v_no_sub, initial_pos) + torch.matmul(ads[-1].lower_than.v_no_sub, initial_neg)

        return self.backprop_layer(layer_prop_A, ads[-1], ads[:-1]) + layer_prop_v

    def clone(self):
        return GreaterThanConstraints(self.A.clone(), self.v.clone(), self.A_no_sub.clone(), self.v_no_sub.clone())



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

    def compute_bounds_backprop(self, ad, ads):
        lower_bounds = []

        for i, node_vals in enumerate(self.A_no_sub.T):
            lower_bounds.append([sum(self.backprop_layer(node_vals, ad, ads)) + self.v_no_sub.T[i]])

        return torch.tensor(lower_bounds)

    def backprop_layer(self, initial, ad, ads):
        initial_pos, initial_neg = TensorUtils.split_positive_negative(initial)

        if len(ads) == 1 or not ads[-1].lower_than:
            return torch.matmul(initial_pos, ads[-1].lower_bounds) + torch.matmul(initial_neg, ads[-1].upper_bounds)

        layer_prop_A = torch.matmul(ads[-1].lower_than.A_no_sub, initial_pos) + torch.matmul(ads[-1].greater_than.A_no_sub, initial_neg)
        layer_prop_v = torch.matmul(ads[-1].lower_than.v_no_sub, initial_pos) + torch.matmul(ads[-1].greater_than.v_no_sub, initial_neg)

        return self.backprop_layer(layer_prop_A, ads[-1], ads[:-1]) + layer_prop_v

    def clone(self):
        return LowerThanConstraints(self.A.clone(), self.v.clone(), self.A_no_sub.clone(), self.v_no_sub.clone())



@dataclass
class AbstractDomain:
    """
        The abstract domain after layer L. For each node x^L_i in layer L, we have:
        - lower_bounds[i] <= x^L_i <= upper_bounds[i]
        - x^L_i >= lower_than.A * x^0 + lower_than.v[i]
        - x^L_i <= greater_than.A * x^0 + greater_than.v[i]
          where x^0 are the nodes of the input layer (i.e. backsubstitution is performed at each step).
    """
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    lower_than: LowerThanConstraints
    greater_than: GreaterThanConstraints


    def preprocess(self, function):
        l = function(self.lower_bounds)
        u = function(self.upper_bounds)
        assert self.lower_than is None and self.greater_than is None
        return AbstractDomain(l, u, None, None)


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
        return AbstractDomain(l, u, None, None)


    @staticmethod
    def last_preprocessing_step(abstract_domains):
        preprocessing_steps = [ad for ad in abstract_domains if not ad.has_constraints()]
        return preprocessing_steps[-1]


    def init(self):
        if self.has_constraints():
            return self

        def idty():
            return torch.eye(len(self.lower_bounds))

        lower_than = LowerThanConstraints(idty(), torch.zeros_like(self.lower_bounds), None, None)
        greater_than = GreaterThanConstraints(idty(), torch.zeros_like(self.upper_bounds), None, None)

        return AbstractDomain(self.lower_bounds, self.upper_bounds, lower_than, greater_than)


    def clone(self):
        lb = self.lower_bounds.clone()
        ub = self.upper_bounds.clone()
        lrc = self.lower_than.clone()
        urc = self.greater_than.clone()
        return AbstractDomain(lb, ub, lrc, urc)


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

