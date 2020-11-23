import torch
import torch.nn as nn
from analyze.deeppoly.analyzer import DeepPoly
from analyze.deeppoly.transform.relu.heuristic import Zero
import unittest
import numpy as np


class ToyNNEx07(nn.Module):

    def __init__(self):
        super().__init__()

        linear1 = nn.Linear(2, 2, False)
        linear1.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, -2.0]]))
        relu = nn.ReLU()
        linear2 = nn.Linear(2, 2, False)
        linear2.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [-1.0, 1.0]]))
        self.layers = nn.Sequential(linear1, relu, linear2)

    def forward(self, x):
        return self.layers(x)



class ToyNNCourse(nn.Module):

    def __init__(self):
        super().__init__()

        linear1 = nn.Linear(2, 2, False)
        linear1.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        relu = nn.ReLU()
        linear2 = nn.Linear(2, 2)
        linear2.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        linear2.bias = torch.nn.Parameter(torch.tensor([-0.5, 0]))
        relu2 = nn.ReLU()
        linear3 = nn.Linear(2, 2)
        linear3.weight = torch.nn.Parameter(torch.tensor([[-1.0, 1.0], [0.0, 1.0]]))
        linear3.bias = torch.nn.Parameter(torch.tensor([3.0, 0]))
        self.layers = nn.Sequential(linear1, relu, linear2, relu2, linear3)

    def forward(self, x):
        return self.layers(x)




class ToyNNsTestCase(unittest.TestCase):


    def test_exercise_07(self):
        """
            Verifying toy NN of the RIAI course, exercise sheet 07 Problem 2
        """
        dp = DeepPoly(save_intermediate_steps=True, relu_heuristics=Zero())
        res, ads, steps = \
            dp.verify(ToyNNEx07(), torch.zeros((2, 1)), 1, 0, domain_bounds=[0, 1], robustness_fn=torch.greater_equal)

        expected_lower_bounds = [[0, 0], [0, -2], [0, 0], [0, -2], [0.0]]
        expected_upper_bounds = [[1, 1], [2, 1], [2, 1], [7 / 3, 2 / 3], None]

        self.assertTrue(res, "could not verify property x_7 - x_8 >= 0")
        self._check(expected_lower_bounds, expected_upper_bounds, ads, steps)


    def test_lecture_deepoly(self):
        """
            Verifying toy NN of the RIAI course - lecture 7 deeppoly slides 8
        """
        dp = DeepPoly(save_intermediate_steps=True, relu_heuristics=Zero())
        res, ads, steps = dp.verify(ToyNNCourse(), torch.zeros((2, 1)), 1, 0, domain_bounds=[-1, 1])

        expected_lower_bounds = [[-1, -1], [-2, -2], [0, 0], [-0.5, -2], [0, 0], [0.5, 0], [0.5]]
        expected_upper_bounds = [[1, 1], [2, 2], [2, 2], [2.5, 2.0], [2.5, 2], [5.0, 2], None]

        self.assertTrue(res, "could not verify property x11 - x12 > 0")
        self._check(expected_lower_bounds, expected_upper_bounds, ads, steps)


    def _check(self, expected_lower_bounds, expected_upper_bounds, ads, steps):
        for i, ad in enumerate(ads):
            if expected_lower_bounds[i] is not None:
                self._check_bounds("upper", expected_lower_bounds[i], ad.lower_bounds, steps[i])
            if expected_upper_bounds[i] is not None:
                self._check_bounds("lower", expected_upper_bounds[i], ad.upper_bounds, steps[i])


    def _check_bounds(self, what, exp, got, step):
        exp = np.array(exp)
        got = got.flatten().numpy()
        self.assertTrue(np.all(np.isclose(got, exp)),
                        f"{what} bounds for layer {step} are not correct (expecting {exp} got {got})")



if __name__ == "__main__":
    unittest.main()


