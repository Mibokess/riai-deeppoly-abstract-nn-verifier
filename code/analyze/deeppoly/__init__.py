import torch


class Setup:
    assert_correct_convolution = True
    robustness_function = torch.greater


setup = Setup()
