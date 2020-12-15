import time
import torch.nn as nn


class TensorUtils:

    @staticmethod
    def split_positive_negative(A):
        """
            A+ is the positive part of A, A+ = max(A, 0)
            A- is the negative part of A, A- = min(A, 0)
            A = A+ + A-
        """
        A_pos = A.clone()
        A_pos[A < 0] = 0
        A_neg = A.clone()
        A_neg[A > 0] = 0
        return A_pos, A_neg



class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed_time(self):
        return time.time() - self._start

    def reset(self):
        self._start = time.time()


class NetworkInspector:

    def __init__(self, net, inputs):
        self.net = net
        self.inputs = inputs

    def get_relu_input_sizes(self):
        sizes = []
        x = self.inputs
        for layer in self.net.layers:
            if isinstance(layer, nn.ReLU):
                sizes.append(x.flatten().size()[0])
            x = layer(x)
        return sizes
