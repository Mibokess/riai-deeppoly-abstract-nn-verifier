
from abc import abstractmethod

import numpy as np
import scipy.linalg
import torch

from analyze.deeppoly import setup
from analyze.deeppoly.domain import AbstractDomain, GreaterThanConstraints, LowerThanConstraints
from analyze.deeppoly.transform import Transformer


class AffineTransformer(Transformer):

    def __init__(self, backsubstitution):
        self.backsubstitution = backsubstitution

    @property
    @abstractmethod
    def weights(self):
        pass

    @property
    @abstractmethod
    def bias(self):
        pass

    def _transform_affine(self, ads, output_shape=None):
        greater_than = GreaterThanConstraints(self.weights, self.bias)
        lower_than = LowerThanConstraints(self.weights, self.bias)

        if self.backsubstitution:
            lower_bounds, upper_bounds = self.compute_bounds(ads, lower_than, greater_than)
        else:
            ad = ads[-1]
            upper_bounds = greater_than.compute_bounds(ad.lower_bounds, ad.upper_bounds)
            lower_bounds = lower_than.compute_bounds(ad.lower_bounds, ad.upper_bounds)

        output_shape = lower_bounds.shape if output_shape is None else output_shape
        ad_lin = AbstractDomain(lower_bounds, upper_bounds, lower_than, greater_than, output_shape)
        return ad_lin


class LinearTransformer(AffineTransformer):

    def __init__(self, layer, backsubstitution):
        super().__init__(backsubstitution)

        w = layer.weight.detach()
        self._weights = w
        if layer.bias is not None:
            self._bias = layer.bias.detach().reshape((layer.out_features, 1))
        else:
            self._bias = torch.zeros((layer.out_features, 1))
        self.N = layer.out_features

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def _transform(self, ads):
        return self._transform_affine(ads)



class Conv2dTransformer(AffineTransformer):

    def __init__(self, layer, backsubstitution, debug=setup.assert_correct_convolution):
        super().__init__(backsubstitution)
        self.conv_weights = layer.weight.detach()
        self.conv_bias = layer.bias.detach()
        self.padding = layer.padding[0]
        self.stride = layer.stride[0]
        self._initialized = False
        self.layer = layer
        self.debug = debug


    def __build(self, input_shape):
        """
            :param input_shape without batch size: #channels x H x W
        """
        if not self._initialized:
            toeplitz, self.output_shape = \
                self.toeplitz(self.conv_weights, input_shape, self.padding, self.stride)
            self._weights = torch.from_numpy(toeplitz).to(torch.float)
            self._bias = AbstractDomain.flatten(self.conv_bias.unsqueeze(-1).unsqueeze(-1).expand(self.output_shape))
            if self.debug:
                x = torch.rand(input_shape).unsqueeze(0)
                expected = self.layer(x).detach()
                xx = x.flatten()
                computed = torch.matmul(self.weights, xx) + self._bias.flatten()
                computed = computed.reshape(self.output_shape).unsqueeze(0)
                assert np.all(np.isclose(expected.numpy(), computed.numpy(), atol=1e-6)), "Conv2d transformer is inaccurate"
                sum = np.sum(((expected - computed) ** 2).numpy())
                assert sum < 1e-8, "Conv2d transformer is inaccurate"
            self._initialized = True


    def _transform(self, ads):
        self.__build(ads[-1].output_shape)
        ad = self._transform_affine(ads, self.output_shape)
        return ad


    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias


    @staticmethod
    def _toeplitz_one_channel(kernel, input_size, padding, stride):
        """ compute the toeplitz matrices for the conv2d operation and one channel only """
        k_h, k_w = kernel.shape
        i_h, i_w = input_size
        i_h_p = i_h + 2 * padding
        i_w_p = i_w + 2 * padding

        o_h = (i_h + 2 * padding - k_h) // stride + 1
        o_w = (i_w + 2 * padding - k_w) // stride + 1

        # construct 1d conv toeplitz matrices for kernel rows
        # the matrices are operating in the Ih x Oh space
        toeplitz = []
        for r in range(k_h):
            toeplitz.append(
                scipy.linalg.toeplitz(c=(kernel[r, 0], *np.zeros(i_w_p - k_w)), r=(*kernel[r], *np.zeros(i_w_p - k_w))))

        W_conv = np.zeros((o_h, o_w, i_h, i_w))

        for j in range(o_h):
            k = stride * j
            for i in range(k_h):
                m = k + i - padding
                if 0 <= m < i_h:
                    T = toeplitz[i]
                    W_conv[j, :, m, :] = T[::stride, padding:-padding]
                    pass

        W_conv.shape = (o_h * o_w, i_h * i_w)

        return W_conv


    @staticmethod
    def toeplitz(kernel, input_size, padding, stride):
        """ Compute toeplitz matrix for 2d conv with multiple in and out channels.
        :param  kernel: shape=(n_out, n_in, H_k, W_k)
        :param  input_size: (n_in, H_i, W_i)
        :returns toeplitz matrix of shape (n_out x H_o x W_out, n_in x H_i x W_i)
        """

        kernel_size = kernel.shape
        o_h = (input_size[1] + 2 * padding - kernel_size[2]) // stride + 1
        o_w = (input_size[2] + 2 * padding - kernel_size[3]) // stride + 1

        output_size = (kernel_size[0], o_h, o_w)

        T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

        for i, ks in enumerate(kernel):  # loop over output channel
            for j, k in enumerate(ks):  # loop over input channel
                T_k = Conv2dTransformer._toeplitz_one_channel(k, input_size[1:], 1, 2)
                T[i, :, j, :] = T_k

        T.shape = (np.prod(output_size), np.prod(input_size))
        return T, output_size


