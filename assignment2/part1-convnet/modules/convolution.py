import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape
        F, CC, HH, WW = self.weight.shape
        assert C == CC

        H_out = 1 + (H + 2 * self.padding - HH) // self.stride
        W_out = 1 + (W + 2 * self.padding - WW) // self.stride
        out = np.zeros((N, F, H_out, W_out))

        # padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

        # convolving
        for i in range(H_out):
            for j in range(W_out):
                pool = x_pad[:, :, i * self.stride : i * self.stride + HH, j * self.stride : j * self.stride + WW]
                out[:, :, i, j] = np.tensordot(pool, self.weight, axes=([1, 2, 3], [1, 2, 3])) + self.bias

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        _, _, HH, WW = self.weight.shape
        _, _, H_out, W_out = dout.shape

        # padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

        self.dx = np.zeros_like(x_pad)
        self.dw = np.zeros_like(self.weight)

        for i in range(H_out):
            for j in range(H_out):
                self.dx[:, :, i * self.stride: i * self.stride + HH, j * self.stride: j * self.stride + WW] += \
                    np.sum(dout[:, :, np.newaxis, i:i + 1, j:j + 1] * self.weight[np.newaxis, :, :, :, :], axis=(1))

                self.dw += np.sum(dout[:, :, np.newaxis, i:i+1, j:j + 1] *
                                  x_pad[:, np.newaxis, :, i * self.stride: i * self.stride + HH, j * self.stride: j * self.stride + WW]
                                  , axis=0)

        self.dx = self.dx[:, :, self.padding : - self.padding, self.padding : - self.padding]
        self.db = dout.sum(axis=(0, 2, 3))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################