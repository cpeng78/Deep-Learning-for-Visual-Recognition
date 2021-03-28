import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        #out = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.kernel_size, self.kernel_size, x.shape[3] // self.kernel_size, self.kernel_size).max(axis=(3, 5))
        n, c, h_in, w_in = x.shape
        H_out = 1 + (h_in - self.kernel_size) // self.stride
        W_out = 1 + (w_in - self.kernel_size) // self.stride
        out = np.zeros((n, c, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                pool = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(pool, axis=(2, 3))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        output = np.zeros_like(x)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                pool = x[:, :, h_start:h_end, w_start:w_end]
                mask = pool == np.max(pool, axis=(2, 3), keepdims=True)
                output[:, :, h_start:h_end, w_start:w_end] += dout[:, :, i:i + 1, j:j + 1] * mask
        self.dx = output
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
