# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        fc1 = np.dot(X, self.weights['W1'])
        relu1 = self.ReLU(fc1)
        sm1 = self.softmax(relu1)
        #print(sm1.shape, y.shape)
        loss = self.cross_entropy_loss(sm1, y)
        accuracy = self.compute_accuracy(sm1, y)
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient with respect to the loss                       #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        m = y.shape[0]
        grad = self.softmax(relu1)
        grad[range(m), y] -= 1
        gradA = grad / m
        relu_dev1 = self.ReLU_dev(fc1)
        gradient = gradA * relu_dev1
        # print(gradA.shape, relu_dev1.shape, X.shape)
        self.gradients['W1'] = np.dot(X.T, gradient)
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE              T               #
        #############################################################################
        return loss, accuracy





        


