import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv = nn.Conv2d(3, 32, (7, 7), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        outs = self.fc(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs