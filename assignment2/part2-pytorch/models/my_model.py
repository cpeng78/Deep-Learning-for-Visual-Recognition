import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='replicate')
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AvgPool2d(6)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)
        #self.fc3 = nn.Linear(84, 10)


        # self.conv1 = nn.Conv2d(3, 32, 5, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        # self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        #
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        #
        # self.batchnorm1 = nn.BatchNorm2d(32)
        # self.batchnorm2 = nn.BatchNorm2d(64)
        # self.batchnorm3 = nn.BatchNorm2d(128)
        #
        # self.pool = nn.AdaptiveAvgPool2d((2,2))
        # self.flatten = nn.Flatten()
        #
        # self.fc1 = nn.Linear(512, 50)
        # self.batchnorm4 = nn.BatchNorm1d(50)
        #
        # self.fc2 = nn.Linear(50, 10)

        # m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        # self.encoder = nn.Sequential(*list(m.children())[:-2])
        # nc = list(m.children())[-1].in_features
        # self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(nc,512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),nn.Linear(512, 10))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # x = self.mask(x)
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.batchnorm1(self.relu(self.conv2(x)))
        x = self.dropout1(self.pool(x))

        x = self.batchnorm2(self.relu(self.conv3(x)))
        x = self.batchnorm2(self.relu(self.conv4(x)))
        x = self.dropout2(self.pool(x))

        x = self.batchnorm3(self.relu(self.conv5(x)))
        x = self.batchnorm3(self.relu(self.conv6(x)))
        x = self.dropout3(self.pool(x))

        x = self.batchnorm4(self.relu(self.fc1(self.flatten(x))))
        x = self.dropout4(x)
        outs = self.fc2(x)


        #
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.batchnorm1(x)
        # x = self.dropout(x)
        #
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        #
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = self.batchnorm3(x)
        # x = self.dropout(x)
        #
        # x = self.pool(x)
        # x = self.flatten(x)
        #
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.batchnorm4(x)
        # x = self.dropout(x)
        #
        # outs = self.fc2(x)

        # x = self.encoder(x)
        # outs = self.head(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs