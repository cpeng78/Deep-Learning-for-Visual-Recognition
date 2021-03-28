import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    effective_number = 1.0 - torch.tensor(cls_num_list).pow(beta)
    per_cls_weights = (1 - beta) / effective_number
    per_cls_weights = per_cls_weights / torch.sum(per_cls_weights)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        #target_one_hot = F.one_hot(target, input.shape[-1]).float()
        #ce_loss = F.cross_entropy(input, target, reduce=False, reduction="none")
        #modulator = (1 - F.softmax(input)).pow(self.gamma) * target_one_hot
        #ce_loss = F.cross_entropy(input, target, reduce=False, reduction="none")
        #print(ce_loss.shape, modulator.shape, self.weight.shape)
        #loss = torch.dot(torch.sum(self.weight*modulator, dim=1), ce_loss)
        #print(target_one_hot)
        #print('loss =', loss)

        softmax = F.softmax(input, dim=1)
        label_one_hot = F.one_hot(target, input.shape[-1]).float()
        pt = softmax
        log_softmax = F.log_softmax(input, dim=1)
        FL = -(1 - pt) ** self.gamma * log_softmax * label_one_hot
        loss = (FL * self.weight).sum(dim=1).mean()
        #fc_loss = modulator * ce_loss
        #loss = torch.dot(self.weight, fc_loss)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

if __name__ == '__main__':
    gamma = 1
    beta = 0.9999
    myweight = torch.tensor([0.0051, 0.0077, 0.0121, 0.0195, 0.0319, 0.0525, 0.0868, 0.1443, 0.2409,
                             0.3993])
    # ?
    yourweight = myweight
    assert yourweight == myweight

    input = torch.tensor([[2.9751, -0.3312, -7.4666, -6.8384, 1.6987, 1.9255, -2.2512, -1.4505,
                           0.7402, 2.1339],
                          [3.3333, -0.6700, -7.6600, -5.8551, 1.3606, 2.1366, -1.8532, -1.4562,
                           0.4192, 3.1757],
                          [3.5510, -0.6564, -8.6301, -5.3536, 1.2964, 1.1514, -2.4029, -0.6956,
                           0.1640, 3.3132],
                          [3.1342, 0.3029, -8.1702, -6.4352, 1.0989, 2.1021, -2.5321, -1.5944,
                           0.4383, 2.3836]])
    target = torch.tensor([0, 6, 2, 0])

    myloss = torch.tensor(0.8269)

    # ?
    yourloss = myloss
    assert myloss == yourloss