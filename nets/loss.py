import torch
from torch import nn
import torch.nn.functional as F


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        #print(input.size(), target.size())
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice
class DDice(nn.Module):
    def __init__(self):
        super(DDice, self).__init__()

    def forward(self, P, Q, L):
        smooth = 1e-5
        input = (1- torch.sigmoid(P))*L
        target = (1- torch.sigmoid(Q))*L
        #print(input.size(), target.size())
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = dice.sum() / num
        return dice
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()  # torch.Tensor([10])

    def forward(self, input, target):
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        return self.loss(input, target)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice
