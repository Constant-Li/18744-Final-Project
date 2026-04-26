import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision.models as models
import cv2
import numpy as np
import pdb

import utils
from modules.temporal_layers import BiLSTMLayer, TemporalConv

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TLD_resnet(nn.Module):
    def __init__(self, loss_weights) -> None:
        super().__init__()
        self.conv2d = getattr(models, 'resnet34')(pretrained=True)
        # self.conv2d = getattr(models, 'vgg16')(pretrained=True)
        # num_ftrs = self.conv2d.classifier[6].in_features
        # self.conv2d.classifier[6] = nn.Identity()
        num_ftrs = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Identity()
        self.fc_brake = nn.Linear(num_ftrs, 2)
        self.fc_turn = nn.Linear(num_ftrs, 5)
        # counts = torch.tensor([197108, 2753, 2073, 1030, 125], dtype=torch.float)
        # counts = torch.tensor([20000, 2753, 2073, 1030, 125], dtype=torch.float)
        # weights = (counts.sum() / counts) ** 0.5
        # self.weights = weights / weights.mean()
        # self.turn_loss = nn.CrossEntropyLoss(weight=self.weights,ignore_index=4)
        # self.turn_loss = nn.CrossEntropyLoss()
        self.turn_loss = FocalLoss()
        self.brake_loss = nn.CrossEntropyLoss()
        # self.loss = nn.CrossEntropyLoss()
        self.loss_weights = loss_weights

    def forward(self, inputs_dict):
        # x = inputs_dict['x'][2]
        # img = x.detach().cpu().numpy()
        # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img_bgr = (img_bgr * 255).clip(0, 255).astype(np.uint8)
        # cv2.imwrite('test.png', img_bgr)
        # pdb.set_trace()
        x = inputs_dict['x'].permute(0, 3, 1, 2)
        x = self.conv2d(x)
        turn_result = self.fc_turn(x)
        brake_result = self.fc_brake(x)
        return{
            'turn_result': turn_result,
            'brake_result': brake_result
        }

    def get_loss(self, ret_dict, label):
        loss, loss_dict = 0, {}
        labels_brake = label[:, 0].long()
        labels_turn = label[:, 1].long()
        for k, weight in self.loss_weights.items():
            if k == 'turn':
                # temp = weight * self.loss(ret_dict['turn_result'], labels_turn)
                temp = weight * self.turn_loss(ret_dict['turn_result'], labels_turn)
                loss += temp
                loss_dict[k] = temp
            elif k == 'brake':
                temp = weight * self.brake_loss(ret_dict['brake_result'], labels_brake)
                # temp = weight * self.loss(ret_dict['brake_result'], labels_brake)
                loss += temp
                loss_dict[k] = temp
        return loss, loss_dict