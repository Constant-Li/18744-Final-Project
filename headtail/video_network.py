import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision.models as models
import cv2
import numpy as np
import pdb

import utils

import torch
import torch.nn as nn

class TaillightBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, transfer_label=False):
        super(TaillightBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # self.lstm = nn.LSTM(
        #     input_size=input_dim,
        #     hidden_size=hidden_dim,
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=False
        # )
        self.transfer_label = transfer_label
        
        if not transfer_label:
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        # avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        if not self.transfer_label:
            return self.classifier(max_pool)
            # return self.classifier(avg_pool)
        else:
            return max_pool
            # return avg_pool

class video_network(nn.Module):
    def __init__(self, loss_weights, transfer_label=False) -> None:
        super().__init__()
        # self.conv2d = getattr(models, 'resnet18')(pretrained=True)
        self.conv2d = getattr(models, 'resnet34')(pretrained=True)
        self.transfer_label = transfer_label
        num_ftrs = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Identity()
        if self.transfer_label:
            self.temporal_model = TaillightBiLSTM(input_dim=num_ftrs, hidden_dim=num_ftrs//2,
                                            num_classes=4, transfer_label=self.transfer_label)
            # self.temporal_model = TaillightBiLSTM(input_dim=num_ftrs, hidden_dim=num_ftrs,
            #                                 num_classes=4, transfer_label=self.transfer_label)
            # self.fc_brake = nn.Linear(num_ftrs*2, 2)
            # self.fc_turn = nn.Linear(num_ftrs*2, 5)
            self.fc_brake = nn.Linear(num_ftrs, 2)
            self.fc_turn = nn.Linear(num_ftrs, 5)
            self.loss_brake = nn.CrossEntropyLoss()
            self.loss_turn = nn.BCEWithLogitsLoss()
        else:
            self.temporal_model = TaillightBiLSTM(input_dim=num_ftrs, hidden_dim=num_ftrs,
                                            num_classes=4)
            self.loss = nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights
        

    def forward(self, inputs_dict):
        # x = inputs_dict['x'][2]
        # img = x.detach().cpu().numpy()
        # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img_bgr = (img_bgr * 255).clip(0, 255).astype(np.uint8)
        # cv2.imwrite('test.png', img_bgr)
        # pdb.set_trace()
        b, t, h, w, c = inputs_dict['x'].shape
        x = inputs_dict['x'].reshape(b*t, h, w, c)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d(x).reshape(b, t, -1)
        # pdb.set_trace()
        tm_outputs = self.temporal_model(x)
        if not self.transfer_label:
            return{
                'result': tm_outputs
            }
        else:
            turn_result = self.fc_turn(tm_outputs)
            brake_result = self.fc_brake(tm_outputs)
            return{
                'turn_result': turn_result,
                'brake_result': brake_result
            }

    def get_loss(self, ret_dict, label):
        loss, loss_dict = 0, {}
        if len(label) == 2:
            turn_label = label[1]
            brake_label = label[0]
        for k, weight in self.loss_weights.items():
            if k == 'all':
                temp = weight * self.loss(ret_dict['result'], label.float())
                loss += temp
                loss_dict[k] = temp
            if k == 'turn':
                temp = weight * self.loss_turn(ret_dict['turn_result'], turn_label.float())
                loss += temp
                loss_dict[k] = temp
            if k == 'brake':
                temp = weight * self.loss_brake(ret_dict['brake_result'], brake_label.squeeze(1).long())
                loss += temp
                loss_dict[k] = temp
        return loss, loss_dict