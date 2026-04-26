import os
import sys
import pdb
import json
import torch
import pickle
import warnings
import itertools
import random
import cv2
from torchvision import transforms
import glob

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import torch.utils.data as data
from utils import video_augmentation
from itertools import chain

sys.path.append("..")

class Video_Feeder(data.Dataset):
    def __init__(
        self,
        mode="train",
        transfer_label=False,
    ):
        self.mode = mode
        self.transfer_label = transfer_label
        with open(f'./datasets/ETR/{self.mode}.json', 'r', encoding='utf-8') as f:
            self.inputs_list = json.load(f)

        print(mode, len(self))
        self.data_aug = self.video_transform()
        self.prefix = '/Users/yyf/Mine_Space/18744/project/dataset/ETR/ALL/'
    
    def __getitem__(self, idx):
        input_data, label, fi = self.read_video(idx)

        input_data = self.normalize_and_crop(input_data)

        if self.transfer_label:
            turn_label = np.zeros(5)
            brake_label = np.array([0])
            if label[0] == 1 and label[1] == 1:
                # turn_label[3] = 1
                brake_label[0] = 1
            elif label[0] == 1 or label[1] == 1:
                brake_label[0] = 1

            if label[2] == 1:
                turn_label[1] = 1
            
            if label[3] == 1:
                turn_label[2] = 1
            
            if label[2] == 0 and label[3] == 0:
                turn_label[0] = 1
            return (
                input_data,
                (torch.LongTensor(brake_label), torch.LongTensor(turn_label), torch.LongTensor(label)),
                fi,
            )
        else:
            return (
                input_data,
                torch.LongTensor(label),
                fi,
            )
            
    def read_video(self, index):
        # load file info
        filename = list(self.inputs_list.keys())[index]
        label = list(self.inputs_list.values())[index]
        img_folder = os.path.join(self.prefix, filename)
        img_folder = os.path.join(img_folder, "*.jpg")
        img_list = sorted(glob.glob(img_folder))
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label, img_folder
    
    def normalize_and_crop(self, video):
        video = [cv2.resize(video_id, (224, 224)) for video_id in video]
        video = self.data_aug(video)
        video = video.float() / 127.5 - 1
        return video

    def video_transform(self):
        if self.mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose(
                    [
                        video_augmentation.ToTensor(),
                    ]
                )                
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose(
                [
                    video_augmentation.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.inputs_list) - 1
    
    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(label[0]) == 3:
            brake_label = torch.stack([img[0] for img in label], dim=0)
            turn_label = torch.stack([img[1] for img in label], dim=0)
            original_label = torch.stack([img[2] for img in label], dim=0)
            padded_video = torch.stack(video, dim=0)
            return {
                'x': padded_video,
                'label': (brake_label, turn_label),
                'origin_info': info,
                'original_label': original_label
                }
        else:
            label = torch.stack(label, dim=0)
            padded_video = torch.stack(video, dim=0)
            return {
                'x': padded_video,
                'label': label,
                'origin_info': info
                }