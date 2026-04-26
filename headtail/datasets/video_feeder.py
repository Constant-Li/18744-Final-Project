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
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import torch.utils.data as data
from utils import video_augmentation
from itertools import chain

sys.path.append("..")

class VideoFeeder(data.Dataset):
    def __init__(
        self,
        mode="train",
    ):
        self.mode = mode
        if self.mode == "test":
             with open(f'./test.json', 'r', encoding='utf-8') as f:
                self.inputs_list = json.load(f)
        else:
            with open(f'./train.json', 'r', encoding='utf-8') as f:
                self.inputs_list = json.load(f)
    
        print(mode, len(self))
        self.data_aug = self.video_transform()
    
    def __getitem__(self, idx):
        input_data, label, coordinate, fi = self.read_video(idx)

        input_data, label = self.normalize_and_crop(input_data, label, coordinate)
        return (
            input_data,
            label,
            fi,
        )
            
    def read_video(self, index):
        # load file info
        info_video = self.inputs_list[index]
        img_path = info_video['file_name']
        coordinate = info_video['coordinate']
        # if img_path.startswith('/TLD'):
        #     img_path = self.prefix_1 + img_path
        #     coordinate = info_video['coordinate']
        # else:
        #     coordinate = None
        label = info_video['label']
        data = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return (
            data,
            label,
            coordinate,
            info_video,
        )
    
    def normalize_and_crop(self, video, label, coordinate):
        video = self.data_aug(video)
        # if coordinate is not None:
        #     coordinate = np.array(coordinate)
        #     x_coords = coordinate[:, 0]
        #     y_coords = coordinate[:, 1]
        #     x1, y1 = int(min(x_coords)), int(min(y_coords))
        #     x2, y2 = int(max(x_coords)), int(max(y_coords))
        #     h, w = video.shape[:2]
        #     x1, y1 = max(0, x1), max(0, y1)
        #     x2, y2 = min(w, x2), min(h, y2)
        #     if x1 >= x2 or y1 >= y2:
        #         return [], []
        #     cropped_img = video[y1:y2, x1:x2, :]
        # else:
        cropped_img = video
        coordinate = np.array(coordinate)
        x_coords = coordinate[:, 0]
        y_coords = coordinate[:, 1]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        h, w = video.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            return [], []
        cropped_img = video[y1:y2, x1:x2, :]
        resize_transform = transforms.Resize((224, 224))
        full_video = resize_transform(cropped_img.permute(2, 0, 1)).permute(1, 2, 0)
        video = full_video.float() / 127.5 - 1
        return video, torch.LongTensor([label])

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
        video = torch.stack([img for img in video if img is not None])
        label = torch.stack([img for img in label if img is not None])
        return {
            'x': video,
            'label': label,
            'origin_info': info
            }