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

class Video_wise_Feeder(data.Dataset):
    def __init__(
        self,
        mode="train",
        transfer_label=False,
    ):
        self.mode = mode
        with open(f'./datasets/TLD/TLD_YT_{self.mode}.json', 'r', encoding='utf-8') as f:
            self.inputs_list = json.load(f)

        def downsample_off(data, target_off):

            off_samples = []
            other_samples = []
            for item in data:
                if item["turn_label"] == "off":
                    off_samples.append(item)
                else:
                    other_samples.append(item)
            if len(off_samples) <= target_off:
                print("Warning: off samples <= target_off, no downsampling applied.")
                return data
            sampled_off = random.sample(off_samples, target_off)
            new_data = sampled_off + other_samples

            random.shuffle(new_data)

            print("Original off:", len(off_samples))
            print("New off:", target_off)
            print("Other labels:", len(other_samples))
            print("Final dataset size:", len(new_data))

            return new_data
        
        if mode == "train":
            self.inputs_list = downsample_off(self.inputs_list, target_off=4000)

        print(mode, len(self))
        self.data_aug = self.video_transform()
        self.prefix = '/Users/yyf/Mine_Space/18744/project/dataset'
    
    def __getitem__(self, idx):
        input_data, label, bbx, fi = self.read_video(idx)

        input_data, label = self.normalize_and_crop(input_data, label, bbx)

        # print(torch.tensor(label, dtype=torch.long))

        return (
            input_data,
            torch.tensor(label, dtype=torch.long),
            fi,
        )
            
    def read_video(self, index):
        # load file info
        # print(self.inputs_list[index])
        filename = self.inputs_list[index]['file_name']
        label = self.inputs_list[index]['turn_label']
        bbx = self.inputs_list[index]['bounding_boxes']
        img_folder = [os.path.join(self.prefix, f.lstrip('/\\')) for f in filename]
        # print(img_folder)
        # img_folder = os.path.join(img_folder, "*.jpg")
        # img_list = sorted(glob.glob(img_folder))
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_folder], label, bbx, self.inputs_list[index]
    
    def normalize_and_crop(self, video_all, label, bbx):
        full_video = []
        for i in range(len(bbx)):
            bbx_temp = bbx[i]
            video = video_all[i]
            coordinate = np.array(bbx_temp)
            x_coords = coordinate[:, 0]
            y_coords = coordinate[:, 1]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            h, w = video.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            cropped_img = video[y1:y2, x1:x2, :]
            full_video.append(cropped_img)
        video = [cv2.resize(video_id, (224, 224)) for video_id in full_video]
        video = self.data_aug(video)
        video = video.float() / 127.5 - 1
        if label == 'off':
            turn_label = 0
        elif label == 'left':
            turn_label = 1
        elif label == 'right':
            turn_label = 2
        elif label == 'both':
            turn_label = 3
        elif label == 'unknow':
            turn_label = 4
        return video, turn_label

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
        # print(label)
        turn_label = torch.stack([img for img in label], dim=0)
        padded_video = torch.stack(video, dim=0)
        return {
            'x': padded_video,
            'label': turn_label,
            'origin_info': info,
            }