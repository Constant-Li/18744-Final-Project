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
        # with open(f'./datasets/TLD_YT_{self.mode}.json', 'r', encoding='utf-8') as f:
        #     self.inputs_list = json.load(f)
        with open(f'./datasets/TLD.json', 'r', encoding='utf-8') as f:
            self.inputs_list = json.load(f)['TLD_YT']

        # def downsample_off(input_list, keep_off=20000):

        #     off_indices = []
        #     for i, item in enumerate(input_list):
        #         if 'car_label' not in item.keys():
        #             continue
        #         for j, car in enumerate(item["car_label"]):
        #             if car["turn_label"] == "off":
        #                 off_indices.append((i, j))
        #     keep_off_indices = set(random.sample(off_indices, keep_off))
        #     new_input_list = []
        #     for i, item in enumerate(input_list):
        #         new_item = copy.deepcopy(item)
        #         new_car_label = []
        #         if 'car_label' not in item.keys():
        #             continue
        #         for j, car in enumerate(item["car_label"]):
        #             if car["turn_label"] == "off":
        #                 if (i, j) in keep_off_indices:
        #                     new_car_label.append(car)
        #             else:
        #                 new_car_label.append(car)
        #         new_item["car_label"] = new_car_label
        #         new_item["car_num"] = len(new_car_label)
        #         if new_item["car_num"] > 0:
        #             new_input_list.append(new_item)
        #     return new_input_list
        
        # if mode == "train":
        #     self.inputs_list = downsample_off(self.inputs_list, keep_off=3000)
    
        print(mode, len(self))
        self.data_aug = self.video_transform()
        self.car_sum_num = sum(item['car_num'] for item in self.inputs_list)
        self.prefix = '/Users/yyf/Mine_Space/18744/project/dataset'
    
    def __getitem__(self, idx):
        input_data, label, car_num, fi = self.read_video(idx)

        input_data, label = self.normalize_and_crop(input_data, label, car_num)
        return (
            input_data,
            label,
            car_num,
            fi,
        )
            
    def read_video(self, index):
        # load file info
        info_video = self.inputs_list[index]
        img_path = info_video['file_name']
        img_path = self.prefix + img_path
        car_num = info_video['car_num']
        label_list = info_video['car_label'] if info_video['car_num'] != 0 else None
        data = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return (
            data,
            label_list,
            car_num,
            info_video,
        )
    
    def normalize_and_crop(self, video, label, car_num):
        video = self.data_aug(video)
        full_label = []
        full_video = []
        if car_num != 0:
            for i in range(car_num):
                car_label = label[i]
                coordinate = np.array(car_label['bounding_boxes']['coordinate'])
                x_coords = coordinate[:, 0]
                y_coords = coordinate[:, 1]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                h, w = video.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x1 >= x2 or y1 >= y2:
                    car_num -= 1
                    continue
                cropped_img = video[y1:y2, x1:x2, :]
                # print(cropped_img.shape)
                full_video.append(cropped_img)
                brake_label = 0 if car_label['brake_label'] == 'car_BrakeOff' else 1
                if car_label['turn_label'] == 'off':
                    turn_label = 0
                elif car_label['turn_label'] == 'left':
                    turn_label = 1
                elif car_label['turn_label'] == 'right':
                    turn_label = 2
                elif car_label['turn_label'] == 'both':
                    turn_label = 3
                elif car_label['turn_label'] == 'unknow':
                    turn_label = 4
                full_label.append((brake_label, turn_label))
            resize_transform = transforms.Resize((224, 224))
            full_video = [resize_transform(video.permute(2, 0, 1)).permute(1, 2, 0) for video in full_video]
            video = [video_temp.float() / 127.5 - 1 for video_temp in full_video]
            return video, [torch.LongTensor(label) for label in full_label]
        else:
            return full_video, full_label

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
        video, label, car_num, info = list(zip(*batch))
        sum_num = sum(car_num)
        if sum_num == 0:
            return None
        video = torch.stack([img for sub_list in video for img in sub_list])
        label = torch.stack([img for sub_list in label for img in sub_list])
        return {
            'x': video,
            'label': label,
            'car_num': sum_num,
            'origin_info': info
            }