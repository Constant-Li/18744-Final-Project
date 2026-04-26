import torch
import random
import numpy as np

EPS = 1e-4


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video


class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video).float()
        return video

class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2) -> None:
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))

        start_idx = 0 if random.uniform(0,1) > 0.5 else 1
        index_ = list(range(start_idx, new_len, 2))
        index_rgb = [index[num] for num in index_]
        return clip[index_rgb]

class TemporalRescale_test(object):
    def __call__(self, clip):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = vid_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        index = [i for i in range(new_len)]
        for i in range(vid_len, new_len):
            index[i] = index[vid_len-1]
        index_rgb = index[::2]
        return clip[index_rgb]
