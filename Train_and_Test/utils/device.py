import os
import torch
import torch.nn as nn

class GpuDataParallel(object):
    def __init__(self):
        self.device = torch.device("cpu")

    def set_device(self, use_mps=True):
        if use_mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS acceleration.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

    def model_to_device(self, model):
        model = model.to(self.device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.Tensor):
            if data.dtype == torch.float64:
                return data.float().to(self.device)
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self.data_to_device(d) for d in data]
        else:
            return data

    def dict_data_to_device(self, data_dict):
        cuda_dict = {}
        for k, v in data_dict.items():
            if 'origin' in k or 'datasets' in k:
                cuda_dict[k] = v
            else:
                cuda_dict[k] = self.data_to_device(v)
        return cuda_dict

    def criterion_to_device(self, loss):
        return loss.to(self.device)

    def occupy_gpu(self):
        """
        Mac 不需要提前占用显存。
        """
        pass