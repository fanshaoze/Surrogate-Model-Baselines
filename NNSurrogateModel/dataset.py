import copy
import random
import cv2
import torch
import torch.utils.data as data
import json
import numpy as np


class CustomDataset(data.Dataset):
    def __init__(self, raw_data=None, raw_labels=None, transform=None, target_transform=None):
        self.raw_data = raw_data
        self.raw_labels = raw_labels

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        _data = self.raw_data[idx]
        _label = self.raw_labels[idx]

        data_tensor = torch.Tensor(_data)
        label_tensor = torch.Tensor(_label).float()
        return data_tensor, label_tensor
