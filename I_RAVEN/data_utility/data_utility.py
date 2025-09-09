# -*- coding:utf-8 -*-
# Adapted from code by 'Vecchio' at https://github.com/VecchioID/DRNet

import os
import glob
import numpy as np
import sys
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms

rpm_folders = {'cs': "center_single",
               'io': "in_center_single_out_center_single",
               'ud': "up_center_single_down_center_single",
               'lr': "left_center_single_right_center_single",
               'd4': "distribute_four",
               'd9': "distribute_nine",
               '4c': "in_distribute_four_out_center_single",
               '*': '*'}


class dataset(Dataset):
    def __init__(self, args, mode, rpm_types, transform=None, transform_p=1.0):
        self.root_dir = args.path
        self.img_size = args.img_size
        # self.set = args.dataset
        self.mode = mode
        self.percent = args.percent if mode != "test" else 100
        # self.shuffle_first = args.shuffle_first
        self.resize = transforms.Resize(self.img_size)
        self.transform = transform if transform else transforms.Compose([
                                             transforms.RandomHorizontalFlip(p=0.3),
                                             transforms.RandomVerticalFlip(p=0.3)]) 


        self.ctx_size = 8
        self.candidate_size = 8 
        self.transform_p = transform_p

        file_names = [[f for f in glob.glob(os.path.join(self.root_dir, rpm_folders[t], "*.npz")) if mode in f] for t in rpm_types]
        [random.shuffle(sublist) for sublist in file_names]
        file_names = [item for sublist in file_names for item in sublist[:int(len(sublist) * self.percent / 100)]]
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def shuffle(self, obj, pos):
        frames_o = []
        frames_p = []
        for f in zip(obj, pos):
            idx = torch.randperm(obj.size(1))
            frames_o.append(f[0][idx])
            frames_p.append(f[1][idx])
        obj = torch.stack(frames_o)
        pos = torch.stack(frames_p)
        return obj, pos

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        target = data["target"]
        images = data["image"]

        if self.mode == "train":
            context = images[:8]
            choices = images[8:]
            idx = list(range(8))
            np.random.shuffle(idx)
            images = np.concatenate((context, choices[idx]))
            target = idx.index(target)
            
        images = torch.tensor(images, dtype=torch.float32)
        images = self.resize(images)

        if self.mode == 'train':
            if self.transform_p > random.random():
                images = self.transform(images)

        target = torch.tensor(target, dtype=torch.long)

        images = images.unsqueeze(1)

        return images, target
