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
import albumentations as A


rpm_folders = {'cs': "center_single",
               'io': "in_center_single_out_center_single",
               'ud': "up_center_single_down_center_single",
               'lr': "left_center_single_right_center_single",
               'd4': "distribute_four",
               'd9': "distribute_nine",
               '4c': "in_distribute_four_out_center_single",
               '*': '*'}

class Augmentor:
    """Applies the same Albumentations transform to a stack of images (panels)."""
    def __init__(self, num_panels, p= 0.0, transforms = ()):
        self.num_panels = num_panels
        self.transform = A.Compose(
            transforms,
            additional_targets={f"image{i + 1}": "image" for i in range(num_panels - 1)},
            p=p,
        )

    def augment(self, images):
        """Applies the transform to all images in the stack."""
        num_panels = images.shape[0]
        kwargs = {f"image{i + 1}": images[i + 1] for i in range(num_panels - 1)}
        augmented = self.transform(image=images[0], **kwargs)
        return np.stack(
            [augmented["image"]] + [augmented[f"image{i+1}"] for i in range(num_panels - 1)]
        )


class dataset(Dataset):
    def __init__(self, args, mode, rpm_types, transform=None, transform_p=1.0, num_panels=16):
        self.root_dir = args.path
        self.img_size = args.img_size
        # self.set = args.dataset
        self.mode = mode
        self.percent = args.percent if mode != "test" else 100
        # self.shuffle_first = args.shuffle_first
        self.resizer = A.Compose([
            A.Resize(self.img_size, self.img_size),
        ])

        self.transform = [
            A.VerticalFlip(p=0.25),
            A.HorizontalFlip(p=0.25),
        ]

        if self.mode == "train" and transform:
            self.transform = transform

        self.augmentor = Augmentor(
            num_panels=num_panels,
            transforms=self.transform
        )

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
            
        resized_panels = [self.resizer(image=panel)['image'] for panel in images]
        images = np.stack(resized_panels)
        images = self.augmentor.augment(images)

        images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize to [0, 1] 
        target = torch.tensor(target, dtype=torch.long)

        images = images.unsqueeze(1)

        return images, target
