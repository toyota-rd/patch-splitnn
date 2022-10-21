"""
This is implemented based on torchivision.datasets.pcam.py

https://github.com/pytorch/vision/blob/main/torchvision/datasets/pcam.py
"""
import numpy as np
import random
from PIL import Image

import torch
from torchvision.datasets import PCAM


class PatchPCAM(PCAM):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        patch_size=48,
        patch_stride=48,
        drop_rate=0.0,
    ):
        super().__init__(root, split, transform, target_transform, download)

        # For Patch SplitNN
        # for dividing to patches
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # culculate total and line number of patches
        if self.patch_stride != 0:
            self.num_line_patches = int(((96-patch_size) / patch_stride) + 1)
            self.total_num_patches = self.num_line_patches * self.num_line_patches
        else:
            self.num_line_patches = 1
            self.total_num_patches = 1
        self.drop_rate = drop_rate


    def __getitem__(self, idx):
        images_file = self._FILES[self._split]["images"][0]
        with self.h5py.File(self._base_folder / images_file) as images_data:
            image = Image.fromarray(images_data["x"][idx]).convert("RGB")

        targets_file = self._FILES[self._split]["targets"][0]
        with self.h5py.File(self._base_folder / targets_file) as targets_data:
            target = int(targets_data["y"][idx, 0, 0, 0])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        # All patches: c h w
        if self._split == 'train':
            if self.drop_rate > 0.0:
                drop_idx = np.arange(self.total_num_patches)
                drop_idx = random.sample(list(drop_idx), int(len(drop_idx) * self.drop_rate))

        # All patches: c h w
        patch_imgs = []
        for i in range(self.num_line_patches):
            h_start = i * self.patch_stride
            h_end = i * self.patch_stride + self.patch_size

            for j in range(self.num_line_patches):
                w_start = j * self.patch_stride
                w_end = j * self.patch_stride + self.patch_size
                
                # tensor = [c h w]
                if self._split == 'train' and self.drop_rate > 0.0:
                    if i*self.num_line_patches + j in drop_idx:
                        patch = torch.zeros(3, self.patch_size, self.patch_size, dtype=torch.float)
                    else:
                        patch = image[:,h_start:h_end, w_start:w_end]
                else:
                    patch = image[:,h_start:h_end, w_start:w_end]

                patch_imgs.append(patch)

        return torch.stack(patch_imgs), target
