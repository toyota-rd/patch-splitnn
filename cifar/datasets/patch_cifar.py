"""
This is implemented based on torchivision.datasets.cifar.py

https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py
"""
import numpy as np
import random
from PIL import Image

import torch
from torchvision.datasets import CIFAR10


class PatchCIFAR10(CIFAR10):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        patch_size=16,
        patch_stride=16,
        drop_rate=0.0,
    ):
        super().__init__(root, train, transform, target_transform, download)

        # for dividing to patches
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # culculate total and line number of patches
        if self.patch_stride != 0:
            self.num_line_patches = int(((32-patch_size) / patch_stride) + 1)
            self.total_num_patches = self.num_line_patches * self.num_line_patches
        else:
            self.num_line_patches = 1
            self.total_num_patches = 1
        self.drop_rate = drop_rate

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # All patches: c h w
        if self.train:
            if self.drop_rate > 0.0:
                drop_idx = np.arange(self.total_num_patches)
                drop_idx = random.sample(list(drop_idx), int(len(drop_idx) * self.drop_rate))

        patch_imgs = []
        for i in range(self.num_line_patches):
            h_start = i * self.patch_stride
            h_end = i * self.patch_stride + self.patch_size

            for j in range(self.num_line_patches):
                w_start = j * self.patch_stride
                w_end = j * self.patch_stride + self.patch_size
                
                # tensor = [c h w]
                if self.train and self.drop_rate > 0.0:
                    if i*self.num_line_patches + j in drop_idx:
                        patch = torch.zeros(3, self.patch_size, self.patch_size, dtype=torch.float)
                    else:
                        patch = img[:,h_start:h_end, w_start:w_end]
                else:
                    patch = img[:,h_start:h_end, w_start:w_end]

                patch_imgs.append(patch)
                
        return torch.stack(patch_imgs), target


class PatchCIFAR100(PatchCIFAR10):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"

    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    
    meta = {"filename": "meta",
            "key": "fine_label_names",
            "md5": "7973b15100ade9c7d40fb424638fde48"}
