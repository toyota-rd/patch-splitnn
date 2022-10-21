"""
This is implemented based on pl_bolts.datamodules.cifar10_datamodule.py

https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/datamodules/cifar10_datamodule.py
"""
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR100


def cifar100_normalization(): 
    normalize = transform_lib.Normalize(
        mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
        std=[x / 255.0 for x in [68.2, 65.4, 70.4]],
    )

    return normalize


class CIFAR100DataModule(VisionDataModule):
    dataset_cls = CIFAR100

    def __init__(
        self,
        data_dir=None,
        val_split=0.2,
        num_workers=16,
        normalize=False,
        batch_size=32,
        seed=42,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )
 
    def num_samples(self):
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    def num_classes(self):
        return 100
 
    def default_transforms(self):
        if self.normalize:
            cifar100_transforms = transforms.Compose([transform_lib.ToTensor(), cifar100_normalization()])
        else:
            cifar100_transforms = transforms.Compose([transform_lib.ToTensor()])
 
        return cifar100_transforms
