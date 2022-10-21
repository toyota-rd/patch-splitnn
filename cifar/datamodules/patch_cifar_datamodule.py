"""
This is implemented based on pl_bolts.datamodules.vision_datamodule.py

https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/datamodules/vision_datamodule.py
"""
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

from datasets.patch_cifar import PatchCIFAR10, PatchCIFAR100


class PatchCIFARDataModule(VisionDataModule):
    def __init__(
        self,
        data_dir=None,
        val_split=0.2,
        num_workers=0,
        normalize=False,
        batch_size=32,
        seed=42,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        patch_size=16,
        patch_stride=16,
        drop_rate=0.0,
        dataset=PatchCIFAR10,
        *args,
        **kwargs,
    ):
        super().__init__(data_dir,
                         val_split,
                         num_workers,
                         normalize,
                         batch_size,
                         seed,
                         shuffle,
                         pin_memory,
                         drop_last,
                         train_transforms,
                         val_transforms,
                         test_transforms,
                         *args,
                         **kwargs)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.drop_rate = drop_rate

        assert dataset in [PatchCIFAR10, PatchCIFAR100], "Set appropriate dataset."
        self.dataset_cls = dataset


    def prepare_data(self, *args, **kwargs):
        self.dataset_cls(self.data_dir, train=True, download=True, patch_size=self.patch_size, patch_stride=self.patch_stride, drop_rate=self.drop_rate)
        self.dataset_cls(self.data_dir, train=False, download=True, patch_size=self.patch_size, patch_stride=self.patch_stride, drop_rate=self.drop_rate)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(self.data_dir, train=True, transform=train_transforms, patch_size=self.patch_size, patch_stride=self.patch_stride, drop_rate=self.drop_rate, **self.EXTRA_ARGS)
            dataset_val = self.dataset_cls(self.data_dir, train=True, transform=val_transforms, patch_size=self.patch_size, patch_stride=self.patch_stride, drop_rate=self.drop_rate, **self.EXTRA_ARGS)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, patch_size=self.patch_size, patch_stride=self.patch_stride, drop_rate=self.drop_rate, **self.EXTRA_ARGS
            )
