import os
import sys
import time

import hydra
from omegaconf import DictConfig

import torch
import torchvision
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datasets.patch_pcam import PatchPCAM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import show_params
from patch_splitnn import PatchSplitNN
from patch_splitnn_plus import PatchSplitNNplus

seed_everything()


@hydra.main(config_path='config', config_name='splitnn_pcam_config.yaml')
def main(args: DictConfig):
    start = time.time()

    # Display config parameters
    show_params(args)

    # Set GPUs
    assert torch.cuda.is_available, 'Please use a machine has GPUs.'
    gpus = 1 if args.backend == 'horovod' else args.gpus

    crop_size = 96
    padding_size = 12
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(crop_size, padding=padding_size),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.7008, 0.5384, 0.6916], std=[0.2350, 0.2774, 0.2128]),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.7008, 0.5384, 0.6916], std =[0.2350, 0.2774, 0.2128]),
        ]
    )

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = PatchPCAM(root=data_dir,
                          split="train",
                          transform=train_transforms,
                          download=False,
                          patch_size=args.patch_size,
                          patch_stride=args.patch_stride,
                          drop_rate=args.drop_rate)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_worker,
                              drop_last=True)

    val_set = PatchPCAM(root=data_dir,
                        split="val",
                        transform=test_transforms,
                        patch_size=args.patch_size,
                        patch_stride=args.patch_stride)

    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_worker,
                            drop_last=False)

    test_set = PatchPCAM(root=data_dir,
                         split="test",
                         transform=test_transforms,
                         patch_size=args.patch_size,
                         patch_stride=args.patch_stride)

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_worker,
                             drop_last=False)

    model_cls = PatchSplitNN if not args.adapt_net else PatchSplitNNplus

    model = model_cls(lr=args.lr,
                      batch_size=args.batch_size,
                      base_model=args.base_model,
                      num_classes=1,
                      patch_size=args.patch_size,
                      patch_stride=args.patch_stride,
                      num_uppmodels=args.num_uppmodels,
                      upp_loss_ratio=args.upp_loss_ratio)

    project_name = 'pcam-patch-splitnn'
    assert args.logger in ['tensorb', 'wandb'], "Set appropriate logger."
    if args.logger == 'tensorb':
        logger = TensorBoardLogger("Logs/", name=project_name)
    elif args.logger == 'wandb':
        logger = WandbLogger(project=project_name)

    trainer = Trainer(
        strategy=args.backend,
        max_epochs=args.epoch,
        gpus=gpus,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    print("Total processing time: %f sec"%(time.time() - start))


if __name__ == '__main__':
    main()
