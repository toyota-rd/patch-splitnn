import time

import hydra
from omegaconf import DictConfig

import torch
import torchvision

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datasets.patch_cifar import PatchCIFAR10, PatchCIFAR100
from datamodules.patch_cifar_datamodule import PatchCIFARDataModule
from datamodules.cifar100_datamodule import cifar100_normalization

from patch_splitnn_cifar import PatchSplitNN
from patch_splitnn_plus_cifar import PatchSplitNNplus
from utils.util import show_params

seed_everything()


@hydra.main(config_path='config', config_name='splitnn_cifar_config.yaml')
def main(args: DictConfig):
    start = time.time()

    # Display config parameters
    show_params(args)
    
    # Check patch size and stride
    assert args.patch_size >= args.patch_stride, 'patch_stride should be smaller than patch_size.'

    # Check patch drop rate
    assert args.drop_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "Set appropriate range."

    # Set GPUs
    assert torch.cuda.is_available, 'Please use a machine has GPUs.'
    gpus = 1 if args.backend == 'horovod' else args.gpus

    assert args.dataset in ["PatchCIFAR10", 'PatchCIFAR100'], "Set appropriate dataset."
    crop_size = 32
    padding_size = 4
    if args.dataset == 'PatchCIFAR10':
        dataset_cls = PatchCIFAR10
        num_classes = 10
        normalization = cifar10_normalization()
    elif args.dataset == 'PatchCIFAR100':
        dataset_cls = PatchCIFAR100
        num_classes = 100
        normalization = cifar100_normalization()

    s = 0.5
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(crop_size, padding=padding_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalization,
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            normalization,
        ]
    )

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    cifar_dm = PatchCIFARDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        drop_rate=args.drop_rate,
        dataset=dataset_cls)

    model_cls = PatchSplitNN if not args.adapt_net else PatchSplitNNplus

    model = model_cls(lr=args.lr,
                      batch_size=args.batch_size,
                      base_model=args.base_model,
                      num_classes=num_classes,
                      patch_size=args.patch_size,
                      patch_stride=args.patch_stride,
                      num_uppmodels=args.num_uppmodels,
                      upp_loss_ratio=args.upp_loss_ratio,
                      drop_rate=args.drop_rate)
    model.datamodule = cifar_dm

    project_name = 'patch-splitnn-{}'.format(args.dataset)
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

    trainer.fit(model, datamodule=cifar_dm)
    trainer.test(model, datamodule=cifar_dm)

    print("Total processing time: %f sec"%(time.time() - start))

if __name__ == '__main__':
    main()
