import os
import sys
import time

import hydra
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datamodules.cifar100_datamodule import CIFAR100DataModule, cifar100_normalization

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import show_params

seed_everything()


class CIFARResnet(LightningModule):
    def __init__(
        self,
        lr=0.05,
        batch_size=64,
        model_name='resnet18',
        num_classes=10,
        ):
        
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = create_model(model_name, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        max_lr = 0.1
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

def create_model(model_name, num_classes):
    assert model_name in ['resnet18', 'resnet34'], "Set appropriate model_name."
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    return model


@hydra.main(config_path='config', config_name='resnet_cifar_config.yaml')
def main(args: DictConfig):
    start = time.time()

    # Display config parameters
    show_params(args)

    # Set GPUs
    assert torch.cuda.is_available, 'Please use a machine has GPUs.'
    gpus = 1 if args.backend == 'horovod' else args.gpus

    assert args.dataset in ['CIFAR10', 'CIFAR100'], "Set appropriate dataset."
    if args.dataset == "CIFAR10":
        num_classes = 10
        normalization = cifar10_normalization()
        datamd_cls = CIFAR10DataModule
    elif args.dataset == "CIFAR100":
        num_classes = 100
        normalization = cifar100_normalization()
        datamd_cls = CIFAR100DataModule

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
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


    cifar_dm = datamd_cls(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_worker,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )

    model = CIFARResnet(lr=args.lr,
                        batch_size=args.batch_size,
                        model_name=args.model_name,
                        num_classes=num_classes)
    model.datamodule = cifar_dm

    project_name = 'reset-18-{}'.format(args.dataset)
    assert args.logger in ['tensorb', 'wandb'], "Set appropriate logger."
    if args.logger == 'tensorb':
        log_name = "{}-{}".format(args.model_name, args.epoch)
        logger = TensorBoardLogger("Logs/", name=log_name)
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
