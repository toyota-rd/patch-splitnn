import os
import sys
import time

import hydra
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.datasets import PCAM
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import accuracy, auroc
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import show_params

seed_everything()

class PCamResnet(LightningModule):
    def __init__(
        self,
        lr=0.05,
        batch_size=64,
        model_name='resnet18',
        ):
        
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.hparams.lr = lr

        print('------ ResNet -----')
        print('LR: {}'.format(self.hparams.lr))
        print('----------------------------')

        self.model = create_model(model_name, num_classes=1)
        self.sig = nn.Sigmoid()
        self.all_logits = torch.FloatTensor([])
        self.all_y = torch.LongTensor([])

    def forward(self, x):
        out = self.model(x)
        out = self.sig(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = y.to(torch.float)
        y_ = torch.cuda.FloatTensor(y_)

        logits = self(x)
        logits = torch.squeeze(logits)
        loss = F.binary_cross_entropy(logits, y_)
        self.log("train_loss", loss)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_ = y.to(torch.float)
        y_ = torch.cuda.FloatTensor(y_)

        logits = self(x)
        logits = torch.squeeze(logits)

        loss = F.binary_cross_entropy(logits, y_)

        if stage == 'val':
            preds = (logits>0.5).float()
            val_acc = accuracy(preds, y)
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", val_acc, prog_bar=True)

        # Calculate AUC at the last epoch
        if stage == 'test':
            self.all_logits = torch.cat((self.all_logits, logits.cpu()))
            self.all_y = torch.cat((self.all_y, y.cpu()))

            self.log(f"{stage}_loss", loss, prog_bar=True)

            if self.current_epoch + 1 == self.trainer.max_epochs:
                acc = accuracy((self.all_logits>0.5).float(), self.all_y)
                auc = auroc(self.all_logits, self.all_y, pos_label=1)
                self.log(f"{stage}_acc", acc, prog_bar=True)
                self.log(f"{stage}_auc", auc, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler_dict = {
            "scheduler": MultiStepLR(
                optimizer,
                milestones=[20, 35, 50],
                gamma=0.1,
            )
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def create_model(model_name, num_classes):
    assert model_name in ['resnet18', 'resnet34'], "Set appropriate model_name."
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)

    return model


@hydra.main(config_path='config', config_name='resnet_pcam_config.yaml')
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
            torchvision.transforms.Normalize(mean=[0.7008, 0.5384, 0.6916], std=[0.2350, 0.2774, 0.2128]),
        ]
    )

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    print(data_dir)
    train_set = PCAM(root=data_dir,
                     split="train",
                     transform=train_transforms,
                     download=False)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_worker,
                              drop_last=True)

    val_set = PCAM(root=data_dir,
                   split="val",
                   transform=test_transforms)

    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_worker,
                            drop_last=False)

    test_set = PCAM(root=data_dir,
                    split="test",
                    transform=test_transforms)

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_worker,
                             drop_last=False)

    model = PCamResnet(lr=args.lr,
                       batch_size=args.batch_size,
                       model_name=args.model_name)

    project_name = 'pcam-resnet-18'
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
