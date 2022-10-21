import math
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import accuracy, auroc

from pytorch_lightning import LightningModule

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import split_resnet


class PatchSplitNNplus(LightningModule):
    def __init__(
        self,
        lr=0.05,
        batch_size=64,
        base_model='resnet18',
        num_classes=1,
        patch_size=48,
        patch_stride=24,
        num_uppmodels=1,
        upp_loss_ratio=1.0,
        ):
        
        super().__init__()

        self.save_hyperparameters()
        self.hparams.lr = lr
        self.batch_size = batch_size
        self.base_model = base_model
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_uppmodels = num_uppmodels
        self.upp_loss_ratio = upp_loss_ratio

        # PCAM image size = (96, 96)
        if patch_size == 96:
            self.num_all_patches = 1
        else:
            self.num_line_patches = int(((96-patch_size) / patch_stride) + 1)
            self.num_all_patches = int(self.num_line_patches * self.num_line_patches)

        self.upper_model, self.lower_model = create_model(self.base_model,
                                                          self.num_uppmodels,)
        self.sig = nn.Sigmoid()

        self.upp_loss_ratio = upp_loss_ratio
        output_size = [int(math.ceil(patch_size/(2*4))), int(math.ceil(patch_size/(2*4)))]
        ch_1 = 512
        ch_2 = 256
        self.upp_fc = nn.Sequential(nn.AdaptiveAvgPool2d((output_size[0], output_size[1])),
                                    nn.Flatten(),
                                    nn.Linear(64 * output_size[0] * output_size[1], ch_1),
                                    nn.BatchNorm1d(ch_1),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(ch_1, ch_2),
                                    nn.BatchNorm1d(ch_2),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(ch_2, self.num_classes),
                                    nn.Sigmoid())

        print('-----')
        print('Num of Img Patches: {}'.format(self.num_all_patches))
        print('Num of Upp Models : {}'.format(self.num_uppmodels))
        print('------')

        self.all_logits = torch.FloatTensor([])
        self.all_y = torch.LongTensor([])

    def forward(self, x, stage):
        uppmodel_idx = 0
        x_perm = x.permute(1, 0, 2, 3, 4)

        upp_out_tmp = []
        upp_out = []

        j = 0
        for i in range(x_perm.size(0)):
            if uppmodel_idx == len(self.upper_model):
                uppmodel_idx = 0

            tmp = self.upper_model[uppmodel_idx](x_perm[i])
            upp_out_tmp.append(self.upp_fc(tmp))

            if i % self.num_line_patches == 0:
                tmp_x = tmp
            else:
                tmp_x = torch.cat([tmp_x, tmp], dim=3)

            if i % self.num_line_patches == self.num_line_patches - 1:
                if j == 0:
                    tmp_img_t = tmp_x
                else:
                    tmp_img_t = torch.cat([tmp_img_t, tmp_x], dim=2)
                j += 1

            uppmodel_idx += 1

        upp_out = upp_out_tmp

        out = self.lower_model(tmp_img_t)
        out = self.sig(out)

        return out, upp_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = y.to(torch.float)
        y_ = torch.cuda.FloatTensor(y_)

        logits, upp_logits = self(x, stage="train")

        logits = torch.squeeze(logits)
        loss = F.binary_cross_entropy(logits, y_)

        upp_loss = 0.0
        tmp_loss = [0.0] * self.num_uppmodels
        # Related number of patches
        for i, val in enumerate(upp_logits):
            val = torch.squeeze(val)
            tmp_loss[int(i % self.num_uppmodels)] += F.binary_cross_entropy(val, y_)

        # Related number of upper models
        for i, val in enumerate(tmp_loss):
            upp_loss += val / (self.num_all_patches / self.num_uppmodels)
            #self.log("train_upp_loss_%d"%(i), val)

        upp_loss = self.upp_loss_ratio * upp_loss

        loss = loss + upp_loss
        
        self.log("train_loss", loss)
        self.log("train_upp_loss", upp_loss)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_ = y.to(torch.float)
        y_ = torch.cuda.FloatTensor(y_)

        logits, upp_logits = self(x, stage="train")

        logits = torch.squeeze(logits)
        loss = F.binary_cross_entropy(logits, y_)

        upp_loss = 0.0
        tmp_loss = [0.0] * self.num_uppmodels
        # Related number of patches
        for i, val in enumerate(upp_logits):
            val = torch.squeeze(val)
            tmp_loss[int(i % self.num_uppmodels)] += F.binary_cross_entropy(val, y_)

        # Related number of upper models
        for i, val in enumerate(tmp_loss):
            upp_loss += val / (self.num_all_patches / self.num_uppmodels)
            #self.log(f"{stage}_upp_loss_%d"%(i), val)

        upp_loss = self.upp_loss_ratio * upp_loss
        full_loss = loss + upp_loss

        if stage == 'val':
            preds = (logits>0.5).float()
            val_acc = accuracy(preds, y)
            self.log(f"{stage}_full_loss", full_loss, prog_bar=True)
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", val_acc, prog_bar=True)

        # Calculate AUC at the last epoch
        if stage == 'test':
            self.all_logits = torch.cat((self.all_logits, logits.cpu()))
            self.all_y = torch.cat((self.all_y, y.cpu()))

            self.log(f"{stage}_full_loss", full_loss, prog_bar=True)
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


def create_model(base_model, num_uppmodels):
    upper_model_list = []

    assert base_model in ['resnet18', 'resnet34'], "Set appropriate model_name."
    for _ in range(num_uppmodels):
        tmp_upper_model = split_resnet.upper_resnet(base_model)
        tmp_upper_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        tmp_upper_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        upper_model_list.append(tmp_upper_model)

    upper_model = nn.ModuleList(upper_model_list)
    lower_model = split_resnet.lower_resnet(base_model, num_classes=1)

    return upper_model, lower_model
