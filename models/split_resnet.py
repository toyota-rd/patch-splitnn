"""
This is implemented based on torchivision.models.resnet.py

https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn

from torchvision.models.resnet import conv1x1, BasicBlock, Bottleneck


class UpperResNet(nn.Module):
    def __init__(self, block, layers, channels):
        super().__init__()
     
        self.inplanes = 64

        # For CIFAR10
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, channels[0], layers[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, norm_layer=nn.BatchNorm2d)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        return x


class LowerResNet(nn.Module):
    def __init__(self, block, layers, channels, num_classes):
        super().__init__()

        self.inplanes = channels[0]

        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, norm_layer=nn.BatchNorm2d)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def upper_resnet(base_model='resnet18'):
    assert base_model in ['resnet18', 'resnet34'], "Set appropriate model_name."
    if base_model == 'resnet18':
        return UpperResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512])
    elif base_model == 'resnet34':
        return UpperResNet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512])

def lower_resnet(base_model='resnet18', num_classes=10):
    assert base_model in ['resnet18', 'resnet34'], "Set appropriate model_name."
    if base_model == 'resnet18':
        return LowerResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], num_classes)
    elif base_model == 'resnet34':
        return LowerResNet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], num_classes)
