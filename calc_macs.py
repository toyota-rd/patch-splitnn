import argparse
import math

import torch
from torch import nn
from thop import profile
from thop import clever_format

from models import split_resnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MACs.')

    parser.add_argument('--dataset', default='cifar10', type=str, help='set the name of dataset.')
    parser.add_argument('--patch-size', default=16, type=int, help='size of patch image')
    parser.add_argument('--patch-stride', default=8, type=int, help='size of patch stride')

    args = parser.parse_args()    

    upper_model = split_resnet.upper_resnet('resnet18')

    assert args.dataset in ['cifar10', 'cifar100', 'pcam']
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        original_size = 32
    elif args.dataset == 'pcam':
        original_size = 96
        upper_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        upper_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    if args.patch_stride != 0:
        num_line_patches = int(((original_size-args.patch_size) / args.patch_stride) + 1)
    else:
        num_line_patches = 1

    lower_model = split_resnet.lower_resnet('resnet18', num_classes=10)

    upper_input = torch.randn(1, 3, args.patch_size, args.patch_size)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        input_size = args.patch_size * num_line_patches
        lower_input = torch.randn(1, 64, input_size, input_size)
    elif args.dataset == 'pcam':
        input_size =  math.ceil(args.patch_size/4) * num_line_patches
        lower_input = torch.randn(1, 64, input_size, input_size)

    upper_macs, upper_params = profile(upper_model, inputs=(upper_input, ))
    lower_macs, lower_params = profile(lower_model, inputs=(lower_input, ))

    upper_macs, upper_params = clever_format([upper_macs, upper_params], "%.3f")
    print('< Upper Model >')
    print("MACs   = {}".format(upper_macs))
    print("Params = {}".format(upper_params))

    lower_macs, lower_params = clever_format([lower_macs, lower_params], "%.3f")
    print('< Lower Model >')
    print("MACs   = {}".format(lower_macs))
    print("Params = {}".format(lower_params))
