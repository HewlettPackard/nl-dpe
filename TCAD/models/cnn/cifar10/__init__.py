###
# Copyright (2026) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .efficientnet_v2 import get_efficientnet_v2
from .efficientnet import EfficientNetB0
from .resnet9 import resnet9
from .resnet import resnet18, resnet34, resnet50
from .senet import SENet18
from .vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


def get_cifar10_model(mode_name, pretrained=True, dropout=None):
    if mode_name == 'efficientnet_v2_s':
        if dropout is not None:
            model = get_efficientnet_v2("efficientnet_v2_s", pretrained, nclass=10, dropout=dropout)
        else:
            model = get_efficientnet_v2("efficientnet_v2_s", pretrained, nclass=10)
    elif mode_name == 'efficientnet':
        if dropout is not None:
            model = EfficientNetB0(pretrained, dropout=dropout)
        else:
            model = EfficientNetB0(pretrained)
    elif mode_name == 'resnet9':
        model = resnet9(pretrained)
    elif mode_name == 'resnet18':
        model = resnet18(pretrained)
    elif mode_name == 'resnet34':
        model = resnet34(pretrained)
    elif mode_name == 'resnet50':
        model = resnet50(pretrained)
    elif mode_name == 'senet':
        model = SENet18(pretrained)
    elif mode_name == 'vgg11_bn':
        model = vgg11_bn(pretrained)
    elif mode_name == 'vgg13_bn':
        model = vgg13_bn(pretrained)
    elif mode_name == 'vgg16_bn':
        model = vgg16_bn(pretrained)
    elif mode_name == 'vgg19_bn':
        model = vgg19_bn(pretrained)
    else:
        print("unrecognized network")
        exit(0)

    return model


def prepare_cifar10_data(batch_size, workers, use_imagenet_input_size=False):
    if use_imagenet_input_size:
        transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.CenterCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = datasets.CIFAR10(root=os.path.join(os.environ['TORCH_HOME'], "cifar10"), train=True, download=True, transform=transform_train)
    valset = datasets.CIFAR10(root=os.path.join(os.environ['TORCH_HOME'], "cifar10"), train=False, download=True, transform=transform_val)

    return trainset, valset
