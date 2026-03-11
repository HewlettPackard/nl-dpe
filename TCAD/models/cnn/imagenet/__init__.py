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

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from .efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from .shufflenetv2 import shufflenet_v2_x1_0
from .densenet import densenet121


def get_imagenet_model(mode_name, pretrained=True, dropout=None, download_pretrained=False):
    if mode_name == 'efficientnet_v2_s':
        if dropout is not None:
            model = efficientnet_v2_s(pretrained, download_pretrained, dropout=dropout)
        else:
            model = efficientnet_v2_s(pretrained, download_pretrained)
    elif mode_name == 'efficientnet_v2_m':
        if dropout is not None:
            model = efficientnet_v2_m(pretrained, download_pretrained, dropout=dropout)
        else:
            model = efficientnet_v2_m(pretrained, download_pretrained)
    elif mode_name == 'efficientnet_v2_l':
        if dropout is not None:
            model = efficientnet_v2_l(pretrained, download_pretrained, dropout=dropout)
        else:
            model = efficientnet_v2_l(pretrained, download_pretrained)
    elif mode_name == 'resnet18':
        model = resnet18(pretrained, download_pretrained)
    elif mode_name == 'resnet34':
        model = resnet34(pretrained, download_pretrained)
    elif mode_name == 'resnet50':
        model = resnet50(pretrained, download_pretrained)
    elif mode_name == 'resnet101':
        model = resnet101(pretrained, download_pretrained)
    elif mode_name == 'resnet152':
        model = resnet152(pretrained, download_pretrained)
    elif mode_name == 'vgg11':
        if dropout is not None:
            model = vgg11(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg11(pretrained, download_pretrained)
    elif mode_name == 'vgg11_bn':
        if dropout is not None:
            model = vgg11_bn(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg11_bn(pretrained, download_pretrained)
    elif mode_name == 'vgg13':
        if dropout is not None:
            model = vgg13(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg13(pretrained, download_pretrained)
    elif mode_name == 'vgg13_bn':
        if dropout is not None:
            model = vgg13_bn(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg13_bn(pretrained, download_pretrained)
    elif mode_name == 'vgg16':
        if dropout is not None:
            model = vgg16(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg16(pretrained, download_pretrained)
    elif mode_name == 'vgg16_bn':
        if dropout is not None:
            model = vgg16_bn(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg16_bn(pretrained, download_pretrained)
    elif mode_name == 'vgg19':
        if dropout is not None:
            model = vgg19(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg19(pretrained, download_pretrained)
    elif mode_name == 'vgg19_bn':
        if dropout is not None:
            model = vgg19_bn(pretrained, download_pretrained, dropout=dropout)
        else:
            model = vgg19_bn(pretrained, download_pretrained)
    elif mode_name == 'shufflenetv2':
        model = shufflenet_v2_x1_0(pretrained, download_pretrained)
    elif mode_name == 'densenet121':
        model = densenet121(pretrained, download_pretrained)
    else:
        print("unrecognized network")
        exit(0)

    return model


def prepare_imagenet_data(batch_size, workers):
    traindir = 'data/imagenet_dataset/imagenet/train'
    valdir = 'data/imagenet_dataset/imagenet/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_dataset, val_dataset
