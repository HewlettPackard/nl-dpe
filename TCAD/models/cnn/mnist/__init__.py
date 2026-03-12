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

from .lenet import lenet


def get_mnist_model(mode_name, pretrained=True, dropout=None):
    if mode_name == 'lenet':
        model = lenet(pretrained)
    else:
        print("unrecognized network")
        exit(0)

    return model


def prepare_mnist_data(batch_size, workers):
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_val_dataset = datasets.MNIST(root=os.path.join(os.environ['TORCH_HOME'], "mnist"), train=True, download=True, transform=mnist_transforms)
    test_dataset = datasets.MNIST(root=os.path.join(os.environ['TORCH_HOME'], "mnist"), train=False, download=True, transform=mnist_transforms)

    return train_val_dataset, test_dataset
