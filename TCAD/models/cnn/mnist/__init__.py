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
