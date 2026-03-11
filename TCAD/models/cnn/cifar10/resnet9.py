'''This version of ResNet is from IBM's paper.
'''
import os

import torch
from torch import nn


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=56, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(56)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=56, out_channels=112, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(112)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(112)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(112)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(224)
        self.relu5 = nn.ReLU(inplace=True)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(224)
        self.relu6 = nn.ReLU(inplace=True)

        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(224)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(224)
        self.relu8 = nn.ReLU(inplace=True)

        self.maxpool8 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc = nn.Linear(224, 10)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x2 = self.maxpool2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)

        x4 = x4 + x2

        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)

        x5 = self.maxpool5(x5)

        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = self.relu6(x6)

        x6 = self.maxpool6(x6)

        x7 = self.conv7(x6)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)

        x8 = self.conv8(x7)
        x8 = self.bn8(x8)
        x8 = self.relu8(x8)

        x8 = x8 + x6

        x8 = self.maxpool8(x8)

        x8 = x8.reshape(x8.size(0), -1)
        x9 = self.fc(x8)

        return x9


def resnet9(pretrained=False, device="cpu"):
    model = ResNet9()
    if pretrained:
        state_dict = torch.load("weights/cifar10_resnet9/resnet9.pt", map_location=device)
        model.load_state_dict(state_dict)
    return model