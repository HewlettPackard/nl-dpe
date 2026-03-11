'''VGG in PyTorch for ImageNet.

Code and pretrained weights from torchvision
'''
from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights: Optional[dict], **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def vgg11(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg11-8a719046.pth")
        else:
            weights = torch.load("weights/imagenet_vgg11/vgg11-8a719046.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("A", False, weights, **kwargs)


def vgg11_bn(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg11_bn-6002323d.pth")
        else:
            weights = torch.load("weights/imagenet_vgg11_bn/vgg11_bn-6002323d.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("A", True, weights, **kwargs)


def vgg13(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg13-19584684.pth")
        else:
            weights = torch.load("weights/imagenet_vgg13/vgg13-19584684.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("B", False, weights, **kwargs)


def vgg13_bn(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg13_bn-abd245e5.pth")
        else:
            weights = torch.load("weights/imagenet_vgg13_bn/vgg13_bn-abd245e5.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("B", True, weights, **kwargs)


def vgg16(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth")
        else:
            weights = torch.load("weights/imagenet_vgg16/vgg16-397923af.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("D", False, weights, **kwargs)


def vgg16_bn(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")
        else:
            weights = torch.load("weights/imagenet_vgg16_bn/vgg16_bn-6c64b313.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("D", True, weights, **kwargs)


def vgg19(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth")
        else:
            weights = torch.load("weights/imagenet_vgg19/vgg19-dcbb9e9d.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("E", False, weights, **kwargs)


def vgg19_bn(pretrained: bool = False, download_pretrained: bool = False, **kwargs: Any) -> VGG:
    if pretrained:
        if download_pretrained:
            weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg19_bn-c79401a0.pth")
        else:
            weights = torch.load("weights/imagenet_vgg19_bn/vgg19_bn-c79401a0.pth", map_location="cpu")
    else:
        weights = None
    return _vgg("E", True, weights, **kwargs)