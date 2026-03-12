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

import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter


def round_half_to_below(x):
    return torch.floor(x+0.5)


class round_ste(torch.autograd.Function):
    """
        Straight-through Estimator(STE) for round_half_to_below()
    """

    @staticmethod
    def forward(ctx, x):
        return round_half_to_below(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class SymmetricTrainableQuantizer(nn.Module):
    """
        Trainable symmetric quantizer
        Training parameters: scale
    """

    def __init__(self, bitW = 8, range = "neg_and_pos") -> None:
        super().__init__()
        self.bitW = bitW
        if range == "neg_and_pos":
            self.quant_min = -(2**(bitW-1))
            self.quant_max = 2**(bitW-1) - 1
        elif range == "neg":
            self.quant_min = -(2**bitW - 1)
            self.quant_max = 0
        elif range == "pos":
            self.quant_min = 0
            self.quant_max = 2**bitW - 1
        self.scale = Parameter(torch.tensor([1.0]))
        self.warmup = True

    def forward(self, input: Tensor, attention_mask: Tensor = None, mode: str = "both") -> Tensor:
        if self.training and self.warmup:
            if attention_mask is not None:
                input_masked = input.data[attention_mask > 0.5]
            else:
                input_masked = input.data
            self.scale.data = torch.max(input_masked.data.min().abs(), input_masked.data.max().abs()) * 2 / (self.quant_max-self.quant_min)
            self.warmup = False
        if mode == "quant":
            y = torch.clamp(round_ste.apply(input / self.scale), min=self.quant_min, max=self.quant_max).to(input.dtype)
        elif mode == "dequant":
            y = input * self.scale
        else:
            y = torch.clamp(round_ste.apply(input / self.scale), min=self.quant_min, max=self.quant_max).to(input.dtype)
            y = y * self.scale
        return y


class AsymmetricTrainableQuantizer(nn.Module):
    """
        Trainable asymmetric quantizer
        Training parameters: scale, zero
    """

    def __init__(self, bitW = 8, range = "neg_and_pos") -> None:
        super().__init__()
        self.bitW = bitW
        if range == "neg_and_pos":
            self.quant_min = -(2**(bitW-1))
            self.quant_max = 2**(bitW-1) - 1
        elif range == "neg":
            self.quant_min = -(2**bitW - 1)
            self.quant_max = 0
        elif range == "pos":
            self.quant_min = 0
            self.quant_max = 2**bitW - 1
        self.scale = Parameter(torch.tensor([1.0]))
        self.zero = Parameter(torch.tensor([0.0]))
        self.warmup = True

    def forward(self, input: Tensor, attention_mask: Tensor = None, mode: str = "both") -> Tensor:
        if self.training and self.warmup:
            if attention_mask is not None:
                input_masked = input.data[attention_mask > 0.5]
            else:
                input_masked = input.data
            if input_masked.data.min() < input_masked.data.max():
                self.scale.data = (input_masked.data.max() - input_masked.data.min()) / (self.quant_max-self.quant_min)
            else:
                self.scale.data = torch.tensor(1 / (self.quant_max-self.quant_min)).to(input.device)
            self.zero.data = self.quant_min - (input_masked.data.min() / self.scale.data)
            self.warmup = False
        if mode == "quant":
            y = torch.clamp(round_ste.apply(input / self.scale + self.zero), min=self.quant_min, max=self.quant_max).to(input.dtype)
        elif mode == "dequant":
            y = (input - self.zero) * self.scale
        else:
            y = torch.clamp(round_ste.apply(input / self.scale + self.zero), min=self.quant_min, max=self.quant_max).to(input.dtype)
            y = (y - self.zero) * self.scale
        return y


class SymmetricStatMovingQuantizer(nn.Module):
    """
        Statistics moving average quantizer, no trainable parameters
        Computing the moving average of scale with new samples
    """

    def __init__(self, bitW = 8, range = "neg_and_pos") -> None:
        super().__init__()
        self.bitW = bitW
        if range == "neg_and_pos":
            self.quant_min = -(2**(bitW-1))
            self.quant_max = 2**(bitW-1) - 1
        elif range == "neg":
            self.quant_min = -(2**bitW - 1)
            self.quant_max = 0
        elif range == "pos":
            self.quant_min = 0
            self.quant_max = 2**bitW - 1
        self.register_buffer("scale", torch.ones(1), persistent=True)
        self.register_buffer("x_min", torch.zeros(1), persistent=True)
        self.register_buffer("x_max", torch.zeros(1), persistent=True)
        self.momentum = 0.95
        self.warmup = True
        self.train_override = None
        self.moving_min_max = False

    def forward(self, input: Tensor, attention_mask: Tensor = None, mode: str = "both") -> Tensor:
        training = False
        if self.train_override is None:
            training = self.training
        else:
            training = self.train_override
        if training:
            if attention_mask is not None:
                input_masked = input.data[attention_mask > 0.5]
            else:
                input_masked = input.data
            if self.warmup and (torch.all(input == 0) == False):
                self.x_min = input_masked.data.min()
                self.x_max = input_masked.data.max()
                self.scale = torch.max(self.x_min.abs(), self.x_max.abs()) * 2 / (self.quant_max-self.quant_min)
                self.warmup = False
            else:
                if self.moving_min_max:
                    self.x_min = self.x_min * self.momentum + input_masked.data.min() * (1 - self.momentum)
                    self.x_max = self.x_max * self.momentum + input_masked.data.max() * (1 - self.momentum)
                else:
                    self.x_min = torch.min(input_masked.data.min(), self.x_min)
                    self.x_max = torch.max(input_masked.data.max(), self.x_max)
                self.scale = torch.max(self.x_min.abs(), self.x_max.abs()) * 2 / (self.quant_max-self.quant_min)
            self.x_min = self.x_min.reshape(1)
            self.x_max = self.x_max.reshape(1)
            self.scale = self.scale.reshape(1)
        if mode == "quant":
            y = torch.clamp(round_ste.apply(input / self.scale), min=self.quant_min, max=self.quant_max).to(input.dtype)
        elif mode == "dequant":
            y = input * self.scale
        else:
            y = torch.clamp(round_ste.apply(input / self.scale), min=self.quant_min, max=self.quant_max).to(input.dtype)
            y = y * self.scale
        return y


class AsymmetricStatMovingQuantizer(nn.Module):
    """
        Statistics moving average quantizer, no trainable parameters
        Computing the moving average of scale with new samples
    """

    def __init__(self, bitW = 8, range = "neg_and_pos") -> None:
        super().__init__()
        self.bitW = bitW
        self.range = range
        if range == "neg_and_pos":
            self.quant_min = -(2**(bitW-1))
            self.quant_max = 2**(bitW-1) - 1
        elif range == "neg":
            self.quant_min = -(2**bitW - 1)
            self.quant_max = 0
        elif range == "pos":
            self.quant_min = 0
            self.quant_max = 2**bitW - 1
        self.register_buffer("scale", torch.ones(1), persistent=True)
        self.register_buffer("zero", torch.zeros(1), persistent=True)
        self.register_buffer("x_min", torch.zeros(1), persistent=True)
        self.register_buffer("x_max", torch.zeros(1), persistent=True)
        self.momentum = 0.95
        self.warmup = True
        self.train_override = None
        self.moving_min_max = False

    def forward(self, input: Tensor, attention_mask: Tensor = None, mode: str = "both") -> Tensor:
        training = False
        if self.train_override is None:
            training = self.training
        else:
            training = self.train_override
        if training:
            if attention_mask is not None:
                input_masked = input.data[attention_mask > 0.5]
            else:
                input_masked = input.data
            if self.warmup and (torch.all(input == 0) == False):
                self.x_min = input_masked.min()
                self.x_max = input_masked.max()
                if self.x_min < self.x_max:
                    self.scale = (self.x_max-self.x_min) / (self.quant_max-self.quant_min)
                else:
                    self.scale = torch.tensor(1 / (self.quant_max-self.quant_min)).to(input.device)
                self.zero = self.quant_min - (self.x_min / self.scale)
                self.warmup = False
            else:
                if self.moving_min_max:
                    self.x_min = self.x_min * self.momentum + input_masked.data.min() * (1 - self.momentum)
                    self.x_max = self.x_max * self.momentum + input_masked.data.max() * (1 - self.momentum)
                else:
                    self.x_min = torch.min(input_masked.data.min(), self.x_min)
                    self.x_max = torch.max(input_masked.data.max(), self.x_max)
                if self.x_min < self.x_max:
                    self.scale = (self.x_max-self.x_min) / (self.quant_max-self.quant_min)
                else:
                    self.scale = torch.tensor(1 / (self.quant_max-self.quant_min)).to(input.device)
                self.zero = self.quant_min - (self.x_min / self.scale)
            self.x_min = self.x_min.reshape(1)
            self.x_max = self.x_max.reshape(1)
            self.scale = self.scale.reshape(1)
            self.zero = self.zero.reshape(1)
        if mode == "quant":
            y = torch.clamp(round_ste.apply(input / self.scale + self.zero), min=self.quant_min, max=self.quant_max).to(input.dtype)
        elif mode == "dequant":
            y = (input - self.zero) * self.scale
        else:
            y = torch.clamp(round_ste.apply(input / self.scale + self.zero), min=self.quant_min, max=self.quant_max).to(input.dtype)
            y = (y - self.zero) * self.scale
        return y


def quantizer_switcher(model, train_override=None, moving_min_max=False):
    with torch.no_grad():
        modules = dict(model.named_modules(remove_duplicate=False))
        for full_name, module in modules.items():
            if isinstance(module, (SymmetricStatMovingQuantizer, AsymmetricStatMovingQuantizer)):
                module.train_override = train_override
                module.moving_min_max = moving_min_max
