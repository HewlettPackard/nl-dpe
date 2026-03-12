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
import numpy as np
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .utils import _parent_name
from .quant import SymmetricTrainableQuantizer, AsymmetricTrainableQuantizer, SymmetricStatMovingQuantizer, AsymmetricStatMovingQuantizer
from .slicing import weight_slicing, weight_recompose
from .noise import DPENoise, ACAMNoise
from .tensor import TensorParameter
from .layers import Matmul, Softmax


class ACAM(nn.Module):
    def __init__(
        self,
        function = "identity",
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        minimun = -float("inf"),
        maximum = float("inf"),
    ):
        super().__init__()
        self.function = function

        self.quant = quant
        self.actbitW = actbitW
        self.output_quant = AsymmetricStatMovingQuantizer(actbitW, range="pos")

        self.acam_noise = acam_noise
        self.acam_g_min = acam_g_min
        self.acam_g_max = acam_g_max

        self.lut = lut
        self.acam_rows = acam_rows
        self.encode = encode
        self.acam_dir = acam_dir
        self.layer_name = layer_name
        self.acam_finetuned = acam_finetuned
        self.trainable = trainable

        self.weight_low = nn.ModuleList()
        self.weight_high = nn.ModuleList()
        for i, rows in enumerate(self.acam_rows):
            self.weight_low.append(TensorParameter(torch.full((rows, 1), maximum)))
            self.weight_high.append(TensorParameter(torch.full((rows, 1), minimun)))

        valid_rows = []
        if self.lut:
            table_path = os.path.join(self.acam_dir, self.layer_name)
            for i in range(self.actbitW):
                if self.acam_finetuned:
                    npy_file_name = os.path.join(table_path, "dt_bit_finetune_"+str(i)+".npy")
                else:
                    npy_file_name = os.path.join(table_path, "dt_bit_"+str(i)+".npy")
                thresholds = np.load(npy_file_name)

                num_rows = thresholds.shape[0]
                thresholds = thresholds.reshape(num_rows, 1, 2)
                if num_rows > self.acam_rows[i]:
                    num_rows = self.acam_rows[i]
                valid_rows.append(num_rows)

                self.weight_low[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 0]
                self.weight_high[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 1]
                self.weight_low[i].acam_weight.data[torch.isnan(self.weight_low[i].acam_weight)] = minimun
                self.weight_high[i].acam_weight.data[torch.isnan(self.weight_high[i].acam_weight)] = maximum

        self.acam_noise_model = ACAMNoise(self.acam_g_min, self.acam_g_max, std_scale, self.weight_low, self.weight_high, stuck_fault, stuck_mean, valid_rows)

    @torch.jit.script
    def optimized_loop(
        x: torch.Tensor,
        noise_weight_low: List[torch.Tensor],
        noise_weight_high: List[torch.Tensor],
        actbitW: int
    ) -> torch.Tensor:
        _, num_elems = x.shape
        max_rows = 0
        for i in range(actbitW):
            max_rows = max(noise_weight_low[i].shape[0], max_rows)

        y = torch.zeros((num_elems,), dtype=torch.int16, device=x.device)
        output_low  = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x.device)
        output_high = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x.device)
        bits = torch.empty((num_elems,), dtype=torch.int16, device=x.device)

        for i in range(actbitW):
            num_rows = noise_weight_low[i].shape[0]

            torch.gt(x, noise_weight_low[i][:, 0].unsqueeze(1), out=output_low[:num_rows])
            torch.gt(noise_weight_high[i][:, 0].unsqueeze(1), x, out=output_high[:num_rows])
            torch.bitwise_and(output_low[:num_rows], output_high[:num_rows], out=output_low[:num_rows])
            for j in range(1, num_rows):
                 torch.bitwise_or(output_low[0], output_low[j], out=output_low[0])
            bits.copy_(output_low[0].to(torch.int16))
            y += bits << i

        return y

    def gray_decode(self, y, bitwidth):
        binary = y.clone()
        for i in range(bitwidth):
            y = y >> 1
            binary = binary ^ y
        return binary

    def forward(self, x):
        if self.lut:
            old_shape = x.shape
            x = x.flatten().unsqueeze(0)

            if self.acam_noise:
                noise_weight_low, noise_weight_high = self.acam_noise_model(self.weight_low, self.weight_high)
            else:
                noise_weight_low = []
                noise_weight_high = []
                for i in range(self.actbitW):
                    noise_weight_low.append(self.weight_low[i].acam_weight)
                    noise_weight_high.append(self.weight_high[i].acam_weight)

            if self.trainable:
                bit_tensors = []
                for i in range(self.actbitW):
                    output_low = x - noise_weight_low[i][:,0].unsqueeze(1)
                    output_high = noise_weight_high[i][:,0].unsqueeze(1) - x
                    output_high_low = F.relu(output_low) * F.relu(output_high)
                    output_high_low = torch.sum(output_high_low, dim=0)
                    output_high_low = output_high_low / (output_high_low + 1e-8)
                    bit_tensors.append(output_high_low)

                if self.encode:
                    b = [None for i in range(self.actbitW)]
                    b[self.actbitW-1] = bit_tensors[self.actbitW-1]
                    for i in range(self.actbitW - 2, -1, -1):
                        b[i] = (bit_tensors[i] - b[i+1]) ** 2
                    y = b[0]
                    for i in range(1, self.actbitW):
                        y += b[i] * 2**i
                else:
                    y = bit_tensors[0]
                    for i in range(1, self.actbitW):
                        y += bit_tensors[i] * 2**i
            else:
                y = self.optimized_loop(x, noise_weight_low, noise_weight_high, self.actbitW)
                if self.encode:
                    y = self.gray_decode(y, self.actbitW)
                y = y.to(torch.float32)

            y = y.reshape(old_shape)

        elif self.function == "identity":
            y = x
        elif self.function == "tanh":
            y = F.tanh(x)
        elif self.function == "sigmoid":
            y = F.sigmoid(x)
        elif self.function == "silu":
            y = F.silu(x)
        elif self.function == "relu":
            y = F.relu(x)
        elif self.function == "gelu":
            y = F.gelu(x)
        elif self.function == "log":
            y = torch.log(x)
        elif self.function == "exp":
            y = torch.exp(x)
        elif self.function == "inv":
            y = 1 / (x+1e-9)

        if self.quant:
            if self.lut:
                y = self.output_quant(y, mode="dequant")
            else:
                y = self.output_quant(y)
        return y


class ACAM1Op(nn.Module):
    def __init__(
        self,
        function = "identity",
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        minimun = -float("inf"),
        maximum = float("inf"),
    ):
        super().__init__()
        self.function = function

        self.quant = quant
        self.actbitW = actbitW
        self.input_quant = AsymmetricStatMovingQuantizer(actbitW, range="pos")
        self.output_quant = AsymmetricStatMovingQuantizer(actbitW, range="pos")

        self.acam_noise = acam_noise
        self.acam_g_min = acam_g_min
        self.acam_g_max = acam_g_max

        self.lut = lut
        self.acam_rows = acam_rows
        self.encode = encode
        self.acam_dir = acam_dir
        self.layer_name = layer_name
        self.acam_finetuned = acam_finetuned
        self.trainable = trainable

        self.weight_low = nn.ModuleList()
        self.weight_high = nn.ModuleList()
        for i, rows in enumerate(self.acam_rows):
            self.weight_low.append(TensorParameter(torch.full((rows, 1), maximum)))
            self.weight_high.append(TensorParameter(torch.full((rows, 1), minimun)))

        if self.lut:
            table_path = os.path.join(self.acam_dir, self.layer_name)
            for i in range(self.actbitW):
                if self.acam_finetuned:
                    npy_file_name = os.path.join(table_path, "dt_bit_finetune_"+str(i)+".npy")
                else:
                    npy_file_name = os.path.join(table_path, "dt_bit_"+str(i)+".npy")
                thresholds = np.load(npy_file_name)

                num_rows = thresholds.shape[0]
                thresholds = thresholds.reshape(num_rows, 1, 2)
                if num_rows > self.acam_rows[i]:
                    num_rows = self.acam_rows[i]

                self.weight_low[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 0]
                self.weight_high[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 1]
                self.weight_low[i].acam_weight.data[torch.isnan(self.weight_low[i].acam_weight)] = minimun
                self.weight_high[i].acam_weight.data[torch.isnan(self.weight_high[i].acam_weight)] = maximum

        self.acam_noise_model = ACAMNoise(self.acam_g_min, self.acam_g_max, std_scale, self.weight_low, self.weight_high, stuck_fault, stuck_mean)

    @torch.jit.script
    def optimized_loop(
        x: torch.Tensor,
        noise_weight_low: List[torch.Tensor],
        noise_weight_high: List[torch.Tensor],
        actbitW: int
    ) -> torch.Tensor:
        _, num_elems = x.shape
        max_rows = 0
        for i in range(actbitW):
            max_rows = max(noise_weight_low[i].shape[0], max_rows)

        y = torch.zeros((num_elems,), dtype=torch.int16, device=x.device)
        output_low  = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x.device)
        output_high = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x.device)
        bits = torch.empty((num_elems,), dtype=torch.int16, device=x.device)

        for i in range(actbitW):
            num_rows = noise_weight_low[i].shape[0]

            torch.gt(x, noise_weight_low[i][:, 0].unsqueeze(1), out=output_low[:num_rows])
            torch.gt(noise_weight_high[i][:, 0].unsqueeze(1), x, out=output_high[:num_rows])
            torch.bitwise_and(output_low[:num_rows], output_high[:num_rows], out=output_low[:num_rows])
            for j in range(1, num_rows):
                 torch.bitwise_or(output_low[0], output_low[j], out=output_low[0])
            bits.copy_(output_low[0].to(torch.int16))
            y += bits << i

        return y

    def gray_decode(self, y, bitwidth):
        binary = y.clone()
        for i in range(bitwidth):
            y = y >> 1
            binary = binary ^ y
        return binary

    def forward(self, x):
        if self.quant:
            if self.lut:
                x = self.input_quant(x, mode="quant")
            else:
                x = self.input_quant(x)

        if self.lut:
            old_shape = x.shape
            x = x.flatten().unsqueeze(0)

            if self.acam_noise:
                noise_weight_low, noise_weight_high = self.acam_noise_model(self.weight_low, self.weight_high)
            else:
                noise_weight_low = []
                noise_weight_high = []
                for i in range(self.actbitW):
                    noise_weight_low.append(self.weight_low[i].acam_weight)
                    noise_weight_high.append(self.weight_high[i].acam_weight)

            if self.trainable:
                bit_tensors = []
                for i in range(self.actbitW):
                    output_low = x - noise_weight_low[i][:,0].unsqueeze(1)
                    output_high = noise_weight_high[i][:,0].unsqueeze(1) - x
                    output_high_low = F.relu(output_low) * F.relu(output_high)
                    output_high_low = torch.sum(output_high_low, dim=0)
                    output_high_low = output_high_low / (output_high_low + 1e-8)
                    bit_tensors.append(output_high_low)

                if self.encode:
                    b = [None for i in range(self.actbitW)]
                    b[self.actbitW-1] = bit_tensors[self.actbitW-1]
                    for i in range(self.actbitW - 2, -1, -1):
                        b[i] = (bit_tensors[i] - b[i+1]) ** 2
                    y = b[0]
                    for i in range(1, self.actbitW):
                        y += b[i] * 2**i
                else:
                    y = bit_tensors[0]
                    for i in range(1, self.actbitW):
                        y += bit_tensors[i] * 2**i
            else:
                y = self.optimized_loop(x, noise_weight_low, noise_weight_high, self.actbitW)
                if self.encode:
                    y = self.gray_decode(y, self.actbitW)
                y = y.to(torch.float32)

            y = y.reshape(old_shape)

        elif self.function == "identity":
            y = x
        elif self.function == "tanh":
            y = F.tanh(x)
        elif self.function == "sigmoid":
            y = F.sigmoid(x)
        elif self.function == "silu":
            y = F.silu(x)
        elif self.function == "relu":
            y = F.relu(x)
        elif self.function == "gelu":
            y = F.gelu(x)
        elif self.function == "log":
            y = torch.log(x)
        elif self.function == "exp":
            y = torch.exp(x)
        elif self.function == "inv":
            y = 1 / (x+1e-9)

        if self.quant:
            if self.lut:
                y = self.output_quant(y, mode="dequant")
            else:
                y = self.output_quant(y)
        return y


class ACAM2Ops(nn.Module):
    def __init__(
        self,
        function = "mul",
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        minimun = -float("inf"),
        maximum = float("inf"),
    ):
        super().__init__()
        self.function = function

        self.quant = quant
        self.actbitW = actbitW
        self.input_quant = SymmetricStatMovingQuantizer(actbitW, range="pos")

        self.acam_noise = acam_noise
        self.acam_g_min = acam_g_min
        self.acam_g_max = acam_g_max

        self.lut = lut
        self.acam_rows = acam_rows
        self.encode = encode
        self.acam_dir = acam_dir
        self.layer_name = layer_name
        self.acam_finetuned = acam_finetuned
        self.trainable = trainable

        self.input1_weight_low = nn.ModuleList()
        self.input1_weight_high = nn.ModuleList()
        self.input2_weight_low = nn.ModuleList()
        self.input2_weight_high = nn.ModuleList()
        for i, rows in enumerate(self.acam_rows):
            self.input1_weight_low.append(TensorParameter(torch.full((rows, 1), maximum)))
            self.input1_weight_high.append(TensorParameter(torch.full((rows, 1), minimun)))
            self.input2_weight_low.append(TensorParameter(torch.full((rows, 1), maximum)))
            self.input2_weight_high.append(TensorParameter(torch.full((rows, 1), minimun)))

        if self.lut:
            table_path = os.path.join(self.acam_dir, self.layer_name)
            for i in range(self.actbitW):
                if self.acam_finetuned:
                    npy_file_name = os.path.join(table_path, "dt_bit_finetune_"+str(i)+".npy")
                else:
                    npy_file_name = os.path.join(table_path, "dt_bit_"+str(i)+".npy")
                thresholds = np.load(npy_file_name)

                num_rows = thresholds.shape[0]
                thresholds = thresholds.reshape(num_rows, 1, 4)
                if num_rows > self.acam_rows[i]:
                    num_rows = self.acam_rows[i]

                self.input1_weight_low[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 0]
                self.input1_weight_high[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 2]
                self.input1_weight_low[i].acam_weight.data[torch.isnan(self.input1_weight_low[i].acam_weight)] = minimun
                self.input1_weight_high[i].acam_weight.data[torch.isnan(self.input1_weight_high[i].acam_weight)] = maximum
                self.input2_weight_low[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 1]
                self.input2_weight_high[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 3]
                self.input2_weight_low[i].acam_weight.data[torch.isnan(self.input2_weight_low[i].acam_weight)] = minimun
                self.input2_weight_high[i].acam_weight.data[torch.isnan(self.input2_weight_high[i].acam_weight)] = maximum

        self.input1_acam_noise_model = ACAMNoise(self.acam_g_min, self.acam_g_max, std_scale, self.input1_weight_low, self.input1_weight_high, stuck_fault, stuck_mean)
        self.input2_acam_noise_model = ACAMNoise(self.acam_g_min, self.acam_g_max, std_scale, self.input2_weight_low, self.input2_weight_high, stuck_fault, stuck_mean)

    @torch.jit.script
    def optimized_loop(
        x1_high: torch.Tensor,
        x1_low: torch.Tensor,
        x2_high: torch.Tensor,
        x2_low: torch.Tensor,
        input1_noise_weight_low: List[torch.Tensor],
        input1_noise_weight_high: List[torch.Tensor],
        input2_noise_weight_low: List[torch.Tensor],
        input2_noise_weight_high: List[torch.Tensor],
        actbitW: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, num_elems = x1_high.shape
        max_rows = 0
        for i in range(actbitW):
            max_rows = max(input1_noise_weight_low[i].shape[0], max_rows)

        y1 = torch.zeros((num_elems,), dtype=torch.int16, device=x1_high.device)
        y2 = torch.zeros((num_elems,), dtype=torch.int16, device=x1_high.device)
        y3 = torch.zeros((num_elems,), dtype=torch.int16, device=x1_high.device)
        y4 = torch.zeros((num_elems,), dtype=torch.int16, device=x1_high.device)
        x1_high_output_low  = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x1_high_output_high = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x1_low_output_low  = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x1_low_output_high = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x2_high_output_low  = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x2_high_output_high = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x2_low_output_low  = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        x2_low_output_high = torch.empty((max_rows, num_elems), dtype=torch.bool, device=x1_high.device)
        bits = torch.empty((num_elems,), dtype=torch.int16, device=x1_high.device)

        for i in range(actbitW):
            num_rows = input1_noise_weight_low[i].shape[0]

            torch.gt(x1_high, input1_noise_weight_low[i][:, 0].unsqueeze(1), out=x1_high_output_low[:num_rows])
            torch.gt(input1_noise_weight_high[i][:, 0].unsqueeze(1), x1_high, out=x1_high_output_high[:num_rows])

            torch.gt(x1_low, input1_noise_weight_low[i][:, 0].unsqueeze(1), out=x1_low_output_low[:num_rows])
            torch.gt(input1_noise_weight_high[i][:, 0].unsqueeze(1), x1_low, out=x1_low_output_high[:num_rows])

            torch.gt(x2_high, input2_noise_weight_low[i][:, 0].unsqueeze(1), out=x2_high_output_low[:num_rows])
            torch.gt(input2_noise_weight_high[i][:, 0].unsqueeze(1), x2_high, out=x2_high_output_high[:num_rows])

            torch.gt(x2_low, input2_noise_weight_low[i][:, 0].unsqueeze(1), out=x2_low_output_low[:num_rows])
            torch.gt(input2_noise_weight_high[i][:, 0].unsqueeze(1), x2_low, out=x2_low_output_high[:num_rows])

            torch.bitwise_and(x1_high_output_low[:num_rows], x1_high_output_high[:num_rows], out=x1_high_output_low[:num_rows])
            torch.bitwise_and(x1_low_output_low[:num_rows], x1_low_output_high[:num_rows], out=x1_low_output_low[:num_rows])

            torch.bitwise_and(x2_high_output_low[:num_rows], x2_high_output_high[:num_rows], out=x2_high_output_low[:num_rows])
            torch.bitwise_and(x2_low_output_low[:num_rows], x2_low_output_high[:num_rows], out=x2_low_output_low[:num_rows])

            torch.bitwise_and(x1_high_output_low[:num_rows], x2_high_output_low[:num_rows], out=x1_high_output_high[:num_rows])
            torch.bitwise_and(x1_high_output_low[:num_rows], x2_low_output_low[:num_rows], out=x1_low_output_high[:num_rows])
            torch.bitwise_and(x1_low_output_low[:num_rows], x2_high_output_low[:num_rows], out=x2_high_output_high[:num_rows])
            torch.bitwise_and(x1_low_output_low[:num_rows], x2_low_output_low[:num_rows], out=x2_low_output_high[:num_rows])

            for j in range(1, num_rows):
                torch.bitwise_or(x1_high_output_high[0], x1_high_output_high[j], out=x1_high_output_high[0])
                torch.bitwise_or(x1_low_output_high[0], x1_low_output_high[j], out=x1_low_output_high[0])
                torch.bitwise_or(x2_high_output_high[0], x2_high_output_high[j], out=x2_high_output_high[0])
                torch.bitwise_or(x2_low_output_high[0], x2_low_output_high[j], out=x2_low_output_high[0])
            bits.copy_(x1_high_output_high[0].to(torch.int16))
            y4 += bits << i
            bits.copy_(x1_low_output_high[0].to(torch.int16))
            y3 += bits << i
            bits.copy_(x2_high_output_high[0].to(torch.int16))
            y2 += bits << i
            bits.copy_(x2_low_output_high[0].to(torch.int16))
            y1 += bits << i

        return y4, y3, y2, y1

    def gray_decode(self, y, bitwidth):
        binary = y.clone()
        for i in range(bitwidth):
            y = y >> 1
            binary = binary ^ y
        return binary

    def forward(self, x1, x2):
        if self.quant:
            if self.lut:
                x1 = self.input_quant(x1, mode="quant")
                x2 = self.input_quant(x2, mode="quant")
            else:
                x1 = self.input_quant(x1)
                x2 = self.input_quant(x2)

        if self.lut:
            x1, x2 = torch.broadcast_tensors(x1, x2)
            old_shape = x1.shape
            x1 = x1.flatten().unsqueeze(0)
            x2 = x2.flatten().unsqueeze(0)

            if self.acam_noise:
                input1_noise_weight_low, input1_noise_weight_high = self.input1_acam_noise_model(self.input1_weight_low, self.input1_weight_high)
                input2_noise_weight_low, input2_noise_weight_high = self.input2_acam_noise_model(self.input2_weight_low, self.input2_weight_high)
            else:
                input1_noise_weight_low = []
                input1_noise_weight_high = []
                input2_noise_weight_low = []
                input2_noise_weight_high = []
                for i in range(self.actbitW):
                    input1_noise_weight_low.append(self.input1_weight_low[i].acam_weight)
                    input1_noise_weight_high.append(self.input1_weight_high[i].acam_weight)
                    input2_noise_weight_low.append(self.input2_weight_low[i].acam_weight)
                    input2_noise_weight_high.append(self.input2_weight_high[i].acam_weight)

            if self.trainable:
                bit_tensors = []
                for i in range(self.actbitW):
                    output_low = x - noise_weight_low[i][:,0].unsqueeze(1)
                    output_high = noise_weight_high[i][:,0].unsqueeze(1) - x
                    output_high_low = F.relu(output_low) * F.relu(output_high)
                    output_high_low = torch.sum(output_high_low, dim=0)
                    output_high_low = output_high_low / (output_high_low + 1e-8)
                    bit_tensors.append(output_high_low)

                if self.encode:
                    b = [None for i in range(self.actbitW)]
                    b[self.actbitW-1] = bit_tensors[self.actbitW-1]
                    for i in range(self.actbitW - 2, -1, -1):
                        b[i] = (bit_tensors[i] - b[i+1]) ** 2
                    y = b[0]
                    for i in range(1, self.actbitW):
                        y += b[i] * 2**i
                else:
                    y = bit_tensors[0]
                    for i in range(1, self.actbitW):
                        y += bit_tensors[i] * 2**i
            else:
                x1 = x1.to(torch.int)
                x2 = x2.to(torch.int)

                x1_high = (x1 >> 4) & 0b1111
                x1_low = x1 & 0b1111

                x2_high = (x2 >> 4) & 0b1111
                x2_low = x2 & 0b1111

                y4, y3, y2, y1 = self.optimized_loop(x1_high, x1_low, x2_high, x2_low, input1_noise_weight_low, input1_noise_weight_high, input2_noise_weight_low, input2_noise_weight_high, self.actbitW)
                if self.encode:
                    y1 = self.gray_decode(y1, 4)
                    y2 = self.gray_decode(y2, 4)
                    y3 = self.gray_decode(y3, 4)
                    y4 = self.gray_decode(y4, 4)
                y = y1 + (y2<<4) + (y3<<4) + (y4<<8)
                y = y.to(torch.float32)

            y = y.reshape(old_shape)

        elif self.function == "mul":
            y = x1 * x2

        if self.quant:
            if self.lut:
                y = self.input_quant(y, mode="dequant")
                y = self.input_quant(y, mode="dequant")
        return y


####################################################################################################################


class AnalogConv2d(nn.Conv2d):
    def __init__(
        self,
        conv_layer = None,
        noise = False,
        g_min = 0,
        g_max = 150,
        activation = None,
        use_gce = False,
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        digital_slicing = False,
        bits_per_cell = 2,
        export_workload = False,
    ):
        assert type(conv_layer) == nn.Conv2d
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size,
                         stride=conv_layer.stride,
                         padding=conv_layer.padding,
                         dilation=conv_layer.dilation,
                         groups=conv_layer.groups,
                         bias=True if conv_layer.bias is not None else False,
                         padding_mode=conv_layer.padding_mode)
        self.dpe_weight = torch.nn.Parameter(conv_layer.weight.detach())  # name it dpe_weight for regularization
        if conv_layer.bias is not None:
            self.bias = torch.nn.Parameter(conv_layer.bias.detach())
        else:
            self.register_parameter('bias', None)

        self.use_gce = use_gce
        self.acam_dir = acam_dir
        self.layer_name = layer_name

        self.noise = noise
        self.g_min = g_min
        self.g_max = g_max
        self.dpe_noise_model = DPENoise(self.g_min, self.g_max, std_scale, conv_layer.weight.shape, True, stuck_fault, stuck_mean)

        self.actbitW = actbitW
        if (activation is not None) and (not self.use_gce):
            assert activation == 'identity', "Only identity activation (to replace ADC) could be in conv2d"
            self.activation = AnalogAct(
                activation=activation,
                use_gce=self.use_gce,
                quant=quant,
                actbitW=self.actbitW,
                acam_noise=acam_noise,
                acam_g_min=acam_g_min,
                acam_g_max=acam_g_max,
                std_scale=std_scale,
                stuck_fault=stuck_fault,
                stuck_mean=stuck_mean,
                lut=lut,
                acam_rows=acam_rows,
                encode=encode,
                acam_dir=self.acam_dir,
                layer_name=self.layer_name+".activation",
                acam_finetuned=acam_finetuned,
                trainable=trainable,
                export_workload=False,
            )
        else:
            self.activation = None

        self.digital_slicing = digital_slicing
        self.bits_per_cell = bits_per_cell
        if self.digital_slicing:
            self.weight_quant = AsymmetricStatMovingQuantizer(self.actbitW, range="neg_and_pos")
            self.dpe_noise_model_digital = DPENoise(self.g_min, self.g_max, std_scale, conv_layer.weight.shape + (self.actbitW // self.bits_per_cell,), False, stuck_fault, stuck_mean)

        self.export_workload = export_workload

    def forward(self, x):
        if self.digital_slicing:
            if self.noise:
                dpe_weight = self.weight_quant(self.dpe_weight, mode="quant")
                dpe_weight_slice, sign = weight_slicing(dpe_weight, self.actbitW, self.bits_per_cell)
                dpe_weight_slice = self.dpe_noise_model_digital(dpe_weight_slice)
                dpe_weight = weight_recompose(dpe_weight_slice, sign, self.actbitW, self.bits_per_cell)
                dpe_weight = self.weight_quant(dpe_weight, mode="dequant")
            else:
                dpe_weight = self.weight_quant(self.dpe_weight)
        else:
            if self.noise:
                dpe_weight = self.dpe_noise_model(self.dpe_weight)
            else:
                dpe_weight = self.dpe_weight

        y = self._conv_forward(x, dpe_weight, self.bias)

        if self.activation is not None:
            y = self.activation(y)

        if self.export_workload:
            self.print_workload(x, y)

        return y

    def print_workload(self, x, y):
        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        if self.activation is None:
            filename = os.path.join(output_dir, self.layer_name + '.act.yaml')
        else:
            filename = os.path.join(output_dir, self.layer_name + '.nothing.yaml')

        C = str(self.in_channels)
        M = str(self.out_channels)
        P = str(y.shape[-2])
        Q = str(y.shape[-1])
        R = str(self.kernel_size[0])
        S = str(self.kernel_size[1])
        HStride = str(self.stride[0])
        WStride = str(self.stride[1])

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {C: "+C+", M: "+M+", P: "+P+", Q: "+Q+", R: "+R+", S: "+S+", HStride: "+HStride+", WStride: "+WStride+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Conv2d\n")


class AnalogLinear(nn.Linear):
    def __init__(
        self,
        linear_layer = None,
        noise = False,
        g_min = 0,
        g_max = 150,
        activation = None,
        use_gce = False,
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        digital_slicing = False,
        bits_per_cell = 2,
        export_workload = False,
    ):
        assert type(linear_layer) == nn.Linear
        super().__init__(linear_layer.in_features, linear_layer.out_features,
                         bias=True if linear_layer.bias is not None else False)
        self.dpe_weight = torch.nn.Parameter(linear_layer.weight.detach())  # name it dpe_weight for regularization
        if linear_layer.bias is not None:
            self.bias = torch.nn.Parameter(linear_layer.bias.detach())
        else:
            self.register_parameter('bias', None)

        self.use_gce = use_gce
        self.acam_dir = acam_dir
        self.layer_name = layer_name

        self.noise = noise
        self.g_min = g_min
        self.g_max = g_max
        self.dpe_noise_model = DPENoise(self.g_min, self.g_max, std_scale, linear_layer.weight.shape, True, stuck_fault, stuck_mean)

        self.actbitW = actbitW
        if (activation is not None) and (not self.use_gce):
            assert activation == 'identity', "Only identity activation (to replace ADC) could be in linear"
            self.activation = AnalogAct(
                activation=activation,
                use_gce=self.use_gce,
                quant=quant,
                actbitW=self.actbitW,
                acam_noise=acam_noise,
                acam_g_min=acam_g_min,
                acam_g_max=acam_g_max,
                std_scale=std_scale,
                stuck_fault=stuck_fault,
                stuck_mean=stuck_mean,
                lut=lut,
                acam_rows=acam_rows,
                encode=encode,
                acam_dir=self.acam_dir,
                layer_name=self.layer_name+".activation",
                acam_finetuned=acam_finetuned,
                trainable=trainable,
                export_workload=False,
            )
        else:
            self.activation = None

        self.digital_slicing = digital_slicing
        self.bits_per_cell = bits_per_cell
        if self.digital_slicing:
            self.weight_quant = AsymmetricStatMovingQuantizer(self.actbitW, range="neg_and_pos")
            self.dpe_noise_model_digital = DPENoise(self.g_min, self.g_max, std_scale, linear_layer.weight.shape + (self.actbitW // self.bits_per_cell,), False, stuck_fault, stuck_mean)

        self.export_workload = export_workload

    def forward(self, x):
        if self.digital_slicing:
            if self.noise:
                dpe_weight = self.weight_quant(self.dpe_weight, mode="quant")
                dpe_weight_slice, sign = weight_slicing(dpe_weight, self.actbitW, self.bits_per_cell)
                dpe_weight_slice = self.dpe_noise_model_digital(dpe_weight_slice)
                dpe_weight = weight_recompose(dpe_weight_slice, sign, self.actbitW, self.bits_per_cell)
                dpe_weight = self.weight_quant(dpe_weight, mode="dequant")
            else:
                dpe_weight = self.weight_quant(self.dpe_weight)
        else:
            if self.noise:
                dpe_weight = self.dpe_noise_model(self.dpe_weight)
            else:
                dpe_weight = self.dpe_weight

        y = F.linear(x, dpe_weight, self.bias)

        if self.activation is not None:
            y = self.activation(y)

        if self.export_workload:
            self.print_workload(x, y)

        return y

    def print_workload(self, x, y):
        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        if self.activation is None:
            filename = os.path.join(output_dir, self.layer_name + '.act.yaml')
        else:
            filename = os.path.join(output_dir, self.layer_name + '.nothing.yaml')

        C = str(self.in_features)
        M = str(self.out_features)

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {C: "+C+", M: "+M+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Linear\n")


class AnalogAct(nn.Module):
    def __init__(
        self,
        activation = "identity",
        use_gce = False,
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        minimun = -float("inf"),
        maximum = float("inf"),
        export_workload = False,
    ):
        super().__init__()

        self.activation = activation

        self.use_gce = use_gce
        self.acam_dir = acam_dir
        self.layer_name = layer_name

        if self.use_gce:
            self.acam_activation = ACAM1Op(
                function = self.activation,
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_activation",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )
        else:
            self.acam_activation = ACAM(
                function = self.activation,
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_activation",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

        self.export_workload = export_workload

    def forward(self, x):
        y = self.acam_activation(x)

        if self.export_workload:
            self.print_workload(x)

        return y

    def print_workload(self, x):
        if self.activation == 'identity': # this means this activation is inside conv/linear as ADC
            return
        if self.activation == 'relu': # this means relu is too easy, it does not use VFU
            return

        ####################
        # this is for baseline

        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self.layer_name + '.activation.yaml')

        Q = str(x.numel())

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {Q: "+Q+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Conv2d\n")

        ####################
        # this is for our design

        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self.layer_name + '.nldpe_activation.yaml')

        product = torch.prod(torch.tensor(x.shape[:-1]))

        N = str(product.item())
        C = str(x.shape[-1])
        M = str(x.shape[-1])

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {N: "+N+", C: "+C+", M: "+M+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Linear\n")


class AnalogMatmul(nn.Module):
    def __init__(
        self,
        use_gce = False,
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        minimun = -float("inf"),
        maximum = float("inf"),
        export_workload = False,
    ):
        super().__init__()

        self.use_gce = use_gce
        self.acam_dir = acam_dir
        self.layer_name = layer_name

        if self.use_gce:
            self.acam_mul = ACAM2Ops(
                function = "mul",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_mul",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )
        else:
            self.acam_log1 = ACAM(
                function = "log",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_log1",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

            self.acam_log2 = ACAM(
                function = "log",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_log2",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

            self.acam_exp = ACAM(
                function = "exp",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_exp",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

        self.export_workload = export_workload

    def forward(self, x, y):
        if self.use_gce:
            sign_x = torch.where(x >= 0, torch.tensor(1), torch.tensor(-1))
            sign_y = torch.where(y >= 0, torch.tensor(1), torch.tensor(-1))
            x = torch.abs(x)
            y = torch.abs(y)

            # Reshape tensors for broadcasting
            # x: [n, c, w, h] -> [n, c, w, h, 1]
            # y: [n, c, h, t] -> [n, c, 1, h, t]
            # sign_x: [n, c, w, h] -> [n, c, w, h, 1]
            # sign_y: [n, c, h, t] -> [n, c, 1, h, t]
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-3)
            sign_x = sign_x.unsqueeze(-1)
            sign_y = sign_y.unsqueeze(-3)

            # Perform element-wise multiplication
            z = self.acam_mul(x, y)  # Shape: [n, c, w, h, t]

            # give the result correct sign
            sign_z = sign_x * sign_y  # Shape: [n, c, w, h, t]
            z = z * sign_z

            # Element-wise addition and sum along dimension h
            result = torch.sum(z, dim=-2)  # Shape: [n, c, w, t]
        else:
            # eliminate 0
            x = x + 1e-9
            y = y + 1e-9

            # make them all positive and extract the sign
            sign_x = torch.where(x >= 0, torch.tensor(1), torch.tensor(-1))
            sign_y = torch.where(y >= 0, torch.tensor(1), torch.tensor(-1))
            x = torch.abs(x)
            y = torch.abs(y)

            # Compute log of x andy
            log_x = self.acam_log1(x)
            log_y = self.acam_log2(y)

            # Reshape tensors for broadcasting
            # log_x: [n, c, w, h] -> [n, c, w, h, 1]
            # log_y: [n, c, h, t] -> [n, c, 1, h, t]
            # sign_x: [n, c, w, h] -> [n, c, w, h, 1]
            # sign_y: [n, c, h, t] -> [n, c, 1, h, t]
            log_x = log_x.unsqueeze(-1)
            log_y = log_y.unsqueeze(-3)
            sign_x = sign_x.unsqueeze(-1)
            sign_y = sign_y.unsqueeze(-3)

            # use addition to replace multiplication, and record the sign
            log_z = log_x + log_y  # Shape: [n, c, w, h, t]
            z = self.acam_exp(log_z)

            # give the result correct sign
            sign_z = sign_x * sign_y  # Shape: [n, c, w, h, t]
            z = z * sign_z

            # Element-wise addition and sum along dimension h
            result = torch.sum(z, dim=-2)  # Shape: [n, c, w, t]

        if self.export_workload:
            self.print_workload(x, y, result)

        return result

    def print_workload(self, x, y, result):
        ####################
        # this is for baseline

        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self.layer_name + '.mult.yaml')

        G = str(x.shape[0] * x.shape[1])
        C = str(y.shape[2])
        M = str(y.shape[3])
        P = str(1)
        Q = str(x.shape[2])
        R = str(1)
        S = str(1)
        HStride = str(1)
        WStride = str(1)

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {G: "+G+", C: "+C+", M: "+M+", P: "+P+", Q: "+Q+", R: "+R+", S: "+S+", HStride: "+HStride+", WStride: "+WStride+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Conv2d\n")

        ####################
        # this is for our design

        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self.layer_name + '.nldpe_mult.yaml')

        product = torch.prod(torch.tensor(x.shape[:-1]))

        N = str(product.item())
        C = str(x.shape[-1])
        M = str(x.shape[-1])

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {N: "+N+", C: "+C+", M: "+M+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Linear\n")


class AnalogSoftmax(nn.Module):
    def __init__(
        self,
        use_gce = False,
        quant = False,
        actbitW = 8,
        acam_noise = False,
        acam_g_min = 0,
        acam_g_max = 150,
        std_scale = 1.0,
        stuck_fault=False,
        stuck_mean=0.04,
        lut = False,
        acam_rows = None,
        encode = False,
        acam_dir = None,
        layer_name = None,
        acam_finetuned = False,
        trainable = False,
        minimun = -float("inf"),
        maximum = float("inf"),
        export_workload = False,
    ):
        super().__init__()

        self.use_gce = use_gce
        self.acam_dir = acam_dir
        self.layer_name = layer_name

        if self.use_gce:
            self.acam_exp = ACAM1Op(
                function = "exp",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_exp",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

            self.acam_inv = ACAM1Op(
                function = "inv",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_inv",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

            self.acam_mul = ACAM2Ops(
                function = "mul",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_mul",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )
        else:
            self.acam_exp1 = ACAM(
                function = "exp",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_exp1",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

            self.acam_exp2 = ACAM(
                function = "exp",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_exp2",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

            self.acam_log = ACAM(
                function = "log",
                quant = quant,
                actbitW = actbitW,
                acam_noise = acam_noise,
                acam_g_min = acam_g_min,
                acam_g_max = acam_g_max,
                std_scale = std_scale,
                stuck_fault = stuck_fault,
                stuck_mean = stuck_mean,
                lut = lut,
                acam_rows = acam_rows,
                encode = encode,
                acam_dir = self.acam_dir,
                layer_name = self.layer_name+".acam_log",
                acam_finetuned = acam_finetuned,
                trainable = trainable,
                minimun = minimun,
                maximum = maximum,
            )

        self.export_workload = export_workload

    def forward(self, input, dim):
        if self.use_gce:
            x = input - torch.max(input, dim, keepdim=True).values

            # it turns out that x could be a very negative value
            # quantizing a very negative value (in acam_exp) makes no meaning, so clamp it to -10 (or other less negative values)
            x = torch.clamp(x, min=-10.0, max=torch.max(x))

            x_exp = self.acam_exp(x)
            x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
            x_exp_inv = self.acam_inv(x_exp_sum)
            y = self.acam_mul(x_exp, x_exp_inv)
        else:
            x = input - torch.max(input, dim, keepdim=True).values

            x_exp = self.acam_exp1(x)
            x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
            x_exp_sum_log = self.acam_log(x_exp_sum)
            x_exp_sum_log_sub = x - x_exp_sum_log
            y = self.acam_exp2(x_exp_sum_log_sub)

        if self.export_workload:
            self.print_workload(input)

        return y

    def print_workload(self, input):
        ####################
        # this is for baseline

        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self.layer_name + '.softmax.yaml')

        Q = str(input.numel())

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {Q: "+Q+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Conv2d\n")

        ####################
        # this is for our design

        output_dir = os.path.join(os.path.dirname(self.acam_dir), 'workload')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self.layer_name + '.nldpe_softmax.yaml')

        product = torch.prod(torch.tensor(input.shape[:-1]))

        N = str(product.item())
        C = str(input.shape[-1])
        M = str(input.shape[-1])

        with open(filename, 'w') as file:
            file.write("{{include_text('../problem_base.yaml')}}\n")
            file.write("problem:\n")
            file.write("  <<<: *problem_base\n")
            file.write("  instance: {N: "+N+", C: "+C+", M: "+M+"}\n")
            file.write("\n")
            #file.write("  name: "+self.layer_name+"\n")
            #file.write("  dnn_name: name_does_not_matter\n")
            #file.write("  notes: Linear\n")


####################################################################################################################


def create_AnalogConv2d(
    conv2d_layer,
    noise,
    g_min,
    g_max,
    activation,
    use_gce,
    quant,
    actbitW,
    acam_noise,
    acam_g_min,
    acam_g_max,
    std_scale,
    stuck_fault,
    stuck_mean,
    lut,
    acam_rows,
    encode,
    acam_dir,
    conv2d_name,
    acam_finetuned,
    trainable,
    digital_slicing,
    bits_per_cell,
    export_workload,
):
    new_module = AnalogConv2d(
        conv_layer=conv2d_layer,
        noise=noise,
        g_min=g_min,
        g_max=g_max,
        activation=activation,
        use_gce=use_gce,
        quant=quant,
        actbitW=actbitW,
        acam_noise=acam_noise,
        acam_g_min=acam_g_min,
        acam_g_max=acam_g_max,
        std_scale=std_scale,
        stuck_fault=stuck_fault,
        stuck_mean=stuck_mean,
        lut=lut,
        acam_rows=acam_rows,
        encode=encode,
        acam_dir=acam_dir,
        layer_name=conv2d_name,
        acam_finetuned=acam_finetuned,
        trainable=trainable,
        digital_slicing=digital_slicing,
        bits_per_cell=bits_per_cell,
        export_workload=export_workload,
    )
    parent_name, name = _parent_name(conv2d_name)
    return new_module, parent_name, name


def create_AnalogLinear(
    linear_layer,
    noise,
    g_min,
    g_max,
    activation,
    use_gce,
    quant,
    actbitW,
    acam_noise,
    acam_g_min,
    acam_g_max,
    std_scale,
    stuck_fault,
    stuck_mean,
    lut,
    acam_rows,
    encode,
    acam_dir,
    linear_name,
    acam_finetuned,
    trainable,
    digital_slicing,
    bits_per_cell,
    export_workload,
):
    new_module = AnalogLinear(
        linear_layer=linear_layer,
        noise=noise,
        g_min=g_min,
        g_max=g_max,
        activation=activation,
        use_gce=use_gce,
        quant=quant,
        actbitW=actbitW,
        acam_noise=acam_noise,
        acam_g_min=acam_g_min,
        acam_g_max=acam_g_max,
        std_scale=std_scale,
        stuck_fault=stuck_fault,
        stuck_mean=stuck_mean,
        lut=lut,
        acam_rows=acam_rows,
        encode=encode,
        acam_dir=acam_dir,
        layer_name=linear_name,
        acam_finetuned=acam_finetuned,
        trainable=trainable,
        digital_slicing=digital_slicing,
        bits_per_cell=bits_per_cell,
        export_workload=export_workload,
    )
    parent_name, name = _parent_name(linear_name)
    return new_module, parent_name, name


def convert_model(
    model,
    noise,
    g_min,
    g_max,
    use_gce,
    quant,
    actbitW,
    acam_noise,
    acam_g_min,
    acam_g_max,
    std_scale,
    stuck_fault,
    stuck_mean,
    lut,
    acam_rows,
    encode,
    acam_dir,
    acam_finetuned,
    trainable,
    digital_slicing,
    bits_per_cell,
    export_workload,
):
    if acam_rows is None:
        acam_rows_file = os.path.join(acam_dir, 'acam_rows.txt')
        if os.path.exists(acam_rows_file):
            with open(acam_rows_file, "r") as f:
                acam_rows = [int(line.strip()) for line in f]
        else:
            assert not lut, "acam_rows.txt must exists when using LUT"
            acam_rows = [1] # a minimum size of acam array, but it is not actually used

    with torch.no_grad():
        modules = dict(model.named_modules(remove_duplicate=False))
        conv2d_layer = None
        conv2d_name = None
        linear_layer = None
        linear_name = None

        prev_in_channels=None
        prev_out_channels=None
        prev_kernel_size=None
        prev_padding=None
        prev_stride=None
        prev_in_features=None
        prev_out_features=None
        for full_name, module in modules.items():
            if isinstance(module, nn.Conv2d):
                if conv2d_layer is not None:
                    new_module, parent_name, name = create_AnalogConv2d(conv2d_layer, noise, g_min, g_max, "identity", use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, conv2d_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    conv2d_layer = None
                    conv2d_name = None
                elif linear_layer is not None:
                    new_module, parent_name, name = create_AnalogLinear(linear_layer, noise, g_min, g_max, "identity", use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, linear_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    linear_layer = None
                    linear_name = None

                conv2d_layer = module
                conv2d_name = full_name

                prev_in_channels=module.in_channels
                prev_out_channels=module.out_channels
                prev_kernel_size=module.kernel_size
                prev_padding=module.padding
                prev_stride=module.stride
                prev_in_features=None
                prev_out_features=None
            elif isinstance(module, nn.Linear):
                if conv2d_layer is not None:
                    new_module, parent_name, name = create_AnalogConv2d(conv2d_layer, noise, g_min, g_max, "identity", use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, conv2d_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    conv2d_layer = None
                    conv2d_name = None
                elif linear_layer is not None:
                    new_module, parent_name, name = create_AnalogLinear(linear_layer, noise, g_min, g_max, "identity", use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, linear_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    linear_layer = None
                    linear_name = None

                linear_layer = module
                linear_name = full_name

                prev_in_channels=None
                prev_out_channels=None
                prev_kernel_size=None
                prev_padding=None
                prev_stride=None
                prev_in_features=module.in_features
                prev_out_features=module.out_features
            elif isinstance(module, nn.BatchNorm2d):
                continue
            elif isinstance(module, (nn.Tanh, nn.Sigmoid, nn.SiLU, nn.ReLU, nn.GELU)):
                do_export_workload = False
                if conv2d_layer is not None:
                    new_module, parent_name, name = create_AnalogConv2d(conv2d_layer, noise, g_min, g_max, None, use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, conv2d_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    conv2d_layer = None
                    conv2d_name = None
                elif linear_layer is not None:
                    new_module, parent_name, name = create_AnalogLinear(linear_layer, noise, g_min, g_max, None, use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, linear_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    linear_layer = None
                    linear_name = None
                else:
                    do_export_workload = True
                if isinstance(module, nn.Tanh):
                    activation = "tanh"
                elif isinstance(module, nn.Sigmoid):
                    activation = "sigmoid"
                elif isinstance(module, nn.SiLU):
                    activation = "silu"
                elif isinstance(module, nn.ReLU):
                    activation = "relu"
                elif isinstance(module, nn.GELU):
                    activation = "gelu"
                new_module = AnalogAct(
                    activation=activation,
                    use_gce=use_gce,
                    quant=quant,
                    actbitW=actbitW,
                    acam_noise=acam_noise,
                    acam_g_min=acam_g_min,
                    acam_g_max=acam_g_max,
                    std_scale=std_scale,
                    stuck_fault=stuck_fault,
                    stuck_mean=stuck_mean,
                    lut=lut,
                    acam_rows=acam_rows,
                    encode=encode,
                    acam_dir=acam_dir,
                    layer_name=full_name,
                    acam_finetuned=acam_finetuned,
                    trainable=trainable,
                    export_workload=do_export_workload,
                )
                parent_name, name = _parent_name(full_name)
                setattr(modules[parent_name], name, new_module)
            else:
                if conv2d_layer is not None:
                    new_module, parent_name, name = create_AnalogConv2d(conv2d_layer, noise, g_min, g_max, None, use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, conv2d_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    conv2d_layer = None
                    conv2d_name = None
                elif linear_layer is not None:
                    new_module, parent_name, name = create_AnalogLinear(linear_layer, noise, g_min, g_max, None, use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                        lut, acam_rows, encode, acam_dir, linear_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
                    setattr(modules[parent_name], name, new_module)
                    linear_layer = None
                    linear_name = None
                if isinstance(module, Matmul):
                    new_module = AnalogMatmul(
                        use_gce=use_gce,
                        quant=quant,
                        actbitW=actbitW,
                        acam_noise=acam_noise,
                        acam_g_min=acam_g_min,
                        acam_g_max=acam_g_max,
                        std_scale=std_scale,
                        stuck_fault=stuck_fault,
                        stuck_mean=stuck_mean,
                        lut=lut,
                        acam_rows=acam_rows,
                        encode=encode,
                        acam_dir=acam_dir,
                        layer_name=full_name,
                        acam_finetuned=acam_finetuned,
                        trainable=trainable,
                        export_workload=export_workload,
                    )
                    parent_name, name = _parent_name(full_name)
                    setattr(modules[parent_name], name, new_module)
                elif isinstance(module, Softmax):
                    new_module = AnalogSoftmax(
                        use_gce=use_gce,
                        quant=quant,
                        actbitW=actbitW,
                        acam_noise=acam_noise,
                        acam_g_min=acam_g_min,
                        acam_g_max=acam_g_max,
                        std_scale=std_scale,
                        stuck_fault=stuck_fault,
                        stuck_mean=stuck_mean,
                        lut=lut,
                        acam_rows=acam_rows,
                        encode=encode,
                        acam_dir=acam_dir,
                        layer_name=full_name,
                        acam_finetuned=acam_finetuned,
                        trainable=trainable,
                        export_workload=export_workload,
                    )
                    parent_name, name = _parent_name(full_name)
                    setattr(modules[parent_name], name, new_module)

        if conv2d_layer is not None:
            new_module, parent_name, name = create_AnalogConv2d(conv2d_layer, noise, g_min, g_max, "identity", use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                lut, acam_rows, encode, acam_dir, conv2d_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
            setattr(modules[parent_name], name, new_module)
            conv2d_layer = None
            conv2d_name = None
        elif linear_layer is not None:
            new_module, parent_name, name = create_AnalogLinear(linear_layer, noise, g_min, g_max, "identity", use_gce, quant, actbitW, acam_noise, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean,
                lut, acam_rows, encode, acam_dir, linear_name, acam_finetuned, trainable, digital_slicing, bits_per_cell, export_workload)
            setattr(modules[parent_name], name, new_module)
            linear_layer = None
            linear_name = None

    return model


####################################################################################################################


def print_params(model, param_file):
    file = open(param_file, "w")
    modules = dict(model.named_modules(remove_duplicate=False))
    for full_name, module in modules.items():
        if isinstance(module, ACAM):
            file.write(full_name+' ')
            file.write(module.function+' ')
            file.write(str(module.output_quant.scale.item())+' ')
            file.write(str(module.output_quant.zero.item())+' ')
            file.write(str(module.output_quant.x_min.item())+' ')
            file.write(str(module.output_quant.x_max.item())+' ')
            file.write(str(module.actbitW)+' ')
            file.write(module.output_quant.range+' ')
            file.write('\n')
        elif isinstance(module, ACAM1Op):
            file.write(full_name+' ')
            file.write(module.function+' ')
            file.write(str(module.input_quant.scale.item())+' ')
            file.write(str(module.input_quant.zero.item())+' ')
            file.write(str(module.input_quant.x_min.item())+' ')
            file.write(str(module.input_quant.x_max.item())+' ')
            file.write(str(module.output_quant.scale.item())+' ')
            file.write(str(module.output_quant.zero.item())+' ')
            file.write(str(module.output_quant.x_min.item())+' ')
            file.write(str(module.output_quant.x_max.item())+' ')
            file.write(str(module.actbitW)+' ')
            file.write(module.input_quant.range+' ')
            file.write(module.output_quant.range+' ')
            file.write('\n')
        elif isinstance(module, ACAM2Ops):
            file.write(full_name+' ')
            file.write(module.function+' ')
            file.write(str(module.actbitW)+' ')
            file.write('\n')
    file.close()