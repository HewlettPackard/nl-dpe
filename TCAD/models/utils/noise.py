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

from .tensor import TensorBuffer, TensorParameter


def write_noise(G_target, training, dist, std_scale, stuck_fault=False, stuck_mean=0.04, stuck_dist=None):
    # use Ruibin's model, but fit with Xia's data
    a_fit = 0.95
    b_fit = -1.86
    write_std = torch.clamp(G_target, min=1e-9, max=2)
    write_std = torch.log(write_std) * a_fit + b_fit
    write_std = torch.exp(write_std) * std_scale
    if training:
        write_noise = write_std * torch.normal(mean=0.0, std=1.0, size=G_target.size(), device=G_target.device)
    else:
        write_noise = write_std * dist

    if stuck_fault:
        bad_rate = stuck_dist
        mask = bad_rate <= stuck_mean
        write_noise[mask] = -G_target[mask]

    return write_noise


def read_noise(G_prog, training, dist, std_scale):
    # use Ruibin's model, but fit with Xia's data
    a_fit = 0.98
    b_fit = -3.1
    read_std = torch.clamp(G_prog, min=1e-9, max=10)
    read_std = torch.log(read_std) * a_fit + b_fit
    read_std = torch.exp(read_std) * std_scale
    if training:
        read_noise = read_std * torch.normal(mean=0.0, std=1.0, size=G_prog.size(), device=G_prog.device)
    else:
        read_noise = read_std * dist
    return read_noise


class DPENoise(torch.nn.Module):
    def __init__(
        self,
        g_min,
        g_max,
        std_scale,
        weight_shape,
        slicing=True,
        stuck_fault=False,
        stuck_mean=0.04,
    ):
        super().__init__()

        self.g_min = g_min
        self.g_max = g_max
        self.std_scale = std_scale
        self.slicing = slicing
        self.stuck_fault = stuck_fault
        self.stuck_mean = stuck_mean
        gen = torch.Generator()
        gen.manual_seed(1111)
        self.register_buffer("stuck_dist_mlc", torch.rand(weight_shape, generator=gen), persistent=False)
        self.register_buffer("stuck_dist_slc", torch.rand(weight_shape, generator=gen), persistent=False)

        self.register_buffer("dist_mlc_write", torch.normal(mean=0.0, std=1.0, size=weight_shape), persistent=False)
        self.register_buffer("dist_mlc_read", torch.normal(mean=0.0, std=1.0, size=weight_shape), persistent=False)
        if self.slicing:
            self.register_buffer("dist_slc_write", torch.normal(mean=0.0, std=1.0, size=weight_shape), persistent=False)
            self.register_buffer("dist_slc_read", torch.normal(mean=0.0, std=1.0, size=weight_shape), persistent=False)

    def forward(self, w):
        g_ratio = (self.g_max-self.g_min) / w.abs().max()

        sign = torch.where(w >= 0, torch.tensor(1), torch.tensor(-1))
        w = torch.abs(w)

        G_target_mlc = w * g_ratio + self.g_min
        G_target_mlc = torch.clamp(G_target_mlc, min=self.g_min, max=self.g_max)
        G_mlc_write_noise = write_noise(G_target_mlc, self.training, self.dist_mlc_write, self.std_scale, self.stuck_fault, self.stuck_mean, self.stuck_dist_mlc)
        G_mlc_prog = G_target_mlc + G_mlc_write_noise
        G_mlc_read_noise = read_noise(G_mlc_prog, self.training, self.dist_mlc_read, self.std_scale)
        G_mlc = G_mlc_prog + G_mlc_read_noise

        if self.slicing:
            g_ratio1 = (self.g_max-self.g_min) / G_mlc_write_noise.abs().max()

            sign1 = torch.where(G_mlc_write_noise >= 0, torch.tensor(1), torch.tensor(-1))
            G_mlc_write_noise = torch.abs(G_mlc_write_noise)

            G_target_lsc = G_mlc_write_noise * g_ratio1 + self.g_min
            G_target_lsc = torch.clamp(G_target_lsc, min=self.g_min, max=self.g_max)
            G_lsc_write_noise = write_noise(G_target_lsc, self.training, self.dist_slc_write, self.std_scale, self.stuck_fault, self.stuck_mean, self.stuck_dist_slc)
            G_slc_prog = G_target_lsc + G_lsc_write_noise
            G_lsc_read_noise = read_noise(G_slc_prog, self.training, self.dist_slc_read, self.std_scale)
            G_slc = G_slc_prog + G_lsc_read_noise

            w = (G_mlc - self.g_min) / g_ratio - (G_slc*sign1 - self.g_min) / g_ratio1 / g_ratio
        else:
            w = (G_mlc - self.g_min) / g_ratio

        w = w * sign
        return w


def get_th_from_g(g):
    # g should be given in uS
    # given a programmed conductance g returns the corresponging DL threshold voltage
    a = 0.50070001310524
    b =-3.756116078076023
    c = 0.6835811044882486

    safe_input = torch.clamp(g, min=1e-19)
    v_th = torch.exp(a*torch.log(safe_input)+b)+c
    return torch.clamp(v_th, min=0.6, max=1)


def get_g_from_th(v_th):
    # given a desired threshold voltage on the DL, returns the conductance that should be programmed
    a = 0.50070001310524
    b =-3.756116078076023
    c = 0.6835811044882486

    safe_input = torch.clamp(v_th-c, min=1e-19)
    return torch.exp((torch.log(safe_input)-b)/a)


class ACAMNoise(torch.nn.Module):
    def __init__(self, acam_g_min, acam_g_max, std_scale, l, h, stuck_fault=False, stuck_mean=0.04, valid_rows=None):
        super().__init__()

        self.acam_g_min = acam_g_min
        self.acam_g_max = acam_g_max
        self.std_scale = std_scale
        self.stuck_fault = stuck_fault
        self.stuck_mean = stuck_mean
        self.stuck_dist_low = nn.ModuleList()
        self.stuck_dist_high = nn.ModuleList()
        gen = torch.Generator()
        gen.manual_seed(2222)
        for i in range(len(l)):
            self.stuck_dist_low.append(TensorBuffer(torch.rand(l[i].size(), generator=gen)))
            self.stuck_dist_high.append(TensorBuffer(torch.rand(h[i].size(), generator=gen)))

        self.stuck_mask = []
        if stuck_fault and (valid_rows is not None):
            for i in range(len(valid_rows)):
                low_mask = self.stuck_dist_low[i].tensor <= stuck_mean
                high_mask = self.stuck_dist_high[i].tensor <= stuck_mean
                stuck_mask = torch.logical_or(low_mask[:, 0], high_mask[:, 0])
                if valid_rows[i] < len(l[i].acam_weight):  # This should only happen when finetuning
                    front_true = torch.nonzero(stuck_mask[:valid_rows[i]], as_tuple=False).flatten()
                    back_false = torch.nonzero(~stuck_mask[valid_rows[i]:], as_tuple=False).flatten()
                    back_false = back_false + valid_rows[i]
                    num_pairs = min(front_true.numel(), back_false.numel())
                    for k in range(num_pairs):
                        src = front_true[k].item()
                        dest = back_false[k].item()
                        l[i].acam_weight.data[dest, :] = l[i].acam_weight.data[src, :]
                        h[i].acam_weight.data[dest, :] = h[i].acam_weight.data[src, :]
                self.stuck_mask.append(stuck_mask)
        else:
            for i in range(len(l)):
                stuck_mask = torch.empty_like(l[i].acam_weight[:, 0], dtype=torch.bool, device=l[i].acam_weight.device)
                self.stuck_mask.append(stuck_mask)
                    
        self.v_dl_min = get_th_from_g(torch.tensor(self.acam_g_min))
        self.v_dl_max = get_th_from_g(torch.tensor(self.acam_g_max))

        low_min = torch.tensor(100000.0)
        low_max = torch.tensor(-100000.0)
        high_min = torch.tensor(100000.0)
        high_max = torch.tensor(-100000.0)
        for i in range(len(l)):
            low = l[i].acam_weight
            low = torch.abs(low)
            low = low[~torch.isinf(low)]
            if len(low) > 0:
                low_min = torch.min(low_min, low.min())
                low_max = torch.max(low_max, low.max())
            high = h[i].acam_weight
            high = torch.abs(high)
            high = high[~torch.isinf(high)]
            if len(high) > 0:
                high_min = torch.min(high_min, high.min())
                high_max = torch.max(high_max, high.max())

        self.low_scaled = TensorParameter((self.v_dl_max - self.v_dl_min) / (low_max - low_min))
        self.low_bias = TensorParameter(self.v_dl_min - low_min * (self.v_dl_max - self.v_dl_min) / (low_max - low_min))
        self.high_scaled = TensorParameter((self.v_dl_max - self.v_dl_min) / (high_max - high_min))
        self.high_bias = TensorParameter(self.v_dl_min - high_min * (self.v_dl_max - self.v_dl_min) / (high_max - high_min))

        self.dist_write_low = nn.ModuleList()
        self.dist_read_low = nn.ModuleList()
        self.dist_write_high = nn.ModuleList()
        self.dist_read_high = nn.ModuleList()
        for i in range(len(l)):
            self.dist_write_low.append(TensorBuffer(torch.normal(mean=0.0, std=1.0, size=l[i].size())))
            self.dist_read_low.append(TensorBuffer(torch.normal(mean=0.0, std=1.0, size=l[i].size())))
            self.dist_write_high.append(TensorBuffer(torch.normal(mean=0.0, std=1.0, size=h[i].size())))
            self.dist_read_high.append(TensorBuffer(torch.normal(mean=0.0, std=1.0, size=h[i].size())))

    def forward(self, l, h):
        noise_l = []
        for i in range(len(l)):
            low = l[i].acam_weight

            sign = torch.where(low >= 0, torch.tensor(1), torch.tensor(-1))
            low = torch.abs(low)

            low_scaled = low * self.low_scaled.acam_weight + self.low_bias.acam_weight
            g_low_target = get_g_from_th(low_scaled)

            g_low_target = torch.clamp(g_low_target, min=self.acam_g_min, max=self.acam_g_max)
            g_low_target_write_noise = write_noise(g_low_target, self.training, self.dist_write_low[i].tensor, self.std_scale, self.stuck_fault, self.stuck_mean, self.stuck_dist_low[i].tensor)
            g_low_prog = g_low_target + g_low_target_write_noise
            g_low_target_read_noise = read_noise(g_low_prog, self.training, self.dist_read_low[i].tensor, self.std_scale)
            g_low = g_low_prog + g_low_target_read_noise

            th_low = get_th_from_g(g_low)
            th_low_scaled = (th_low - self.low_bias.acam_weight) / self.low_scaled.acam_weight

            th_low_scaled = th_low_scaled * sign
            th_low_scaled[torch.isnan(th_low_scaled)] = float('-inf')
            th_low_scaled[self.stuck_mask[i]] = float('inf')
            noise_l.append(th_low_scaled)

        noise_h = []
        for i in range(len(h)):
            high = h[i].acam_weight

            sign = torch.where(high >= 0, torch.tensor(1), torch.tensor(-1))
            high = torch.abs(high)

            high_scaled = high * self.high_scaled.acam_weight + self.high_bias.acam_weight
            g_high_target = get_g_from_th(high_scaled)

            g_high_target = torch.clamp(g_high_target, min=self.acam_g_min, max=self.acam_g_max)
            g_high_target_write_noise = write_noise(g_high_target, self.training, self.dist_write_high[i].tensor, self.std_scale, self.stuck_fault, self.stuck_mean, self.stuck_dist_high[i].tensor)
            g_high_prog = g_high_target + g_high_target_write_noise
            g_high_target_read_noise = read_noise(g_high_prog, self.training, self.dist_read_high[i].tensor, self.std_scale)
            g_high = g_high_prog + g_high_target_read_noise

            th_high = get_th_from_g(g_high)       
            th_high_scaled = (th_high - self.high_bias.acam_weight) / self.high_scaled.acam_weight

            th_high_scaled = th_high_scaled * sign
            th_high_scaled[torch.isnan(th_high_scaled)] = float('inf')
            th_high_scaled[self.stuck_mask[i]] = float('-inf')
            noise_h.append(th_high_scaled)

        return noise_l, noise_h
