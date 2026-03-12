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


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

        # Just as a reference, the following code is the one that replaces multiplications
        # with log-add-exp operations, this is the one we will implement in ACAM

        # eliminate 0
        x = x + 1e-9
        y = y + 1e-9

        # make them all positive and extract the sign
        sign_x = torch.where(x >= 0, torch.tensor(1), torch.tensor(-1))
        sign_y = torch.where(y >= 0, torch.tensor(1), torch.tensor(-1))
        x = torch.abs(x)
        y = torch.abs(y)

        # Compute log of x andy
        log_x = torch.log(x)  # Shape: [n, c, w, h]
        log_y = torch.log(y)  # Shape: [n, c, h, t]
    
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
        z = torch.exp(log_z)  # Shape: [n, c, w, h, t]

        # give the result correct sign
        sign_z = sign_x * sign_y  # Shape: [n, c, w, h, t]
        z = z * sign_z

        # Element-wise addition and sum along dimension h
        result = torch.sum(z, dim=-2)  # Shape: [n, c, w, t]


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dim):
        return nn.functional.softmax(input, dim=dim)

        # Just as a reference, the following code is the one that replaces softmax operations,
        # this is the one we will implement in ACAM

        x = input - torch.max(input, dim, keepdim=True).values

        x_exp = torch.exp(x)
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        x_exp_sum_log = torch.log(x_exp_sum)
        x_exp_sum_log_sub = x - x_exp_sum_log
        y = torch.exp(x_exp_sum_log_sub)
