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


class TensorBuffer(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("tensor", tensor, persistent=False)

    def size(self):
        return self.tensor.size()

    def device(self):
        return self.tensor.device


class TensorParameter(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.acam_weight = torch.nn.Parameter(tensor)

    def size(self):
        return self.acam_weight.size()

    def device(self):
        return self.acam_weight.device