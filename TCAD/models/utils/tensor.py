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