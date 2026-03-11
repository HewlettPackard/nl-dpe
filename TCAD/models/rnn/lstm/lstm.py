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

import math
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class Project(nn.Module):
    """
        Wrap the two matmuls in LSTM, to be replaced by QuantProject
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_size = 4 * hidden_size
        self.W_ih = nn.Parameter(torch.empty((gate_size, input_size)))
        self.W_hh = nn.Parameter(torch.empty((gate_size, hidden_size)))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))
        self.bachnorm = nn.BatchNorm1d(gate_size, affine=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
        self.bachnorm.weight.data.fill_(1)
        self.bachnorm.bias.data.zero_()

    def forward(self, input: Tensor, h: Tensor) -> Tensor:
        return self.bachnorm(input @ self.W_ih.T + h @ self.W_hh.T + self.bias_ih + self.bias_hh)


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.project = Project(input_size, hidden_size)

        self.sigmoid_in = nn.Sigmoid()
        self.sigmoid_forget = nn.Sigmoid()
        self.tanh_cell = nn.Tanh()
        self.sigmoid_out = nn.Sigmoid()
        self.tanh_out = nn.Tanh()

    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor], ) -> Tuple[Tensor, Tensor]:
        h, c = hx

        gates = self.project(input, h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.sigmoid_in(ingate)
        forgetgate = self.sigmoid_forget(forgetgate)
        cellgate = self.tanh_cell(cellgate)
        outgate = self.sigmoid_out(outgate)

        c1 = forgetgate * c
        c2 = ingate * cellgate
        c = c1 + c2
        c3 = self.tanh_out(c)
        h = outgate * c3

        return h, c


class CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([CustomLSTMCell(input_size if layer == 0 else hidden_size, hidden_size) for layer in range(num_layers)])

    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        seq_len = input.size()[0]
        out = []
        h_n = [hx[0][i] for i in range(self.num_layers)]
        c_n = [hx[1][i] for i in range(self.num_layers)]
        for s in range(seq_len):
            token = input[s]
            for layer in range(self.num_layers):
                h, c = self.layers[layer](token, (h_n[layer], c_n[layer]))
                h_n[layer] = h
                c_n[layer] = c
                token = h
            out.append(token)
        return torch.stack(out), (torch.stack(h_n), torch.stack(c_n))


class CustomRNNModel(nn.Module):

    def __init__(self, vocab_size, ninp, nhid, nlayers, dropout=0.5):
        super(CustomRNNModel, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, ninp)#, _freeze=True) # freeze embedding layer during PTQ and QAT
        self.rnn = CustomLSTM(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.drop(emb)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)