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

import time
import math
import torch


def repackage_hidden(h):
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    return data, target


def evaluate_lstm(data_source, model, criterion, device, interval, batch_size, bptt=35):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        hidden = model.init_hidden(batch_size)
        hidden[0].to(device)
        hidden[1].to(device)
        for i in range(0, data_source.size(0) - 1, bptt):# iterate over every timestep
            data, targets = get_batch(data_source, i, bptt)
            data = data.to(device)
            targets = targets.to(device)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


def train_lstm(train_data, model, criterion, opt, epoch, device, interval, batch_size, bptt=35, clip=0.25):
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    hidden[0].to(device)
    hidden[1].to(device)
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        data = data.to(device)
        targets = targets.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        opt.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total_loss += loss.data

        if batch % interval == 0 and batch > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt,
                elapsed * 1000 / interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
