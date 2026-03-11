import os
import argparse
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models.utils.fuse import ACAM
torch.manual_seed(0)


###############################################################################################################


def identity(x):
    return x


def gelu(x):
    return F.gelu(x)


def sigmoid(x):
    return F.sigmoid(x)


def tanh(x):
    return F.tanh(x)


def silu(x):
    return F.silu(x)


def relu(x):
    return F.relu(x)


def log(x):
    x = torch.clamp(x, min=0.00000001)
    return torch.log(x)


def exp(x):
    return torch.exp(x)


function_list = {
    "identity": identity,
    "gelu": gelu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "silu": silu,
    "relu": relu,
    "log": log,
    "exp": exp,
}


###############################################################################################################


def load_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    first_numbers = [float(line.split()[0]) for line in lines]
    tensor = torch.tensor(first_numbers)
    return tensor


def save_model(model, save_path):
    for i in range(model.actbitW):
        npy_file_name = os.path.join(save_path, "dt_bit_finetune_"+str(i)+".npy")
        weight_tensor = torch.cat((model.weight_low[i].acam_weight.data, model.weight_high[i].acam_weight.data), dim=1)
        np.save(npy_file_name, weight_tensor.numpy())


def load_model(model, save_path):
    for i in range(model.actbitW):
        npy_file_name = os.path.join(save_path, "dt_bit_finetune_"+str(i)+".npy")
        thresholds = np.load(npy_file_name)

        num_rows = thresholds.shape[0]
        thresholds = thresholds.reshape(num_rows, 1, 2)
        if num_rows > model.acam_rows[i]:
            num_rows = model.acam_rows[i]

        model.weight_low[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 0]
        model.weight_high[i].acam_weight.data[:num_rows, :] = torch.from_numpy(thresholds)[:num_rows, :, 1]


def round_half_to_below(x):
    return torch.floor(x+0.5)


def quant(x, scale, zero, quant_min, quant_max):
    y = torch.clamp(round_half_to_below(x / scale + zero), min=quant_min, max=quant_max).to(x.dtype)
    y = (y - zero) * scale
    return y


def ground_truth(func_name, x):
    y = function_list[func_name](x)
    return y


def finetune_tree(tree_dir, infinity, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean, acam_rows, encode, lr, momentum, weight_decay, epochs, batch_size,
                  module_fullname, func_name, scale, zero, x_min, x_max, bits, out_range):
        tree_path = os.path.join(tree_dir, module_fullname)
        data_path = os.path.join(tree_path, "train_data.txt")

        train_data = load_data(data_path)
        minimun = train_data.min().item() - infinity
        maximum = train_data.max().item() + infinity
        target_data = ground_truth(func_name, train_data)
        target_data = quant(target_data, scale, zero, quant_min=0, quant_max=2**bits - 1)

        dataset = TensorDataset(train_data, target_data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = ACAM(
            function = func_name,
            quant = True,
            actbitW = bits,
            acam_noise = True,
            acam_g_min = acam_g_min,
            acam_g_max = acam_g_max,
            std_scale = std_scale,
            stuck_fault = stuck_fault,
            stuck_mean = stuck_mean,
            lut = True,
            acam_rows = acam_rows,
            encode = encode,
            acam_dir = tree_dir,
            layer_name = module_fullname,
            acam_finetuned = False,
            trainable = True,
            minimun = minimun,
            maximum = maximum,
        )
        model.output_quant.scale = torch.tensor([scale])
        model.output_quant.zero = torch.tensor([zero])
        model.output_quant.train_override = False

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_data)/batch_size)

        print('======================================', module_fullname, func_name)

        model.eval()
        with torch.no_grad():
            y = model(train_data)
        loss = criterion(y, target_data)
        pre_loss = loss.item()
        print("pre_loss: ", pre_loss)

        save_model(model, tree_path)

        best_loss = pre_loss
        for epoch in range(epochs):
            model.train()
            for batch_data, batch_target in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                scheduler.step()

            model.eval()
            with torch.no_grad():
                y = model(train_data)
            loss = criterion(y, target_data)
            test_loss = loss.item()
            if test_loss <= best_loss:
                best_loss = test_loss
                save_model(model, tree_path)

        load_model(model, tree_path)

        model.eval()
        with torch.no_grad():
            y = model(train_data)
        loss = criterion(y, target_data)
        post_loss = loss.item()/len(y)
        print("post_loss:", post_loss)


def finetune(tree_dir, only, infinity, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean, acam_rows, encode, lr, momentum, weight_decay, epochs, batch_size):
    params_file = os.path.join(tree_dir, "params.txt")
    input_file = open(params_file, "r")
    while True:
        line = input_file.readline()
        if not line:
            break

        tokens = line.split()
        module_fullname = tokens[0]     # full name of the activation layer in the model
        func_name = tokens[1]           # function name
        scale = float(tokens[2])        # scale of output quantization
        zero = float(tokens[3])         # zero point of output quantization
        x_min = float(tokens[4])        # minimum input value
        x_max = float(tokens[5])        # maximum input value
        bits = int(tokens[6])           # bit width of the output
        out_range = tokens[7]           # range of output quantization

        if only is not None:
            if only != func_name:
                continue

        if acam_rows is None:
            acam_rows_file = os.path.join(tree_dir, 'acam_rows.txt')
            with open(acam_rows_file, "r") as f:
                acam_rows = [int(line.strip()) for line in f]

        finetune_tree(tree_dir, infinity, acam_g_min, acam_g_max, std_scale, stuck_fault, stuck_mean, acam_rows, encode, lr, momentum, weight_decay, epochs, batch_size,
                      module_fullname, func_name, scale, zero, x_min, x_max, bits, out_range)
