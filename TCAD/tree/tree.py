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
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn import metrics
from matplotlib import pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F

from .compiler import extract


################################################################
# functions to be implemented in ACAM


def identity(x):
    y = x
    return y
def iidentity(y):
    x = y
    return x


def gelu(x):
    x = torch.tensor(x)
    y = F.gelu(x)
    y = y.numpy()
    return y
def igelu(y, iterations=10):
    y = torch.tensor(y)
    y = torch.clamp(y, min=-0.348)
    x = y.clone().detach().requires_grad_(True)
    for _ in range(iterations):
        gelu_x = F.gelu(x)
        grad = autograd.grad(gelu_x, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        x = x - (gelu_x - y) / grad
    x = x.detach().numpy()
    return x


def sigmoid(x):
    x = torch.tensor(x)
    y = F.sigmoid(x)
    y = y.numpy()
    return y
def isigmoid(y):
    y = torch.tensor(y)
    y = torch.clamp(y, min=1e-7, max=1 - 1e-7)
    x = torch.logit(y)
    x = x.numpy()
    return x


def tanh(x):
    x = torch.tensor(x)
    y = F.tanh(x)
    y = y.numpy()
    return y
def itanh(y):
    y = torch.tensor(y)
    y = torch.clamp(y, min=-0.999999, max=0.999999)
    x = 0.5 * torch.log((1 + y) / (1 - y))
    x = x.numpy()
    return x


def silu(x):
    x = torch.tensor(x)
    y = F.silu(x)
    y = y.numpy()
    return y
def isilu(y, iterations=10):
    y = torch.tensor(y)
    y = torch.clamp(y, min=-0.278)
    x = y.clone().detach().requires_grad_(True)
    for _ in range(iterations):
        silu_x = F.silu(x)
        grad = autograd.grad(silu_x, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        x = x - (silu_x - y) / grad
    x = x.detach().numpy()
    return x


def relu(x):
    x = torch.tensor(x)
    y = F.relu(x)
    y = y.numpy()
    return y
def irelu(y):
    y = torch.tensor(y)
    x = y
    x = x.numpy()
    return x


def log(x):
    if x == 0:
        x += 0.00000001
    x = torch.tensor(x)
    y = torch.log(x)
    y = y.numpy()
    return y
def ilog(y):
    y = torch.tensor(y)
    x = torch.exp(y)
    x = x.numpy()
    return x


def exp(x):
    x = torch.tensor(x)
    y = torch.exp(x)
    y = y.numpy()
    return y
def iexp(y):
    if y == 0:
        y += 0.00000001
    y = torch.tensor(y)
    x = torch.log(y)
    x = x.numpy()
    return x


def inv(x):
    x = torch.tensor(x)
    x[x == 0] += 1e-9
    y = 1 / x
    y = y.numpy()
    return y
def iinv(y):
    y = torch.tensor(y)
    y[y == 0] += 1e-9
    x = 1 / y
    x = x.numpy()
    return x


function_list = {
    "identity": (identity, identity),
    "gelu": (gelu, igelu),
    "sigmoid": (sigmoid, isigmoid),
    "tanh": (tanh, itanh),
    "silu": (silu, isilu),
    "relu": (relu, irelu),
    "log": (log, ilog),
    "exp": (exp, iexp),
    "inv": (inv, iinv),
}


################################################################


def round_half_to_below(x):
    return np.floor(x + 0.5)


# -> int32
def quant(r, scale, zero, quant_min, quant_max):
    q = np.clip(round_half_to_below(r.astype(np.float32) / scale + zero), a_min=quant_min, a_max=quant_max).astype(np.int32)
    return q


# -> float32
def dequant(q, scale, zero):
    r = (q.astype(np.float32) - zero) * scale
    return r


# -> int32
def gray_encode(y):
    y = y.astype(np.int32)
    g = y ^ (y >> 1)
    return g


# -> int32
def gray_decode(g, bits):
    g = g.astype(np.int32)
    y = g.copy()
    for i in range(bits-1):
        g = g >> 1
        y = y ^ g
    return y


def generate_raw_data(filename, func_name, scale, zero, x_min, x_max, bits, out_range, encode):
    """
    Generate the raw data of input-output pairs.

    Args:
        filename    : Filename to save.
        func_name   : Type of Function to approximate.
        bits        : Number of output bits.
        scale       : Output scaling factor.
        zero        : Output zero point.
        x_min       : Minimum input.
        x_max       : Maximum input.
        out_range   : Range of output quantization.
        encode      : True if using gray encoding
    """

    file = open(filename, "w")

    if out_range == "neg_and_pos":
        quant_min = -(2**(bits-1))    # the minimun binary value of output
        quant_max = 2**(bits-1) - 1   # the maximum binary value of output
    elif out_range == "pos":
        quant_min = 0                 # the minimun binary value of output
        quant_max = 2**bits - 1       # the maximum binary value of output
    mask = 2**bits - 1          # mask to get the binarized output
    format_string = '{0:0' + str(bits) + 'b}'

    y_min = dequant(np.array(quant_min), scale, zero)
    y_max = dequant(np.array(quant_max), scale, zero)
    y_x_min = function_list[func_name][1](y_min)
    y_x_max = function_list[func_name][1](y_max)
    x_min = function_list[func_name][1](x_min)
    x_max = function_list[func_name][1](x_max)
    x_min = min(x_min, y_x_min)
    x_max = max(x_max, y_x_max)
    num_samples = 5000
    if func_name in ["gelu"]: # these functions tend to have a much larger input range, so use a relatively larger input range for sampling
        x_min *= 10
        x_max *= 10
        num_samples *= (10 * 2)
    samples = np.linspace(x_min, x_max, num_samples)
    for s in samples:
        x = np.array(s)
        y = function_list[func_name][0](x)
        q = quant(y, scale, zero, quant_min, quant_max)
        g = gray_encode(q)

        qm = q & mask
        gm = g & mask

        if encode:
            file.write(str(x) + " " + format_string.format(gm) + '\n')
        else:
            file.write(str(x) + " " + format_string.format(qm) + '\n')

    file.close()


def prepare_training_data(filename):
    """
    Prepare the training data for decision tree.

    Args:
        filename : Filename that contains training data pairs.

    Returns:
        features: Training features, shape [num_samples, num_features].
        labels  : Labels, shape [num_samples, num_out_bits].
    """

    file = open(filename, "r")

    features = []
    labels = []

    while True:
        line = file.readline()
        if not line:
            break

        tokens = line.split()

        num_features = len(tokens) - 1
        idx = 0

        features.append([])
        for i in range(num_features):
            features[-1].append(float(tokens[i]))

        labels.append([])
        output_c = tokens[num_features]

        for i in reversed(output_c):
            labels[-1].append(i)

    file.close()

    features = np.array(features)
    labels = np.array(labels)

    #############################################################################################
    # scale and move the dataset to the range of -100 to 100, to avoid precision issues in sklearn

    target_min = -100
    target_max = 100
    data_min, data_max = np.min(features), np.max(features)
    scaling_factor = (target_max - target_min) / (data_max - data_min)
    zero_point = target_min - data_min * scaling_factor
    features = features * scaling_factor + zero_point

    return features, labels, scaling_factor, zero_point


def train_dt(features, labels, out_bit_index, figname, print_tree=False, **kwargs):
    """
    Train a decision tree for a output bit.

    Args:
        features      : Training data. All features.
        labels        : Training data. All labels.
        out_bit_index : Which output bit to train on.
        figname       : If not None, save the trained tree structure as a picture
        print_tree    : Set to True to print the trained tree structure
        kwargs        : Additional arguments passed to DecisionTreeClassifier

    Returns:
        clf: The trained decision tree.
        acc: The accuracy of the trained decision tree.
    """

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_labels = labels.shape[1]

    assert out_bit_index < num_labels, 'out_bit_index out of range'

    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(features, labels[:, out_bit_index])

    y_pred = clf.predict(features)
    acc = metrics.accuracy_score(labels[:, out_bit_index], y_pred)

    if print_tree:
        text_representation = export_text(clf)
        print(text_representation)

    if figname is not None:

        feature_names = []
        for i in range(num_features):
            feature_names.append('group_'+str(i))

        target_names = []
        for i in range(num_labels):
            target_names.append('target_'+str(i))

        fig = plt.figure(figsize=(25,20))
        _ = plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True)
        fig.savefig(figname)
        plt.close(fig)

    return clf, acc


################################################################


def train_tree(params_dir, module_fullname, func_name, scale, zero, x_min, x_max, bits, out_range, max_leaf_nodes, encode, row_list_total):
    if max_leaf_nodes is not None:
        assert len(max_leaf_nodes) == bits, 'If you specific the max leaf nodes, you must specific for every output bit.'

    # the result of this tree are in a separate folder
    tree_dir = os.path.join(params_dir, module_fullname)
    os.makedirs(tree_dir, exist_ok=True)

    train_file = os.path.join(tree_dir, 'train_data.txt')      # data to train the classifier

    generate_raw_data(filename=train_file, func_name=func_name, scale=scale, zero=zero, x_min=x_min, x_max=x_max, bits=bits, out_range=out_range, encode=encode)

    features, labels, scaling_factor, zero_point = prepare_training_data(filename=train_file)

    total_cells = 0

    dt_list = []
    acc_list = []
    row_list = []
    col_list = []
    cell_list = []
    for i in range(bits):
        png_file = os.path.join(tree_dir, "dt_bit_"+str(i)+".png")
        npy_file = os.path.join(tree_dir, "dt_bit_"+str(i)+".npy")
        # we multiply max_leaf_nodes by 2, because only half of the leaf nodes are programmed into ACAM
        dt, acc = train_dt(features=features, labels=labels, out_bit_index=i, figname=png_file,
            max_leaf_nodes=None if max_leaf_nodes is None else max_leaf_nodes[i]*2,
        )
        dt_list.append(dt)
        acc_list.append(acc)
        if dt.get_depth() == 0:
            rows = 1
            cols = features.shape[1]
            cells = rows * cols
            total_cells += cells
            cam_map = array_nan = np.full((1, 2*features.shape[1]), -1.0)
        else:
            cam_map = extract(dt)
            rows = len(cam_map)
            cols = features.shape[1]
            cells = rows * cols
            total_cells += cells
        row_list.append(rows)
        col_list.append(cols)
        cell_list.append(cells)

        cam_map = (cam_map - zero_point) / scaling_factor
        np.save(npy_file, cam_map)

    print('===================================================================================', module_fullname, func_name)
    print(f"acc: {acc_list}")
    print(f"rows: {row_list}")
    print(f"cols: {col_list}")
    print(f"cells: {cell_list}")
    print("total cells:", total_cells)

    if row_list_total is None:
        row_list_total = [row_list[i] for i in range(bits)]
    else:
        for i in range(bits):
            if row_list_total[i] < row_list[i]:
                row_list_total[i] = row_list[i]
    return row_list_total


def train_trees(params_dir, max_leaf_nodes, encode, seed):
    if seed is not None:
        np.random.seed(seed)

    file = os.path.join(params_dir, 'params.txt')

    # this will store the maximum number of rows for each bit
    row_list_total = None

    # each line is for different tree
    input_file = open(file, "r")
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

        row_list_total = train_tree(params_dir, module_fullname, func_name, scale, zero, x_min, x_max, bits, out_range, max_leaf_nodes, encode, row_list_total)

    print('-----------------------------------------------------------------')
    print(row_list_total)

    acam_rows_file = os.path.join(params_dir, 'acam_rows.txt')
    with open(acam_rows_file, "w") as f:
        for x in row_list_total:
            f.write(f"{x}\n")

    input_file.close()
