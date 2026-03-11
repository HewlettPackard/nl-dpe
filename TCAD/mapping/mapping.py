import os
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn.functional as F


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
    x = torch.tensor(x)
    x[x == 0] += 0.00000001
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


def map_1op_function(mapping_dir, func_name, x_scale, x_zero, x_min, x_max, y_scale, y_zero, y_min, y_max, bits, in_range, out_range, encode):
    print('=====================', mapping_dir, func_name)

    if out_range == "neg_and_pos":
        quant_min = -(2**(bits-1))    # the minimun binary value of output
        quant_max = 2**(bits-1) - 1   # the maximum binary value of output
    elif out_range == "pos":
        quant_min = 0                 # the minimun binary value of output
        quant_max = 2**bits - 1       # the maximum binary value of output
    input_levels = 2**bits
    mask = 2**bits - 1

    table = []
    for x in range(0, input_levels):
        x = np.array(x)
        x_float = dequant(x, x_scale, x_zero)
        y_float = function_list[func_name][0](x_float)
        y = quant(y_float, y_scale, y_zero, quant_min, quant_max)
        if encode:
            y = gray_encode(y)
        y = y & mask
        table.append((x, [int(b) for b in format(y, f'0{bits}b')]))

    # map for each bit
    acam_rows = []
    for b in range(bits):
        ranges = []
        prev_bit = ""
        prev_x = ""
        start = ""
        end = ""
        for t in table:
            if t[1][b] != prev_bit:
                if t[1][b] == 1:
                    start = t[0]
                elif t[1][b] == 0:
                    end = prev_x
                    if start != "":
                        ranges.append((start, end))
            prev_bit = t[1][b]
            prev_x = t[0]
        if prev_bit == 1:
            ranges.append((start, input_levels-1))

        threshold_array = np.array(ranges).astype(np.float32)
        if (len(threshold_array)) == 0:
            threshold_array = np.append(threshold_array, [0, input_levels-1]).reshape(1, 2)
        threshold_array[:, 0] = threshold_array[:, 0] - 0.5
        threshold_array[:, 1] = threshold_array[:, 1] + 0.5
        npy_file = os.path.join(mapping_dir, "dt_bit_"+str(bits-1-b)+".npy")
        np.save(npy_file, threshold_array)
        acam_rows.append(len(threshold_array))

    #########################################################
    loaded_bits = []
    for b in range(bits):
        npy_file = os.path.join(mapping_dir, "dt_bit_"+str(bits-1-b)+".npy")
        loaded_bits.append(np.load(npy_file))

    x = np.linspace(0, input_levels-1, input_levels)

    V = np.zeros(input_levels).astype(np.int32)
    for b in range(bits):
        low = loaded_bits[b][:, 0][:, None]
        high = loaded_bits[b][:, 1][:, None]

        B = (x >= low) & (high >= x)
        B = B.astype(np.int32)
        B = B.sum(axis=0)
        B = B > 0
        B = B.astype(np.int32)

        V += B * 2**(7-b)

    V = V & mask
    if encode:
        V = gray_decode(V, bits)
    V = dequant(V, y_scale, y_zero)

    x_float = dequant(x, x_scale, x_zero)
    T = function_list[func_name][0](x_float).reshape(-1)
    T = quant(T, y_scale, y_zero, quant_min, quant_max)
    T = dequant(T, y_scale, y_zero)

    error = np.abs(T-V).sum()
    print("Error:", error)
    #########################################################

    return acam_rows


def find_largest_rectangle(matrix, uncovered):
    rows, cols = matrix.shape
    best_rect = None

    for r1 in range(rows):
        for r2 in range(r1, rows):
            valid_cols = []
            for c in range(cols):
                if all(matrix[r, c] == 1 and uncovered[r, c] for r in range(r1, r2 + 1)):
                    valid_cols.append(c)
                else:
                    if valid_cols:
                        c1, c2 = valid_cols[0], valid_cols[-1]
                        height = r2 - r1 + 1
                        width = c2 - c1 + 1
                        area = height * width
                        if best_rect is None or area > best_rect[-1] or \
                            (area == best_rect[-1] and height > best_rect[2] - best_rect[0] + 1):
                            best_rect = (r1, c1, r2, c2, area)
                        valid_cols = []
            if valid_cols:
                c1, c2 = valid_cols[0], valid_cols[-1]
                height = r2 - r1 + 1
                width = c2 - c1 + 1
                area = height * width
                if best_rect is None or area > best_rect[-1] or \
                    (area == best_rect[-1] and height > best_rect[2] - best_rect[0] + 1):
                    best_rect = (r1, c1, r2, c2, area)
    return best_rect


def cover_ones_with_rectangles(matrix):
    matrix = np.array(matrix)
    uncovered = (matrix == 1)
    rectangles = []

    while np.any(uncovered):
        rect = find_largest_rectangle(matrix, uncovered)
        if rect is None:
            break  # No more rectangles possible
        r1, c1, r2, c2, _ = rect
        rectangles.append((r1, c1, r2, c2))
        uncovered[r1:r2+1, c1:c2+1] = False  # Mark covered 1s

    return rectangles


def map_2ops_function(mapping_dir, func_name, bits, encode):
    print('=====================', mapping_dir, func_name)

    assert bits == 8
    input_bits = bits // 2
    input_levels = 2**input_bits
    mask = 2**bits - 1

    table = []
    for x1 in range(0, input_levels):
        inner = []
        for x2 in range(0, input_levels):
            y = x1 * x2
            y = np.array(y)
            if encode:
                y = gray_encode(y)
            bit_list = [int(b) for b in format(y, f'0{bits}b')]
            inner.append(bit_list)
        table.append(inner)
    table = np.array(table)

    tables = []
    for i in range(bits):
        tables.append(table[:,:,i])

    acam_rows = []
    total_rectangles = 0
    for b, t in enumerate(tables):
        rectangles = cover_ones_with_rectangles(t)
        rectangles = np.array(rectangles).astype(np.float32)
        rectangles[:, 0] = rectangles[:, 0] - 0.5
        rectangles[:, 1] = rectangles[:, 1] - 0.5
        rectangles[:, 2] = rectangles[:, 2] + 0.5
        rectangles[:, 3] = rectangles[:, 3] + 0.5
        total_rectangles += len(rectangles)
        npy_file = os.path.join(mapping_dir, "dt_bit_"+str(bits-1-b)+".npy")
        np.save(npy_file, rectangles)
        acam_rows.append(len(rectangles))

    #########################################################
    loaded_bits = []
    for b in range(bits):
        npy_file = os.path.join(mapping_dir, "dt_bit_"+str(bits-1-b)+".npy")
        loaded_bits.append(np.load(npy_file))

    x = np.linspace(0, input_levels-1, input_levels)
    x1 = np.repeat(x, input_levels)[None, :]
    x2 = np.tile(x, 16)[None, :]

    V = np.zeros(input_levels*input_levels).astype(np.int32)
    for b in range(bits):
        x1_low = loaded_bits[b][:, 0][:, None]
        x2_low = loaded_bits[b][:, 1][:, None]
        x1_high = loaded_bits[b][:, 2][:, None]
        x2_high = loaded_bits[b][:, 3][:, None]

        B = (x1 >= x1_low) & (x1_high >= x1) & (x2 >= x2_low) & (x2_high >= x2)
        B = B.astype(np.int32)
        B = B.sum(axis=0)
        B = B > 0
        B = B.astype(np.int32)

        V += B * 2**(7-b)

    V = V & mask
    if encode:
        V = gray_decode(V, bits)

    T = x1 * x2

    error = np.abs(T-V).sum()
    print("Error:", error)
    #########################################################

    return acam_rows


def mapping(params_dir, encode, seed):

    if seed is not None:
        np.random.seed(seed)

    file = os.path.join(params_dir, 'params.txt')

    acam_rows = []

    # each line is for different mapping
    input_file = open(file, "r")
    while True:
        line = input_file.readline()
        if not line:
            break

        tokens = line.split()
        module_fullname = tokens[0]     # full name of the activation layer in the model
        func_name = tokens[1]           # function name
        if func_name == "mul":
            bits = int(tokens[2])           # bit width of the output

            # the result of this mapping are in a separate folder
            mapping_dir = os.path.join(params_dir, module_fullname)
            os.makedirs(mapping_dir, exist_ok=True)

            func_acam_rows = map_2ops_function(mapping_dir, func_name, bits, encode)
            for i, r in enumerate(func_acam_rows):
                if len(acam_rows) < i+1:
                    acam_rows.append(r)
                elif acam_rows[i] < r:
                    acam_rows[i] = r
        else:
            x_scale = float(tokens[2])      # scale of input quantization
            x_zero = float(tokens[3])       # zero point of input quantization
            x_min = float(tokens[4])        # minimum input value
            x_max = float(tokens[5])        # maximum input value
            y_scale = float(tokens[6])      # scale of output quantization
            y_zero = float(tokens[7])       # zero point of output quantization
            y_min = float(tokens[8])        # minimum output value
            y_max = float(tokens[9])        # maximum output value
            bits = int(tokens[10])          # bit width of the output
            in_range = tokens[11]           # range of input quantization
            out_range = tokens[12]          # range of output quantization

            # the result of this mapping are in a separate folder
            mapping_dir = os.path.join(params_dir, module_fullname)
            os.makedirs(mapping_dir, exist_ok=True)

            func_acam_rows = map_1op_function(mapping_dir, func_name, x_scale, x_zero, x_min, x_max, y_scale, y_zero, y_min, y_max, bits, in_range, out_range, encode)
            for i, r in enumerate(func_acam_rows):
                if len(acam_rows) < i+1:
                    acam_rows.append(r)
                elif acam_rows[i] < r:
                    acam_rows[i] = r

    input_file.close()
    acam_rows.reverse() # print in the same order of --acam_rows option in commandline
    print(acam_rows)

    acam_rows_file = os.path.join(params_dir, 'acam_rows.txt')
    with open(acam_rows_file, "w") as f:
        for x in acam_rows:
            f.write(f"{x}\n")