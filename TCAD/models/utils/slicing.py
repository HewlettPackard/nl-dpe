import torch


def int_to_base(x_int, base, n_digits):
    x_int = x_int.long()
    x_int = x_int.unsqueeze(-1)

    shape_len = len(x_int.shape)
    new_shape = (1,) * (shape_len - 1) + (n_digits,)

    exponents = torch.arange(n_digits - 1, -1, -1, device=x_int.device).view(new_shape)
    powers = base ** exponents

    digits = (x_int // powers) % base
    digits = digits.to(x_int.dtype)
    return digits


def weight_slicing(W, bit_width, bits_per_cell):
    levels_per_cell = 2**bits_per_cell             # we can also think of it as the scale between different weight slices
    cells_per_weight = bit_width // bits_per_cell  # how many cells to store one weight

    # remember sign and convert all weights to non-negative
    sign = torch.where(W >= 0, torch.tensor(1), torch.tensor(-1))
    W = torch.abs(W)

    # slice weight matrix
    W_slice = int_to_base(W, levels_per_cell, cells_per_weight)

    return W_slice, sign


def weight_recompose(W_slice, sign, bit_width, bits_per_cell):
    levels_per_cell = 2**bits_per_cell             # we can also think of it as the scale between different weight slices
    cells_per_weight = bit_width // bits_per_cell  # how many cells to store one weight

    # prepare weight shifts that are used to recompose the weights from slices
    shape_len = len(W_slice.shape)
    new_shape = (1,) * (shape_len - 1) + (cells_per_weight,)
    weight_shifts = torch.flip(levels_per_cell ** torch.arange(cells_per_weight), dims=(0,)).to(W_slice.device).to(W_slice.dtype).view(new_shape)

    # recompose
    W_shifted = W_slice * weight_shifts
    W_recomposed = W_shifted.sum(dim=-1)

    # restore sign
    W_recomposed = W_recomposed * sign

    return W_recomposed