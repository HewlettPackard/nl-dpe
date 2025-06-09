###
# Copyright (2025) Hewlett Packard Enterprise Development LP
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
from jaxtyping import Float

__all__ = ["attention_head_index", "enabled", "call"]


_BERT_LAYER_INDEX: int = 1
"""1-based layer index (1, 2, 3, 4) or -1 to disable DPE call."""

_ATTENTION_HEAD_INDEX: int = 2
"""1-based attention head index [1, 2, ..., 12] inclusive or -1 to disable DPE call."""


def attention_head_index() -> int:
    return _ATTENTION_HEAD_INDEX


def enabled(bert_layer_index: int) -> bool:
    """Check if DPE is enabled.

    Args:
        bert_layer_index: Index of the BERT layer for which we check if DPE is enabled.

    Returns:
        True if DPE is enabled, False otherwise.
    """
    return bert_layer_index == _BERT_LAYER_INDEX and _ATTENTION_HEAD_INDEX >= 1


def call(
        attention_input: Float[torch.Tensor, "batch sequence input_sz"]
) -> tuple[
        Float[torch.Tensor, "batch sequence output_sz"],
        Float[torch.Tensor, "batch sequence output_sz"],
        Float[torch.Tensor, "batch sequence output_sz"]
]:
    """Call the DPE engine that computes query, key and value matrices for dot-product attention for one attention head.

    Args:
        attention_input: Rank-3 tensor containing input for attention module. First dimension is batch dimension,
            its size could be 32. Second dimension is length of the input sequence dimension, its size could be 64.
            Last dimension is input size dimension. Typically, that's the same as hidden size, but here that will be
            hidden_size // num_attention_heads. It could be 26 (12 (num_attention_heads) * 26 (input attention size) = 
            312 (hidden size)).
    
    Returns:
        A tuple of three tensors - query, key and value. For models in this repository, this will be something like
        (32, 64, 312). Output dimension is determined by the weight tensors previously exported.
    """

    # Just for example, placeholder implementation returns random tensors.
    output_dim: int = 26
    output_shape: torch.Size = torch.Size((attention_input.shape[0], attention_input.shape[1], output_dim))

    def _random_tensor() -> Float[torch.Tensor, "batch sequence output_sz"]:
        return torch.randn(output_shape, dtype=attention_input.dtype, device=attention_input.device)

    return (_random_tensor(), _random_tensor(), _random_tensor())
