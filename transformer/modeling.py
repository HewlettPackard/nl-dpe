# coding=utf-8
# Copyright (2025) Hewlett Packard Enterprise Development LP
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import copy
import json
import logging
import math
import os
import re
from io import open
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

import nl_dpe.dpe as dpe

from .file_utils import CONFIG_NAME, WEIGHTS_NAME

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "",
}

BERT_CONFIG_NAME = "bert_config.json"
TF_WEIGHTS_NAME = "model.ckpt"

# Regular expression to extract global step fromm file names such as step_1000_pytorch_model.bin
_checkpoint_file_name_pattern = re.compile(r"step_(\d+)_pytorch_model\.bin")


def _extract_global_step_from_checkpoint(path: Path) -> int:
    match = _checkpoint_file_name_pattern.search(path.name)
    return int(match.group(1)) if match else -1


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print(f"Converting TensorFlow checkpoint from {tf_path}.")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print(f"Loading TF weight {name} with shape {shape}.")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping:", "/".join(name))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                try:
                    pointer = getattr(pointer, "bias")
                except AttributeError:
                    print("Skipping:", "/".join(name))
                    continue
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif l[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping:", "/".join(name))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print(f"Initialize PyTorch weight {namme}.")
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
            """Construct a layernorm module in the TF style (epsilon inside the square root)."""
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x: Float[torch.Tensor, "batch seq model_dim"]) -> Float[torch.Tensor, "batch seq model_dim"]:
            u: Float[torch.Tensor, "batch seq 1"] = x.mean(-1, keepdim=True)
            s: Float[torch.Tensor, "batch seq 1"] = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class HeadAttention(nn.Module):
    def __init__(self, config, hidden_size, head_num, head_used) -> None:
        super(HeadAttention, self).__init__()
        self.head_num = head_num
        self.head_used = head_used
        self.hidden_size = hidden_size
        if self.hidden_size % self.head_num != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.head_num})."
            )

        self.attention_head_size = int(self.hidden_size / self.head_num)
        self.all_head_size = self.num_heads_used * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads_used, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.num_heads_used != self.num_attention_heads:
            pad_shape = context_layer.size()[:-1] + (
                (self.num_attention_heads - self.num_heads_used) * self.attention_head_size,
            )

            pad_layer = torch.zeros(*pad_shape).to(context_layer.device)
            context_layer = torch.cat((context_layer, pad_layer), -1)
        return context_layer


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}
NORM = {"layer_norm": BertLayerNorm}


class BertArchitecture:
    DEFAULT = 1
    AVERAGE_HIDDEN_STATES = 2
    REDUCED_HIDDEN_DIMENSIONALITY = 3


class DimReductionLayerArchitecture:
    DEFAULT = 1  # Simple one-layer linear projection


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        pre_trained: str = "",
        training="",
        embedding_size: int = -1,
        arch: int = 1,
        dim_reduction_arch: int = 1,
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pre_trained = pre_trained
            self.training = training
            self.embedding_size = embedding_size
            self.arch = arch
            self.dim_reduction_arch = dim_reduction_arch
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

        if self.embedding_size < 0:
            self.embedding_size = self.hidden_size

    @classmethod
    def from_dict(cls, json_object) -> "BertConfig":
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        if config.embedding_size < 0:
            config.embedding_size = config.hidden_size
        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "BertConfig":
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self) -> Dict:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str) -> None:
        """Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig) -> None:
        super(BertEmbeddings, self).__init__()
        embedding_dim: int = config.hidden_size
        if config.arch == BertArchitecture.REDUCED_HIDDEN_DIMENSIONALITY:
            embedding_dim = config.embedding_size
            if embedding_dim < config.hidden_size:
                raise ValueError(
                    "The embedding_size configuration parameter must be less than hidden_size. "
                    f"Got embedding_size: {config.embedding_size} and hidden_size: {config.hidden_size}."
                )

        self.dim_reduction_layer: Optional[nn.Module] = None
        if embedding_dim > config.hidden_size:
            print(f"[BertEmbeddings] adding dim reduction {embedding_dim} -> {config.hidden_size}.")
            if config.dim_reduction_arch == DimReductionLayerArchitecture.DEFAULT:
                self.dim_reduction_layer = nn.Linear(embedding_dim, config.hidden_size)
            else:
                raise ValueError(f"Unknown dimension reduction layer architecture: {config.dim_reduction_arch}. ")

        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, embedding_dim)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, embedding_dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: Int[torch.Tensor, "batch seq"],
            token_type_ids: Optional[Int[torch.Tensor, "batch seq"]] = None
    ) -> Float[torch.Tensor, "batch seq model_dim"]:
        seq_length: int = input_ids.size(1)
        position_ids: Int[torch.Tensor, " seq"] = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids: Int[torch.Tensor, "batch seq"] = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings: Float[torch.Tensor, "batch seq model_dim"] = self.word_embeddings(input_ids)
        position_embeddings: Float[torch.Tensor, "batch seq model_dim"] = self.position_embeddings(position_ids)
        token_type_embeddings: Float[torch.Tensor, "batch seq model_dim"] = self.token_type_embeddings(token_type_ids)

        embeddings: Float[torch.Tensor, "batch seq model_dim"] = \
            words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.dim_reduction_layer is not None:
            embeddings = self.dim_reduction_layer(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig, bert_layer_index: int) -> None:
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads: int = config.num_attention_heads  # TinyBERT 12
        self.attention_head_size: int = int(config.hidden_size / config.num_attention_heads)  # TinyBERT 26
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size  # TinyBERT 312

        self.arch = config.arch
        attention_input_dim: int = config.hidden_size
        if self.arch == BertArchitecture.AVERAGE_HIDDEN_STATES:
            attention_input_dim = self.attention_head_size

        self.query = nn.Linear(attention_input_dim, self.all_head_size)  # TinyBERT [312, 312] all attention heads
        self.key = nn.Linear(attention_input_dim, self.all_head_size)  # TinyBERT [312, 312] all attention heads
        self.value = nn.Linear(attention_input_dim, self.all_head_size)  # TinyBERT [312, 312] all attention heads

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.bert_layer_index = bert_layer_index

    def transpose_for_scores(
            self, x: Float[torch.Tensor, "batch seq model_dim"]
    ) -> Float[torch.Tensor, "batch head seq model_dim"]:
        new_x_shape: torch.Size = cast(torch.Size, x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: Float[torch.Tensor, "batch seq model_dim"],
            attention_mask: Float[torch.Tensor, "batch 1 1 seq"],
            output_att: bool = False
    ) -> Tuple[
            Float[torch.Tensor, "batch seq model_dim"],
            Float[torch.Tensor, "batch head seq seq"]
    ]:
        # head_in_dim could be different from model_dim.
        attention_input: Float[torch.Tensor, "batch seq head_in_dim"]
        # hidden_states: [Batch, SeqLen, HiddenSz] attention_mask: [Batch, 1, 1, SeqLen]
        if self.arch == BertArchitecture.AVERAGE_HIDDEN_STATES:
            hidden_states = hidden_states.reshape(
                hidden_states.shape[0],  # Batch
                hidden_states.shape[1],  # SeqLen
                self.num_attention_heads,  #
                self.attention_head_size,  #
            )
            attention_input = hidden_states.mean(dim=2)
        else:
            attention_input = hidden_states

        # These linear layers will bring input dimension back to hidden size if input was reduced above.
        mixed_query_layer: Float[torch.Tensor, "batch seq model_dim"] = self.query(attention_input)
        mixed_key_layer: Float[torch.Tensor, "batch seq model_dim"] = self.key(attention_input)
        mixed_value_layer: Float[torch.Tensor, "batch seq model_dim"] = self.value(attention_input)

        if dpe.enabled(self.bert_layer_index):
            dpe_query, dpe_key, dpe_value = dpe.call(attention_input)
            start_idx: int = (dpe.attention_head_index() - 1) * self.attention_head_size
            end_idx: int = start_idx + self.attention_head_size

            mixed_query_layer[: , :, start_idx:end_idx] = dpe_query
            mixed_key_layer[: , :, start_idx:end_idx] = dpe_key
            mixed_value_layer[: , :, start_idx:end_idx] = dpe_value

        query_layer: Float[torch.Tensor, "batch head seq head_out_dim"] = self.transpose_for_scores(
            mixed_query_layer
        )
        key_layer: Float[torch.Tensor, "batch head seq head_out_dim"] = self.transpose_for_scores(
            mixed_key_layer
        )
        value_layer: Float[torch.Tensor, "batch head seq head_out_dim"] = self.transpose_for_scores(
            mixed_value_layer
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores: Float[torch.Tensor, "batch head seq seq"] = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # [Batch, NumAttHeads, SeqLen, SeqLen]

        # Normalize the attention scores to probabilities.
        attention_probs: Float[torch.Tensor, "batch head seq seq"] = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer: Float[torch.Tensor, "batch head seq head_out_dim"] = torch.matmul(
            attention_probs, value_layer
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape: torch.Size = cast(torch.Size, context_layer.size()[:-2] + (self.all_head_size, ))
        context_layer: Float[torch.Tensor, "batch seq model_dim"] = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig, bert_layer_index: int) -> None:
        super(BertAttention, self).__init__()

        self.self = BertSelfAttention(config, bert_layer_index)
        self.output = BertSelfOutput(config)
        self.bert_layer_index = bert_layer_index

    def forward(
            self,
            input_tensor: Float[torch.Tensor, "batch seq model_dim"],
            attention_mask: Float[torch.Tensor, "batch 1 1 seq"]
    ) -> Tuple[
            Float[torch.Tensor, "batch seq model_dim"],
            Float[torch.Tensor, "batch head seq seq"]
    ]:
        self_output: Float[torch.Tensor, "batch seq model_dim"]
        layer_att: Float[torch.Tensor, "batch head seq seq"]
        self_output, layer_att = self.self(input_tensor, attention_mask)
        attention_output: Float[torch.Tensor, "batch seq model_dim"] = self.output(self_output, input_tensor)
        return attention_output, layer_att


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            hidden_states: Float[torch.Tensor, "batch seq model_dim"],
            input_tensor: Float[torch.Tensor, "batch seq model_dim"]
    ) -> Float[torch.Tensor, "batch seq model_dim"]:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig, intermediate_size=-1) -> None:
        super(BertIntermediate, self).__init__()
        if intermediate_size < 0:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        else:
            self.dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(
            self, hidden_states: Float[torch.Tensor, "batch seq model_dim"]
    ) -> Float[torch.Tensor, "batch seq intermediate_dim"]:
        y: Float[torch.Tensor, "batch seq intermediate_dim"] = self.dense(hidden_states)
        y = self.intermediate_act_fn(y)
        return y


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig, intermediate_size: int = -1) -> None:
        super(BertOutput, self).__init__()
        if intermediate_size < 0:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, 
            hidden_states: Float[torch.Tensor, "batch seq intermediate_dim"], 
            input_tensor: Float[torch.Tensor, "batch seq model_dim"]
    ) -> Float[torch.Tensor, "batch seq model_dim"]:
        y: Float[torch.Tensor, "batch seq model_dim"] = self.dense(hidden_states)
        y = self.dropout(y)
        y = self.LayerNorm(y + input_tensor)
        return y


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig, layer_index: int) -> None:
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, layer_index)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.layer_index = layer_index

    def forward(
            self,
            hidden_states: Float[torch.Tensor, "batch seq model_dim"],
            attention_mask: Float[torch.Tensor, "batch 1 1 seq"]
    ) -> Tuple[
            Float[torch.Tensor, "batch seq model_dim"], # layer output
            Float[torch.Tensor, "batch head seq seq"] # attention
    ]:
        attention_output: Float[torch.Tensor, "batch seq model_dim"]
        layer_att: Float[torch.Tensor, "batch head seq seq"]

        attention_output, layer_att = self.attention(hidden_states, attention_mask)
        intermediate_output: Float[torch.Tensor, "batch seq intermediate_dim"] = self.intermediate(
            attention_output
        )
        layer_output: Float[torch.Tensor, "batch seq model_dim"] = self.output(
            intermediate_output, attention_output
        )

        return layer_output, layer_att


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, idx + 1) for idx in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: Float[torch.Tensor, "batch seq model_dim"],
            attention_mask: Float[torch.Tensor, "batch 1 1 seq"]
    ) -> Tuple[
            list[Float[torch.Tensor, "batch seq model_dim"]], # hidden states
            list[Float[torch.Tensor, "batch head seq seq"]] # attention outputs
    ]:
        """
        hidden_states: [Batch, Length, EmbDim]
        """
        all_encoder_layers: list[Float[torch.Tensor, "batch seq model_dim"]] = []  # hidden states
        all_encoder_atts: list[Float[torch.Tensor, "batch head seq seq"]] = []
        layer_att: Float[torch.Tensor, "batch head seq seq"]

        layer_module: BertLayer
        for layer_module in self.layer:
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(hidden_states, attention_mask)
            all_encoder_atts.append(layer_att)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig, recurs=None) -> None:
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(
            self, hidden_states: list[Float[torch.Tensor, "batch seq model_dim"]]
    ) -> Float[torch.Tensor, "batch model_dim"]:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. "-1" refers to last layer
        pooled_output: Float[torch.Tensor, "batch model_dim"]= hidden_states[-1][:, 0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None:
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            cls_name: str = self.__class__.__name__
            raise ValueError(
                f"Parameter config in `{cls_name}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                f"`model = {cls_name}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path: str, *inputs, **kwargs):
        if pretrained_model_name_or_path.endswith(".json"):
            resolved_config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            resolved_config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            raise FileNotFoundError(f"Cannot find config file at {pretrained_model_name_or_path}.")

        config = BertConfig.from_json_file(resolved_config_file)

        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of 
                        Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict: Optional[Dict] = kwargs.get("state_dict", None)
        kwargs.pop("state_dict", None)
        from_tf: bool = kwargs.get("from_tf", False)
        kwargs.pop("from_tf", None)

        # This handles case when input is a path to a model weights file.
        weights_file = WEIGHTS_NAME
        if os.path.isfile(pretrained_model_name_or_path):
            pretrained_model_name_or_path, weights_file = os.path.split(pretrained_model_name_or_path)

        # Load config
        config_file: str = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.

        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            # Try pre-defined name, then see if maybe we are loading from a training run.
            weights_path: str = os.path.join(pretrained_model_name_or_path, weights_file)
            if not os.path.isfile(weights_path):
                checkpoints: List[Path] = list(Path(pretrained_model_name_or_path).glob("*_pytorch_model.bin"))
                # The checkpoints list contain paths to model checkpoints. Each file name has t he following pattern:
                # step_{global_step}_pytorch_model.bin. We select the checkpoint with the highest global_step value.
                if len(checkpoints) > 0:
                    weights_path = max(checkpoints, key=_extract_global_step_from_checkpoint).as_posix()
                else:
                    logger.warning(f"Could not find the model weights file at {pretrained_model_name_or_path}.")

            logger.info("Loading model {}".format(weights_path))
            state_dict = torch.load(weights_path, map_location="cpu")

        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(s.startswith("bert.") for s in state_dict.keys()):
            start_prefix = "bert."

        logger.info("loading model...")
        load(model, prefix=start_prefix)
        logger.info("done!")
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )

        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described 
            below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig) -> None:
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        token_type_ids: Optional[Int[torch.Tensor, "batch seq"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch seq"]] = None,
        output_all_encoded_layers: bool = True,
        output_att: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        # input_ids, token_type_ids, attention_mask: [Batch, Length]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask: Float[torch.Tensor, "batch 1 1 seq"] = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output: Float[torch.Tensor, "batch seq model_dim"] = self.embeddings(input_ids, token_type_ids)

        encoded_layers: list[Float[torch.Tensor, "batch seq model_dim"]]  # hidden states
        layer_atts: list[Float[torch.Tensor, "batch head seq seq"]]  # attention outputs
        encoded_layers, layer_atts = self.encoder(embedding_output, extended_attention_mask)

        pooled_output: Float[torch.Tensor, "batch model_dim"] = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output

        return encoded_layers, layer_atts, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape 
            [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are 
            ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None
    ):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        elif masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config: BertConfig, fit_size: int = 768) -> None:
        super(TinyBertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        masked_lm_labels=None,
        next_sentence_label=None,
        labels=None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # input_ids, token_type_ids, attention_mask: [Batch, Length]
        sequence_output: List[Tensor]  # embeddings + encoder layers?
        att_output: List[Tensor]  # num encoder layers
        pooled_output: Tensor  # [Batch, EmbDim]
        sequence_output, att_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, output_att=False, infer=False
    ):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, output_att=output_att
        )

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            if not output_att:
                return masked_lm_loss
            else:
                return masked_lm_loss, att_output
        else:
            if not output_att:
                return prediction_scores
            else:
                return prediction_scores, att_output


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSentencePairClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSentencePairClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        a_input_ids,
        b_input_ids,
        a_token_type_ids=None,
        b_token_type_ids=None,
        a_attention_mask=None,
        b_attention_mask=None,
        labels=None,
    ):
        _, a_pooled_output = self.bert(a_input_ids, a_token_type_ids, a_attention_mask, output_all_encoded_layers=False)
        # a_pooled_output = self.dropout(a_pooled_output)

        _, b_pooled_output = self.bert(b_input_ids, b_token_type_ids, b_attention_mask, output_all_encoded_layers=False)
        # b_pooled_output = self.dropout(b_pooled_output)

        logits = self.classifier(
            torch.relu(torch.cat((a_pooled_output, b_pooled_output, torch.abs(a_pooled_output - b_pooled_output)), -1))
        )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class TinyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, fit_size=768):
        super(TinyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_bert_weights)

    def forward(
            self,
            input_ids: Int[torch.Tensor, "batch seq"],
            token_type_ids: Optional[Int[torch.Tensor, "batch seq"]] = None,
            attention_mask: Optional[Int[torch.Tensor, "batch seq"]] = None,
            labels = None,
            is_student: bool = False
    ) -> tuple[
            Float[torch.Tensor, "batch class"],
            list[Float[torch.Tensor, "batch head_size seq seq"]],
            list[Float[torch.Tensor, "batch seq model_dim"]]
    ]:
        sequence_output: list[Float[torch.Tensor, "batch seq model_dim"]]
        att_output: list[Float[torch.Tensor, "batch head_size seq seq"]]
        pooled_output: Float[torch.Tensor, "batch model_dim"]

        sequence_output, att_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, output_att=True
        )

        logits: Float[torch.Tensor, "batch class"] = self.classifier(torch.relu(pooled_output))

        if is_student:
            tmp = []
            for sequence_layer in sequence_output:
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output
