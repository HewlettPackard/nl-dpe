import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from .data import TextDataset, CustomCollateFn
from .tokenizer import Tokenizer
from .llama import LlamaForCausalLM, ForCausalLMLoss


class LlamaHead(nn.Module):
    def __init__(self, llama_model):
        super().__init__()

        self.embed_tokens = llama_model.model.embed_tokens
        self.rotary_emb = llama_model.model.rotary_emb

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        labels = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return hidden_states, causal_mask, position_ids, position_embeddings

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else sequence_length + 1

        batch_size=input_tensor.shape[0]

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask


class LlamaLayers(nn.Module):
    def __init__(self, llama_model, start, end):
        super().__init__()

        self.layers = nn.ModuleList()
        for layer in llama_model.model.layers[start:end+1]:
            self.layers.append(layer)

    def forward(
        self,
        hidden_states,
        causal_mask,
        position_ids,
        position_embeddings,
    ):
        for decoder_layer in self.layers:
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_output
        return hidden_states, causal_mask, position_ids, position_embeddings


class LlamaTail(nn.Module):
    def __init__(self, llama_model):
        super().__init__()

        self.norm = llama_model.model.norm
        self.lm_head = llama_model.lm_head

    def forward(
        self,
        hidden_states,
        labels = None,
    ):
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=128256)

        return (loss, logits) if loss is not None else (logits,)


def get_llama_model(model_type):
    if model_type == "llama3.2_1B":
        model_id = "meta-llama/Llama-3.2-1B"
        config = {
            "attention_dropout": 0.0,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "attention_bias": False,
            "intermediate_size": 8192,
            "mlp_bias": False,
            "partial_rotary_factor": False,
            "rope_scaling": {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'},
            "pad_token_id": None,
            "num_hidden_layers": 16,
            "rms_norm_eps": 1e-05,
            "vocab_size": 128256,
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "head_dim": 64,
            "rope_theta": 500000.0,
        }
    elif model_type == "llama3.2_3B":
        model_id = "meta-llama/Llama-3.2-3B"
        config = {
            "attention_dropout": 0.0,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "attention_bias": False,
            "intermediate_size": 8192,
            "mlp_bias": False,
            "partial_rotary_factor": False,
            "rope_scaling": {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'},
            "pad_token_id": None,
            "num_hidden_layers": 28,
            "rms_norm_eps": 1e-05,
            "vocab_size": 128256,
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "head_dim": 128,
            "rope_theta": 500000.0,
        }
    else:
        print("unrecognized llama model type")
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_id)

    model = LlamaForCausalLM(config=config)
    model.load_state_dict(pretrained_model.state_dict())

    dataset = load_dataset("ptb_text_only")
    train_texts = dataset["train"]["sentence"]
    val_texts = dataset["validation"]["sentence"]
    test_texts = dataset["test"]["sentence"]

    def create_dataset(texts):
        seq_len = 512
        tokens = tokenizer(" ".join(texts), return_tensors="pt", return_attention_mask=False)
        input_ids = tokens["input_ids"][0]
        n_tokens = input_ids.shape[0]

        input_id_list = []
        for i in range(0, n_tokens - seq_len, seq_len):
            input_chunk = input_ids[i:i+seq_len]
            input_id_list.append(input_chunk)
        return input_id_list

    train_list = create_dataset(train_texts)
    test_list = create_dataset(test_texts)

    return model, train_list, test_list, tokenizer


def split_llama(model, parts=None):
    if parts == "head":
        head = LlamaHead(model)
        model = head
    elif parts == "tail":
        tail = LlamaTail(model)
        model = tail
    elif parts is not None:
        stard_end = parts.split("-")
        start_layer = int(stard_end[0])
        end_layer = int(stard_end[1])
        layers = LlamaLayers(model, start_layer, end_layer)
        model = layers
    return model