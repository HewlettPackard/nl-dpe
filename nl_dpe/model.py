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
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, OrderedDict

import click
import numpy as np
import torch


def _validate_model_path(ctx: click.Context, param: str, model_path: Path) -> Path:
    """Check that the path points to a directory with a model."""
    if not model_path.is_dir():
        raise click.BadParameter(f"Model path does not exist or not directory: {model_path}.")

    _expected_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
    _missing_files = [file_name for file_name in _expected_files if not (model_path / file_name).is_file()]
    if _missing_files:
        raise click.BadParameter(f"Model path is missing required files: {', '.join(_missing_files)} in {model_path}.")

    return model_path


def _validate_output_path(ctx: click.Context, param: str, model_path: Path) -> Path:
    """Check that the output file path does not exist."""
    if model_path.exists():
        raise click.BadParameter(f"Output file or directory already exists: {model_path}.")
    return model_path


def _validate_layer_index(ctx: click.Context, param: str, layer: int) -> int:
    """Check that the 1-based layer index is in the expected range."""
    if not (1 <= layer <= 4):
        raise click.BadParameter(f"Layer index must be in the range [1, 4], got {layer}.")
    return layer


def _validate_head_index(ctx: click.Context, param: str, head: int) -> int:
    """Check that the 1-based head index is the expected range."""
    if not (1 <= head <= 12):
        raise click.BadParameter(f"Head index must be in the range [1, 12], got {head}.")
    return head


def _check_numpy_array(arr: np.ndarray, expected_shape: tuple) -> np.ndarray:
    """Check that input is a numpy array with the expected shape."""
    assert isinstance(arr, np.ndarray), f"Expected a numpy array. Got: {type(arr)}"
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got {arr.shape}."
    return arr


model_path_option = click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the PyTorch model.",
    callback=_validate_model_path,
)


weights_file_option = click.option(
    "--weights-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to attention weights to load in pickle format (dictionary).",
)


layer_option = click.option(
    "--layer",
    type=int,
    required=True,
    callback=_validate_layer_index,
    help="1-based index of the encoder layer [1, ..., 4].",
)


head_option = click.option(
    "--head",
    type=int,
    required=True,
    callback=_validate_head_index,
    help="1-based index of the attention head layer [1, ..., 12].",
)


@click.group(name="model", help="CLI to interact with PyTorch models (extract weights, update models).")
def cli() -> None: ...


@cli.command(name="extract-attention-weights", help="Extract attention weights from a PyTorch model.")
@model_path_option
@click.option(
    "--output-file",
    type=click.Path(exists=False, path_type=Path),
    required=True,
    help="Path to save the extracted attention weights in pickle format (dictionary).",
    callback=_validate_output_path,
)
@layer_option
@head_option
def extract_attention_weights(model_path: Path, output_file: Path, layer: int, head: int) -> None:
    """Extract attention weights for a specific layer and head from a PyTorch model.

    Args:
        model_path: Path to the PyTorch model directory containing `pytorch_model.bin`.
        output_file: Path to save the extracted attention weights in pickle format (dictionary). This is a pickle
                     file containing a dictionary with 6 entries (weights and biases for key, query, and value matrices
                     Matrix names will be the same as in the original model).
        layer: 1-based index of the encoder layer to extract weights from (1 to 4).
        head: 1-based index of the attention head to extract weights from (1 to 12).
    """
    att_head_dim: int = 26
    start_idx = (head - 1) * att_head_dim
    end_idx = start_idx + att_head_dim

    model_weights: OrderedDict[str, torch.Tensor] = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
    assert isinstance(model_weights, OrderedDict), "Model weights should be an OrderedDict."

    att_matrices: dict = {}

    for att_mat_name in ["key", "query", "value"]:
        weight_key: str = f"bert.encoder.layer.{layer - 1}.attention.self.{att_mat_name}.weight"
        bias_key: str = f"bert.encoder.layer.{layer - 1}.attention.self.{att_mat_name}.bias"
        if weight_key not in model_weights or bias_key not in model_weights:
            print(f"Weight matrix ({weight_key}) or bias vector ({bias_key}) not found in model weights.")
            sys.exit(1)

        mha_weights: torch.Tensor = model_weights[weight_key]  # [312, 26]
        att_weights: np.ndarray = _check_numpy_array(mha_weights[start_idx:end_idx, :].numpy(), expected_shape=(26, 26))

        mha_biases: torch.Tensor = model_weights[bias_key]  # [312,]
        att_biases: np.ndarray = _check_numpy_array(mha_biases[start_idx:end_idx].numpy(), expected_shape=(26,))

        att_matrices.update({weight_key: att_weights, bias_key: att_biases})

    with output_file.open("wb") as f:
        pickle.dump(att_matrices, f)
    print(f"Saved {len(att_matrices)} attention matrices (for [{layer}, {head}] attention head) to: {output_file}")


@cli.command(name="update-attention-weights", help="Update attention weights in a PyTorch model.")
@model_path_option
@weights_file_option
@head_option
@click.option(
    "--output-dir",
    type=click.Path(exists=False, path_type=Path),
    required=True,
    help="Path where a model with updated attention weights will be saved.",
    callback=_validate_output_path,
)
def update_model(model_path: Path, weights_file: Path, head: int, output_dir: Path) -> None:
    """Update model weights for a specific attention head in a PyTorch model.

    This function loads the attention heads for one head, updates the model and saves it in a new directory.

    Args:
        model_path: Path to the PyTorch model directory containing `pytorch_model.bin`.
        weights_file: Path to the pickle file containing attention weights for the specified head. It should contain
                      exactly the same dictionary that was extracted by the `extract-attention-weights` command - keys
                      must be the same, but values of course can be different.
        head: 1-based index of the attention head to update.
        output_dir: Path where the updated model will be saved. The directory will be created if it does not exist.
                    Other files will be copied to this directory (e.g., config.json and vocab.txt).
    """
    att_head_dim: int = 26
    start_idx = (head - 1) * att_head_dim
    end_idx = start_idx + att_head_dim

    model_weights: OrderedDict[str, torch.Tensor] = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
    assert isinstance(model_weights, OrderedDict), "Model weights should be an OrderedDict."
    print(f"Loaded weights of model to be updated from {model_path / 'pytorch_model.bin'}.")

    with weights_file.open("rb") as f:
        att_matrices: Dict = pickle.load(f)
    print(f"Loaded {len(att_matrices)} new attention matrices from {weights_file}.")

    with torch.no_grad():
        key: str
        value: np.ndarray
        for key, value in att_matrices.items():
            assert key in model_weights, f"Key {key} not found in model weights."
            if key.endswith("weight"):
                value = _check_numpy_array(value, expected_shape=(att_head_dim, att_head_dim))
                model_weights[key][start_idx:end_idx, :] = torch.from_numpy(value)
            elif key.endswith("bias"):
                _ = _check_numpy_array(value, expected_shape=(att_head_dim,))
                model_weights[key][start_idx:end_idx] = torch.from_numpy(value)
            else:
                assert False, f"Unexpected key format: {key}. Expected 'weight' or 'bias' suffix."
            print(f"Updated weight tensor {key}")
    print(f"Updated attention weights for head {head} in one of encoder layers.")

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model_weights, output_dir / "pytorch_model.bin")
    print(f"Updated model weights saved to: {output_dir / 'pytorch_model.bin'}")

    for file_ in ["config.json", "vocab.txt"]:
        shutil.copy(model_path / file_, output_dir / file_)
        print(f"Copied {file_} to {output_dir / file_}")


@cli.command(name="add-noise-to-attention-weights", help="Add Gaussian noise to attention weights.")
@weights_file_option
@click.option("--mean", type=float, default=0.0, help="Mean of the Gaussian noise to add. Default: 0.0")
@click.option("--std", type=float, default=0.1, help="Standard deviation of the Gaussian noise to add. Default: 0.1")
def add_noise_to_attention_weights(weights_file: Path, mean: float, std: float) -> None:
    """Add Gaussian noise to attention weights in a pickle file.

    This is used mostly for testing purposes to verify the attention weights are actually updated, and the model with
    the updated weights indeed demonstrate different performance.

    Args:
        weights_file: Path to the pickle file containing attention weights.
        mean: Mean of the Gaussian noise to add. Default: 0.0
        std: Standard deviation of the Gaussian noise to add. Default: 0.1
    """
    print(f"Loading weights from {weights_file}")
    with weights_file.open("rb") as f:
        weights_dict = pickle.load(f)
    print(f"Loaded {len(weights_dict.keys())} weight tensors from {weights_file}.")

    for key, value in weights_dict.items():
        # Generate noise with the same shape as the tensor
        noise = np.random.normal(mean, std, value.shape).astype(value.dtype)

        # Add noise to the tensor
        weights_dict[key] = value + noise
        print(f"Added noise (mean={mean}, std={std}) to {key}, shape={value.shape}")

    # Save the modified weights back to the file
    with weights_file.open("wb") as f:
        pickle.dump(weights_dict, f)
    print(f"Saved weight tensors with added noise to back to {weights_file}.")


if __name__ == "__main__":
    cli()
