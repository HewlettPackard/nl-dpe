# Finetuning CNN/LLM for NL-DPE

This project provides an implementation of Noise-Aware Fine-Tuning (NAF) for the NL-DPE (Nonlinear Dot-Product Engine). 
It supports both CNN and Transformer models.

For CNN models, the supported datasets include MNIST, CIFAR-10, and ImageNet. All ImageNet models use the 
implementations provided by torchvision and directly load the pretrained weights from torchvision.

For MNIST and CIFAR-10 models, the networks need to be trained from scratch. Due to GitHub's file size 
limitations, the pretrained weights for these models are not included in this repository. However, since 
these models are relatively small, training them from scratch should be straightforward.

This project also supports two Transformer models: BERT-base and BERT-tiny. Their pretrained weights and 
the GLUE dataset are obtained from Hugging Face.

## Setup environment

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

Assume `ROOT` refers to the current directory of this README file. 
All the following commands should be executed inside `ROOT`.

First, make sure there is a `data` directory under `ROOT`:
```
mkdir -p data
```

Set the environment variable `RACE_IT_PATH` to point to this directory. 
All generated data, downloaded datasets, and pretrained weights will be stored in `data`:
```
export RACE_IT_PATH=$(pwd)/data
```

If you plan to test on the ImageNet dataset, place the ImageNet data under `data/imagenet_dataset`.

Finally, the commands for running each model and dataset are listed in: `ROOT/doc`.