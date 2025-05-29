# Non-Linear Dot Product Engine (NL-DPE)
This project is based on this [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT) repository.
Some of the changes we introduced:
- New model architectures based on original TinyBERT implmenentation.
- Support for newer Python versions.
- Code formatting.

This repository does not include the training code - only code to run inference.


## Install
```shell
# Clone the NL-DPE repository.
git clone https://github.com/HewlettPackard/nl-dpe
cd ./nl-dpe

# Create Python virtual environment.
virtualenv ./.env --python=3
source ./.env/bin/activate
pip install poetry==2.1.2

# Install project and its dependencies.
poetry install --without=dev

# Make sure project modules are importable.
export PYTHONPATH=$(pwd)
```


## Environment variables
`MLFLOW_TRACKING_URI` \[optional\]: MLflow tracking server URI to load models from (when model URI is `mlflow:{MLFLOW_RUN_ID}`). 
This only works now when running evaluation on a system where the models under test were trained.


## Datasets
Evaluation script uses the GLUE dataset. See the README file from the TinyBERT repository ([Data Augmentation section](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT#data-augmentation)) for more details. From that README file:

> Before running data augmentation of GLUE tasks you should download the [GLUE](https://gluebenchmark.com/tasks) data by running this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory.


## Models
Three models are located in the [models](./models/) directory.

- [wise-tern-584](./models/wise-tern-584/): **SST-2**. 
  - From paper
    - BERT_base_teacher = 93.4, TinyBERT_4 = 92.6, TinyBERT_6 = 93.1
  - Our experiments
    - TinyBERT = 92.77
    - TinyBERT with hidden state averaging (run_name = wise-tern-584, run_id = 70457740292c4ff58fa8cb7ca009f154) = 91.74
- [blushing-dove-984](./models/blushing-dove-984/): **CoLA**
  - From paper
    - BERT_base_teacher = 52.8, TinyBERT_4 = 44.1, TinyBERT_6 = 51.1
  - Our experiments
    - TinyBERT = 40.92
    - TinyBERT with hidden state averaging (run_name = blushing-dove-984, run_id = 37059330c45b4a259ad5fadd812beaa5) = 31.18
- [spiffy-snake-501](./models/spiffy-snake-501/): **MRPC**
  - From paper
    - BERT_base_teacher = 87.5, TinyBERT_4 = 86.4, TinyBERT_6 = 87.3
  - Our experiments
    - TinyBERT = 87.36
    - TinyBERT with hidden state averaging (run_name = spiffy-snake-501, run_id = b3c699617932420da22862c37698724a) = 78.72


## Model evaluation
Look at [nl_dpe/evaluate.py](./nl_dpe/evaluate.py) file:
```shell
python ./nl_dpe/evaluate.py <task_name> <model_uri> <data_dir>
#      <task_name>: One of 'cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli'
#                   Task name must be consistent with the model being evaluated.
#      <model_uri>: Path to a model directory
#      <data_dir>:  Path to the data directory (e.g., ${GLUE_DIR}\\SST-2).
```
