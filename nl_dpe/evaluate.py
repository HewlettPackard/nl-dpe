# coding=utf-8
# Copyright (2025) Hewlett Packard Enterprise Development LP
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
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
# BERT finetuning runner.

# This file has been adopted from the huawei-noah/Pretrained-Language-Model/ repository:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/task_distill.py


import csv
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from nl_dpe.error import NlDpeError
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler("debug_layer_loss.log")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger(__name__)


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid: str, text_a: str, text_b: Optional[str] = None, label: Optional[str] = None) -> None:
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None) -> None:
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


@dataclass
class DefaultParams:
    num_train_epochs: int
    max_seq_length: int


class OutputMode:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Task:
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir: str, output_mode: str, default_params: Optional[DefaultParams]) -> None:
        self.data_dir = data_dir
        self.output_mode = output_mode
        self.default_params = default_params

    def get_train_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def num_labels(self) -> int:
        return len(self.get_labels())

    @classmethod
    def _read_tsv(cls, input_file: str, quotechar: Optional[str] = None) -> List[List[str]]:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines: List[List[str]] = []
            for line in reader:
                lines.append(line)
            return lines


class Sst2Task(Task):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir, OutputMode.CLASSIFICATION, DefaultParams(10, 64))

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "train.tsv")), "train")

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self) -> List[InputExample]:
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "train_aug.tsv")), "aug")

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], dataset_split: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (dataset_split, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ColaTask(Task):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir, OutputMode.CLASSIFICATION, DefaultParams(50, 64))

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "train.tsv")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self):
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MrpcTask(Task):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir, OutputMode.CLASSIFICATION, DefaultParams(20, 128))

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "train.tsv")), "train")

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self) -> List[InputExample]:
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, "train_aug.tsv")), "aug")

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples: List[InputExample] = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def get_task(task_name: str, data_dir: str) -> Task:
    if task_name == "sst-2":
        return Sst2Task(data_dir)
    elif task_name == "cola":
        return ColaTask(data_dir)
    elif task_name == "mrpc":
        return MrpcTask(data_dir)
    else:
        raise NlDpeError(f"Unknown task: '{task_name}'.")


def get_local_path(model_uri: str) -> str:
    if model_uri.startswith("mlflow:"):
        from nl_dpe.contrib import mlflow_toolkit

        return mlflow_toolkit.get_artifact_path(model_uri[7:]).as_posix()

    return model_uri


def convert_examples_to_features(
    examples: List[InputExample], tokenizer: BertTokenizer, task: Task
) -> List[InputFeatures]:
    """Loads a data file into a list of `InputBatch`s."""
    assert task.default_params is not None, "Task default parameters must be defined."

    label_map: dict[str, int] = {label: i for i, label in enumerate(task.get_labels())}
    max_seq_length: int = task.default_params.max_seq_length

    features: List[InputFeatures] = []

    ex_index: int
    example: InputExample
    for ex_index, example in enumerate(examples):
        assert example.label is not None, "Example label must be defined."
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if task.output_mode == OutputMode.CLASSIFICATION:
            label_id: int = label_map[example.label]
        elif task.output_mode == OutputMode.REGRESSION:
            label_id = int(example.label)
        else:
            raise KeyError(task.output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                seq_length=seq_length,
            )
        )
    return features


def get_tensor_data(output_mode: str, features: List[InputFeatures]) -> Tuple[TensorDataset, torch.Tensor]:
    if output_mode == OutputMode.CLASSIFICATION:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == OutputMode.REGRESSION:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        raise ValueError(f"Unknown output mode: {output_mode}.")

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths)

    return tensor_data, all_label_ids


def _truncate_seq_pair(tokens_a: List, tokens_b: List, max_length: int) -> None:
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def evaluate(task_name: str, model_uri: str, data_dir: str, eval_batch_size: int = 32) -> None:
    task_name = task_name.lower()
    task = get_task(task_name, data_dir)

    model_path: str = get_local_path(model_uri)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

    eval_examples: List[InputExample] = task.get_dev_examples()
    eval_features: List[InputFeatures] = convert_examples_to_features(eval_examples, tokenizer, task)
    eval_data, eval_labels = get_tensor_data(task.output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model = TinyBertForSequenceClassification.from_pretrained(model_path, num_labels=task.num_labels())
    model.to(device)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    result: Dict = do_eval(model, task_name, eval_dataloader, device, task.output_mode, eval_labels, task.num_labels())

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))


def do_eval(
    model: TinyBertForSequenceClassification,
    task_name: str,
    eval_dataloader: DataLoader,
    device: torch.device,
    output_mode: str,
    eval_labels: torch.Tensor,
    num_labels: int,
) -> Dict:
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == OutputMode.CLASSIFICATION:
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == OutputMode.REGRESSION:
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        else:
            raise ValueError(f"Unknown output mode: {output_mode}.")

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == OutputMode.CLASSIFICATION:
        preds = np.argmax(preds, axis=1)
    elif output_mode == OutputMode.REGRESSION:
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result["eval_loss"] = eval_loss

    return result


def compute_metrics(task_name: str, preds, labels) -> Dict:
    assert len(preds) == len(labels)

    def simple_accuracy(preds, labels) -> float:
        return (preds == labels).mean()

    def acc_and_f1(preds, labels) -> Dict:
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels) -> Dict:
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        # FIXME sergey: check `corr` below is computed correctly.
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


if __name__ == "__main__":
    try:
        args: list[str] = sys.argv[1:]
        if (len(args) != 3):
            print("Usage: python nl_dpe/evaluate.py <task_name> <model_uri> <data_dir>")
            print("       <task_name>: One of 'cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli'.")
            print("                    Task name must be consistent with the model being evaluated.")
            print("       <model_uri>: Path to a model directory.")
            print("       <data_dir>:  Path to the data directory (e.g., ${GLUE_DIR}\\SST-2).")
            sys.exit(1)

        task_name: str = args[0].lower()
        model_uri: str = args[1]
        data_dir: str = args[2]

        evaluate(task_name, model_uri, data_dir)
    except NlDpeError as err:
        print(str(err))
        exit(1)
