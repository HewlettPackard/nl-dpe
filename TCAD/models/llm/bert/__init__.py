import os
import math

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    get_scheduler,
    default_data_collator,
)

from .bert import BertForQuestionAnswering, BertForSequenceClassification


transformers.logging.set_verbosity_error()


def preprocess_training_examples(examples, tokenizer, max_length, stride):

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer, max_length, stride):

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def get_bert_model_for_squad():
    model_checkpoint = 'csarron/bert-base-uncased-squad-v1'

    config = AutoConfig.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    pretrained_model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model = BertForQuestionAnswering(config=config)
    model.load_state_dict(pretrained_model.state_dict(), strict=True)

    raw_datasets = load_dataset('squad')

    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 384, "stride": 128}
    )
    train_dataset.set_format("torch")

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 384, "stride": 128}
    )
    eval_dataset = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataset.set_format("torch")

    metric = evaluate.load('evaluate/metrics/squad/squad.py')

    return model, raw_datasets, validation_dataset, train_dataset, eval_dataset, tokenizer, default_data_collator, metric


def get_bert_model_for_glue(model_type, task_name, max_length=128, download_pretrained=False):
    ##############################################################################################
    # Dataset, tokenizer and model 

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset("nyu-mll/glue", task_name)

    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    task_to_weights = {
        "bert_base": {
            "cola": ["weights/cola_bert_base", "Ruizhou/bert-base-uncased-finetuned-cola"],
            "mnli": ["weights/mnli_bert_base", "yoshitomo-matsubara/bert-base-uncased-mnli"],
            "mrpc": ["weights/mrpc_bert_base", "google-bert/bert-base-cased-finetuned-mrpc"],
            "qnli": ["weights/qnli_bert_base", "textattack/bert-base-uncased-QNLI"],
            "qqp":  ["weights/qqp_bert_base", "textattack/bert-base-uncased-QQP"],
            "rte":  ["weights/rte_bert_base", "howey/bert-base-uncased-rte"],
            "sst2": ["weights/sst2_bert_base", "textattack/bert-base-uncased-SST-2"],
            "stsb": ["weights/stsb_bert_base", "textattack/bert-base-uncased-STS-B"],
            "wnli": ["weights/wnli_bert_base", "textattack/bert-base-uncased-WNLI"],
        },
        "bert_tiny": {
            "mnli": ["weights/mnli_bert_tiny", "M-FAC/bert-tiny-finetuned-mnli"],
            "mrpc": ["weights/mrpc_bert_tiny", "M-FAC/bert-tiny-finetuned-mrpc"],
            "qnli": ["weights/qnli_bert_tiny", "M-FAC/bert-tiny-finetuned-qnli"],
            "qqp":  ["weights/qqp_bert_tiny", "M-FAC/bert-tiny-finetuned-qqp"],
            "sst2": ["weights/sst2_bert_tiny", "M-FAC/bert-tiny-finetuned-sst2"],
            "stsb": ["weights/stsb_bert_tiny", "M-FAC/bert-tiny-finetuned-stsb"],
        },
    }

    # Load configuration, pretrained model and tokenizer
    idx = 1 if download_pretrained else 0
    config = AutoConfig.from_pretrained(
        task_to_weights[model_type][task_name][idx],
        num_labels=num_labels,
        finetuning_task=task_name,
        _attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(task_to_weights[model_type][task_name][idx], use_fast=False)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        task_to_weights[model_type][task_name][idx],
        config=config,
    )
    model = BertForSequenceClassification(config=config)
    model.load_state_dict(pretrained_model.state_dict())

    ##############################################################################################
    # Preprocessing

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp":  ("question1", "question2"),
        "rte":  ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id and not is_regression):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            print(f"The configuration of the model provided the following label correspondence: {label_name_to_id}. Using it!")
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(f"Your model seems to have been trained with labels, but they don't match the dataset: model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}.\nIgnoring the model labels as a result.",)

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    ##############################################################################################
    # Metric

    metric = evaluate.load("glue", task_name)

    return model, train_dataset, eval_dataset, tokenizer, data_collator, metric, is_regression
