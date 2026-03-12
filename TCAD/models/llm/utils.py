###
# Copyright (2026) Hewlett Packard Enterprise Development LP
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

import time
from enum import Enum
from tqdm.auto import tqdm
import numpy as np
import collections
import math

import torch


# type of meter
class Summary(Enum):

    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


# computes and stores the average and current value
class AverageMeter(object):

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


# show progress
class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


#################################################################################################


def compute_squad_metrics(start_logits, end_logits, features, examples, metric, n_best=20, max_answer_length=30):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def evaluate_squad(eval_dataloader, model, raw_datasets, validation_dataset, device, metric):
    model.eval()
    start_logits = []
    end_logits = []
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_squad_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"], metric)
    print(metrics)
    return torch.tensor(metrics.get('f1'), device=device) 


def train_squad(train_dataloader, model, optimizer, lr_scheduler, weight_norm, epoch, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    lr = AverageMeter('lr', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, lr, losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for step, batch in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        lr.update(optimizer.param_groups[0]['lr'])

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        # regularization
        dpe_weights = []
        for name, param in model.module.named_parameters():
            if 'dpe_weight' in name:
                dpe_weights.append(param)
        max_weight_norm = max(param.abs().max() for param in dpe_weights)
        loss = loss + weight_norm * max_weight_norm

        losses.update(loss.item())

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0:
            progress.display(step + 1)


#################################################################################################


def evaluate_glue(eval_dataloader, model, device, is_regression, metric, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(eval_dataloader), [batch_time, data_time], prefix='Test: ')

    model.eval()

    end = time.time()
    for step, batch in enumerate(eval_dataloader):
        data_time.update(time.time() - end)

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = (predictions, batch["labels"])
        metric.add_batch(predictions=predictions, references=references,)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0:
            progress.display(step + 1)

    eval_metric = metric.compute()
    print(eval_metric)
    return torch.tensor(eval_metric.get('matthews_correlation', eval_metric.get('pearson', eval_metric.get('accuracy'))), device=device)


def train_glue(train_dataloader, model, optimizer, lr_scheduler, weight_norm, epoch, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    lr = AverageMeter('lr', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, lr, losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for step, batch in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        lr.update(optimizer.param_groups[0]['lr'])

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        # regularization
        dpe_weights = []
        for name, param in model.module.named_parameters():
            if 'dpe_weight' in name:
                dpe_weights.append(param)
        max_weight_norm = max(param.abs().max() for param in dpe_weights)
        loss = loss + weight_norm * max_weight_norm

        losses.update(loss.item())

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0:
            progress.display(step + 1)


#################################################################################################


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def evaluate_ptb(test_list, model, device, batch_size, parts=None):
    model.eval()

    if parts is None:
        iters = (len(test_list) + batch_size - 1) // batch_size
        losses = []
        with torch.no_grad():
            for i in tqdm(range(iters)):
                input_chunk = torch.stack(test_list[i*batch_size:(i+1)*batch_size]).to(device)
                target_chunk = input_chunk.clone()
                loss, logits = model(input_chunk, labels=target_chunk)
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        perplexity = math.exp(mean_loss)
        print(f"Perplexity on PTB (test split): {perplexity:.2f}")

        return torch.tensor(perplexity, device=device)
    elif parts == "head":
        iters = (len(test_list) + batch_size - 1) // batch_size
        results = []
        with torch.no_grad():
            for i in tqdm(range(iters)):
                input_chunk = torch.stack(test_list[i*batch_size:(i+1)*batch_size]).to(device)
                target_chunk = input_chunk.clone()
                result = model(input_chunk, labels=target_chunk)
                results.append(result)
        results_cpu = move_to_device(results, "cpu")
        torch.save(results_cpu, "results.pt")
        exit(0)
    elif parts == "tail":
        iters = (len(test_list) + batch_size - 1) // batch_size
        inputs_cpu = torch.load("results.pt")
        losses = []
        with torch.no_grad():
            for i in tqdm(range(iters)):
                target_chunk = torch.stack(test_list[i*batch_size:(i+1)*batch_size]).to(device)
                inputs = move_to_device(inputs_cpu[i], device)
                loss, logits = model(inputs[0], labels=target_chunk)
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        perplexity = math.exp(mean_loss)
        print(f"Perplexity on PTB (test split): {perplexity:.2f}")
        exit(0)
    else:
        iters = (len(test_list) + batch_size - 1) // batch_size
        inputs_cpu = torch.load("results.pt")
        results = []
        with torch.no_grad():
            for i in tqdm(range(iters)):
                inputs = move_to_device(inputs_cpu[i], device)
                inputs = move_to_device(inputs, device)
                result = model(*inputs)
                results.append(result)
        results_cpu = move_to_device(results, "cpu")
        torch.save(results_cpu, "results.pt")
        exit(0)


def train_ptb(test_list, model, device, batch_size):
    raise NotImplementedError("Training on PTB not implemented yet")
