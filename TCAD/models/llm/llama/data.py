import csv
import copy

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path, header_rows=0):
        self.file_path = file_path
        self.header_rows = header_rows
        self.index_map = []
        self._build_index()

    def _build_index(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            while True:
                pos = file.tell()
                line = file.readline()
                if not line:
                    break
                self.index_map.append(pos)

    def __len__(self):
        return len(self.index_map) - self.header_rows

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        index = idx + self.header_rows

        with open(self.file_path, 'r', encoding='utf-8') as file:
            file.seek(self.index_map[index])
            line = file.readline().strip()
            pair = line.split("\",\"")
        
        return {"question": pair[0][1:], "answer": pair[1][:-1]}


class CustomCollateFn:
    def __init__(self, tokenizer, prompt=""):
        self.tokenizer = tokenizer
        self.prompt = prompt

    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []

        max_length = 0
        for pair in batch:
            question = pair['question']
            answer = pair['answer']

            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                 + f"{self.prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                 + f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
                 + f"{answer}<|eot_id|>\n\n"

            text = text.encode("ascii", "ignore").decode("ascii")

            token_ids = self.tokenizer.encode(text)

            length = len(token_ids)
            max_length = length if length > max_length else max_length

            mask = [1] * length
            label = copy.copy(token_ids)

            input_ids.append(token_ids)
            attention_mask.append(mask)
            labels.append(label)

        for i in range(len(input_ids)):
            padding = max_length - len(input_ids[i])
            input_ids[i].extend([self.tokenizer.special_tokens['<|eot_id|>']] * padding)
            attention_mask[i].extend([0] * padding)
            labels[i].extend([-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
