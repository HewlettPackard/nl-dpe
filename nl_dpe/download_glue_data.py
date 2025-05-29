"""Script for downloading all GLUE data.

# Copyright (2025) Hewlett Packard Enterprise Development LP
This file is the updated version of the original file from this gist:
    https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e.
This version fixes some issues, add python type annotations and ensures it works with Python 3.10.

# Originally, was developed by W4ngatang GitHub user.
Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized,
or you can download the original data from:
    https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi
and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract'
(see below for an example). You should then rename and place specific files in a folder (see below for an example).

mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
1/30/19: It looks like SentEval is no longer hosting their extracted and tokenized MRPC data, so you'll need to download
         the data from the original source for now.
2/11/19: It looks like SentEval actually *is* hosting the extracted data. Hooray!
"""

import argparse
import io
import os
import sys
import urllib
import urllib.request
import zipfile
from typing import List
from urllib.error import HTTPError

TASKS = [
    "CoLA",
    "SST",
    "MRPC",
    "QQP",
    "STS",
    "MNLI",
    "QNLI",
    "RTE",
    "WNLI",
    "diagnostic",
]
TASK2PATH = {
    "CoLA": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    "SST": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "QQP": "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    "STS": "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
    "MNLI": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    "QNLI": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    "WNLI": "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
    "diagnostic": "https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
    "MRPC": "https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv"
    
}

MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"


def download_and_extract(task: str, data_dir: str) -> None:
    print(f"Downloading and extracting {task}...")
    if task == "MNLI":
        print(
            "\tNote (12/10/20): This script no longer downloads SNLI. "
            "You will need to manually download and format the data to use SNLI."
        )
    data_file = f"{task}.zip"
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


def format_mrpc(data_dir: str, path_to_data: str) -> None:
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        try:
            mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
            mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
            urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
            urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
        except HTTPError:
            print("Error downloading MRPC")
            return
    assert os.path.isfile(mrpc_train_file), f"Train data not found at {mrpc_train_file}"
    assert os.path.isfile(mrpc_test_file), f"Test data not found at {mrpc_test_file}"

    with (
        io.open(mrpc_test_file, encoding="utf-8") as data_fh,
        io.open(os.path.join(mrpc_dir, "test.tsv"), "w", encoding="utf-8") as test_fh,
    ):
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split("\t")
            test_fh.write(f"{idx}\t{id1}\t{id2}\t{s1}\t{s2}\n")
    try:
        urllib.request.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))
    except (KeyError, HTTPError) as err:
        print(
            "\tError downloading standard development IDs for MRPC. "
            f"You will need to manually split your data. The error was {err}."
        )
        return

    dev_ids = []
    with io.open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf-8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split("\t"))

    with (
        io.open(mrpc_train_file, encoding="utf-8") as data_fh,
        io.open(os.path.join(mrpc_dir, "train.tsv"), "w", encoding="utf-8") as train_fh,
        io.open(os.path.join(mrpc_dir, "dev.tsv"), "w", encoding="utf-8") as dev_fh,
    ):
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] in dev_ids:
                dev_fh.write(f"{label}\t{id1}\t{id2}\t{s1}\t{s2}\n")
            else:
                train_fh.write(f"{label}\t{id1}\t{id2}\t{s1}\t{s2}\n")

    print("\tCompleted!")


def download_diagnostic(data_dir: str) -> None:
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")


def get_tasks(task_names: str) -> List[str]:
    task_names_list: List[str] = task_names.split(",")
    if "all" in task_names_list:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names_list:
            assert task_name in TASKS, f"Task {task_name} not found!"
            tasks.append(task_name)
    return tasks


def main(arguments: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory to save data to", type=str, default="glue_data")
    parser.add_argument(
        "--tasks",
        help="tasks to download data for as a comma separated string",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--path_to_mrpc",
        help="path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt",
        type=str,
        default="",
    )
    args: argparse.Namespace = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == "MRPC":
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == "diagnostic":
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
