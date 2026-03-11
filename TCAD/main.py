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

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HOME'] = os.path.join(os.environ['RACE_IT_PATH'], 'hf_download')
os.environ['TORCH_HOME'] = os.path.join(os.environ['RACE_IT_PATH'], 'torch_download')

import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from models.cnn.utils import train_cnn, validate_cnn
from models.cnn.cifar10 import get_cifar10_model, prepare_cifar10_data
from models.cnn.mnist import get_mnist_model, prepare_mnist_data
from models.cnn.imagenet import get_imagenet_model, prepare_imagenet_data

from models.llm.utils import train_glue, evaluate_glue, train_squad, evaluate_squad, train_ptb, evaluate_ptb
from models.llm.bert import get_bert_model_for_squad, get_bert_model_for_glue

from models.utils.fuse import convert_model, print_params
from models.utils.quant import quantizer_switcher

from tree.tree import train_trees
from tree.finetune import finetune
from mapping.mapping import mapping


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="Noise Aware Finetuning")

    # common args
    parser.add_argument("--model_type", type=str, help="Model type name.",)
    parser.add_argument("--task_name", type=str, default=None, help="The name of dataset.",)
    parser.add_argument("--download_pretrained", action='store_true', help="Download weights from the web.",)
    parser.add_argument("--load_pretrained", action='store_true', help="Load pretrained weights of the original model.",)
    parser.add_argument("--load", action='store_true', help="Load model from load_dir.",)
    parser.add_argument("--load_dir", type=str, default=None, help="Directory name to load model.",)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory name to store model.",)

    # training args
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_norm", type=float, default=1e-3, help="Coefficient for weight regularization.",)
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.",)
    parser.add_argument('--evaluate', action='store_true', help='Only Evaluate without training',)
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers (default: 8)',)
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency (default: 10)',)
    parser.add_argument("--gpu", nargs='+', type=int, default=None, help="Which GPU to use.",)
    parser.add_argument("--port", type=str, default=None, help="If specified, overwrite auto generated port for multiple GPU.",)
    parser.add_argument("--seed", type=int, default=None, help="Seed to reproduce the same result.",)

    # difference between GCE and DT:
    # 1. ADCs: GCE still uses ADCs, DT uses ACAM for ADC. So GCE do not implement the identity function
    # 2. activations: same for GCE and DT
    # 3. matmul: GCE implements bivariate functions, DT decompose to log-add-exp sequences
    # 4. softmax: exp and log are the same for GCE and DT, but the division is different, similar to matmul
    parser.add_argument('--use_gce', action='store_true', help='If specified, use GCE mapping instead of DTs.',)

    # quantization
    parser.add_argument('--quant', action='store_true', help='Quantize model.',)
    parser.add_argument("--bitwidth", type=int, default=8, help="Quantization bit width.",)
    parser.add_argument('--calibrate', action='store_true', help='Calibrate PTQ.',)
    parser.add_argument('--moving_min_max', action='store_true', help='Calculate moving min and max for input, instead of the min and max of the whole dataset.',)

    # noise
    parser.add_argument('--noise', action='store_true', help='Add noise to DPE.',)
    parser.add_argument('--acam_noise', action='store_true', help='Add noise to ACAM.',)
    parser.add_argument("--g_min", type=float, default=0.0, help="Minimum conductance in DPE.",)
    parser.add_argument("--g_max", type=float, default=150.0, help="Maximum conductance in DPE.",)
    parser.add_argument("--acam_g_min", type=float, default=0.0, help="Minimum conductance in ACAM.",)
    parser.add_argument("--acam_g_max", type=float, default=150.0, help="Maximum conductance in ACAM.",)
    parser.add_argument("--std_scale", type=float, default=1.0, help="Scaling factor on std of DPE/ACAM noises.",)
    parser.add_argument("--stuck_fault", action='store_true', help='Has stuck-at fault.',)
    parser.add_argument("--stuck_mean", type=float, default=0.04, help="Mean of stuck-at fault.",)

    # ACAM
    parser.add_argument('--lut', action='store_true', help='Use ACAM for activation.',)
    parser.add_argument("--acam_rows", nargs='+', type=int, default=None, help="Equivalent to the max size in max_leaf_nodes.",)
    parser.add_argument('--encode', action='store_true', help='Use gray encoding for output.',)
    parser.add_argument('--acam_finetuned', action='store_true', help='Use finetuned trees.',)

    # Digital slicing for DPE
    parser.add_argument('--digital_slicing', action='store_true', help='Use digital slicing for weights.',)
    parser.add_argument("--bits_per_cell", type=int, default=2, help="How many digital bits can a ReRAM cell store.",)

    parser.add_argument('--print_params', action='store_true', help='Print out the parameters for ACAM DT training or mapping.',)
    # DT
    parser.add_argument("--tree_dir", type=str, default="trees", help="Directory name to store DT parameters.",)
    parser.add_argument('--train_trees', action='store_true', help='Train decision trees',)
    parser.add_argument('--max_leaf_nodes', nargs='+', type=int, default=None, help="Max leaf nodes for each output bit, from MSB to LSB.",)
    # Finetune DT
    parser.add_argument('--finetune', action='store_true', help='Finetune the DTs',)
    parser.add_argument("--finetune_learning_rate", type=float, default=0.0001, help="Learning rate.",)
    parser.add_argument('--finetune_momentum', type=float, default=0, help='momentum')
    parser.add_argument('--finetune_weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Total number of training epochs to perform.",)
    parser.add_argument("--finetune_batch_size", type=int, default=5000, help="atch size for finetuning (default: 5000 means no batch since training data size is 5000).",)
    parser.add_argument("--finetune_only", type=str, default=None, help="Only finetune a specific function.",)
    parser.add_argument("--finetune_infinity", type=float, default=10, help="A very large number that can not be present in trees.",)
    # mapping
    parser.add_argument("--mapping_dir", type=str, default="mappings", help="Directory name to store mapping parameters.",)
    parser.add_argument('--mapping', action='store_true', help='Perform mapping',)

    # Export for Timeloop+Accelergy
    parser.add_argument('--export_workload', action='store_true', help='Export each layer into a format that Timeloop can understand.',)

    args = parser.parse_args()

    return args


def run(args, device):
    # Load data and build the model
    if args.task_name == "cifar10":
        pretrained_model = get_cifar10_model(args.model_type, args.load_pretrained, dropout=0.0)
        train_dataset, val_dataset = prepare_cifar10_data(args.batch_size, args.workers, use_imagenet_input_size=args.model_type=="efficientnet_v2_s")
    elif args.task_name == "mnist":
        pretrained_model = get_mnist_model(args.model_type, args.load_pretrained)
        train_dataset, val_dataset = prepare_mnist_data(args.batch_size, args.workers)
    elif args.task_name == "imagenet":
        pretrained_model = get_imagenet_model(args.model_type, args.load_pretrained, download_pretrained=args.download_pretrained)
        train_dataset, val_dataset = prepare_imagenet_data(args.batch_size, args.workers)
    elif args.task_name == "squad":
        pretrained_model, raw_datasets, validation_dataset, train_dataset, val_dataset, tokenizer, data_collator, metric = get_bert_model_for_squad()
        # put these in args, the training thread will use these
        args.raw_datasets = raw_datasets
        args.validation_dataset = validation_dataset
        args.data_collator = data_collator
        args.metric = metric
    elif args.model_type in ["bert_base", "bert_tiny"]:
        pretrained_model, train_dataset, val_dataset, tokenizer, data_collator, metric, is_regression = get_bert_model_for_glue(args.model_type, args.task_name, download_pretrained=args.download_pretrained)
        # put these in args, the training thread will use these
        args.data_collator = data_collator
        args.metric = metric
        args.is_regression = is_regression
    else:
        print("unrecognized dataset")
        exit(0)

    if args.use_gce:
        acam_dir = args.mapping_dir
    else:
        acam_dir = args.tree_dir
    model = convert_model(pretrained_model, noise=args.noise, g_min=args.g_min, g_max=args.g_max, use_gce=args.use_gce, quant=args.quant, actbitW=args.bitwidth,
                          acam_noise=args.acam_noise, acam_g_min=args.acam_g_min, acam_g_max=args.acam_g_max, std_scale=args.std_scale,
                          stuck_fault=args.stuck_fault, stuck_mean=args.stuck_mean,
                          lut=args.lut, acam_rows=args.acam_rows, encode=args.encode, acam_dir=acam_dir, acam_finetuned=args.acam_finetuned, trainable=False,
                          digital_slicing=args.digital_slicing, bits_per_cell=args.bits_per_cell,
                          export_workload = args.export_workload)

    if args.load:
        load_dir = os.path.join(args.load_dir, 'model_weights.pth')
        print("===== Loading weights from ", load_dir)
        state_dict = torch.load(load_dir)
        if args.lut:
            keys_to_delete = [key for key in state_dict if "acam_weight" in key]
            for key in keys_to_delete:
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()

    ##############################################################################################
    # PTQ

    if args.calibrate:
        assert args.quant or args.digital_slicing, "Calibrating without quantizing the model"
        model.to(device)
        if args.task_name == "squad":
            val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=data_collator)
            quantizer_switcher(model, True, args.moving_min_max)
            evaluate_squad(val_loader, model, raw_datasets, validation_dataset, device, metric)
        elif args.model_type in ["bert_base", "bert_tiny"]:
            val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=data_collator)
            quantizer_switcher(model, True, args.moving_min_max)
            evaluate_glue(val_loader, model, device, is_regression, metric, args.print_freq)
        else:
            val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            quantizer_switcher(model, True, args.moving_min_max)
            validate_cnn(val_loader, model, criterion, device, args.print_freq)

        print("===== Saving weights to ", os.path.join(args.save_dir, 'model_weights.pth'))
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_weights.pth'))
        return

    ##############################################################################################
    # Evaluate

    if args.evaluate:
        model.to(device)
        if args.task_name == "squad":
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=data_collator)
            evaluate_squad(val_loader, model, raw_datasets, validation_dataset, device, metric)
        elif args.model_type in ["bert_base", "bert_tiny"]:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=data_collator)
            evaluate_glue(val_loader, model, device, is_regression, metric, args.print_freq)
        else:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            validate_cnn(val_loader, model, criterion, device, args.print_freq)

        if args.print_params:
            if args.use_gce:
                os.makedirs(args.mapping_dir, exist_ok=True)
                params_file = os.path.join(args.mapping_dir, 'params.txt')
            else:
                os.makedirs(args.tree_dir, exist_ok=True)
                params_file = os.path.join(args.tree_dir, 'params.txt')
            print("===== Saving parameters to ", params_file)
            print_params(model, params_file)
        return

    ##############################################################################################
    # Train

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    world_size = len(args.gpu)
    mp.spawn(
        train_process,
        args=(world_size, model, optimizer, criterion, train_dataset, val_dataset, args),
        nprocs=world_size,
        join=True
    )


def train_process(rank, world_size, model, optimizer, criterion, train_dataset, val_dataset, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '123'+str(args.gpu[0])+str(args.gpu[0]) if args.port is None else args.port
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    torch.cuda.set_device(rank)
    init_process_group("nccl", rank=rank, world_size=world_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset)
    if args.model_type in ["bert_base", "bert_tiny"]:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=train_sampler, num_workers=args.workers, collate_fn=args.data_collator)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=val_sampler, num_workers=args.workers, collate_fn=args.data_collator)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=train_sampler, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=val_sampler, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)

    model = DDP(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)

    quantizer_switcher(model.module, False)

    for epoch in range(0, args.num_train_epochs):
        train_loader.sampler.set_epoch(epoch)
        if args.task_name == "squad":
            train_squad(train_loader, model, optimizer, scheduler, args.weight_norm, epoch, rank, args.print_freq)
            acc1 = evaluate_squad(val_loader, model, args.raw_datasets, args.validation_dataset, rank, args.metric)
        elif args.model_type in ["bert_base", "bert_tiny"]:
            train_glue(train_loader, model, optimizer, scheduler, args.weight_norm, epoch, rank, args.print_freq)
            acc1 = evaluate_glue(val_loader, model, rank, args.is_regression, args.metric, args.print_freq)
        else:
            train_cnn(train_loader, model, criterion, optimizer, args.weight_norm, epoch, rank, args.print_freq)
            acc1 = validate_cnn(val_loader, model, criterion, rank, args.print_freq)

        acc1_tensor = acc1.clone().detach().to(rank)
        torch.distributed.all_reduce(acc1_tensor, op=torch.distributed.ReduceOp.SUM)
        acc1 = acc1_tensor.item() / world_size

        scheduler.step()

        if rank == 0:
            print("Test accuracy:", acc1)

    if rank == 0:
        print("===== Saving weights to ", os.path.join(args.save_dir, 'model_weights.pth'))
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'model_weights.pth'))

    destroy_process_group()


def main():
    args = parse_args()

    args.tree_dir = os.path.join(os.environ['RACE_IT_PATH'], args.task_name, args.model_type, args.tree_dir)
    args.mapping_dir = os.path.join(os.environ['RACE_IT_PATH'], args.task_name, args.model_type, args.mapping_dir)
    if args.train_trees:
        train_trees(args.tree_dir, args.max_leaf_nodes, args.encode, args.seed)
        return
    elif args.mapping:
        mapping(args.mapping_dir, args.encode, args.seed)
        return
    elif args.finetune:
        finetune(
            tree_dir=args.tree_dir,
            only=args.finetune_only,
            infinity=args.finetune_infinity,
            acam_g_min=args.acam_g_min,
            acam_g_max=args.acam_g_max,
            std_scale=args.std_scale,
            stuck_fault=args.stuck_fault,
            stuck_mean=args.stuck_mean,
            acam_rows=args.acam_rows,
            encode=args.encode,
            batch_size=args.finetune_batch_size,
            lr=args.finetune_learning_rate,
            momentum=args.finetune_momentum,
            weight_decay=args.finetune_weight_decay,
            epochs=args.finetune_epochs,
        )
        return

    if args.load_dir is None:
        args.load_dir = os.path.join(os.environ['RACE_IT_PATH'], args.task_name, args.model_type, "checkpoints")
    else:
        args.load_dir = os.path.join(os.environ['RACE_IT_PATH'], args.task_name, args.model_type, "checkpoints_" + args.load_dir)

    if args.save_dir is None:
        args.save_dir = os.path.join(os.environ['RACE_IT_PATH'], args.task_name, args.model_type, "checkpoints")
    else:
        args.save_dir = os.path.join(os.environ['RACE_IT_PATH'], args.task_name, args.model_type, "checkpoints_" + args.save_dir)

    os.makedirs(args.load_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.gpu is None:
        args.gpu = [0, 1, 2, 3, 4, 5, 6, 7]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
    device = torch.device("cuda") # only used for non-training, which only uses one GPU

    if args.seed is not None:
        torch.manual_seed(args.seed)

    run(args, device)


if __name__ == "__main__":
    main()