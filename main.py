import logging
from os.path import dirname

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import algorithms
from train_tools import *
from utils import *

import numpy as np
import argparse
import warnings
import random
import pprint
import os
import torch.nn.functional as F


def tinyimagenet_transforms(x):
    return F.pad(x.unsqueeze(0), (8, 8, 8, 8), mode="reflect").data.squeeze()


def pad_and_squeeze(x):
    return F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").data.squeeze()


warnings.filterwarnings("ignore")

# Set torch base logging.info precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fed_contrast": algorithms.fed_contrast.Server,
    "fed_classifier": algorithms.fed_classifier.Server
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    np.random.seed(opt.train_setups.seed)
    random.seed(opt.train_setups.seed)

    # Distribute the data to clients
    data_distributed = data_distributer(**args.data_setups)

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
    )

    # Optimization setups
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    algo_params = args.train_setups.algo.params
    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        **args.train_setups.scenario,
    )

    return server


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, file_name):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)

    # Conduct FL
    server.run()

    # Save the final global model
    torch.save(server.model.state_dict(), f'{log_file_path}_model.pth')
    xx = np.asarray(server.server_results['test_accuracy'])
    top_idx = xx.argsort()[-1:-6:-1]
    TopBER = xx[top_idx]
    logging.info(f'top_idx:{top_idx}')
    logging.info(f'test acc :{TopBER}')
    np.save(file_name + '.npz', TopBER)
    # Upload model to wandb


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedavg_if01.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--data_alpha", type=str, choices=['05', '10'], default='05')
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--lam1", type=int)
parser.add_argument("--lam2", type=int)
parser.add_argument("--tau", type=float)
parser.add_argument("--epoch_classifier", type=int)
parser.add_argument("--dyn_alpha", type=float)
args = parser.parse_args()


#######################################################################################


def set_log():
    logger = logging.getLogger()
    pid = os.getgid()
    path_basename = os.path.basename(args.config_path)
    log_file = os.path.join('log', opt.data_setups.dataset_name, f'{args.partition_method}')
    if args.partition_method is not None:
        log_file = os.path.join(log_file, str(path_basename).split('_if')[0])
    # strftime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f'_{pid}'
    # double 调参
    if args.lam1 is not None:
        log_file = log_file + f'lam_[{args.lam1}][{args.lam2}]'
    # if args.lam2 is not None:
    #     log_file = log_file + f'second_{args.lam2}_'
    if args.tau is not None:
        log_file = log_file + f'tau[{args.tau}]'
    if args.sample_ratio is not None and str(args.sample_ratio) == '1.0':
        log_file = log_file + f'full'
    if args.epoch_classifier is not None and args.epoch_classifier != 10:
        log_file = log_file + f'EC_[{args.epoch_classifier}]'
    if args.dyn_alpha is not None:
        log_file = log_file + f'alpha_[{args.dyn_alpha}]'
    if args.seed is not None:
        log_file = log_file + f'seed[{args.seed}]'
    log_file = log_file + f'_alpha[{args.data_alpha}]'
    print(f'str(args.sample_ratio): {str(args.sample_ratio)}')
    file_name = log_file + f"_{pid}.log"
    os.makedirs(dirname(file_name), exist_ok=True)
    print(f'file_name create:{file_name}')
    # dir_name = log_path(args_)
    fhandler = logging.FileHandler(filename=file_name, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)  # debug 使用 只在终端显现
    return log_file, file_name


if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt
    # Overwrite config by parsed arguments
    opt = config_overwriter(opt, args)

    log_file_path, file_name = set_log()
    logging.getLogger().setLevel(logging.INFO)  # debug 使用 只在终端显现
    # logging.info configuration dictionary pretty
    logging.info("")
    logging.info("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    logging.info(pp.pformat(opt))
    logging.info("=" * 120)

    # Execute experiment
    main(opt, file_name)

    #   # debug 使用 只在终端显现
    # # logging.info configuration dictionary pretty
    # logging.info("")
    # logging.info("=" * 50 + " Configuration " + "=" * 50)
    # pp = pprint.PrettyPrinter(compact=True)
    # logging.info(pp.pformat(opt))
    # logging.info("=" * 120)
    # print('a')
    # # Execute experiment
    # main(opt)
