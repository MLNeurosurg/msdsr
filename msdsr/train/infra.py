"""Infrastructure utils for training.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
import argparse
from shutil import copy2
from datetime import datetime
from functools import partial
from typing import Dict, Tuple

import uuid

import pytorch_lightning as pl


def parse_args():
    """Extract config file path for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


def get_exp_name(cf):
    """Generate experiment name with a hash, time, and comments in config."""
    time = datetime.now().strftime("%b%d-%H-%M-%S")
    return "-".join([uuid.uuid4().hex[:8], time, cf["infra"]["comment"]])


def setup_ddp_exp_name(exp_name: str):
    """Setup DDP experiment name."""
    if pl.utilities.rank_zero.rank_zero_only.rank != 0:
        return os.path.join(exp_name, "high_rank")
    else:
        return exp_name


def setup_output_dirs(cf: Dict, get_exp_name: callable,
                      cmt_append: str) -> Tuple[str, str, callable]:
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_name = setup_ddp_exp_name(cf["infra"]["exp_name"])
    exp_root = os.path.join(log_root, exp_name, instance_name)

    model_dir = os.path.join(exp_root, 'models')
    config_dir = os.path.join(exp_root, 'config')

    for dir_name in [model_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    return exp_root, model_dir, partial(copy2, dst=config_dir)


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_path"].split("/")[0]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name, "evals",
                            eval_instance_name)

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    for dir_name in [pred_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    return exp_root, pred_dir, partial(copy2, dst=config_dir)


def scale_lr(config):
    """Scale learning rate based off config."""
    bs = config["loader"]["direct_params"]["train"]["batch_size"]
    base_lr = config["training"]["base_learning_rate"]
    accum_grad_batches = config["training"].get("accumulate_grad_batches", 1)
    ngpu = config["infra"]["SLURM_GPUS_ON_NODE"] * config["infra"][
        "SLURM_JOB_NUM_NODES"]
    if config["training"]["scale_lr"]:
        lr = accum_grad_batches * ngpu * bs * base_lr
        logging.info(f"Computed learning rate to {lr} = " +
                     f"{accum_grad_batches} (accumulate_grad_batches) * " +
                     f"{ngpu} (num_gpus) * " + f"{bs} (batchsize) * " +
                     f"{base_lr} (base_lr)")
        return lr
    else:
        logging.info(f"Base learning rate = {base_lr:.2e}")
        return base_lr


def config_loggers(exp_root):
    """Config logger for the experiments

    Sets string format and where to save.
    """

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(exp_root, 'train.log')),
                            logging.StreamHandler()
                        ],
                        force=True)
    logging.info("Exp root {}".format(exp_root))

    formatter = logging.Formatter(logging_format_str, datefmt="%H:%M:%S")
    logger = logging.getLogger("pytorch_lightning.core")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(exp_root, 'train.log')))
    for h in logger.handlers:
        h.setFormatter(formatter)
