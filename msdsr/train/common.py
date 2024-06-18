"""Common utilities for training.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

import logging
from typing import Dict

import torch


def get_num_it_per_ep(train_loader, cf=None):
    """Return training relevant information.

    Returns number of iterations per epoch, number available gpus,
    and whether gradient accumulation is enabled.
    """
    if cf:
        numgpu = (cf["infra"]["devices"] * cf["infra"]["num_nodes"])
        grad_accum = cf["training"].get("accumulate_grad_batches", 1)
        logging.info(f"num gpus total: {numgpu}")
    else:
        numgpu = torch.cuda.device_count()
        logging.warning(f"num gpus (fallback) total: {numgpu}")
        grad_accum = 1

    numgpu = max(1, numgpu)
    return {
        "num_it_per_ep": len(train_loader) // numgpu // grad_accum,
        "num_gpu": numgpu,
        "grad_accum": grad_accum
    }


def scale_lr(config: Dict, training_params: Dict):
    """Scale learning rate based off config."""
    bs = config["loader"]["direct_params"]["train"]["batch_size"]
    base_lr = config["training"]["base_learning_rate"]
    accum_grad_batches = training_params["grad_accum"]
    ngpu = training_params["num_gpu"]

    if config["training"]["scale_lr"]:
        lr = accum_grad_batches * ngpu * bs * base_lr / 256.0
        logging.info(f"Computed learning rate to {lr} = " +
                     f"{accum_grad_batches} (accumulate_grad_batches) * " +
                     f"{ngpu} (num_gpus) * " + f"{bs} (batchsize) * " +
                     f"{base_lr} (base_lr) / 256")
        return lr
    else:
        logging.info(f"Base learning rate = {base_lr:.2e}")
        return base_lr


def convert_epoch_to_iter(unit: str, steps: int, num_it_per_ep: int) -> int:
    """Converts number of epochs / iterations to number of iterations."""
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")
