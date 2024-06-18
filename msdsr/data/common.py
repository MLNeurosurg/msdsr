"""Data processing utilities designed to work with SRH datasets.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

from torch.utils.data import DataLoader, RandomSampler

from opensrh.datasets.srh_dataset import SRHClassificationDataset

from msdsr.data.zsrh_dataset import ZSRHDataset
from msdsr.data.db_improc import get_srh_augs, get_srh_base_aug, get_zsrh_base_aug


def setup_data(cf):
    """Setup dataloaders for ingest.

    Args:
        cf: config parameters from yaml

    Returns:
        train_loader: training dataloader
        val_loader: validation dataloader
    """

    # Load training dataloader
    if ("train" in cf["data"]["direct_params"]) and (
            "train" in cf["loader"]["direct_params"]):
        train_dset = SRHClassificationDataset(
            transform=get_srh_augs(get_srh_base_aug,
                                   cf["data"]["train_augmentation"],
                                   cf["data"]["rand_aug_prob"]),
            **cf["data"]["direct_params"]["train"])

        train_loader = DataLoader(train_dset,
                                  **cf["loader"]["direct_params"]["train"])
    else:
        train_loader = None

    # Load validation dataloader
    if ("val" in cf["data"]["direct_params"]) and (
            "val" in cf["loader"]["direct_params"]):
        val_dset = SRHClassificationDataset(
            transform=get_srh_augs(get_srh_base_aug,
                                   cf["data"]["valid_augmentation"],
                                   cf["data"]["rand_aug_prob"]),
            **cf["data"]["direct_params"]["val"])

        if "val_sampler" in cf["loader"]:
            sampler_params = {
                "sampler": RandomSampler(val_dset,
                                         **cf["loader"]["val_sampler"])
            }
        else:
            sampler_params = {}

        val_loader = DataLoader(val_dset,
                                **cf["loader"]["direct_params"]["val"],
                                **sampler_params)
    else:
        val_loader = None

    return train_loader, val_loader


def setup_3d_data(cf):
    """Create validation dataloader for evaluation.

    Args:
        cf: config parameters from yaml

    Returns:
        val_loader: validation dataloader
    """

    val_dset = ZSRHDataset(transform=get_srh_augs(
        get_zsrh_base_aug, cf["data"]["valid_augmentation"],
        cf["data"]["rand_aug_prob"]),
                           **cf["data"]["direct_params"]["val"])

    val_loader = DataLoader(val_dset, **cf["loader"]["direct_params"]["val"])

    return val_loader
