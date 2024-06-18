"""Training script for MSDSR.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
import torch
import pytorch_lightning as pl

from msdsr.train.common import scale_lr, get_num_it_per_ep
from msdsr.train.infra import (setup_output_dirs, parse_args, get_exp_name,
                               config_loggers)
from msdsr.data.common import setup_data
from msdsr.ldm.models.diffusion.ddpm import LatentDiffusion


def main():
    # setup infrastructure
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])
    torch.set_float32_matmul_precision('medium')

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # setup dataloaders
    train_loader, val_loader = setup_data(cf)
    training_params = get_num_it_per_ep(train_loader, cf)
    logging.info(training_params)

    # setup lightning module
    lm = LatentDiffusion(learning_rate=scale_lr(cf, training_params),
                         **cf["model"]["params"])

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv"),
    ]

    # config callbacks
    epoch_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=-1,
        every_n_epochs=cf["training"]["eval_ckpt_ep_freq"],
        filename="ckpt-epoch{epoch}",
        auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    if not lm.automatic_optimization:
        train_cf = {}
    else:
        train_cf = {
            "gradient_clip_val":
            0.5,
            "accumulate_grad_batches":
            cf["training"].get("accumulate_grad_batches", 1),
        }

    trainer = pl.Trainer(
        accelerator="gpu",
        default_root_dir=exp_root,
        enable_progress_bar=True,
        strategy="ddp",
        logger=logger,
        log_every_n_steps=10,
        callbacks=[epoch_ckpt, lr_monitor],
        max_epochs=cf["training"]["num_epochs"],
        check_val_every_n_epoch=cf["training"]["eval_ckpt_ep_freq"],
        devices=cf["infra"]["devices"],
        num_nodes=cf["infra"]["num_nodes"],
        **train_cf)

    trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
