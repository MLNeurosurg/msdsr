"""Generate volumes for (unpaired) 3D super-resolution evaluation.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

import logging
from os.path import join as opj
import yaml
import gzip
import torch
import pytorch_lightning as pl
import einops

from msdsr.ldm.models.diffusion.ddpm import LatentDiffusion
from msdsr.data.common import setup_3d_data
from msdsr.train.infra import (parse_args, setup_eval_paths, get_exp_name,
                               config_loggers)


class Slice2VolumeEvalSystem(pl.LightningModule):
    """Lightning system for 3D evaluation."""

    def __init__(self, cf):
        """Inits evaluation system.

        Args:
            cf: config parameters from yaml
        """
        super().__init__()

        self.cf_ = cf

        ckpt_path = opj(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                        cf["eval"]["ckpt_path"])
        self.ddpm_model = LatentDiffusion(**cf["model"]["params"])
        self.ddpm_model.init_from_ckpt(ckpt_path)
        self.ddpm_model.eval()

        self.scale_factor = cf["eval"]["scale_factor"]
        self.transpose_pred = cf["eval"]["transpose_pred"]

    @torch.inference_mode()
    def predict_step(self, batch, _):
        """Super-resolve a batch of 3D volume.

        Args:
            batch: batch of volumes to super-resolve, cropped to 48x48x48.

        Returns:
            dict of conditioned layers and predicted layers per volume
        """
        out = []
        cond = []

        im = batch["image"].squeeze()

        if self.transpose_pred:
            im = einops.rearrange(im, "b z c h w -> b z c w h")

        for i in range(48):  # predict each row per volume in batch
            # get the condition (imaged rows)
            im_i = einops.rearrange(im[:, ::self.scale_factor // 2, :, i, :],
                                    "b z c h -> b c z h")

            # get index of the row condition and the rows to predict
            total_len, imaged_len = im_i.shape[-1], im_i.shape[-2]
            reserve = (total_len - imaged_len * self.scale_factor) // 2
            alternate = torch.arange(reserve, total_len - reserve)

            imaged = alternate[0::self.scale_factor]  # the condition

            to_pred = torch.stack([d for d in alternate if d not in imaged])
            to_pred = torch.cat((torch.arange(reserve), to_pred,
                                 torch.arange(total_len - reserve, total_len)))

            x_mask = torch.tile(to_pred, (im_i.shape[0], 1))
            c_mask = torch.tile(imaged, (im_i.shape[0], 1))
            cond.append([im_i, x_mask, c_mask])

            # predict interpolation for one layer in volume
            out.append(
                self.ddpm_model.pred_2d([im_i, x_mask, c_mask],
                                        shape=(3, 48, 48)).detach().cpu())

        return {"cond": cond, "pred": out}


def main():
    """Main driver for 3D volume generation."""
    # setting up infra
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config = setup_eval_paths(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get predictions
    logging.info("generating predictions")

    # prediction forward passes
    val_loader = setup_3d_data(cf)

    model = Slice2VolumeEvalSystem(cf)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         logger=False,
                         default_root_dir=exp_root,
                         inference_mode=True,
                         deterministic=True)

    val_pred = trainer.predict(model, dataloaders=val_loader)

    # save predictions
    val_pred_fname = opj(pred_dir, "val_predictions.pt.gz")
    with gzip.open(val_pred_fname, "w") as fd:
        torch.save(val_pred, fd)


if __name__ == "__main__":
    main()
