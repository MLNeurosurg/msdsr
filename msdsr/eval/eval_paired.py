"""Generate 2D paired evaluation metrics.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from os.path import join as opj
import gzip
import torch
import pytorch_lightning as pl
import einops

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure

from msdsr.ldm.models.diffusion.ddpm import LatentDiffusion
from msdsr.data.common import setup_data
from msdsr.train.infra import (parse_args, setup_eval_paths, get_exp_name,
                               config_loggers)
from msdsr.eval.common import viz_rescale_hist


class PairedSREvalSystem(pl.LightningModule):
    """Lightning system for 2D inference."""

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

    @torch.inference_mode()
    def predict_step(self, batch, _):
        """Predict 2D high res given low res input.

        Args:
            batch: batch of images to super-resolve

        Returns:
            dict of:
                cond: conditioned rows
                pred: predicted rows
                path: path to batch
                gt: ground truth image
        """

        im = batch["image"].squeeze()
        low_res_im = im[:, :, ::self.scale_factor, :]
        total_len = low_res_im.shape[-1]

        alternate = torch.arange(total_len)
        imaged = alternate[0::self.scale_factor]
        to_pred = torch.stack([d for d in alternate if d not in imaged])

        x_mask = torch.tile(to_pred, (low_res_im.shape[0], 1))
        c_mask = torch.tile(imaged, (low_res_im.shape[0], 1))

        cond = [low_res_im, x_mask, c_mask]

        # Predict new rows between low-res ground truth rows
        out = self.ddpm_model.pred_2d([low_res_im, x_mask, c_mask],
                                      shape=(3, 48, 48)).detach().cpu()
        return {"cond": cond, "pred": out, "path": batch["path"][0], "gt": im}


def cal_fid(gt, pred, batch_size=256):
    """Calculate FID score.

    Args:
        gt: ground truth image
        pred: predicted image
        batch_size: batch size

    Returns:
        FID score for ground truth versus predicted image
    """
    fid = FrechetInceptionDistance(normalize=True).cuda()

    gt_batches = torch.split(gt, split_size_or_sections=batch_size)
    fake_batches = torch.split(pred, split_size_or_sections=batch_size)

    for f in fake_batches:
        fid.update(f, real=False)

    for g in gt_batches:
        fid.update(g, real=True)

    return fid.compute().item()


def cal_ssim(gt, pred, batch_size=256):
    """Calculate SSIM score.

    Args:
        gt: ground truth image
        pred: predicted image
        batch_size: batch size

    Returns:
        SSIM score for ground truth versus predicted image
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    gt_batches = torch.split(gt, split_size_or_sections=batch_size)
    fake_batches = torch.split(pred, split_size_or_sections=batch_size)

    scores = 0
    total = 0
    for g, f in zip(gt_batches, fake_batches):
        scores += ssim(f, g).item() * len(gt_batches)
        total += len(gt_batches)

    return scores / total


def report_metrics(val_pred):
    """Calculate metrics for predictions.

    Args:
        val_pred: ground truth and predicted 2D images

    Returns:
        dict of L1 scores, FID scores, and SSIM scores per image
    """
    gt = torch.cat([i["gt"] for i in val_pred])
    pred = torch.cat([i["pred"] for i in val_pred])

    gt = torch.stack([torch.tensor(viz_rescale_hist(d)) for d in gt])
    pred = torch.stack([torch.tensor(viz_rescale_hist(d)) for d in pred])

    gt = einops.rearrange(gt, "b h w c -> b c h w").cuda()
    pred = einops.rearrange(pred, "b h w c -> b c h w").cuda()

    fid_score = cal_fid(gt, pred)

    gt = gt.to(float) / 255
    pred = pred.to(float) / 255

    l1_score = torch.abs(gt - pred).mean().item()

    ssim_score = cal_ssim(gt, pred)

    return {"l1": l1_score, "fid": fid_score, "ssim": ssim_score}


def main():
    """Main driver for 2D paired evaluation."""
    # setting up infra
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config = setup_eval_paths(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    cp_config(cf_fd.name)
    config_loggers(exp_root)
    logging.info("generating predictions")

    # inference
    _, val_loader = setup_data(cf)

    model = PairedSREvalSystem(cf)
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

    # compute metrics
    print(report_metrics(val_pred))


if __name__ == "__main__":
    main()
