"""Evaluate 3D super-resolved volumes.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

from os.path import join as opj
import logging
import gzip
import yaml

import numpy as np
import scipy

import torch
import einops
import torchmetrics

from msdsr.train.infra import parse_args
from msdsr.eval.common import viz_rescale_hist


def setup_logging():
    """Setup logger."""
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Metrics log")


def read_data(cf):
    """Load predicted volumes for evaluation.

    Args:
        cf: config parameters from yaml

    Returns:
        pred_first: 3D prediction in initial orientation
        pred_transpose: 3D prediction in transposed orientation
        pred_avg: average of pred_first and pred_transpose
        cond: conditioned slices in volume
        gt2d: 2D ground truth
    """
    eval_root = opj(cf["log_dir"], cf["exp_name"], cf["train_exp"], "evals")

    # load first prediction
    first_pred_fname = opj(eval_root, cf["pred_eval"]["first"],
                           "predictions/val_predictions.pt.gz")

    with gzip.open(first_pred_fname) as fd:
        first_data = torch.load(fd, map_location="cpu")
    logging.info(f"OK - {first_pred_fname}")

    pred_first = torch.cat([
        einops.rearrange(torch.stack(d["pred"]), "w b c z h -> b c z h w")
        for d in first_data
    ])

    cond = torch.cat([
        einops.rearrange(torch.stack([c[0] for c in d["cond"]]),
                         "w b c z h -> b c z h w") for d in first_data
    ])

    # load transposed prediction
    xpose_pred_fname = opj(eval_root, cf["pred_eval"]["xpose"],
                           "predictions/val_predictions.pt.gz")
    with gzip.open(xpose_pred_fname) as fd:
        transposed_data = torch.load(fd, map_location="cpu")
    logging.info(f"OK - {xpose_pred_fname}")

    pred_transpose = torch.cat([
        einops.rearrange(torch.stack(d["pred"]), "h b c z w -> b c z h w")
        for d in transposed_data
    ])
    pred_avg = (pred_first + pred_transpose) / 2

    # load 2d groundtruth
    gt2d_fname = opj(eval_root, cf["pred_eval"]["2d_gt"],
                     "predictions/val_predictions.pt.gz")
    with gzip.open(gt2d_fname) as fd:
        left = torch.load(fd)
    logging.info(f"OK - {gt2d_fname}")
    gt2d = torch.cat([i["gt"] for i in left])

    return pred_first, pred_transpose, pred_avg, cond, gt2d


def build_simple_gaussian_kernel(kernel_size=3, sigma=0.5):
    """Generate 3D gaussian kernel for blurring.

    Args:
        kernel_size: kernel size (odd value)
        sigma: sigma value for gaussian

    Returns:
        A 3D gaussian kernel.
    """
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
    y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
    z = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def blur_volumes(im_in):
    """Blur volumes with gaussian kernel.

    Args:
        im_in: image input

    Returns:
        Blurred image with a 3D gaussian kernel.
    """
    kernel = build_simple_gaussian_kernel()
    return torch.tensor(
        np.stack([
            np.stack([scipy.ndimage.convolve(j, kernel) for j in i])
            for i in im_in
        ]))


@torch.no_grad()
def get_metrics(gt_data, fake_data, batch_size=512):
    """Calculate SliceFID scores for generated volumes.

    Args:
        gt_data: ground truth 2D image
        fake_data: sliced super-resolved volume
        batch_size: batch size (for grouping)

    Returns:
        SliceFID score for super-resolved volume versus ground truth.
    """
    fake_data = einops.rearrange(
        torch.stack([torch.tensor(viz_rescale_hist(d)) for d in fake_data]),
        "b h w c -> b c h w")

    gt_data = einops.rearrange(
        torch.stack([torch.tensor(viz_rescale_hist(d)) for d in gt_data]),
        "b h w c -> b c h w")

    gt_batches = torch.split(gt_data, split_size_or_sections=batch_size)
    fake_batches = torch.split(fake_data, split_size_or_sections=batch_size)

    torch.cuda.empty_cache()
    fid = torchmetrics.image.fid.FrechetInceptionDistance().cuda()

    for f in fake_batches:
        fid.update(f.cuda(), real=False)

    for g in gt_batches:
        fid.update(g.cuda(), real=True)

    return fid.compute().item()


def main():
    """Main driver for 3D volume metric evaluation."""
    # setup infra
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    setup_logging()

    scale_factor = cf["scale_factor"]
    logging.info("scale_factor = %d", scale_factor)

    out_first, out_transpose, out_avg, cond, gt = read_data(cf)

    # crop out the slices above and below first observation
    crop_first = out_first[:, :, 4:-4, 4:-4, 4:-4]
    crop_xpose = out_transpose[:, :, 4:-4, 4:-4, 4:-4]
    crop_avg = out_avg[:, :, 4:-4, 4:-4, 4:-4]
    crop_cond = cond[..., 4:-4, 4:-4]

    # perform linear and linear interpolation
    crop_lin = torch.nn.functional.interpolate(crop_cond,
                                               scale_factor=(4, 1, 1),
                                               mode="trilinear")
    crop_nn = torch.nn.functional.interpolate(crop_cond,
                                              scale_factor=(4, 1, 1),
                                              mode="nearest")

    if cf["do_blur"]:
        crop_first = blur_volumes(crop_first)
        crop_xpose = blur_volumes(crop_xpose)
        crop_avg = blur_volumes(crop_avg)
        crop_nn = blur_volumes(crop_nn)
        crop_lin = blur_volumes(crop_lin)

    # Get XY metrics for all 5 models
    metrics = {}
    nn_xy = einops.rearrange(crop_nn, "b c z h w -> (b z) c h w")
    metrics["nn_xy_fid"] = get_metrics(gt, nn_xy)
    lin_xy = einops.rearrange(crop_lin, "b c z h w -> (b z) c h w")
    metrics["lin_xy_fid"] = get_metrics(gt, lin_xy)
    first_xy = einops.rearrange(crop_first, "b c z h w -> (b z) c h w")
    metrics["first_xy_fid"] = get_metrics(gt, first_xy)
    xpose_xy = einops.rearrange(crop_xpose, "b c z h w -> (b z) c h w")
    metrics["xpose_xy_fid"] = get_metrics(gt, xpose_xy)
    avg_xy = einops.rearrange(crop_avg, "b c z h w -> (b z) c h w")
    metrics["avg_xy_fid"] = get_metrics(gt, avg_xy)

    # Get YZ metrics for all 5 models
    nn_yz = einops.rearrange(crop_nn, "b c z h w -> (b h) c z w")
    metrics["nn_yz_fid"] = get_metrics(gt, nn_yz)
    lin_yz = einops.rearrange(crop_lin, "b c z h w -> (b h) c z w")
    metrics["lin_yz_fid"] = get_metrics(gt, lin_yz)
    first_yz = einops.rearrange(crop_first, "b c z h w -> (b h) c z w")
    metrics["first_yz_fid"] = get_metrics(gt, first_yz)
    xpose_yz = einops.rearrange(crop_xpose, "b c z h w -> (b h) c z w")
    metrics["xpose_yz_fid"] = get_metrics(gt, xpose_yz)
    avg_yz = einops.rearrange(crop_avg, "b c z h w -> (b h) c z w")
    metrics["avg_yz_fid"] = get_metrics(gt, avg_yz)

    # Get XZ metrics for all 5 models
    nn_xz = einops.rearrange(crop_nn, "b c z h w -> (b w) c h z")
    metrics["nn_xz_fid"] = get_metrics(gt, nn_xz)
    lin_xz = einops.rearrange(crop_lin, "b c z h w -> (b w) c h z")
    metrics["lin_xz_fid"] = get_metrics(gt, lin_xz)
    first_xz = einops.rearrange(crop_first, "b c z h w -> (b w) c h z")
    metrics["first_xz_fid"] = get_metrics(gt, first_xz)
    xpose_xz = einops.rearrange(crop_xpose, "b c z h w -> (b w) c h z")
    metrics["xpose_xz_fid"] = get_metrics(gt, xpose_xz)
    avg_xz = einops.rearrange(crop_avg, "b c z h w -> (b w) c h z")
    metrics["avg_xz_fid"] = get_metrics(gt, avg_xz)

    logging.info(metrics)


if __name__ == "__main__":
    main()
