"""Common utils for evaluation.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

import numpy as np
from typing import List
import torch


def histogram_equalization_(im: np.ndarray, bits: int = 8):
    """Rescale one channel of image using histogram equalization.

    Bits is desired number of bits in output (number of bins in the
    histogram). It can be a decimal, and the number of bins is floored
    """
    # adapted from https://ianwitham.wordpress.com/tag/histogram-equalization/
    num_bins = np.floor(2**bits).astype(np.uint16)
    hist, bins = np.histogram(im.flatten(), num_bins)
    hist[0] = 0

    cdf = hist.cumsum()
    cdf = (num_bins - 1) * cdf / cdf[-1]  # normalize

    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return np.array(im2).reshape(im.shape)


def viz_rescale_hist(im: torch.Tensor,
                     bits: List[float] = [7.5, 8, 8]) -> np.ndarray:
    """Rescale color channels for more realistic image vizualiation.

    Args:
        im: input image
        bits: scaling factor per RGB channels

    Returns a scaled image according to bits.
    """
    im_numpy = im.transpose(0, -1).numpy()
    stacked = np.dstack([
        histogram_equalization_(im_numpy[..., 0], bits[0]).T,
        histogram_equalization_(im_numpy[..., 1], bits[1]).T,
        histogram_equalization_(im_numpy[..., 2], bits[2]).T,
    ])
    return stacked.astype(np.uint8)
