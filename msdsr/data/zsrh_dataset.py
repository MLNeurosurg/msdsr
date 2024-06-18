"""PyTorch datasets for 3D SRH data.

Copyright (c) 2024 University of Michigan.
Licensed under the MIT License. See LICENSE for license information.
"""

from os.path import join as opj
from typing import List
import einops
import torch


class ZSRHDataset():
    """Z-stacked SRH volumetric dataset with PyTorch."""

    def __init__(self, data_root: str, studies: List[str],
                 transform: callable):
        """Inits the zSRH Dataset.

        Populate each attribute and load all slides in studies.

        Args:
            data_root: root zSRH directory
            studies: a list of pt files containing raw SRH images
            transform: a callable object for image transformation
        """
        self.data_root_ = data_root
        self.transform_ = transform

        # Walk through and load each study
        def load_one_slide(path):
            data = torch.load(opj(data_root, path))
            return einops.rearrange(data, "px py z c h w -> (px py) z c h w")

        self.instances_ = torch.cat([load_one_slide(s)
                                     for s in studies]).to(torch.float32)

    def __getitem__(self, idx: int):
        """Retrieve a patch specified by idx."""
        im = self.instances_[idx]

        if self.transform_ is not None:
            im = self.transform_(im)

        return {"image": im}

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.instances_)
