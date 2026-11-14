"""Input transform container used by the publication configurations."""

from __future__ import annotations

import torch


class Compose(torch.nn.Module):
    """Apply configured transforms sequentially during training."""

    def __init__(self, transform_list=None, transforms=None):
        super().__init__()
        transform_list = [] if transform_list is None else list(transform_list)
        transforms = {} if transforms is None else transforms
        self.transforms = torch.nn.Sequential(
            *transform_list,
            *[value for value in transforms.values()],
        )

    def forward(self, x):
        if self.training:
            return self.transforms(x)
        return x
