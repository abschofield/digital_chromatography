"""Latent interface used by the published spectral models."""

import torch


class IdentityLatentInterface(torch.nn.Module):
    @torch.no_grad()
    def encode(self, x):
        return x

    @torch.no_grad()
    def decode(self, x):
        return x

    @torch.no_grad()
    def quantize(self, x):
        return x

    @torch.no_grad()
    def dequantize(self, x):
        return x
