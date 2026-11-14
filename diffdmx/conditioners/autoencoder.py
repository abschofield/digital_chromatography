import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence

from diffdmx.conditioners.base import BaseConditioner
from diffdmx.a_unet.blocks import Packed, Attention


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, num_groups: int = 8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        groups = min(num_groups, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConditionerAutoencoder(BaseConditioner):
    """
    Autoencoder-style conditioner with residual blocks and optional attention.
    Produces multi-resolution embeddings for generator injection.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        factors: Sequence[int],
        context_channels: Sequence[int],
        attentions: Optional[Sequence[int]] = None,
        attention_heads: int = 8,
        attention_features: int = 64,
        num_groups: int = 8,
        kernel_size: int = 3,
        latent=None,
        **kwargs,
    ):
        super().__init__(latent=latent, **kwargs)
        self.in_channels = in_channels
        self.channels = list(channels)
        self.factors = list(factors)
        self.context_channels = list(context_channels)
        self.attentions = list(attentions) if attentions is not None else [0] * len(self.channels)
        self.attention_heads = attention_heads
        self.attention_features = attention_features
        self.num_groups = num_groups
        self.kernel_size = kernel_size

        if not (len(self.channels) == len(self.factors) == len(self.context_channels) == len(self.attentions)):
            raise ValueError("channels, factors, context_channels, attentions must have same length")

        self.input_proj = nn.Conv1d(in_channels, self.channels[0], kernel_size=1)

        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.gate_init = -2.0

        num_stages = len(self.channels)
        for idx, ch in enumerate(self.channels):
            next_ch = self.channels[idx + 1] if idx + 1 < num_stages else ch
            self.enc_blocks.append(ResidualBlock(ch, kernel_size=self.kernel_size, num_groups=self.num_groups))
            if self.attentions[idx]:
                self.enc_attn.append(
                    Packed(
                        Attention(
                            features=ch,
                            head_features=self.attention_features,
                            num_heads=self.attention_heads,
                        )
                    )
                )
            else:
                self.enc_attn.append(nn.Identity())

            factor = self.factors[idx]
            if factor > 1:
                self.downsamples.append(
                    nn.Conv1d(ch, next_ch, kernel_size=factor, stride=factor, padding=0)
                )
            elif next_ch != ch:
                self.downsamples.append(nn.Conv1d(ch, next_ch, kernel_size=1))
            else:
                self.downsamples.append(nn.Identity())

        bottleneck_ch = self.channels[-1]
        self.aux_projs = nn.ModuleList()
        self.aux_gates = nn.ParameterList()
        for ch in self.channels:
            self.aux_projs.append(
                nn.Sequential(
                    nn.GroupNorm(min(self.num_groups, ch), ch),
                    nn.SiLU(),
                    nn.Conv1d(ch, bottleneck_ch, kernel_size=1),
                )
            )
            self.aux_gates.append(nn.Parameter(torch.tensor(self.gate_init)))

        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(self.channels))
        for idx, ch in enumerate(rev_channels):
            next_ch = rev_channels[idx + 1] if idx + 1 < len(rev_channels) else ch
            self.dec_blocks.append(ResidualBlock(ch, kernel_size=self.kernel_size, num_groups=self.num_groups))
            if self.attentions[len(self.channels) - idx - 1]:
                self.dec_attn.append(
                    Packed(
                        Attention(
                            features=ch,
                            head_features=self.attention_features,
                            num_heads=self.attention_heads,
                        )
                    )
                )
            else:
                self.dec_attn.append(nn.Identity())

            factor = self.factors[len(self.channels) - idx - 1]
            if factor > 1:
                self.upsamples.append(
                    nn.ConvTranspose1d(ch, next_ch, kernel_size=factor, stride=factor, padding=0)
                )
            elif next_ch != ch:
                self.upsamples.append(nn.Conv1d(ch, next_ch, kernel_size=1))
            else:
                self.upsamples.append(nn.Identity())

        self.context_projs = nn.ModuleList()
        self.context_gates = nn.ParameterList()
        for idx, ctx_ch in enumerate(self.context_channels):
            if ctx_ch > 0:
                ch = self.channels[idx]
                self.context_projs.append(
                    nn.Sequential(
                        nn.GroupNorm(min(self.num_groups, ch), ch),
                        nn.SiLU(),
                        nn.Conv1d(ch, ctx_ch, kernel_size=1),
                    )
                )
                self.context_gates.append(nn.Parameter(torch.tensor(self.gate_init)))
            else:
                self.context_projs.append(nn.Identity())
                self.context_gates.append(nn.Parameter(torch.tensor(self.gate_init)))

    def _gate_value(self, gate: nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(gate)

    def _encode(self, mix: torch.Tensor) -> List[torch.Tensor]:
        x = self.latent.encode(mix)
        x = self.input_proj(x)
        enc_outputs: List[torch.Tensor] = []

        for block, attn, down in zip(self.enc_blocks, self.enc_attn, self.downsamples):
            x = block(x)
            x = attn(x)
            enc_outputs.append(x)
            x = down(x)
        return enc_outputs

    def _bottleneck(self, enc_outputs: List[torch.Tensor]) -> torch.Tensor:
        x = enc_outputs[-1]
        target_len = x.shape[-1]
        for out, proj, gate in zip(enc_outputs, self.aux_projs, self.aux_gates):
            aux = proj(out)
            if aux.shape[-1] != target_len:
                aux = F.interpolate(aux, size=target_len, mode="linear", align_corners=False)
            x = x + self._gate_value(gate) * aux
        return x

    def _decode(self, bottleneck: torch.Tensor) -> List[torch.Tensor]:
        x = bottleneck
        dec_embeddings: List[Optional[torch.Tensor]] = [None] * len(self.channels)

        for idx, (block, attn, up) in enumerate(zip(self.dec_blocks, self.dec_attn, self.upsamples)):
            depth = len(self.channels) - idx - 1
            x = block(x)
            x = attn(x)
            dec_embeddings[depth] = x
            x = up(x)

        return dec_embeddings  # type: ignore

    def encode(self, mix, target=None):
        enc_outputs = self._encode(mix)
        bottleneck = self._bottleneck(enc_outputs)
        dec_embeddings = self._decode(bottleneck)

        channels: List[Optional[torch.Tensor]] = []
        for idx, ctx_ch in enumerate(self.context_channels):
            if ctx_ch <= 0 or dec_embeddings[idx] is None:
                channels.append(None)
                continue
            context = self.context_projs[idx](dec_embeddings[idx])
            channels.append(self._gate_value(self.context_gates[idx]) * context)

        return {"channels": channels}

    def training_step(self, x, target=None):
        return self.encode(x, target), {}
