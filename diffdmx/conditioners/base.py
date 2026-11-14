import torch
from typing import Sequence
from diffdmx.latents import IdentityLatentInterface


class BaseConditioner(torch.nn.Module):
    def __init__(self, latent=IdentityLatentInterface(), trainable=False, *args, **kwargs):
        super().__init__()
        self.trainable = trainable
        self.latent = latent
        
    def training_step(self, x, target=None):
        if self.trainable:
            raise NotImplementedError
        return {}, {}
        
    def encode(self, x, target=None):
        raise NotImplementedError
        
    def forward(self, x, target=None):
        if self.training and self.trainable:
            return self.training_step(x, target)
        return self.encode(x, target)
    
    
class EmptyConditioner(BaseConditioner):   
    def encode(self, x, target=None):
        return {}


class BaseAppendChannels(BaseConditioner):
    def encode(self, x, target=None):
        return {'append_channels': self.latent.encode(x)}
    
    
class BaseFeatures(BaseConditioner):
    def encode(self, x, target=None):
        return {'features': x}
    
    
class BaseInjectChannels(BaseConditioner):
    def __init__(self, context_channels: Sequence[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_channels = context_channels
        self.inject_depth = [
            idx for idx, chan in enumerate(context_channels) if chan>0
        ]

    def encode(self, x, target=None):
        x = self.latent.encode(x)
        channels = [None] * len(self.context_channels)
        for i in self.inject_depth:
            channels[i] = x
        return {'channels': channels}