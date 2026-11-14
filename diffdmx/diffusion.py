import torch
import scipy

from torch import Tensor
from einops import repeat
from tqdm import tqdm
from typing import Optional

import numpy as np

from diffdmx.audio_diffusion_pytorch.diffusion import (
    VSampler,
    VDiffusion,
    extend_dim,
)



class VDiffusionL(VDiffusion):
    def __init__(
        self,
        *args,
        loss='mse',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.MSELoss()
        if loss == 'l1':
            self.loss = torch.nn.L1Loss()

    def forward(self, x, prior_cond = None, **kwargs):  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        if prior_cond is not None:
            noise = noise + prior_cond.mean(1, keepdim=True)
            noise = noise * prior_cond.std(1, keepdim=True)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return self.loss(v_pred, v_target)
    
    
class VDiffusionLL(VDiffusionL):
    def __init__(
        self,
        *args,
        loss=torch.nn.MSELoss(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss = loss


class VSamplerL(VSampler):
    def __init__(
        self,
        sample_rate=44100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate

    def forward(  # type: ignore
        self,
        x_noisy: Tensor,
        num_steps: int,
        show_progress: bool = False,
        eta: float = 0.0,
        eta_freq_range: Optional[float] = None,
        **kwargs
    ) -> Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        x_pred = x_noisy
        for i in progress_bar:
            with torch.no_grad():
                v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred

            if i < num_steps - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = eta * (betas[i + 1]**2 / betas[i]**2).sqrt() * \
                    (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (betas[i + 1]**2 - ddim_sigma**2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                x_noisy = x_pred * alphas[i + 1] + noise_pred * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    add_noise = torch.randn_like(x_noisy)
                    if eta_freq_range is not None:
                        add_noise = high_pass_filter(
                            add_noise,
                            self.sample_rate,
                            eta_freq_range)
                    x_noisy += add_noise.to(x_noisy.device) * ddim_sigma

                # Update progress bar
                progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")
        return x_pred

    
def high_pass_filter(signal, sample_rate, cutoff_freq):
    # Design a high-pass filter using scipy
    sos = scipy.signal.butter(1, cutoff_freq, btype='highpass', fs=sample_rate, output='sos')
    
    # Apply the filter for each channel separately
    filtered_signal_list = []
    for channel in signal.squeeze(0):
        filtered_signal_list.append(scipy.signal.sosfilt(sos, channel.cpu().numpy()))
    
    # Convert the list of numpy arrays to a single numpy array before converting to tensor
    filtered_signal = np.array(filtered_signal_list)
    
    # Convert back to torch tensor
    return torch.tensor(filtered_signal, dtype=signal.dtype).unsqueeze(0)
