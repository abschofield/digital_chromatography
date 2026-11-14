import hydra
import torch
import pytorch_lightning as pl


class DiffusionSeparator(pl.LightningModule):
    """Diffusion Model for Source Separation"""
    def __init__(self,
                 latent_interface,
                 conditioner,
                 diffusion,
                 optimizer,
                 lr_scheduler=None,
                 lr_scheduler_monitor=None,
                 augmentation=None,
                 ema=None,
                 skip_nan_loss=False,
                 n_signal=596136,
                 num_oversample=1,
                 target_stem=0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.augment = hydra.utils.instantiate(augmentation)
        self.latent = hydra.utils.instantiate(latent_interface)
        self.conditioner = hydra.utils.instantiate(conditioner, latent=self.latent)
        self.diffusion = hydra.utils.instantiate(diffusion)
        self.diffusion_ema = hydra.utils.instantiate(ema,
                                                     self.diffusion,
                                                     include_online_model=False)
        self.skip_nan_loss = skip_nan_loss
        self.num_oversample = num_oversample
        self.target_stem = target_stem
        self.norm_floor = 1e-5

    ### Training
    def check_nan_in_loss(self, loss):
        # Skip/Stop on nan
        has_nan_loss = torch.isnan(loss).any().float()
        has_nan_loss = self.trainer.strategy.reduce(has_nan_loss,
                                                    reduce_op='sum') > 0
        if has_nan_loss:
            print('#### NAN LOSS DETECTED. Stopping training...')
            self.trainer.should_stop = has_nan_loss
        return loss

    @torch.autocast('cuda')
    def _step(self, batch, diffusion=None):
        mix = batch[:, 0, ...]
        stems = batch[:, 1:]
        target = stems[:, self.target_stem, ...]
        conditioners, conditioners_log = self.conditioner.training_step(mix, target=target)
        conditioners.pop('target', target)
        target = conditioners.pop('target') if conditioners.get('target', None) is not None else target
        target = self.latent.encode(target) # Encode target sample
        diff_loss = self.diffusion(x=target, **conditioners) # Diffusion loss on target sample
        cond_loss = conditioners_log.get('loss', torch.tensor(0.0).to(self.device))
        loss = diff_loss + cond_loss # Sum aux losses from conditioner
        return loss, {'diff_loss': diff_loss, **{
            'cond_' + k: v for k, v in conditioners_log.items()}}

    def training_step(self, batch, batch_idx):
        loss, losses_to_log = self._step(batch, self.diffusion)
        if not self.skip_nan_loss:
            loss = self.check_nan_in_loss(loss)
        loss = torch.nan_to_num(loss)
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()
            self.log('ema_decay', self.diffusion_ema.get_current_decay())
        self.log('train/loss', loss.detach(), prog_bar=True)
        for key, val in losses_to_log.items():
            self.log('train/%s' % key, val.detach(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses_to_log = self._step(batch, self.diffusion)
        self.log('val/loss', loss.detach())
        for key, val in losses_to_log.items():
            self.log('val/%s' % key, val.detach())
            
        if self.diffusion_ema is not None:
            loss_ema, extra_outputs = self._step(batch, self.diffusion_ema.ema_model)
            self.log('val/ema/loss', loss_ema, prog_bar=True)
            for key, val in extra_outputs.items():
                self.log('val/ema/%d' + key, val, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=filter(lambda p: p.requires_grad, self.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler,
            optimizer=optimizer
        )
        if lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.hparams.lr_scheduler_monitor,
                "interval": "epoch",
                "frequency": 1
            },
        }

    ### Inference
    @property
    def auto_select_diffusion(self):
        diffusion = self.diffusion
        if self.diffusion_ema is not None:
            diffusion = self.diffusion_ema.ema_model
        return diffusion

    def norm(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (self.norm_floor + std), mean, std

    def denorm(self, y, mean, std):
        return y * std + mean

    @torch.no_grad()
    @torch.autocast('cuda')
    def sample(self,
               mix,
               target=None,
               norm=False,
               num_oversample=None,
               **kwargs):
        num_steps = kwargs.get("num_steps")
        if num_steps is None:
            raise ValueError(
                "num_steps must be provided explicitly to DiffusionSeparator.sample(); no default is used."
            )
        if norm:
            mix, mean, std = self.norm(mix)
        b_size, a_channels, n_samples = mix.shape
        conditioners = self.conditioner.encode(mix, target) # Get conditioners
        target_for_size = self.latent.encode(mix) # Encode mix and get size
        y = torch.zeros([b_size, a_channels, n_samples], device=mix.device) # Placeholder for output
        num_oversample = self.num_oversample if num_oversample is None else num_oversample
        for _ in range(num_oversample):
            noise = torch.randn_like(target_for_size) # Input noise
            if mix in conditioners:
                mix = conditioners.pop('mix') # Remove mix from conditioners
            v = self.auto_select_diffusion.sample(noise, **conditioners, **kwargs) # Sample!
            y += self.latent.decode(v)[..., :n_samples] # Decode generated latents (if so)
            y = torch.nan_to_num(y, nan=0.0) # Setting to silence NaN values
        y = (y / num_oversample)
        if norm:
            y = self.denorm(y, mean, std)
        return y.squeeze(1), mix.squeeze(1)

    def forward(self, x, sample_kwargs=None):
        if sample_kwargs is None:
            sample_kwargs = {}
        if sample_kwargs.get("num_steps") is None:
            raise ValueError(
                "sample_kwargs['num_steps'] must be provided explicitly to DiffusionSeparator.forward(); no default is used."
            )
        sample_kwargs = {
            "show_progress": False,
            "num_oversample": 1,
            **sample_kwargs,
        }
        return self.sample(
            mix=x,
            **sample_kwargs
        )[0].unsqueeze(1)
