from typing import Any, Callable, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from .base import SamplerBase
from components.samplers import register_sampler


@register_sampler("ddim")
class DDIMSampler(SamplerBase):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        zt = self.random_latent(latent_shape)

        for step, t in enumerate(tqdm(self.model.timesteps.int(), desc="ddim")):
            at = self.model.alpha(t)
            at_next = self.model.alpha(t - self.model.skip)
            with torch.no_grad():
                out = self.model.predict_noise(zt, t, uc, c, model_kwargs)
                noise_pred = out.noise_uc + cfg_guidance * (out.noise_c - out.noise_uc)
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            zt  = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_pred
            zt, z0t = self._callback(step, t, zt, z0t, callback_fn)

        return z0t

    def _callback(self, step, t, zt, z0t, callback_fn):
        if callback_fn is None:
            return zt, z0t
        kw = callback_fn(step, t, {"z0t": z0t.detach(), "zt": zt.detach(), "decode": self.model.decode})
        return kw["zt"], kw["z0t"]


@register_sampler("ddim_cfg++")
class DDIMCFGppSampler(SamplerBase):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        zt = self.random_latent(latent_shape)

        for step, t in enumerate(tqdm(self.model.timesteps.int(), desc="ddim_cfg++")):
            at = self.model.alpha(t)
            at_next = self.model.alpha(t - self.model.skip)
            with torch.no_grad():
                out = self.model.predict_noise(zt, t, uc, c, model_kwargs)
                noise_pred = out.noise_uc + cfg_guidance * (out.noise_c - out.noise_uc)
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            zt  = at_next.sqrt() * z0t + (1 - at_next).sqrt() * out.noise_uc  # CFG++ key
            if callback_fn:
                kw = callback_fn(step, t, {"z0t": z0t.detach(), "zt": zt.detach(), "decode": self.model.decode})
                zt, z0t = kw["zt"], kw["z0t"]

        return z0t


@register_sampler("ddim_inversion")
class DDIMInversionSampler(DDIMSampler):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        assert src_latent is not None, "ddim_inversion requires src_latent"
        zt = self.inversion(src_latent, uc, c, cfg_guidance, model_kwargs)
        return super().sample(uc, c, cfg_guidance, latent_shape, model_kwargs, zt, callback_fn)
