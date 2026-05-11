from typing import Any, Callable, Dict, Optional, Tuple

import torch
from diffusers import EulerAncestralDiscreteScheduler
from tqdm import tqdm

from utils.schedule import get_ancestral_step
from .base import SamplerBase
from components.samplers import register_sampler


@register_sampler("euler")
class EulerSampler(SamplerBase):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        sigmas = self._get_sigmas()
        zt = self.random_latent_kdiffusion(latent_shape, sigmas[0].item()).to(torch.float16)

        for step in tqdm(range(len(sigmas) - 1), desc=f"euler [{self.schedule}]"):
            sigma = sigmas[step]
            t = self.model.timestep(sigma)
            with torch.no_grad():
                z0t, _ = self.model.kdiffusion_zt_to_denoised(zt, sigma, uc, c, cfg_guidance, t, model_kwargs)
            d  = self.model.to_d(zt, sigma, z0t)
            zt = z0t + d * sigmas[step + 1]
            if callback_fn:
                kw = callback_fn(step, t, {"z0t": z0t.detach(), "zt": zt.detach(), "decode": self.model.decode})
                zt, z0t = kw["zt"], kw["z0t"]

        return z0t


@register_sampler("euler_a")
class EulerASampler(SamplerBase):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        sigmas = self._get_sigmas()
        zt = self.random_latent_kdiffusion(latent_shape, sigmas[0].item()).to(torch.float16)

        for step in tqdm(range(len(sigmas) - 1), desc=f"euler_a [{self.schedule}]"):
            sigma = sigmas[step]
            t = self.model.timestep(sigma)
            sigma_down, sigma_up = get_ancestral_step(sigma, sigmas[step + 1])
            with torch.no_grad():
                z0t, _ = self.model.kdiffusion_zt_to_denoised(zt, sigma, uc, c, cfg_guidance, t, model_kwargs)
            d  = self.model.to_d(zt, sigma, z0t)
            zt = z0t + d * sigma_down
            if sigmas[step + 1] > 0:
                zt = zt + torch.randn_like(zt) * sigma_up
            if callback_fn:
                kw = callback_fn(step, t, {"z0t": z0t.detach(), "zt": zt.detach(), "decode": self.model.decode})
                zt, z0t = kw["zt"], kw["z0t"]

        return z0t


@register_sampler("euler_a_native")
class EulerANativeSampler(SamplerBase):
    """
    Uses diffusers EulerAncestralDiscreteScheduler.step() directly.
    v→x0 conversion is handled inside the scheduler — matches diffusers pipeline exactly.
    Schedule type is ignored (uses diffusers' own timestep spacing).
    """
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        sched_cfg = self.model.scheduler_config
        if sched_cfg is None:
            raise ValueError("euler_a_native requires model.scheduler_config (SDXLModel)")

        native_sched = EulerAncestralDiscreteScheduler.from_config(
            sched_cfg, prediction_type=self.model.prediction_type
        )
        native_sched.set_timesteps(len(self.model.timesteps), device=self.model.device)

        zt = torch.randn(latent_shape, device=self.model.device, dtype=torch.float16)
        zt = zt * native_sched.init_noise_sigma

        for step, t in enumerate(tqdm(native_sched.timesteps, desc="euler_a_native")):
            zt_input = native_sched.scale_model_input(zt, t)
            with torch.no_grad():
                out = self.model.predict_raw(zt_input, t, uc, c, model_kwargs)
                noise_pred = out.noise_uc + cfg_guidance * (out.noise_c - out.noise_uc)
            zt = native_sched.step(noise_pred, t, zt).prev_sample

        return zt


@register_sampler("euler_cfg++")
class EulerCFGppSampler(SamplerBase):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        sigmas = self._get_sigmas()
        zt = self.random_latent_kdiffusion(latent_shape, sigmas[0].item()).to(torch.float16)

        for step in tqdm(range(len(sigmas) - 1), desc=f"euler_cfg++ [{self.schedule}]"):
            sigma = sigmas[step]
            t = self.model.timestep(sigma)
            with torch.no_grad():
                z0t, z0t_unc = self.model.kdiffusion_zt_to_denoised(zt, sigma, uc, c, cfg_guidance, t, model_kwargs)
            d  = self.model.to_d(zt, sigma, z0t_unc)   # CFG++: move along uncond direction
            zt = z0t + d * sigmas[step + 1]
            if callback_fn:
                kw = callback_fn(step, t, {"z0t": z0t.detach(), "zt": zt.detach(), "decode": self.model.decode})
                zt, z0t = kw["zt"], kw["z0t"]

        return z0t
