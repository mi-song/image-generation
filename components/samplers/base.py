from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
from tqdm import tqdm

from components.models.base import ModelBase
from utils.schedule import get_sigmas_karras, get_sigmas_exponential, get_sigmas_ays

ScheduleType = Literal["linear", "karras", "exponential", "ays"]


class SamplerBase(ABC):
    """
    Receives a ModelBase and operates only through its interface.
    Responsible for latent → latent transformation.
    VAE encode/decode is the Pipeline's responsibility.
    """

    def __init__(self, model: ModelBase, schedule: ScheduleType = "linear"):
        self.model = model
        self.schedule = schedule

    @abstractmethod
    def sample(
        self,
        uc: torch.Tensor,
        c: torch.Tensor,
        cfg_guidance: float,
        latent_shape: Tuple[int, int, int, int],   # (B, C, H_lat, W_lat)
        model_kwargs: Optional[Dict[str, Any]] = None,
        src_latent: Optional[torch.Tensor] = None,
        callback_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Returns a denoised latent tensor."""
        ...

    # ── latent initialization ─────────────────────────────────────────────────

    def random_latent(self, latent_shape: Tuple) -> torch.Tensor:
        return torch.randn(latent_shape, device=self.model.device)

    def random_latent_kdiffusion(self, latent_shape: Tuple, sigma_max: float) -> torch.Tensor:
        z = torch.randn(latent_shape, device=self.model.device)
        return z * (sigma_max ** 2 + 1) ** 0.5

    # ── DDIM inversion ────────────────────────────────────────────────────────

    @torch.no_grad()
    def inversion(
        self,
        z0: torch.Tensor,
        uc: torch.Tensor,
        c: torch.Tensor,
        cfg_guidance: float,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        zt = z0.clone().to(self.model.device)
        for _, t in enumerate(tqdm(reversed(self.model.timesteps), desc="DDIM inversion")):
            at = self.model.alpha(t)
            at_prev = self.model.alpha(t - self.model.skip)
            out = self.model.predict_noise(zt, t, uc, c, model_kwargs)
            noise_pred = out.noise_uc + cfg_guidance * (out.noise_c - out.noise_uc)
            z0t = (zt - (1 - at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1 - at).sqrt() * noise_pred
        return zt

    # ── sigma schedule ────────────────────────────────────────────────────────

    def _native_sigmas(self) -> torch.Tensor:
        ts = self.model.timesteps.cpu().long()
        alphas = self.model.total_alphas[ts]
        sigmas = ((1 - alphas) / alphas).sqrt()
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def _get_sigmas(self) -> torch.Tensor:
        """Return sigma schedule according to self.schedule."""
        base = self._native_sigmas()
        sigma_max, sigma_min = base[0], base[-2]  # last real sigma before the appended 0
        n = len(self.model.timesteps)

        if self.schedule == "karras":
            return get_sigmas_karras(n, sigma_min, sigma_max, device=self.model.device)
        elif self.schedule == "exponential":
            return get_sigmas_exponential(n, sigma_min, sigma_max, device=self.model.device)
        elif self.schedule == "ays":
            is_sdxl = hasattr(self.model, "tokenizer_2")
            return get_sigmas_ays(n, self.model.total_alphas, is_sdxl=is_sdxl, device=self.model.device)
        else:  # "linear"
            return base

    # ── img2img (hires fix second pass) ──────────────────────────────────────

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def img2img(
        self,
        src_latent: torch.Tensor,
        strength: float,
        uc: torch.Tensor,
        c: torch.Tensor,
        cfg_guidance: float,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        VE-style img2img. Adds sigma noise to src_latent and denoises the tail.
        strength=1.0 → full denoising (= txt2img), strength=0.5 → half the steps.
        """
        sigmas = self._get_sigmas()
        n = len(sigmas) - 1
        start_step = max(0, int(n * (1.0 - strength)))
        sigma_start = sigmas[start_step]

        zt = src_latent.to(torch.float16) + sigma_start * torch.randn_like(src_latent)

        for step in tqdm(range(start_step, n), desc="hires fix"):
            sigma = sigmas[step]
            t = self.model.timestep(sigma)
            with torch.no_grad():
                z0t, _ = self.model.kdiffusion_zt_to_denoised(
                    zt, sigma, uc, c, cfg_guidance, t, model_kwargs
                )
            d  = self.model.to_d(zt, sigma, z0t)
            zt = z0t + d * sigmas[step + 1]

        return z0t
